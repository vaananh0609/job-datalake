import logging
import os
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("train_model")


# ===============================
# SPARK SESSION
# ===============================
def build_spark():
    spark = (
        SparkSession.builder
        .appName("SalaryPredictionTraining")

        # ===== AWS S3 SUPPORT =====
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4"
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )

        # ===== 🔥 FIX NumberFormatException: 60s =====
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "3")
        .config("spark.hadoop.fs.s3a.retry.limit", "3")
        .config("spark.hadoop.fs.s3a.retry.interval", "1000")
        .config("spark.hadoop.fs.s3a.threads.max", "10")
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.socket.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.ttl", "0")
        .config("spark.hadoop.fs.s3a.idle.connection.timeout", "60000")

        # ===== STABILITY =====
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.hive.metastore.jars", "builtin")

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


# ===============================
# AUTO DETECT COLUMNS
# ===============================
def detect_columns(df):
    loc_candidates = ["location", "city", "location_name", "locationV2.cityName"]
    lvl_candidates = ["level", "jobLevel", "job_level", "seniority"]

    loc_col = next((c for c in loc_candidates if c in df.columns), None)
    lvl_col = next((c for c in lvl_candidates if c in df.columns), None)

    return loc_col, lvl_col


def derive_processed_prefix(prefix: str) -> str:
    """Derive a processed data prefix from the provided prefix.

    Rules:
    - If prefix already contains 'processed' (case-insensitive) -> return as-is
    - If one of the path segments equals 'raw' -> replace that segment with 'processed'
    - Otherwise append '/processed' to the prefix
    """
    if not prefix:
        return prefix

    lower = prefix.lower()
    if "processed" in lower:
        return prefix

    # Split into segments and replace any segment exactly equal to 'raw'
    parts = prefix.split("/")
    replaced = False
    for i, p in enumerate(parts):
        if p.lower() == "raw":
            parts[i] = "processed"
            replaced = True
            break

    if replaced:
        return "/".join(parts)

    # No 'raw' segment found; append processed without duplicating slashes
    prefix = prefix.rstrip("/")
    return f"{prefix}/processed"


# ===============================
# MAIN
# ===============================
def main():
    # ===== ENV =====
    bucket = os.environ.get("S3_BUCKET_NAME")
    prefix = os.environ.get("S3_PREFIX")
    endpoint = os.environ.get("S3_ENDPOINT")
    region = os.environ.get("AWS_REGION")


    if not bucket or not prefix:
        raise SystemExit("❌ Missing S3_BUCKET_NAME or S3_PREFIX env")

    # If the repository secret `S3_PREFIX` points to raw data, derive the
    # processed prefix automatically so training uses processed/jobs_fact.
    processed_prefix = derive_processed_prefix(prefix)
    if processed_prefix != prefix:
        LOG.info("Derived processed prefix from S3_PREFIX: %s -> %s", prefix, processed_prefix)

    read_path = f"s3a://{bucket}/{processed_prefix}/jobs_fact/"
    model_output = f"s3a://{bucket}/models/salary_prediction_model"

    print(f"📥 Reading parquet from: {read_path}")
    print(f"💾 Model output: {model_output}")

    spark = build_spark()

    # Apply endpoint/region if provided
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    if endpoint:
        hadoop_conf.set("fs.s3a.endpoint", endpoint)
    if region:
        hadoop_conf.set("fs.s3a.region", region)

    # Detect and normalize S3A timeout values that may be expressed with units
    # (environments may set values like '60s', '24h' which Hadoop's longOption can't parse)
    candidate_keys = [
        "fs.s3a.connection.timeout",
        "fs.s3a.connection.establish.timeout",
        "fs.s3a.threads.keepalivetime",
        "fs.s3a.multipart.purge.age",
        "fs.s3a.connection.maximum",
        "fs.s3a.socket.timeout",
        "fs.s3a.connection.request.timeout",
        "fs.s3a.idle.connection.timeout",
    ]

    unit_map = {
        "ms": 1,
        "s": 1000,
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
    }

    for key in candidate_keys:
        try:
            val = hadoop_conf.get(key)
            if not val:
                continue
            val = val.strip()
            # if plain integer string, nothing to do
            if re.fullmatch(r"\d+", val):
                continue
            m = re.fullmatch(r"(\d+)(ms|s|m|h|d)", val, flags=re.IGNORECASE)
            if m:
                num = int(m.group(1))
                unit = m.group(2).lower()
                ms = str(num * unit_map.get(unit, 1))
                hadoop_conf.set(key, ms)
                LOG.info("Normalized %s: %s -> %s", key, val, ms)
            else:
                LOG.debug("S3A config %s has non-numeric value: %s", key, val)
        except Exception:
            LOG.debug("Could not normalize key %s", key, exc_info=True)

    # ===== READ DATA =====
    df = spark.read.parquet(read_path)

    # ===== SALARY =====
    if "salary_avg" not in df.columns:
        if "salary_min" in df.columns and "salary_max" in df.columns:
            df = df.withColumn(
                "salary_avg",
                (col("salary_min") + col("salary_max")) / 2
            )
        else:
            raise SystemExit("❌ No salary_avg or salary_min/max")

    df = df.filter(col("salary_avg").isNotNull())

    # ===== FEATURES =====
    loc_col, lvl_col = detect_columns(df)

    stages = []
    features = []

    if loc_col:
        idx = f"{loc_col}_idx"
        vec = f"{loc_col}_vec"
        stages.append(StringIndexer(inputCol=loc_col, outputCol=idx, handleInvalid="skip"))
        stages.append(OneHotEncoder(inputCol=idx, outputCol=vec))
        features.append(vec)

    if lvl_col:
        idx = f"{lvl_col}_idx"
        vec = f"{lvl_col}_vec"
        stages.append(StringIndexer(inputCol=lvl_col, outputCol=idx, handleInvalid="skip"))
        stages.append(OneHotEncoder(inputCol=idx, outputCol=vec))
        features.append(vec)

    if not features:
        raise SystemExit("❌ No usable categorical features found")

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    stages.append(assembler)

    rf = RandomForestRegressor(
        labelCol="salary_avg",
        featuresCol="features",
        numTrees=50,
        maxDepth=10,
        seed=42
    )
    stages.append(rf)

    pipeline = Pipeline(stages=stages)

    # ===== TRAIN / TEST =====
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(f"📊 Training rows: {train_df.count()}")
    print("🚀 Training model...")

    model = pipeline.fit(train_df)

    # ===== EVALUATE =====
    preds = model.transform(test_df)
    evaluator = RegressionEvaluator(
        labelCol="salary_avg",
        predictionCol="prediction",
        metricName="rmse"
    )
    rmse = evaluator.evaluate(preds)

    print(f"📉 RMSE = {rmse:.2f}")
    preds.select(
        *(c for c in [loc_col, lvl_col] if c),
        "salary_avg",
        "prediction"
    ).show(5, truncate=False)

    # ===== SAVE =====
    model.write().overwrite().save(model_output)
    print("✅ Model saved successfully")

    spark.stop()


if __name__ == "__main__":
    main()
