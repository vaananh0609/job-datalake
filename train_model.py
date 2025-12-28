import logging
import os
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F
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

        # TIMEOUTS: numeric milliseconds only
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.socket.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.idle.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "3")

        .config("spark.sql.files.ignoreCorruptFiles", "true")
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


# ===============================
# MAIN
# ===============================
def main():
    bucket = os.environ.get("S3_BUCKET_NAME")
    prefix = os.environ.get("S3_PREFIX", "processed")
    endpoint = os.environ.get("S3_ENDPOINT")
    region = os.environ.get("AWS_REGION")

    if not bucket:
        raise SystemExit("❌ S3_BUCKET_NAME is missing")

    read_path = f"s3a://{bucket}/{prefix}/jobs_fact/"
    model_output = f"s3a://{bucket}/models/salary_prediction_model"

    LOG.info("📥 Reading parquet from: %s", read_path)
    LOG.info("💾 Model output: %s", model_output)

    spark = build_spark()

    # Apply endpoint / region if needed (MinIO / custom S3)
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    if endpoint:
        hadoop_conf.set("fs.s3a.endpoint", endpoint)
    if region:
        hadoop_conf.set("fs.s3a.region", region)

    # ===== READ DATA =====
    df = spark.read.parquet(read_path)

    # ===== SANITIZE COLUMN NAMES =====
    def sanitize_column_names(df):
        cols = df.columns
        mapping = []
        for c in cols:
            new = c.strip() if c is not None else ""
            if new == "":
                mapping.append((c, None))
                continue
            # replace chars not allowed in identifiers with underscore
            safe = re.sub(r"[^0-9A-Za-z_]+", "_", new)
            if safe != c:
                mapping.append((c, safe))

        # Drop empty-name columns
        for old, new in mapping:
            if new is None:
                LOG.warning("Dropping column with empty name: %r", old)
                df = df.drop(old)
            else:
                LOG.info("Rename column: %s -> %s", old, new)
                df = df.withColumnRenamed(old, new)
        return df

    try:
        df = sanitize_column_names(df)
    except Exception:
        LOG.exception("Failed sanitizing column names")

    # ===== CLEAN CATEGORICAL VALUES =====
    cat_candidates = ["location", "city", "location_name", "locationV2.cityName",
                      "level", "jobLevel", "job_level", "seniority"]
    present_cat_cols = [c for c in cat_candidates if c in df.columns]
    for c in present_cat_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), F.lit("__MISSING__")).otherwise(F.col(c)))

    # ===== SALARY =====
    if "salary_avg" not in df.columns:
        if "salary_min" in df.columns and "salary_max" in df.columns:
            df = df.withColumn(
                "salary_avg",
                (col("salary_min") + col("salary_max")) / 2
            )
        else:
            raise SystemExit("❌ salary columns not found")

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
        raise SystemExit("❌ No categorical features found")

    stages.append(VectorAssembler(inputCols=features, outputCol="features"))

    stages.append(
        RandomForestRegressor(
            labelCol="salary_avg",
            featuresCol="features",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
    )

    pipeline = Pipeline(stages=stages)

    # ===== TRAIN / TEST =====
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    LOG.info("📊 Training rows: %d", train_df.count())
    LOG.info("🚀 Training model...")

    try:
        model = pipeline.fit(train_df)
    except Exception as exc:
        LOG.exception("Model training failed: %s", exc)
        try:
            LOG.error("Dataframe columns: %s", df.columns)
            LOG.error("Dataframe schema: %s", df.schema.simpleString())
            LOG.error("Training sample (5 rows):")
            train_df.show(5, truncate=False)
        except Exception:
            LOG.exception("Failed to log dataframe sample")
        raise SystemExit("❌ Training failed — see logs for dataframe schema and sample")

    # ===== EVALUATE =====
    preds = model.transform(test_df)
    rmse = RegressionEvaluator(
        labelCol="salary_avg",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(preds)

    LOG.info("📉 RMSE = %.2f", rmse)

    # ===== SAVE =====
    model.write().overwrite().save(model_output)
    LOG.info("✅ Model saved successfully")

    spark.stop()


if __name__ == "__main__":
    main()
