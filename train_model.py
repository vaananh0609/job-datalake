import argparse
import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

LOG = logging.getLogger("train_model")


# =========================
# SPARK BUILDER
# =========================
def build_spark(
    app_name: str = "SalaryPredictionML",
    endpoint: str | None = None,
    region: str | None = None
):
    builder = (
        SparkSession.builder
        .appName(app_name)

        # Hadoop AWS
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

        # ===== BẮT BUỘC: milliseconds =====
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000")
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")
        .config("spark.hadoop.fs.s3a.threads.max", "50")
        .config("spark.hadoop.fs.s3a.retry.limit", "3")
        .config("spark.hadoop.fs.s3a.multipart.purge", "false")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400000")

        # Credentials provider
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )
    )

    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.access.key", ak)
            .config("spark.hadoop.fs.s3a.secret.key", sk)
        )

    if region:
        builder = builder.config("spark.hadoop.fs.s3a.region", region)

    if endpoint:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint", endpoint)

    return builder.getOrCreate()

# =========================
# COLUMN DETECTION
# =========================
def detect_columns(df):
    loc_candidates = ["location", "city", "locationV2.cityName", "location_name"]
    lvl_candidates = ["level", "jobLevel", "job_level", "seniority"]

    loc = next((c for c in loc_candidates if c in df.columns), None)
    lvl = next((c for c in lvl_candidates if c in df.columns), None)
    return loc, lvl


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-path")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix")
    parser.add_argument("--model-output")
    parser.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"))
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    # -------- derive read path --------
    if args.s3_path:
        read_path = args.s3_path
    elif args.s3_bucket and args.s3_prefix:
        read_path = f"s3a://{args.s3_bucket}/{args.s3_prefix.rstrip('/')}/jobs_fact/"
    else:
        raise SystemExit("❌ Specify --s3-path or --s3-bucket + --s3-prefix")

    # -------- model output --------
    model_output = args.model_output
    if not model_output:
        bucket = args.s3_bucket or read_path[6:].split("/", 1)[0]
        model_output = f"s3a://{bucket}/models/salary_prediction_model"

    spark = build_spark(endpoint=args.endpoint, region=args.region)

    print(f"📥 Reading parquet from {read_path}")
    df = spark.read.parquet(read_path)

    # -------- ensure salary_avg --------
    if "salary_avg" not in df.columns:
        if {"salary_min", "salary_max"}.issubset(df.columns):
            df = df.withColumn("salary_avg", (col("salary_min") + col("salary_max")) / 2)
        else:
            raise SystemExit("❌ salary_avg missing")

    # -------- base clean --------
    df_clean = df.filter(col("salary_avg") > 0)

    # -------- detect columns --------
    loc_col, lvl_col = detect_columns(df_clean)

    if loc_col:
        df_clean = df_clean.filter(col(loc_col).isNotNull())
    if lvl_col:
        df_clean = df_clean.filter(col(lvl_col).isNotNull())

    count = df_clean.count()
    print(f"✅ Training records: {count}")
    if count < args.min_samples:
        raise SystemExit("❌ Not enough samples")

    # -------- features --------
    stages = []
    features = []

    if loc_col:
        stages += [
            StringIndexer(inputCol=loc_col, outputCol=f"{loc_col}_idx", handleInvalid="skip"),
            OneHotEncoder(inputCol=f"{loc_col}_idx", outputCol=f"{loc_col}_vec"),
        ]
        features.append(f"{loc_col}_vec")

    if lvl_col:
        stages += [
            StringIndexer(inputCol=lvl_col, outputCol=f"{lvl_col}_idx", handleInvalid="skip"),
            OneHotEncoder(inputCol=f"{lvl_col}_idx", outputCol=f"{lvl_col}_vec"),
        ]
        features.append(f"{lvl_col}_vec")

    if not features:
        raise SystemExit("❌ No usable feature columns")

    stages.append(VectorAssembler(inputCols=features, outputCol="features"))

    stages.append(
        RandomForestRegressor(
            featuresCol="features",
            labelCol="salary_avg",
            numTrees=50,
            maxDepth=10,
            seed=42,
        )
    )

    pipeline = Pipeline(stages=stages)

    train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

    print("🚀 Training model...")
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)
    rmse = RegressionEvaluator(
        labelCol="salary_avg",
        predictionCol="prediction",
        metricName="rmse",
    ).evaluate(preds)

    print("📉 RMSE:", rmse)
    preds.select(*(c for c in [loc_col, lvl_col, "salary_avg", "prediction"] if c)).show(5)

    print(f"💾 Saving model to {model_output}")
    model.write().overwrite().save(model_output)

    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
