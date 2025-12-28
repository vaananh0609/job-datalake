import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("train_model")


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

        # numeric milliseconds only
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


def detect_columns(df):
    loc_candidates = ["location", "city", "location_name", "locationV2.cityName"]
    lvl_candidates = ["level", "jobLevel", "job_level", "seniority"]

    loc_col = next((c for c in loc_candidates if c in df.columns), None)
    lvl_col = next((c for c in lvl_candidates if c in df.columns), None)

    return loc_col, lvl_col


def main():
    bucket = os.environ.get("S3_BUCKET_NAME")
    prefix = os.environ.get("S3_PREFIX", "processed")
    endpoint = os.environ.get("S3_ENDPOINT")
    region = os.environ.get("AWS_REGION")

    if not bucket:
        raise SystemExit("❌ S3_BUCKET_NAME missing")

    read_path = f"s3a://{bucket}/{prefix}/jobs_fact/"
    model_output = f"s3a://{bucket}/models/salary_prediction_model"

    LOG.info("📥 Reading parquet from: %s", read_path)
    LOG.info("💾 Model output: %s", model_output)

    spark = build_spark()

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    if endpoint:
        hadoop_conf.set("fs.s3a.endpoint", endpoint)
    if region:
        hadoop_conf.set("fs.s3a.region", region)

    df = spark.read.parquet(read_path)

    if "salary_avg" not in df.columns:
        df = df.withColumn(
            "salary_avg",
            (col("salary_min") + col("salary_max")) / 2
        )

    df = df.filter(col("salary_avg").isNotNull())

    loc_col, lvl_col = detect_columns(df)

    stages, features = [], []

    if loc_col:
        stages += [
            StringIndexer(inputCol=loc_col, outputCol=f"{loc_col}_idx", handleInvalid="skip"),
            OneHotEncoder(inputCol=f"{loc_col}_idx", outputCol=f"{loc_col}_vec")
        ]
        features.append(f"{loc_col}_vec")

    if lvl_col:
        stages += [
            StringIndexer(inputCol=lvl_col, outputCol=f"{lvl_col}_idx", handleInvalid="skip"),
            OneHotEncoder(inputCol=f"{lvl_col}_idx", outputCol=f"{lvl_col}_vec")
        ]
        features.append(f"{lvl_col}_vec")

    if not features:
        raise SystemExit("❌ No features found")

    stages += [
        VectorAssembler(inputCols=features, outputCol="features"),
        RandomForestRegressor(
            labelCol="salary_avg",
            featuresCol="features",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
    ]

    pipeline = Pipeline(stages=stages)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    LOG.info("🚀 Training model...")
    model = pipeline.fit(train_df)

    rmse = RegressionEvaluator(
        labelCol="salary_avg",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(model.transform(test_df))

    LOG.info("📉 RMSE = %.2f", rmse)

    model.write().overwrite().save(model_output)
    LOG.info("✅ Model saved")

    spark.stop()


if __name__ == "__main__":
    main()
