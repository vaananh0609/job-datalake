import argparse
import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("train_model")


# -------------------------------------------------
# Spark builder (AWS S3A – GitHub Actions safe)
# -------------------------------------------------
def build_spark(app_name="SalaryPrediction"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )
        .getOrCreate()
    )

    return spark


# -------------------------------------------------
# Detect usable feature columns
# -------------------------------------------------
def detect_columns(df):
    loc_candidates = ["location", "city", "location_name"]
    lvl_candidates = ["level", "job_level", "seniority"]

    loc = next((c for c in loc_candidates if c in df.columns), None)
    lvl = next((c for c in lvl_candidates if c in df.columns), None)

    return loc, lvl


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--s3-prefix", required=True)
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    # ---- SAFE PATH BUILD ----
    prefix = args.s3_prefix.strip("/")
    read_path = f"s3a://{args.s3_bucket}/{prefix}/jobs_fact/"
    model_output = f"s3a://{args.s3_bucket}/models/salary_prediction_model"

    print(f"📥 Reading parquet from {read_path}")
    print(f"💾 Model output: {model_output}")

    spark = build_spark()

    # ---- READ DATA ----
    df = spark.read.parquet(read_path)

    # ---- SALARY ----
    if "salary_avg" not in df.columns:
        if "salary_min" in df.columns and "salary_max" in df.columns:
            df = df.withColumn(
                "salary_avg",
                (col("salary_min") + col("salary_max")) / 2
            )
        else:
            raise RuntimeError("❌ No salary column found")

    loc_col, lvl_col = detect_columns(df)

    if not loc_col and not lvl_col:
        raise RuntimeError("❌ No usable categorical columns found")

    df = df.filter(col("salary_avg").isNotNull())

    if loc_col:
        df = df.filter(col(loc_col).isNotNull())
    if lvl_col:
        df = df.filter(col(lvl_col).isNotNull())

    count = df.count()
    print(f"📊 Training rows: {count}")

    if count < args.min_samples:
        raise RuntimeError("❌ Not enough samples to train")

    # ---- PIPELINE ----
    stages = []
    features = []

    if loc_col:
        idx = f"{loc_col}_idx"
        vec = f"{loc_col}_vec"
        stages += [
            StringIndexer(inputCol=loc_col, outputCol=idx, handleInvalid="skip"),
            OneHotEncoder(inputCol=idx, outputCol=vec)
        ]
        features.append(vec)

    if lvl_col:
        idx = f"{lvl_col}_idx"
        vec = f"{lvl_col}_vec"
        stages += [
            StringIndexer(inputCol=lvl_col, outputCol=idx, handleInvalid="skip"),
            OneHotEncoder(inputCol=idx, outputCol=vec)
        ]
        features.append(vec)

    stages.append(VectorAssembler(inputCols=features, outputCol="features"))

    stages.append(
        RandomForestRegressor(
            featuresCol="features",
            labelCol="salary_avg",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
    )

    pipeline = Pipeline(stages=stages)

    # ---- TRAIN ----
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print("🚀 Training model...")
    model = pipeline.fit(train_df)

    # ---- EVAL ----
    preds = model.transform(test_df)
    rmse = RegressionEvaluator(
        labelCol="salary_avg",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(preds)

    print(f"📉 RMSE = {rmse}")

    preds.select(
        *(c for c in [loc_col, lvl_col] if c),
        "salary_avg",
        "prediction"
    ).show(5, truncate=False)

    # ---- SAVE ----
    print("💾 Saving model...")
    model.write().overwrite().save(model_output)

    print("✅ TRAINING DONE")
    spark.stop()


if __name__ == "__main__":
    main()
