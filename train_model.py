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


def build_spark(app_name: str = "SalaryPredictionML", endpoint: str | None = None, region: str | None = None):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    )

    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        builder = builder.config("spark.hadoop.fs.s3a.access.key", ak).config(
            "spark.hadoop.fs.s3a.secret.key", sk
        )

    if region:
        builder = builder.config("spark.hadoop.fs.s3a.region", region)

    if endpoint:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint", endpoint)

    return builder.getOrCreate()


def detect_columns(df):
    # prefer these names if present
    loc_candidates = ["location", "city", "locationV2.cityName", "location_name"]
    lvl_candidates = ["level", "jobLevel", "job_level", "seniority"]

    loc = next((c for c in loc_candidates if c in df.columns), None)
    lvl = next((c for c in lvl_candidates if c in df.columns), None)
    return loc, lvl


def main():
    parser = argparse.ArgumentParser(description="Train salary prediction model from jobs_fact parquet")
    parser.add_argument("--s3-path", help="S3 path to jobs_fact parquet (s3a://bucket/prefix/)")
    parser.add_argument("--s3-bucket", help="Bucket name (alternative to --s3-path)")
    parser.add_argument("--s3-prefix", help="Prefix inside bucket (alternative to --s3-path)")
    parser.add_argument("--model-output", default=None, help="S3 path to save trained model (default: s3a://<bucket>/models/salary_prediction_model)")
    parser.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT"), help="S3 endpoint (optional)")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"), help="AWS region (optional)")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum samples required to train")
    args = parser.parse_args()

    # derive read path
    if args.s3_path:
        read_path = args.s3_path
    elif args.s3_bucket and args.s3_prefix:
        read_path = f"s3a://{args.s3_bucket}/{args.s3_prefix.rstrip('/')}/jobs_fact/"
    else:
        raise SystemExit("Specify --s3-path or both --s3-bucket and --s3-prefix")

    model_output = args.model_output
    if not model_output:
        # default under same bucket
        if args.s3_path and args.s3_path.startswith("s3a://"):
            body = args.s3_path[6:].rstrip("/")
            bucket = body.split("/", 1)[0]
        elif args.s3_bucket:
            bucket = args.s3_bucket
        else:
            bucket = None
        if bucket:
            model_output = f"s3a://{bucket}/models/salary_prediction_model"
        else:
            raise SystemExit("Cannot determine default model output path; provide --model-output")

    spark = build_spark(endpoint=args.endpoint, region=args.region)
    LOG.info("Reading parquet from %s", read_path)
    df = spark.read.parquet(read_path)

    # ensure salary_avg exists
    if "salary_avg" not in df.columns:
        # try compute from salary_min/salary_max
        if "salary_min" in df.columns and "salary_max" in df.columns:
            df = df.withColumn("salary_avg", (col("salary_min") + col("salary_max")) / 2)
        else:
            raise SystemExit("salary_avg not found and cannot be computed")

    df_clean = df.filter(col("salary_avg") > 0)
    count = df_clean.count()
    print(f"Training records: {count}")
    if count < args.min_samples:
        raise SystemExit(f"Not enough samples to train (found {count}, need {args.min_samples})")

    loc_col, lvl_col = detect_columns(df_clean)
    features = []
    stages = []

    if loc_col:
        idx_loc = f"{loc_col}_idx"
        vec_loc = f"{loc_col}_vec"
        stages.append(StringIndexer(inputCol=loc_col, outputCol=idx_loc, handleInvalid="skip"))
        stages.append(OneHotEncoder(inputCol=idx_loc, outputCol=vec_loc))
        features.append(vec_loc)

    if lvl_col:
        idx_lvl = f"{lvl_col}_idx"
        vec_lvl = f"{lvl_col}_vec"
        stages.append(StringIndexer(inputCol=lvl_col, outputCol=idx_lvl, handleInvalid="skip"))
        stages.append(OneHotEncoder(inputCol=idx_lvl, outputCol=vec_lvl))
        features.append(vec_lvl)

    if not features:
        raise SystemExit("No categorical feature column found (expected location/city and/or level). Add columns or update detect list.")

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    stages.append(assembler)

    rf = RandomForestRegressor(featuresCol="features", labelCol="salary_avg")
    stages.append(rf)

    pipeline = Pipeline(stages=stages)

    # split
    train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)

    print("Training model...")
    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="salary_avg", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("RMSE:\t", rmse)
    print("Sample predictions:")
    sel_cols = []
    if loc_col:
        sel_cols.append(loc_col)
    if lvl_col:
        sel_cols.append(lvl_col)
    sel_cols += ["salary_avg", "prediction"]
    predictions.select(*sel_cols).show(5)

    # save model
    print(f"Saving model to {model_output}")
    model.write().overwrite().save(model_output)
    print("Model saved.")

    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
