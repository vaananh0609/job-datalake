import argparse
import logging
import os
import re
from typing import Tuple

import boto3
from pyspark.sql import SparkSession

LOG = logging.getLogger("etl_job")


# =========================
# SPARK BUILDER (SAFE)
# =========================
def build_spark(endpoint: str | None, path_style_access: bool, region: str | None):
    builder = (
        SparkSession.builder
        .appName("Test S3A Connection")

        # Hadoop AWS
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4"
        )
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )

        # ---- FIX BUG "24h" ----
        # Hadoop expects milliseconds (NUMBER), not duration string
        .config("spark.hadoop.fs.s3a.multipart.purge", "false")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400000")

        # ---- Numeric-only configs ----
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000")
        .config("spark.hadoop.fs.s3a.threads.max", "50")
        .config("spark.hadoop.fs.s3a.retry.limit", "3")

        # ---- Credentials provider (AWS SDK v1) ----
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )
    )

    # =========================
    # CREDENTIALS
    # =========================
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if ak and sk:
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.access.key", ak)
            .config("spark.hadoop.fs.s3a.secret.key", sk)
        )

    # =========================
    # REGION / ENDPOINT
    # =========================
    if region:
        builder = builder.config("spark.hadoop.fs.s3a.region", region)

    if endpoint:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint", endpoint)

    if path_style_access:
        builder = builder.config("spark.hadoop.fs.s3a.path.style.access", "true")

    return builder.getOrCreate()


# =========================
# S3 PATH PARSER
# =========================
def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    if s3_path.startswith("s3a://"):
        body = s3_path[6:]
    elif s3_path.startswith("s3://"):
        body = s3_path[5:]
    else:
        raise ValueError("s3_path phải bắt đầu bằng s3a:// hoặc s3://")

    parts = body.split("/", 1)
    bucket = parts[0]
    key = "" if len(parts) == 1 else parts[1]

    if key and not key.endswith("/") and not re.search(r"\.(json|parquet)$", key):
        key += "/"

    return bucket, key


# =========================
# FIND LATEST YYYY/MM/DD
# =========================
def find_latest_ymd_prefix(bucket: str, prefix: str, region: str | None):
    s3 = boto3.client("s3", region_name=region)

    def list_prefix(pfx):
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=pfx, Delimiter="/")
        return [cp["Prefix"] for p in pages for cp in p.get("CommonPrefixes", [])]

    years = list_prefix(prefix)
    y = sorted(y.rstrip("/").split("/")[-1] for y in years)[-1]

    months = list_prefix(f"{prefix}{y}/")
    m = sorted(m.rstrip("/").split("/")[-1] for m in months)[-1]

    days = list_prefix(f"{prefix}{y}/{m}/")
    d = sorted(d.rstrip("/").split("/")[-1] for d in days)[-1]

    return f"{prefix}{y}/{m}/{d}/"


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-path", required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--path-style-access", action="store_true")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"))
    parser.add_argument("--write-output", action="store_true", help="Write transformed Parquet to S3")
    parser.add_argument("--out-prefix", default="processed", help="Output prefix under bucket, e.g. processed/")
    args = parser.parse_args()

    spark = build_spark(args.endpoint, args.path_style_access, args.region)

    bucket, key = parse_s3_path(args.s3_path)

    if not re.search(r"\.(json|parquet)$", key):
        latest = find_latest_ymd_prefix(bucket, key, args.region)
        read_path = f"s3a://{bucket}/{latest}*.json"
    else:
        read_path = args.s3_path

    LOG.info("Reading: %s", read_path)

    df = spark.read.option("multiline", "true").json(read_path)
    df.printSchema()
    total = df.count()
    print(f"✅ CONNECT OK | Rows = {total}")

    # -------------------------
    # Transform: jobs_fact
    # -------------------------
    def parse_salary(col_name):
        if col_name in df.columns:
            return regexp_replace(col(col_name).cast("string"), "[^0-9.]", "").cast("double")
        else:
            return lit(None)

    salary_min_expr = parse_salary("salaryMin")
    salary_max_expr = parse_salary("salaryMax")

    df_jobs = df.select(
        col("jobId").cast("string").alias("job_id"),
        col("jobTitle").alias("job_title"),
        col("companyName").alias("company_name"),
        col("locationV2.cityName").alias("city"),
        when(salary_min_expr.isNull(), lit(0)).otherwise(salary_min_expr).alias("salary_min"),
        when(salary_max_expr.isNull(), lit(0)).otherwise(salary_max_expr).alias("salary_max"),
        to_date(col("approvedOn")).alias("posted_date")
    )
    df_jobs = df_jobs.withColumn("salary_avg", (col("salary_min") + col("salary_max")) / 2)

    # -------------------------
    # Transform: job_skills
    # -------------------------
    if "skills" in df.columns:
        exploded = df.select(col("jobId").cast("string").alias("job_id"), explode(col("skills")).alias("skill_obj"))
        skill_name = when(col("skill_obj").cast(StringType()).isNotNull(), col("skill_obj").cast(StringType())).otherwise(col("skill_obj.skillName"))
        df_skills = exploded.select(col("job_id"), skill_name.alias("skill_name")).where(col("skill_name").isNotNull())
    else:
        df_skills = spark.createDataFrame([], schema="job_id string, skill_name string")

    # -------------------------
    # Write outputs (Parquet) if requested
    # -------------------------
    if args.write_output:
        out_prefix = args.out_prefix.rstrip('/')
        jobs_path = f"s3a://{bucket}/{out_prefix}/jobs_fact/"
        skills_path = f"s3a://{bucket}/{out_prefix}/job_skills/"

        print(f"Ghi jobs_fact -> {jobs_path}")
        df_jobs.write.mode("overwrite").option("compression", "snappy").parquet(jobs_path)

        print(f"Ghi job_skills -> {skills_path}")
        df_skills.write.mode("overwrite").option("compression", "snappy").parquet(skills_path)
        print("GHI XONG! Parquet đã được lưu vào S3.")

    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
