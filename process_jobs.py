"""
process_jobs.py

PySpark script to read the newest daily JSON file from S3 (multiline JSON array),
clean & transform job data, and write results into PostgreSQL.

Configuration is via environment variables (no secrets in source):
- S3_BUCKET, S3_PREFIX (prefix where daily files are stored)
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (optional)
- POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
- JDBC_WRITE_MODE (default: append) or --mode CLI flag

Dependencies:
- pyspark
- boto3

Usage example:
    pip install pyspark boto3
    python process_jobs.py --s3-prefix "raw-data/jobs/" --s3-bucket my-bucket

"""

import os
import argparse
import logging
from urllib.parse import quote_plus

import boto3
from botocore.exceptions import ClientError

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, when, lit
from pyspark.sql.types import StringType


LOG = logging.getLogger("process_jobs")


def get_env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"Environment variable {name} is required but not set")
    return val


def find_latest_s3_object(bucket: str, prefix: str, aws_region: str = None):
    """Return the S3 key of the most recently modified object under prefix.
    Requires boto3 credentials via env or IAM role.
    """
    session_kwargs = {}
    if aws_region:
        session_kwargs["region_name"] = aws_region
    s3 = boto3.client("s3", **session_kwargs)

    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    latest = None

    for page in page_iterator:
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            # skip prefixes
            if key.endswith("/"):
                continue
            if latest is None or obj["LastModified"] > latest["LastModified"]:
                latest = obj

    if not latest:
        raise FileNotFoundError(f"No objects found in s3://{bucket}/{prefix}")

    return latest["Key"], latest["LastModified"]


def build_spark_session(aws_access_key, aws_secret_key, aws_endpoint=None, aws_region=None):
    jars = "org.apache.hadoop:hadoop-aws:3.3.4,org.postgresql:postgresql:42.6.0"

    builder = SparkSession.builder.appName("JobsProcessing")
    builder = builder.config("spark.jars.packages", jars)

    # Configure S3 access via hadoop fs.s3a
    if aws_access_key and aws_secret_key:
        builder = builder.config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        builder = builder.config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
    if aws_region:
        builder = builder.config("spark.hadoop.fs.s3a.region", aws_region)
    if aws_endpoint:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint", aws_endpoint)
    builder = builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    spark = builder.getOrCreate()
    return spark


def normalize_skills(df_raw):
    """Return a DataFrame with columns (jobId, skill_name) handling
    skills stored as list of strings or list of objects {"skillName": ...}
    """
    # If skills is array of strings, explode directly
    # If array of objects, extract skillName
    from pyspark.sql.functions import coalesce

    # Create a skills column normalized to array of structs with field 'skillName'
    # Try to map strings -> struct(skillName=string) by casting
    # For generality, handle both cases with when/otherwise

    # First, create a column skills_str which will be null when skills are objects
    # We'll attempt to read skill as string and fallback to object path
    skills_col = "skills"

    df = df_raw.select(col("jobId").cast("string"), col(skills_col))

    # Explode and then normalize
    exploded = df.select(col("jobId"), explode(col(skills_col)).alias("skill_obj"))

    # If skill_obj is string, keep it; if it's struct with skillName, extract
    skill_name = when(col("skill_obj").cast(StringType()).isNotNull(), col("skill_obj").cast(StringType())).otherwise(col("skill_obj.skillName"))

    df_skills = exploded.select(col("jobId"), skill_name.alias("skill_name")).where(col("skill_name").isNotNull())
    return df_skills


def transform_jobs(df_raw):
    # Select and normalize fields
    df_jobs = df_raw.select(
        col("jobId").cast("string"),
        col("jobTitle"),
        col("companyName"),
        when(col("salaryMin").isNull(), lit(0)).otherwise(col("salaryMin")).alias("salary_min"),
        when(col("salaryMax").isNull(), lit(0)).otherwise(col("salaryMax")).alias("salary_max"),
        col("approvedOn").alias("posted_date")
    )

    df_jobs = df_jobs.withColumn("salary_avg", (col("salary_min") + col("salary_max")) / 2)
    return df_jobs


def write_to_postgres(df, table: str, db_url: str, db_props: dict, mode: str = "append"):
    LOG.info(f"Writing {table} -> {db_url} (mode={mode})")
    df.write.jdbc(url=db_url, table=table, mode=mode, properties=db_props)


def main():
    parser = argparse.ArgumentParser(description="Process jobs JSON from S3 and write to Postgres")
    parser.add_argument("--s3-bucket", default=get_env("S3_BUCKET"), help="S3 bucket name")
    parser.add_argument("--s3-prefix", default=get_env("S3_PREFIX"), help="S3 prefix where daily files are stored (example: raw-data/jobs/)" )
    parser.add_argument("--s3-endpoint", default=get_env("S3_ENDPOINT", None), help="Optional S3 endpoint")
    parser.add_argument("--mode", default=get_env("JDBC_WRITE_MODE", "append"), choices=["append", "overwrite"], help="JDBC write mode")

    args = parser.parse_args()

    if not args.s3_bucket or not args.s3_prefix:
        parser.error("--s3-bucket and --s3-prefix must be provided or set via env vars S3_BUCKET and S3_PREFIX")

    aws_access_key = get_env("AWS_ACCESS_KEY_ID", None)
    aws_secret_key = get_env("AWS_SECRET_ACCESS_KEY", None)
    aws_region = get_env("AWS_REGION", None)

    LOG.info("Finding latest S3 object...")
    try:
        latest_key, last_modified = find_latest_s3_object(args.s3_bucket, args.s3_prefix, aws_region)
    except (FileNotFoundError, ClientError) as e:
        LOG.error("Could not find latest object on S3: %s", e)
        raise

    s3a_path = f"s3a://{args.s3_bucket}/{latest_key}"
    LOG.info(f"Latest file: {s3a_path} (LastModified={last_modified})")

    spark = build_spark_session(aws_access_key, aws_secret_key, aws_endpoint=args.s3_endpoint, aws_region=aws_region)

    LOG.info("Reading JSON from S3 (multiline)")
    df_raw = spark.read.option("multiline", "true").json(s3a_path)
    df_raw.cache()
    LOG.info("Raw schema:")
    df_raw.printSchema()

    LOG.info("Transforming jobs table...")
    df_jobs = transform_jobs(df_raw)

    LOG.info("Normalizing skills...")
    df_skills = normalize_skills(df_raw)

    # Prepare DB connection
    pg_host = get_env("POSTGRES_HOST", required=True)
    pg_port = get_env("POSTGRES_PORT", "5432")
    pg_db = get_env("POSTGRES_DB", required=True)
    pg_user = get_env("POSTGRES_USER", required=True)
    pg_pass = get_env("POSTGRES_PASSWORD", required=True)

    # Note: password should be URL encoded
    jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"
    db_props = {
        "user": pg_user,
        "password": pg_pass,
        "driver": "org.postgresql.Driver"
    }

    write_to_postgres(df_jobs, "jobs_fact", jdbc_url, db_props, mode=args.mode)
    write_to_postgres(df_skills, "job_skills", jdbc_url, db_props, mode=args.mode)

    LOG.info("Processing completed successfully")
    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
