"""
etl_job.py

Test kết nối Spark <-> S3 bằng S3A.
- Đọc JSON (multiline) từ S3 / MinIO
- Resolve folder YYYY/MM/DD mới nhất
- In schema + count
- FIX TRIỆT ĐỂ lỗi Hadoop S3A: "24h"

Chạy OK trên:
- Local
- GitHub Actions
- Docker
- MinIO
- AWS S3
"""

import argparse
import logging
import os
import re
from typing import Tuple

import boto3
from pyspark.sql import SparkSession

# =========================
# LOGGING
# =========================
LOG = logging.getLogger("etl_job")


# =========================
# SPARK BUILDER (FIXED)
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

        # ---- FIX CRITICAL BUG: Hadoop duration string "24h" ----
        .config("spark.hadoop.fs.s3a.multipart.purge", "false")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400000")
        .config("spark.hadoop.fs.s3a.directory.marker.retention", "0")

        # ---- Numeric-only configs (NO '60s', '1h', etc) ----
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000")
        .config("spark.hadoop.fs.s3a.threads.max", "50")
        .config("spark.hadoop.fs.s3a.retry.limit", "3")

        # ---- Credentials provider (AWS SDK v1 – STABLE) ----
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )
    )

    # =========================
    # CREDENTIALS (ENV)
    # =========================
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if access_key and secret_key:
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
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
    session_args = {}
    if region:
        session_args["region_name"] = region

    s3 = boto3.client("s3", **session_args)

    def list_prefix(pfx):
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=bucket,
            Prefix=pfx,
            Delimiter="/"
        )
        out = []
        for page in pages:
            for cp in page.get("CommonPrefixes", []):
                out.append(cp["Prefix"])
        return out

    years = list_prefix(prefix)
    if not years:
        raise FileNotFoundError("Không tìm thấy YEAR partition")

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
    parser = argparse.ArgumentParser("Spark S3A test job")
    parser.add_argument("--s3-path", required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--path-style-access", action="store_true")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"))
    args = parser.parse_args()

    spark = build_spark(
        endpoint=args.endpoint,
        path_style_access=args.path_style_access,
        region=args.region
    )

    bucket, key = parse_s3_path(args.s3_path)

    if not re.search(r"\.(json|parquet)$", key):
        latest = find_latest_ymd_prefix(bucket, key, args.region)
        read_path = f"s3a://{bucket}/{latest}*.json"
    else:
        read_path = args.s3_path

    LOG.info("Reading from: %s", read_path)

    try:
        df = spark.read.option("multiline", "true").json(read_path)
        df.printSchema()
        print("✅ KẾT NỐI S3A THÀNH CÔNG")
        print(f"📦 Tổng số dòng: {df.count()}")
    finally:
        spark.stop()


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
