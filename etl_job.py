"""etl_job.py

Script test kết nối Spark <-> S3 bằng S3A.
- Đọc 1 file JSON (multiline) từ S3 và in schema.
- Chưa transform gì, chỉ để verify môi trường Java/PySpark và config S3.

Cách dùng (Windows PowerShell ví dụ):
  pip install -r requirements.txt

  $env:AWS_ACCESS_KEY_ID="..."
  $env:AWS_SECRET_ACCESS_KEY="..."
  $env:AWS_REGION="ap-southeast-1"   # optional

  python etl_job.py --s3-path "s3a://ten-bucket/raw_data/jobs_data.json" --endpoint "https://s3.amazonaws.com"

Gợi ý:
- Nếu bạn dùng MinIO: endpoint thường là http://localhost:9000 và cần thêm --path-style-access.
"""

import argparse
import logging
import os
import re
from typing import Tuple

import boto3

from pyspark.sql import SparkSession

LOG = logging.getLogger("etl_job")


def build_spark(endpoint: str | None, path_style_access: bool, region: str | None):
    builder = (
        SparkSession.builder.appName("Test S3 Connection")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    )

    # Credentials: ưu tiên env AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        builder = builder.config("spark.hadoop.fs.s3a.access.key", access_key)
        builder = builder.config("spark.hadoop.fs.s3a.secret.key", secret_key)

    if region:
        builder = builder.config("spark.hadoop.fs.s3a.region", region)

    if endpoint:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint", endpoint)

    if path_style_access:
        builder = builder.config("spark.hadoop.fs.s3a.path.style.access", "true")

    # Workaround: enforce numeric timeouts / thread settings to avoid
    # NumberFormatException when Hadoop config contains values like "60s".
    # These keys are set as numeric milliseconds or integer counts.
    builder = builder.config("spark.hadoop.fs.s3a.connection.maximum", "100")
    builder = builder.config("spark.hadoop.fs.s3a.connection.timeout", "60000")
    builder = builder.config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
    builder = builder.config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000")
    builder = builder.config("spark.hadoop.fs.s3a.threads.max", "50")
    builder = builder.config("spark.hadoop.fs.s3a.retry.limit", "3")

    return builder.getOrCreate()


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse s3a://bucket/key... -> (bucket, key_prefix) without leading slash"""
    if s3_path.startswith("s3a://"):
        body = s3_path[len("s3a://"):]
    elif s3_path.startswith("s3://"):
        body = s3_path[len("s3://"):]
    else:
        raise ValueError("s3_path must start with s3a:// or s3://")
    parts = body.split('/', 1)
    bucket = parts[0]
    key = '' if len(parts) == 1 else parts[1]
    # ensure prefix ends with /
    if key and not key.endswith('/'):
        # if looks like a file (endswith .json), keep as-is
        if not re.search(r"\.(json|parquet)$", key):
            key = key + '/'
    return bucket, key


def find_latest_ymd_prefix(bucket: str, prefix: str, aws_region: str | None = None) -> str:
    """Find latest YYYY/MM/DD/ prefix under given prefix. Returns full prefix (with trailing slash).
    Uses S3 common-prefixes listing by levels (year -> month -> day)."""
    session_kwargs = {}
    if aws_region:
        session_kwargs['region_name'] = aws_region
    s3 = boto3.client('s3', **session_kwargs)

    def list_common_prefixes(pfx: str):
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=pfx, Delimiter='/')
        prefixes = []
        for page in pages:
            for cp in page.get('CommonPrefixes', []):
                prefixes.append(cp.get('Prefix'))
        return prefixes

    # level 1: years
    years = list_common_prefixes(prefix)
    if not years:
        raise FileNotFoundError(f'No year prefixes found under s3://{bucket}/{prefix}')
    latest_year = sorted([y.rstrip('/').split('/')[-1] for y in years])[-1]
    year_prefix = prefix + latest_year + '/'

    months = list_common_prefixes(year_prefix)
    if not months:
        raise FileNotFoundError(f'No month prefixes under {year_prefix}')
    latest_month = sorted([m.rstrip('/').split('/')[-1] for m in months])[-1]
    month_prefix = year_prefix + latest_month + '/'

    days = list_common_prefixes(month_prefix)
    if not days:
        raise FileNotFoundError(f'No day prefixes under {month_prefix}')
    latest_day = sorted([d.rstrip('/').split('/')[-1] for d in days])[-1]
    day_prefix = month_prefix + latest_day + '/'

    return day_prefix


def main():
    parser = argparse.ArgumentParser(description="Test Spark read JSON from S3 via s3a")
    parser.add_argument("--s3-path", required=True, help="Ví dụ: s3a://bucket/raw_data/jobs_data.json")
    parser.add_argument("--endpoint", default=None, help="S3 endpoint. AWS: https://s3.amazonaws.com. MinIO: http://localhost:9000")
    parser.add_argument("--path-style-access", action="store_true", help="Bật path-style access (thường cần cho MinIO)")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"), help="AWS region (optional)")
    args = parser.parse_args()

    spark = build_spark(endpoint=args.endpoint, path_style_access=args.path_style_access, region=args.region)

    # If args.s3_path is a prefix (ends with /) or doesn't point to a json file,
    # try to resolve the latest YYYY/MM/DD/ subfolder.
    s3_path = args.s3_path
    try:
        bucket, key = parse_s3_path(s3_path)
    except ValueError as e:
        print("LOI: s3_path không hợp lệ:", e)
        raise

    if not re.search(r"\.(json|parquet)$", key):
        # treat as base prefix -> find latest date partition
        try:
            latest_suffix = find_latest_ymd_prefix(bucket, key, aws_region=args.region)
            full_key = latest_suffix
            full_s3a = f"s3a://{bucket}/{full_key}"
            LOG.info("Resolved latest daily prefix: %s", full_s3a)
            # try to read any .json under this prefix by using wildcard
            read_path = full_s3a + "*.json"
        except Exception as e:
            print("LOI khi tim folder ngay moi nhat:", e)
            spark.stop()
            raise
    else:
        read_path = s3_path

    LOG.info("Đang thử đọc dữ liệu từ: %s", read_path)
    try:
        df = spark.read.option("multiline", "true").json(read_path)
        df.printSchema()
        print("KET NOI THANH CONG! :D")
        print(f"So dong (count) = {df.count()}")
    except Exception as e:
        print("LOI KET NOI:")
        print(e)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
