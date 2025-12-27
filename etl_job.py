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

    return builder.getOrCreate()


def main():
    parser = argparse.ArgumentParser(description="Test Spark read JSON from S3 via s3a")
    parser.add_argument("--s3-path", required=True, help="Ví dụ: s3a://bucket/raw_data/jobs_data.json")
    parser.add_argument("--endpoint", default=None, help="S3 endpoint. AWS: https://s3.amazonaws.com. MinIO: http://localhost:9000")
    parser.add_argument("--path-style-access", action="store_true", help="Bật path-style access (thường cần cho MinIO)")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"), help="AWS region (optional)")
    args = parser.parse_args()

    spark = build_spark(endpoint=args.endpoint, path_style_access=args.path_style_access, region=args.region)

    LOG.info("Đang thử đọc dữ liệu từ: %s", args.s3_path)
    try:
        df = spark.read.option("multiline", "true").json(args.s3_path)
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
