import os
import json
import boto3
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import json_util

# Cấu hình lấy từ biến môi trường (Set trong GitHub Secrets)
MONGO_URI = os.getenv("MONGO_URI")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def get_mongo_data():
    """Kết nối MongoDB và lấy dữ liệu"""
    try:
        client = MongoClient(MONGO_URI)
        db = client["job_datalake_buffer"]  
        
        # Lấy từ 3 collection chính
        collections = ["raw_vietnamworks", "raw_joboko", "raw_topcv"] 
        data_buffers = {}

        for col_name in collections:
            col = db[col_name]
            # Lấy toàn bộ data (hoặc lọc theo ngày crawl nếu muốn tối ưu)
            # Ví dụ lọc data cào trong 24h qua:
            # yesterday = datetime.now() - timedelta(days=1)
            # query = {"crawled_at": {"$gte": yesterday}}
            query = {} 
            
            cursor = col.find(query)
            
            # Chuyển BSON sang JSON (xử lý ObjectId và Date)
            data_list = list(cursor)
            if data_list:
                # json_util.dumps giúp convert ObjectId và ISODate của Mongo
                data_buffers[col_name] = json_util.dumps(data_list, ensure_ascii=False)
                print(f"✅ {col_name}: Lấy được {len(data_list)} bản ghi")
            else:
                print(f"⚠️ {col_name}: Không có dữ liệu mới")
        
        return data_buffers

    except Exception as e:
        print(f"❌ Lỗi MongoDB: {e}")
        return {}

def upload_to_s3(data_buffers):
    """Upload dữ liệu lên AWS S3"""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    today_str = datetime.now().strftime("%Y/%m/%d") # Tạo folder theo ngày (Partitioning)

    for col_name, json_data in data_buffers.items():
        try:
            # Đặt tên file theo cấu trúc: raw/tên_bảng/năm/tháng/ngày/tên_file.json
            file_key = f"raw/{col_name}/{today_str}/{col_name}_{datetime.now().strftime('%H%M%S')}.json"
            
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=file_key,
                Body=json_data,
                ContentType='application/json'
            )
            print(f"🚀 Upload thành công: s3://{S3_BUCKET_NAME}/{file_key}")
        except Exception as e:
            print(f"❌ Lỗi Upload S3 ({col_name}): {e}")

if __name__ == "__main__":
    print("--- BẮT ĐẦU ETL MONGO TO S3 ---")
    data = get_mongo_data()
    if data:
        upload_to_s3(data)
    else:
        print("Không có dữ liệu để upload.")
    print("--- HOÀN THÀNH ---")
