import os
import sys
import json
import io
from datetime import datetime, date
from minio import Minio
from minio.error import S3Error
from bson import ObjectId

# Thêm đường dẫn để import database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database

# --- CẤU HÌNH MINIO ---
# Bạn nên thay đổi thông tin này khớp với cài đặt MinIO của bạn
MINIO_ENDPOINT = "localhost:9000"  # Địa chỉ MinIO Server
MINIO_ACCESS_KEY = "minioadmin"    # User mặc định
MINIO_SECRET_KEY = "minioadmin"    # Password mặc định
MINIO_SECURE = False               # False nếu chạy localhost không có HTTPS

# Tên Bucket để chứa dữ liệu (Data Lake - Raw Zone)
BUCKET_NAME = "job-datalake-raw"

# --- HELPER JSON SERIALIZER ---
# MongoDB trả về datetime và ObjectId mà json.dumps không hiểu, cần hàm này xử lý
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def sync_collection_to_minio(db, collection_name, minio_client):
    """
    Đồng bộ dữ liệu từ 1 collection MongoDB sang MinIO
    """
    col = db[collection_name]
    
    # Chỉ lấy các job chưa được đồng bộ
    cursor = col.find({"synced_to_datalake": False})
    
    count = 0
    print(f"\n🔄 Đang xử lý collection: {collection_name}...")
    
    for doc in cursor:
        try:
            # 1. Chuẩn bị dữ liệu
            job_id = doc.get("jobId", str(doc["_id"]))
            source = doc.get("source", "unknown")
            
            # Lấy ngày cào để phân thư mục (Partitioning)
            # Cấu trúc: source/year/month/day/job_id.json
            crawled_at = doc.get("crawled_at", datetime.now())
            if isinstance(crawled_at, str):
                try:
                    crawled_at = datetime.fromisoformat(crawled_at)
                except:
                    crawled_at = datetime.now()
            
            object_name = f"{source}/{crawled_at.year}/{crawled_at.month:02d}/{crawled_at.day:02d}/{job_id}.json"
            
            # Chuyển document thành JSON bytes
            # Dùng json_serial để xử lý ngày tháng và ObjectId
            data_bytes = json.dumps(doc, default=json_serial, ensure_ascii=False).encode('utf-8')
            data_stream = io.BytesIO(data_bytes)
            
            # 2. Upload lên MinIO (Put Object)
            minio_client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=object_name,
                data=data_stream,
                length=len(data_bytes),
                content_type="application/json"
            )
            
            # 3. Cập nhật trạng thái trong MongoDB để không sync lại lần sau
            col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"synced_to_datalake": True}}
            )
            
            count += 1
            if count % 10 == 0:
                print(f"   -> Đã sync {count} jobs...")
                
        except Exception as e:
            print(f"❌ Lỗi sync job {doc.get('jobId')}: {e}")

    print(f"✅ Hoàn thành {collection_name}: Tổng cộng {count} jobs đã đẩy lên MinIO.")

def main():
    print("--- BẮT ĐẦU ĐỒNG BỘ DATA LAKE (MONGODB -> MINIO) ---")
    
    # 1. Kết nối MongoDB
    try:
        db = get_database()
        print("✅ Kết nối MongoDB thành công.")
    except Exception as e:
        print(f"❌ Lỗi kết nối MongoDB: {e}")
        return

    # 2. Kết nối MinIO
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        print("✅ Kết nối MinIO thành công.")
        
        # Kiểm tra bucket, nếu chưa có thì tạo
        found = minio_client.bucket_exists(BUCKET_NAME)
        if not found:
            minio_client.make_bucket(BUCKET_NAME)
            print(f"   -> Đã tạo bucket mới: {BUCKET_NAME}")
        else:
            print(f"   -> Bucket '{BUCKET_NAME}' đã tồn tại.")
            
    except S3Error as e:
        print(f"❌ Lỗi kết nối MinIO: {e}")
        return

    # 3. Thực hiện Sync cho từng nguồn dữ liệu
    collections_to_sync = [
        "raw_vietnamworks", 
        "raw_topcv", 
        "raw_joboko"
    ]
    
    for col_name in collections_to_sync:
        sync_collection_to_minio(db, col_name, minio_client)

    print("\n🏁 QUÁ TRÌNH ĐỒNG BỘ HOÀN TẤT.")

if __name__ == "__main__":
    main()