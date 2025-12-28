import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# --- 1. CẤU HÌNH MÔI TRƯỜNG & KẾT NỐI SPARK ---
# Lấy biến môi trường từ GitHub Actions workflow
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://s3.amazonaws.com")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
# Prefix đầu vào (ví dụ: processed/)
PREFIX_IN = os.getenv("S3_PREFIX", "processed/")

if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME]):
    print("❌ LỖI: Thiếu biến môi trường AWS/S3. Kiểm tra lại GitHub Secrets.")
    sys.exit(1)

# Khởi tạo Spark Session với cấu hình S3A
spark = SparkSession.builder \
    .appName("TrainSalaryModel_CI") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT) \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

print("🚀 Spark Session đã khởi tạo thành công!")

# --- 2. ĐỌC DỮ LIỆU TỪ S3 (SILVER LAYER) ---
# Đảm bảo prefix có/không dấu slash đều xử lý đúng
prefix_clean = PREFIX_IN.rstrip('/')
# Đường dẫn file parquet đầu vào (kết quả từ bước ETL trước)
input_path = f"s3a://{BUCKET_NAME}/{prefix_clean}/jobs_fact"
print(f"📂 Đang đọc dữ liệu từ: {input_path}")

try:
    df = spark.read.parquet(input_path)
    # Chỉ lấy các bản ghi có lương > 0 để train
    df_train_source = df.filter(col("salary_avg") > 0)
    count = df_train_source.count()
    print(f"📊 Số lượng bản ghi hợp lệ để train: {count}")
    if count == 0:
        print("⚠️ Không có dữ liệu hợp lệ (salary_avg>0). Dừng job.")
        spark.stop()
        sys.exit(0)
except Exception as e:
    print(f"❌ Lỗi đọc file Parquet: {str(e)}")
    spark.stop()
    sys.exit(1)

# --- 3. XÂY DỰNG PIPELINE MACHINE LEARNING ---

# Bước A: Xử lý dữ liệu Categorical (Biến chữ thành số)
# Xây pipeline động: chỉ dùng các cột tồn tại trong dataframe
stages = []
feature_cols = []

if 'location' in df_train_source.columns:
    stages.append(StringIndexer(inputCol='location', outputCol='loc_idx', handleInvalid='keep'))
    feature_cols.append('loc_idx')

if 'level' in df_train_source.columns:
    stages.append(StringIndexer(inputCol='level', outputCol='lvl_idx', handleInvalid='keep'))
    feature_cols.append('lvl_idx')

if not feature_cols:
    print("❌ Không có cột categorical (location/level) để làm feature. Cần ít nhất 1 cột.")
    spark.stop()
    sys.exit(1)

# Bước B: Gom các đặc trưng (Features) thành 1 vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
stages.append(assembler)

# Bước C: Khai báo thuật toán (Random Forest)
rf = RandomForestRegressor(featuresCol='features', labelCol='salary_avg', numTrees=50)
stages.append(rf)

# Gom tất cả vào 1 Pipeline
pipeline = Pipeline(stages=stages)

# --- 4. HUẤN LUYỆN & ĐÁNH GIÁ ---
print("⏳ Đang chia tập dữ liệu Train/Test...")
train_data, test_data = df_train_source.randomSplit([0.8, 0.2], seed=42)

print("🏋️‍♂️ Bắt đầu Training...")
model = pipeline.fit(train_data)

print("mag  Đang Evaluate trên tập Test...")
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="salary_avg", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print("="*40)
print(f"✅ Training Hoàn tất!")
print(f"📉 Sai số trung bình (RMSE): {rmse:,.2f}")
print("="*40)

# --- 5. LƯU MODEL (MODEL REGISTRY) ---
# Lưu model ra S3 để Web App (Streamlit/API) có thể tải về dùng
model_output_path = f"s3a://{BUCKET_NAME}/models/salary_prediction_v1"
print(f"💾 Đang lưu model vào: {model_output_path}")

try:
    model.write().overwrite().save(model_output_path)
    print("✅ Lưu Model thành công!")
except Exception as e:
    print(f"❌ Lỗi khi lưu model: {str(e)}")
    sys.exit(1)

spark.stop()
