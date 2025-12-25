import requests
import json
import time
import sys
import os
from datetime import datetime

# ==============================
# IMPORT DATABASE
# ==============================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database


# ==============================
# HELPER FUNCTIONS (CHỐNG NoneType)
# ==============================
def safe_list(value):
    """Đảm bảo luôn trả về list"""
    return value if isinstance(value, list) else []

def safe_dict(value):
    """Đảm bảo luôn trả về dict"""
    return value if isinstance(value, dict) else {}

def clean_text(text):
    if not text:
        return ""
    return text.replace("\r\n", " ").replace("\n", " ").strip()


# ==============================
# MAIN CRAWLER
# ==============================
def crawl_vietnamworks():
    print("🚀 BẮT ĐẦU CRAWL VIETNAMWORKS (FULL FIELDS - SAFE MODE)")

    url = "https://ms.vietnamworks.com/job-search/v1.0/search"
    page = 0
    total_crawled = 0

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Origin": "https://www.vietnamworks.com",
        "Referer": "https://www.vietnamworks.com/"
    }

    # ==============================
    # DATABASE
    # ==============================
    db = get_database()
    collection = db["raw_vietnamworks"]

    while True:
        try:
            print(f"\n⏳ Đang crawl page {page} ...")

            payload = {
                "query": "",
                "filter": [],
                "ranges": [],
                "order": [],
                "hitsPerPage": 50,
                "page": page,
                "retrieveFields": []  # để rỗng -> lấy tối đa field
            }

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=20
            )

            if response.status_code != 200:
                print(f"❌ API lỗi {response.status_code}, đợi 10s...")
                time.sleep(10)
                continue

            data_json = response.json()
            jobs_list = safe_list(data_json.get("data"))

            if not jobs_list:
                print("🏁 HẾT DỮ LIỆU – DỪNG CRAWL")
                break

            parsed_jobs = []

            for job in jobs_list:
                job = safe_dict(job)

                # ==============================
                # WORKING LOCATIONS
                # ==============================
                locations_clean = []
                cities_clean = []

                for loc in safe_list(job.get("workingLocations")):
                    loc = safe_dict(loc)
                    addr = loc.get("address")
                    city = loc.get("cityName") or loc.get("cityNameVI")

                    if addr:
                        locations_clean.append(addr)
                    if city and city not in cities_clean:
                        cities_clean.append(city)

                # ==============================
                # JOB FUNCTION
                # ==============================
                job_func = safe_dict(job.get("jobFunction"))
                parent_name = job_func.get("parentName", "")

                children_names = [
                    c.get("name")
                    for c in safe_list(job_func.get("children"))
                    if isinstance(c, dict) and c.get("name")
                ]

                job_function_str = ""
                if parent_name and children_names:
                    job_function_str = f"{parent_name} > {', '.join(children_names)}"
                elif parent_name:
                    job_function_str = parent_name
                else:
                    job_function_str = ", ".join(children_names)

                # ==============================
                # BENEFITS
                # ==============================
                benefits_clean = []
                for ben in safe_list(job.get("benefits")):
                    ben = safe_dict(ben)
                    name = ben.get("benefitName")
                    val = clean_text(ben.get("benefitValue"))
                    if name:
                        benefits_clean.append(f"{name}: {val}")

                # ==============================
                # SKILLS
                # ==============================
                skills_list = [
                    s.get("skillName")
                    for s in safe_list(job.get("skills"))
                    if isinstance(s, dict) and s.get("skillName")
                ]

                # ==============================
                # INDUSTRIES
                # ==============================
                industries = [
                    i.get("industryV3Name")
                    for i in safe_list(job.get("industriesV3"))
                    if isinstance(i, dict) and i.get("industryV3Name")
                ]

                # ==============================
                # SALARY
                # ==============================
                salary_info = safe_dict(job.get("salary"))
                salary_min = 0
                salary_max = 0
                currency = salary_info.get("currency", "USD")

                if salary_info.get("type") == "Range":
                    salary_min = salary_info.get("min", 0)
                    salary_max = salary_info.get("max", 0)

                # ==============================
                # FINAL JOB OBJECT
                # ==============================
                job_obj = {
                    "jobId": str(job.get("jobId")),
                    "source": "VietnamWorks",
                    "jobUrl": job.get("jobUrl") or f"https://www.vietnamworks.com/job-{job.get('jobId')}-jv",

                    "jobTitle": job.get("jobTitle"),
                    "jobDescription": job.get("jobDescription", ""),
                    "jobRequirement": job.get("jobRequirement", ""),
                    "jobLevel": job.get("jobLevel", ""),
                    "jobFunction": job_function_str,
                    "industries": industries,

                    "companyName": job.get("companyName"),
                    "companyProfile": job.get("companyProfile", ""),
                    "companySize": job.get("companySize", ""),
                    "companyLogo": job.get("companyLogo", ""),
                    "contactName": job.get("contactName", ""),

                    "workingLocations": locations_clean,
                    "cities": cities_clean,
                    "skills": skills_list,
                    "benefits": benefits_clean,

                    "salaryMin": salary_min,
                    "salaryMax": salary_max,
                    "salaryCurrency": currency,
                    "prettySalary": job.get("prettySalary", ""),
                    "yearsOfExperience": job.get("yearsOfExperience", 0),

                    "postedDate": job.get("onlineOn"),
                    "expiredDate": job.get("expiredDate"),
                    "crawled_at": datetime.utcnow()
                }

                parsed_jobs.append(job_obj)

            # ==============================
            # SAVE TO DATABASE (UPSERT)
            # ==============================
            new_count = 0
            for job in parsed_jobs:
                try:
                    res = collection.update_one(
                        {"jobId": job["jobId"], "source": "VietnamWorks"},
                        {"$set": job},
                        upsert=True
                    )
                    if res.upserted_id:
                        new_count += 1
                except Exception as e:
                    print(f"⚠️ Lỗi lưu job {job['jobId']}: {e}")

            total_crawled += len(parsed_jobs)
            print(f"✅ Page {page}: {len(parsed_jobs)} jobs | Mới: {new_count} | Tổng: {total_crawled}")

            page += 1
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 DỪNG THỦ CÔNG (Ctrl+C)")
            break
        except Exception as e:
            print(f"🔥 LỖI KHÔNG MONG MUỐN: {e}")
            time.sleep(10)


if __name__ == "__main__":
    crawl_vietnamworks()
