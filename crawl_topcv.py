import asyncio
import random
from datetime import datetime
import os
import sys

from playwright.async_api import async_playwright, TimeoutError

# =========================
# MongoDB
# =========================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database

# =========================
# CONFIG
# =========================
BASE_URL = "https://www.topcv.vn/tim-viec-lam-moi-nhat"
SAFETY_MAX_PAGES = 200
MAX_CONSECUTIVE_EMPTY = 6
MAX_CONCURRENT_JOB_TASKS = 4

USER_DATA_DIR = os.path.abspath("browser_profile_topcv")

# =========================
# BLOCK DETECTION (CI)
# =========================
async def is_blocked_by_topcv(page):
    content = await page.content()
    keywords = [
        "xác minh bạn là con người",
        "verify you are human",
        "captcha",
        "cloudflare"
    ]
    content = content.lower()
    return any(k in content for k in keywords)

# =========================
# JOB LIST EXTRACTION
# =========================
async def extract_job_links(page):
    return await page.evaluate("""
    () => {
        const links = Array.from(
            document.querySelectorAll('a[href*="/viec-lam/"]')
        ).map(a => a.href);
        return [...new Set(links)];
    }
    """)

# =========================
# JOB DETAIL (JSON-LD ONLY)
# =========================
async def parse_job_jsonld(context, job_url, sem):
    async with sem:
        page = await context.new_page()
        try:
            await page.goto(job_url, timeout=45000)
            await page.wait_for_timeout(1500)

            if await is_blocked_by_topcv(page):
                print(f"⛔ CI BLOCKED: {job_url}")
                return None

            json_ld = await page.evaluate("""
            () => {
                try {
                    const s = document.querySelector(
                        'script[type="application/ld+json"]'
                    );
                    return s ? JSON.parse(s.innerText) : null;
                } catch (e) {
                    return null;
                }
            }
            """)

            if not json_ld:
                print(f"⛔ NO JSON-LD: {job_url}")
                return None

            return json_ld

        except Exception as e:
            print(f"⚠️ Job error {job_url}: {e}")
            return None
        finally:
            await page.close()

# =========================
# STRICT VALIDATION
# =========================
def is_valid_job(json_ld: dict) -> bool:
    if not json_ld:
        return False

    if not json_ld.get("title"):
        return False

    hiring = json_ld.get("hiringOrganization") or {}
    if not hiring.get("name"):
        return False

    if not json_ld.get("jobLocation"):
        return False

    return True

# =========================
# MAIN CRAWLER
# =========================
async def crawl_topcv_clean_final():
    print("\n🚀 TOPCV CRAWLER – CLEAN FINAL")

    db = get_database()
    col = db["raw_topcv"]
    print("✅ MongoDB connected")

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ],
            viewport={"width": 1920, "height": 1080},
            locale="vi-VN"
        )

        page = context.pages[0] if context.pages else await context.new_page()

        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        await page.goto("https://www.topcv.vn/", timeout=60000)
        await page.wait_for_timeout(4000)

        current_page = 1
        consecutive_empty = 0

        total_pages = 0
        total_found = 0
        total_saved = 0

        sem = asyncio.Semaphore(MAX_CONCURRENT_JOB_TASKS)

        while current_page <= SAFETY_MAX_PAGES:
            url = f"{BASE_URL}?page={current_page}"
            print(f"\n⏳ PAGE {current_page}")

            try:
                await page.goto(url, timeout=60000)
            except TimeoutError:
                print("⚠️ Timeout page")
                current_page += 1
                continue

            await page.wait_for_timeout(random.randint(3000, 5000))

            for _ in range(4):
                await page.mouse.wheel(0, random.randint(1500, 2500))
                await page.wait_for_timeout(random.randint(800, 1400))

            job_links = await extract_job_links(page)
            found = len(job_links)

            print(f"🔎 Found {found} jobs")

            if found == 0:
                consecutive_empty += 1
                print(f"⚠️ Empty page {consecutive_empty}/{MAX_CONSECUTIVE_EMPTY}")
                if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                    print("⛔ Real end detected")
                    break
                current_page += 1
                continue
            else:
                consecutive_empty = 0

            async def process_job(link):
                raw_id = link.split("/")[-1].split(".")[0]
                job_id = f"topcv_{raw_id}"

                json_ld = await parse_job_jsonld(context, link, sem)

                if not json_ld:
                    return 0

                if not is_valid_job(json_ld):
                    print(f"⛔ INVALID JSON-LD: {job_id}")
                    return 0

                doc = {
                    "jobId": job_id,
                    "jobUrl": link,
                    "source": "TopCV",
                    "crawled_at": datetime.now(),
                    "json_ld": json_ld
                }

                res = col.update_one(
                    {"jobId": job_id},
                    {"$setOnInsert": doc},
                    upsert=True
                )

                if res.upserted_id:
                    print(f"✅ Saved {job_id}")
                    return 1

                return 0

            tasks = [process_job(link) for link in job_links]
            results = await asyncio.gather(*tasks)

            saved = sum(results)

            total_pages += 1
            total_found += found
            total_saved += saved

            print(f"💾 Page saved: {saved}")

            current_page += 1
            await page.wait_for_timeout(random.randint(4000, 7000))

        print("\n🏁 DONE")
        print(f"📄 Pages crawled: {total_pages}")
        print(f"📦 Jobs found: {total_found}")
        print(f"✅ Jobs saved: {total_saved}")

        await context.close()
        print("✅ Browser closed")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    asyncio.run(crawl_topcv_clean_final())
