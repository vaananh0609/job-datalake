import asyncio
import random
from datetime import datetime
import os
import sys
import re

# Thêm đường dẫn để import database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database

from playwright.async_api import async_playwright, TimeoutError

# Cấu hình URL gốc của Joboko (listing jobs dùng nút "Xem thêm việc làm")
BASE_URL = "https://vn.joboko.com/jobs"
MAX_SCROLL_ROUNDS = 100  # số vòng scroll/load-more để lấy thêm jobs
USER_DATA_DIR = os.path.abspath("browser_profile_joboko")
MAX_LINKS_PER_PAGE = 50  # số job tối đa xử lý mỗi vòng scroll
REQUEST_DELAY_RANGE = (0.6, 1.4)  # delay giữa các truy vấn chi tiết

SCROLL_PAUSE_MS = 1200
NO_NEW_LINKS_STOP_ROUNDS = 3

DETAIL_GOTO_TIMEOUT_MS = 60000
DETAIL_RETRIES = 2

LOAD_MORE_WAIT_MS = 15000


def joboko_job_id(job_url: str) -> str | None:
    # Example: https://vn.joboko.com/viec-lam-...-xvi6228392
    m = re.search(r"-xvi(\d+)(?:$|[?#])", job_url)
    if m:
        return f"joboko_{m.group(1)}"
    return None


def filter_joboko_job_links(links: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for u in links or []:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Only job detail pages (must contain -xvi<digits>)
        if not u.startswith("https://vn.joboko.com/viec-lam"):
            continue
        if not re.search(r"-xvi\d+(?:$|[?#])", u):
            continue
        if "#" in u:
            u = u.split("#", 1)[0]
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


async def setup_block_resources(page):
    async def _route(route):
        if route.request.resource_type in ["image", "font", "media", "stylesheet"]:
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/*", _route)


async def fetch_raw_job_detail(detail_page, job_url: str):
    await detail_page.goto(job_url, timeout=DETAIL_GOTO_TIMEOUT_MS, wait_until="domcontentloaded")
    await detail_page.wait_for_timeout(1200)

    json_ld_all = await detail_page.evaluate(
        """
        () => {
            const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
            const parsed = [];
            for (const s of scripts) {
                try {
                    parsed.push(JSON.parse(s.innerText));
                } catch (e) {
                    // ignore
                }
            }
            return parsed;
        }
        """
    )

    # choose JobPosting object
    json_ld = None
    if isinstance(json_ld_all, list):
        for item in json_ld_all:
            if not isinstance(item, dict):
                continue
            t = item.get("@type")
            if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
                json_ld = item
                break
        if json_ld is None and json_ld_all:
            first = json_ld_all[0]
            json_ld = first if isinstance(first, dict) else None

    # If this isn't a job page (e.g., JSON-LD is WebSite), treat as missing
    if isinstance(json_ld, dict):
        t = json_ld.get("@type")
        if not (t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t)):
            json_ld = None

    dom = await detail_page.evaluate(
        """
        () => {
            const getText = s => document.querySelector(s)?.innerText?.trim() || "";
            const getHTML = s => document.querySelector(s)?.innerHTML || "";
            return {
                title: getText('h1') || getText('.nw-company-hero__title'),
                company: getText('.nw-company-hero__text a') || getText('.nw-company-hero__text') || getText('.nw-sidebar-company__title'),
                locations: getText('.nw-company-hero__address') || getText('.text-left.job-work-places'),
                salary: getText('.text-left.job-base-infos') || getText('.nw-job-list__main') || "",
                description_html: getHTML('.text-left.job-desc') || "",
                requirements_html: getHTML('.text-left.job-requirement') || "",
                benefits_html: getHTML('.text-left.job-benefit') || "",
            };
        }
        """
    )

    html_length = len(await detail_page.content())
    return json_ld, dom, html_length


async def extract_job_links_from_listing(page) -> list[str]:
    links = await page.evaluate("""
    () => {
        const sel = Array.from(document.querySelectorAll('a[href*="/viec-lam"], a[href*="/vacancies"], a[href*="/viec-lam-"]'));
        return [...new Set(sel.map(a => a.href))];
    }
    """)
    return filter_joboko_job_links(links)


async def try_load_more_listing(page, prev_total_links: int) -> bool:
    # 1) scroll to bottom (some sites only render the button after scroll)
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    except Exception:
        pass
    await page.wait_for_timeout(SCROLL_PAUSE_MS)

    # 2) try clicking a visible load-more button
    clicked = False
    try:
        # Prefer role-based button lookup
        btn = page.get_by_role("button", name=re.compile(r"(xem\s*thêm|tải\s*thêm|xem\s*tiếp)", re.I)).first
        if await btn.count() > 0 and await btn.is_visible():
            await btn.click(timeout=5000)
            clicked = True
    except Exception:
        pass

    if not clicked:
        try:
            # Fallback: any element containing text (looser regex; works with "XEM THÊM VIỆC LÀM")
            btn2 = page.locator("text=/(xem\\s*thêm|tải\\s*thêm|xem\\s*tiếp)/i").first
            if await btn2.count() > 0 and await btn2.is_visible():
                await btn2.click(timeout=5000)
                clicked = True
        except Exception:
            pass

    if not clicked:
        # As a last resort, try one more scroll to trigger infinite scroll.
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        except Exception:
            pass

    # 3) wait until more links appear
    try:
        await page.wait_for_function(
            """(prev) => {
                const set = new Set(Array.from(document.querySelectorAll('a[href*="/viec-lam"], a[href*="/vacancies"], a[href*="/viec-lam-"]')).map(a => a.href));
                return set.size > prev;
            }""",
            arg=prev_total_links,
            timeout=LOAD_MORE_WAIT_MS,
        )
        return True
    except Exception:
        return False


async def crawl_joboko():
    print("--- BẮT ĐẦU CRAWL JOB JOBOKO (FULL DATA) ---")
    
    # 1. Kết nối DB
    try:
        db = get_database()
        collection = db["raw_joboko_test"]
    except Exception as e:
        print(f"❌ Lỗi kết nối Database: {e}")
        return

    # 2. Khởi tạo trình duyệt
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False, # Để False để theo dõi quá trình chạy
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 800}
        )
        
        page = context.pages[0] if context.pages else await context.new_page()
        detail_page = await context.new_page()
        
        # Chặn hình ảnh/font để load nhanh hơn
        await setup_block_resources(page)
        await setup_block_resources(detail_page)

        seen_job_urls: set[str] = set()
        scroll_round = 1
        no_new_rounds = 0
        
        try:
            print(f"\n📄 Mở trang listing: {BASE_URL}")
            await page.goto(BASE_URL, timeout=60000, wait_until="domcontentloaded")
            await page.wait_for_timeout(random.randint(1500, 2500))

            while scroll_round <= MAX_SCROLL_ROUNDS:
                print(f"\n📄 Vòng scroll: {scroll_round} ({BASE_URL})...")
                
                try:
                    links = await extract_job_links_from_listing(page)

                    if not links:
                        print("⚠️ Không tìm thấy job nào. Có thể cần cập nhật CSS Selector hoặc hết trang.")
                        # Thử debug bằng cách in ra HTML nếu không thấy job
                        # content = await page.content()
                        # print(content[:500]) 
                        break

                    # limit links processed to avoid overloading browser/driver
                    new_links = [u for u in links if u not in seen_job_urls]
                    for u in new_links:
                        seen_job_urls.add(u)

                    print(f"   -> Tìm thấy {len(links)} job links (mới: {len(new_links)}).")

                    links = new_links[:MAX_LINKS_PER_PAGE]
                    count_new = 0

                    for job_url in links:
                        job_id = joboko_job_id(job_url)
                        if not job_id:
                            continue

                        json_ld = None
                        dom = None
                        html_length = None

                        for attempt in range(1, DETAIL_RETRIES + 1):
                            try:
                                json_ld, dom, html_length = await fetch_raw_job_detail(detail_page, job_url)
                                break
                            except Exception as e:
                                msg = str(e)
                                print(f"⚠️ Detail error (attempt {attempt}/{DETAIL_RETRIES}) {job_url}: {msg}")
                                # if detail page got closed, recreate it
                                if 'Target page, context or browser has been closed' in msg or 'TargetClosed' in msg:
                                    try:
                                        detail_page = await context.new_page()
                                        await setup_block_resources(detail_page)
                                    except Exception as e2:
                                        print(f"⚠️ Cannot recreate detail page: {e2}")
                                        break
                                await asyncio.sleep(2.0)

                        raw_doc = {
                            "jobId": job_id,
                            "url": job_url,
                            "fetched_at": datetime.now(),
                            "json_ld": json_ld,
                            "dom": dom,
                            "html_length": html_length,
                            "source": "Joboko",
                            "synced_to_datalake": False,
                        }

                        # Skip non-job pages (no JobPosting JSON-LD and no DOM title)
                        if raw_doc["json_ld"] is None and (not raw_doc["dom"] or not raw_doc["dom"].get("title")):
                            continue

                        try:
                            res = collection.update_one(
                                {"jobId": job_id},
                                {"$set": raw_doc},
                                upsert=True,
                            )
                            if res.upserted_id:
                                count_new += 1
                        except Exception as e:
                            print(f"⚠️ DB write failed for {job_id}: {e}")

                        await asyncio.sleep(random.uniform(*REQUEST_DELAY_RANGE))

                    print(f"   ✅ Đã lưu/cập nhật {len(links)} jobs (Mới: {count_new})")

                    # Wait for the listing to actually grow on-page (link count increases)
                    loaded_more = await try_load_more_listing(page, prev_total_links=len(links))

                    if not new_links and not loaded_more:
                        no_new_rounds += 1
                    else:
                        no_new_rounds = 0

                    if no_new_rounds >= NO_NEW_LINKS_STOP_ROUNDS:
                        print(f"⚠️ {NO_NEW_LINKS_STOP_ROUNDS} vòng liên tiếp không có link mới và không load thêm được. Dừng.")
                        break

                    scroll_round += 1
                    
                except TimeoutError:
                    print("❌ Timeout tải trang, thử lại...")
                    await asyncio.sleep(5)
                    continue
                except Exception as e:
                    msg = str(e)
                    if 'Target page, context or browser has been closed' in msg or 'TargetClosed' in msg:
                        print(f"❌ Browser/context bị đóng khi đang crawl listing: {msg}")
                        break
                    raise

        except KeyboardInterrupt:
            print("\n🛑 Dừng chương trình thủ công.")
        except Exception as e:
            print(f"❌ Lỗi không mong muốn: {e}")
        finally:
            try:
                await context.close()
            except Exception as e:
                print(f"⚠️ Browser close error: {e}")
            print("🏁 Kết thúc crawl Joboko.")

if __name__ == "__main__":
    asyncio.run(crawl_joboko())
