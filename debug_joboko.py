import asyncio
import json
from playwright.async_api import async_playwright

# 🔴 DÁN LINK JOB JOBOKO BẤT KỲ VÀO ĐÂY
JOB_URL = "https://vn.joboko.com/viec-lam-chuyen-vien-quan-he-khach-hang-sales-representative-xvi6228392"

async def debug_dump_joboko():
    print("\n--- 🧪 DEBUG JOBOKO – DUMP ALL RAW DATA OF 1 JOB ---")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = await browser.new_context(
            locale="vi-VN",
            viewport={"width": 1920, "height": 1080}
        )

        page = await context.new_page()

        # Né bot detection
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        print(f"🌍 Open job page:\n{JOB_URL}")
        await page.goto(JOB_URL, timeout=60000)
        await page.wait_for_timeout(5000)

        # =====================================================
        # 1️⃣ DUMP JSON EMBEDDED (LD+JSON)
        # =====================================================
        json_ld = await page.evaluate("""
        () => Array.from(
            document.querySelectorAll('script[type="application/ld+json"]')
        ).map(s => s.innerText)
        """)

        print("\n================ JSON EMBEDDED =================")
        if json_ld:
            for i, block in enumerate(json_ld):
                print(f"\n--- JSON #{i+1} ---")
                try:
                    print(json.dumps(json.loads(block), indent=2, ensure_ascii=False))
                except:
                    print(block)
        else:
            print("❌ No JSON-LD found")

        # =====================================================
        # 2️⃣ DUMP CÁC FIELD DOM LIÊN QUAN JOB
        # =====================================================
        dom_dump = await page.evaluate("""
        () => {
            const keywords = [
                'job', 'salary', 'company', 'address',
                'location', 'skill', 'benefit',
                'description', 'require', 'level',
                'experience'
            ];

            const result = [];

            document.querySelectorAll('*').forEach(el => {
                const cls = el.className?.toString() || '';
                const id = el.id || '';
                const text = el.innerText?.trim();

                if (!text || text.length < 10) return;

                const hit = keywords.some(k =>
                    cls.toLowerCase().includes(k) ||
                    id.toLowerCase().includes(k)
                );

                if (hit) {
                    result.push({
                        tag: el.tagName,
                        id,
                        class: cls,
                        text: text.slice(0, 600)
                    });
                }
            });

            return result;
        }
        """)

        print("\n================ DOM FIELD DUMP =================")
        print(json.dumps(dom_dump, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(debug_dump_joboko())
