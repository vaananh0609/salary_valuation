import asyncio
import logging
import os
import random
import re
import sys
import unicodedata
from typing import Dict, List

from playwright.async_api import Page, async_playwright

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config.db_settings import ensure_raw_collection_index, get_mongo_client, get_raw_collection
from src.extractors.mongo_upsert import bulk_upsert_jobs

BASE_URL = "https://www.vietnamworks.com/viec-lam"
INDUSTRY_ID = int(os.getenv("VW_INDUSTRY_ID", "5"))
MAX_PAGES = int(os.getenv("VW_MAX_PAGES", "80"))
MAX_NO_NEW_PAGES = int(os.getenv("VW_MAX_NO_NEW_PAGES", "5"))
HEADLESS = os.getenv("VW_HEADLESS", "0") == "1"
CHALLENGE_WAIT_SECONDS = int(os.getenv("VW_CHALLENGE_WAIT_SECONDS", "60"))

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text or "") if unicodedata.category(ch) != "Mn")


def _clean_link(link: str) -> str:
    return (link or "").split("?")[0].split("#")[0].strip()


def has_quantitative_salary(salary_raw: str) -> bool:
    s = _strip_accents(salary_raw).lower().strip()
    return bool(s) and not any(k in s for k in ("thoa thuan", "thuong luong", "negotiable", "canh tranh"))


async def detect_challenge(page: Page) -> bool:
    html = (await page.content()).lower()
    return any(k in html or k in _strip_accents(html) for k in ("checking your browser", "verify you are human", "cloudflare", "xác minh", "xac minh"))


async def parse_listing(page: Page) -> List[Dict]:
    for _ in range(5):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)

    try:
        await page.wait_for_selector(
            "div.search_list.view_job_item, div.job-card-container, div[data-testid='job-card']",
            timeout=30000,
        )
    except Exception:
        return []

    return await page.evaluate(
        r"""
        () => {
            const cards = Array.from(document.querySelectorAll('div.search_list.view_job_item'));
            const seen = new Set();
            const results = [];

            const text = (el) => el ? ((el.getAttribute('title') || el.textContent || '').trim()) : '';
            const normalize = (s) => (s || '').replace(/\s+/g, ' ').trim();

            const isSalaryLike = (s) => {
                const t = normalize(s).toLowerCase();
                if (!t) return false;
                return /\d|usd|vnd|thương lượng|thuong luong|negotiable|cạnh tranh|canh tranh/.test(t);
            };

            for (const card of cards) {
                const titleEl = card.querySelector('h2 a[href]') || card.querySelector('a.img_job_card[href]');
                if (!titleEl) continue;

                let link = titleEl.href || '';
                link = link.split('?')[0].split('#')[0];
                if (!link || seen.has(link)) continue;
                seen.add(link);

                const spans = Array.from(card.querySelectorAll('span')).map(s => normalize(text(s))).filter(Boolean);
                let salary_raw = spans.find(isSalaryLike) || '';
                let location_text = '';
                for (const s of spans) {
                    if (s !== salary_raw && !isSalaryLike(s)) {
                        location_text = s;
                        break;
                    }
                }

                results.push({
                    title: normalize(text(titleEl)),
                    company: normalize(text(card.querySelector("a[href*='/nha-tuyen-dung/']"))),
                    salary_raw,
                    location_text,
                    link,
                });
            }

            return results;
        }
        """
    )


async def click_next_page(page: Page, next_page: int) -> bool:
    btn = page.locator("ul.pagination li.page-item button, ul.pagination li.page-item a", has_text=str(next_page)).first
    if await btn.count() == 0:
        # Fallback if the site does not render pagination controls in current response.
        next_url = f"{BASE_URL}?g={INDUSTRY_ID}&page={next_page}&sorting=lasted"
        try:
            await page.goto(next_url, timeout=60000, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)
            return True
        except Exception:
            return False

    before = await page.evaluate(
        """() => {
            const a = document.querySelector('h2 a[href], h3 a[href], a[href*="/viec-lam/"]');
            return a ? (a.href || '') : '';
        }"""
    )

    try:
        await btn.click(timeout=15000)
        await page.wait_for_function(
            """(prev) => {
                const a = document.querySelector('h2 a[href], h3 a[href], a[href*="/viec-lam/"]');
                const cur = a ? (a.href || '') : '';
                return cur && cur !== prev;
            }""",
            before,
            timeout=15000,
        )
        return True
    except Exception:
        return False


def build_page_url(page_num: int) -> str:
    if page_num <= 1:
        return f"{BASE_URL}?g={INDUSTRY_ID}&sorting=lasted"
    return f"{BASE_URL}?g={INDUSTRY_ID}&page={page_num}&sorting=lasted"


async def main() -> None:
    client = get_mongo_client()
    collection = get_raw_collection(client)
    ensure_raw_collection_index(collection)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS, slow_mo=50 if not HEADLESS else 0)
        context = await browser.new_context(
            locale="vi-VN",
            timezone_id="Asia/Ho_Chi_Minh",
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        start_url = build_page_url(1)
        await page.goto(start_url, timeout=60000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        seen_links = set()
        no_new_pages = 0
        page_num = 1

        while page_num <= MAX_PAGES:
            logger.info("VietnamWorks page %s", page_num)

            if await detect_challenge(page):
                logger.warning("Challenge detected on page %s. Waiting %ss...", page_num, CHALLENGE_WAIT_SECONDS)
                await asyncio.sleep(CHALLENGE_WAIT_SECONDS)

            items = await parse_listing(page)
            if not items:
                no_new_pages += 1
                logger.info("No items parsed on page %s", page_num)
            else:
                jobs_batch = []
                new_count = 0
                for item in items:
                    link = _clean_link(item.get("link"))
                    if not link or link in seen_links:
                        continue
                    seen_links.add(link)
                    new_count += 1
                    jobs_batch.append(
                        {
                            "title": re.sub(r"^moi\s*[:\-–—]?\s*", "", item.get("title", ""), flags=re.IGNORECASE).strip(),
                            "company": item.get("company", ""),
                            "salary_raw": item.get("salary_raw", ""),
                            "location_text": item.get("location_text", ""),
                            "link": link,
                            "industry_id": "5",
                            "source": "VietnamWorks",
                            "has_quantitative_salary": has_quantitative_salary(item.get("salary_raw", "")),
                        }
                    )
                bulk_upsert_jobs(collection, jobs_batch)
                logger.info("VietnamWorks page %s -> %s new jobs", page_num, new_count)
                no_new_pages = no_new_pages + 1 if new_count == 0 else 0

            if no_new_pages >= MAX_NO_NEW_PAGES:
                logger.info("Stop VietnamWorks: %s pages without new jobs.", MAX_NO_NEW_PAGES)
                break

            next_page = page_num + 1
            moved = await click_next_page(page, next_page)
            if not moved:
                # Fallback for layouts without clickable pagination controls.
                next_url = build_page_url(next_page)
                try:
                    await page.goto(next_url, timeout=60000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(1500)
                    logger.info("VietnamWorks fallback URL pagination -> page %s", next_page)
                    moved = True
                except Exception as exc:
                    logger.info("No next page. Stop VietnamWorks. Reason: %s", exc)
                    moved = False

            if not moved:
                break

            page_num = next_page
            await page.wait_for_timeout(random.randint(1200, 2500))

        await context.close()
        await browser.close()

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
