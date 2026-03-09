import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional
import re
import unicodedata
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongo.errors import ServerSelectionTimeoutError
from playwright.async_api import Page, async_playwright

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

# ================= CONFIG =================

BASE_URL = "https://vn.joboko.com/jobs"

INDUSTRY_ID = int(os.getenv("JOBOKO_INDUSTRY_ID", "30"))
SALARY_RANGES = os.getenv("JOBOKO_SALARY_RANGES", "x,0-10,10-15,15-20,20-30,30-0").split(",")
EXP_LEVELS = os.getenv("JOBOKO_EXP_LEVELS", "1,3,5,6,100").split(",")

MAX_PAGES_PER_RANGE = int(os.getenv("JOBOKO_MAX_PAGES_PER_RANGE", "0"))   # 0 = no limit
MAX_LOAD_MORE_CLICKS = int(os.getenv("JOBOKO_MAX_LOAD_MORE_CLICKS", "0")) # 0 = no limit

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("JOBOKO_DB_NAME") or os.getenv("RAW_DB_NAME") or os.getenv("DB_NAME", "jobs")
COLLECTION_NAME = os.getenv("RAW_COLLECTION_NAME") or os.getenv("JOBOKO_COLLECTION") or "raw_jobs"


# ================= TEXT NORMALIZE =================

def _strip_accents(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def is_multi_location(location_text: str) -> bool:
    """Return True when listing indicates multiple locations."""
    if not location_text:
        return False

    normalized = _strip_accents(location_text).lower().strip()
    normalized = " ".join(normalized.split())

    if "toan quoc" in normalized:
        return True
    if "noi khac" in normalized:
        return True
    if re.search(r"\b\d+\s*noi\s*khac\b", normalized):
        return True

    separators = ["&", ",", ";", "|", "/", " va "]
    return any(sep in normalized for sep in separators)


def has_quantitative_salary(salary_raw: str) -> bool:
    """True when salary is numeric/range; False for thoa thuan/canh tranh/blank."""
    if not salary_raw:
        return False

    normalized = _strip_accents(salary_raw).lower().strip()
    normalized = " ".join(normalized.split())
    qualitative_tokens = ["thoa thuan", "thuong luong", "canh tranh", "negotiable"]
    if any(token in normalized for token in qualitative_tokens):
        return False
    return True


# ================= URL HELPERS =================

def build_range_url(industry_id: int, salary_range: str, exp_level: str) -> str:
    return f"{BASE_URL}?ind={industry_id}&sal={salary_range}&exp={exp_level}"


def merge_query(url: str, updates: Dict[str, str]) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for k, v in updates.items():
        query[k] = [str(v)]
    new_query = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


# ================= PARSER =================

async def parse_listing(page: Page) -> List[Dict]:
    """Parse current Joboko listing page."""
    try:
        await page.wait_for_selector(".nw-job-list__list .item[data-jid]", timeout=30000)
    except Exception:
        return []

    items = await page.evaluate(
        """
        () => {
            const cards = Array.from(document.querySelectorAll('.nw-job-list__list .item[data-jid]'));
            const results = [];
            const seen = new Set();

            const text = (el) => el ? (el.textContent || '').trim() : '';

            for (const card of cards) {
                const a = card.querySelector('h2.item-title a');
                if (!a) continue;

                let href = a.href || '';
                if (!href) continue;
                if (href.includes('#')) href = href.split('#')[0];
                if (href.includes('?')) href = href.split('?')[0];

                if (seen.has(href)) continue;
                seen.add(href);

                const title = text(a);
                if (!title) continue;

                const company =
                    text(card.querySelector('.item-company span')) ||
                    text(card.querySelector('.item-company'));

                const location_text =
                    text(card.querySelector('.item-address span')) ||
                    text(card.querySelector('.item-address'));

                const salary_raw =
                    text(card.querySelector('.item-rate span')) ||
                    text(card.querySelector('.item-rate'));

                results.push({
                    title,
                    company,
                    location_text,
                    salary_raw,
                    link: href
                });
            }
            return results;
        }
        """
    )
    return list(items or [])


# ================= LOAD MORE =================

async def get_next_page_url_from_button(page: Page, current_url: str) -> Optional[str]:
    btn = page.locator(".nw-job-list__more a.btn.btn-nw-success-ol").first
    if await btn.count() == 0:
        return None

    href = (await btn.get_attribute("href")) or ""
    href = href.strip()
    if not href:
        return None

    if href.startswith("/"):
        cur = urlparse(current_url)
        href = f"{cur.scheme}://{cur.netloc}{href}"

    cur = urlparse(current_url)
    cur_q = parse_qs(cur.query)
    nxt = urlparse(href)
    nxt_q = parse_qs(nxt.query)

    for k in ("ind", "sal"):
        if k in cur_q and k not in nxt_q:
            nxt_q[k] = cur_q[k]

    new_query = urlencode(nxt_q, doseq=True)
    return urlunparse((nxt.scheme, nxt.netloc, nxt.path, nxt.params, new_query, nxt.fragment))


async def click_load_more_append(page: Page, item_selector: str) -> bool:
    btn = page.locator(".nw-job-list__more a.btn.btn-nw-success-ol").first
    if await btn.count() == 0:
        return False

    before = await page.evaluate(f"document.querySelectorAll('{item_selector}').length")
    try:
        await btn.click(timeout=15000)
    except Exception:
        return False

    try:
        await page.wait_for_function(
            f"document.querySelectorAll('{item_selector}').length > {before}",
            timeout=20000,
        )
        return True
    except Exception:
        return False


# ================= CRAWL =================

async def crawl_salary_range(
    page: Page,
    mongo_col,
    industry_id: int,
    salary_range: str,
    exp_level: str,
    global_seen: set,
):
    base_url = build_range_url(industry_id, salary_range, exp_level)

    try:
        await page.goto(base_url, timeout=60000, wait_until="domcontentloaded")
    except Exception as e:
        logger.warning("JobOKO: timeout/error loading base url sal=%s exp=%s: %s", salary_range, exp_level, e)
        return
    await page.wait_for_timeout(1200)

    item_selector = ".nw-job-list__list .item[data-jid]"
    pages = 1
    load_more_clicks = 0

    while True:
        logger.info("JobOKO page %d [sal=%s, exp=%s]: %s", pages, salary_range, exp_level, page.url)

        items = await parse_listing(page)
        if not items:
            logger.info("JobOKO page %d [sal=%s, exp=%s]: khong co job nao duoc parse.", pages, salary_range, exp_level)
            break

        # LẤY CẢ JOB THỎA THUẬN: không filter salary_raw.
        # Giữ filter multi-location như logic cũ.
        eligible = [it for it in items if not is_multi_location(it.get("location_text", ""))]
        new_items = [it for it in eligible if it.get("link") and it["link"] not in global_seen]

        for it in new_items:
            global_seen.add(it["link"])
            now = datetime.now(timezone.utc)
            doc = {
                "title": it.get("title", ""),
                "company": it.get("company", ""),
                "salary_raw": it.get("salary_raw", ""),
                "location_text": it.get("location_text", ""),
                "link": it.get("link", ""),
                "industry_id": str(industry_id),
                "source": "Joboko",
                "has_quantitative_salary": has_quantitative_salary(it.get("salary_raw", "")),
            }
            mongo_col.update_one(
                {"link": doc["link"]},
                {
                    "$set": {**doc, "last_seen_at": now},
                    "$setOnInsert": {"first_crawled_at": now},
                },
                upsert=True,
            )

        logger.info("JobOKO page %d [sal=%s, exp=%s] -> %d new jobs", pages, salary_range, exp_level, len(new_items))

        if MAX_PAGES_PER_RANGE > 0 and pages >= MAX_PAGES_PER_RANGE:
            logger.info(
                "JobOKO: dat MAX_PAGES_PER_RANGE=%d, dung range sal=%s exp=%s.",
                MAX_PAGES_PER_RANGE,
                salary_range,
                exp_level,
            )
            break
        if MAX_LOAD_MORE_CLICKS > 0 and load_more_clicks >= MAX_LOAD_MORE_CLICKS:
            logger.info(
                "JobOKO: dat MAX_LOAD_MORE_CLICKS=%d, dung range sal=%s exp=%s.",
                MAX_LOAD_MORE_CLICKS,
                salary_range,
                exp_level,
            )
            break

        next_url = await get_next_page_url_from_button(page, page.url)
        if next_url:
            if next_url == page.url:
                break
            pages += 1
            try:
                await page.goto(next_url, timeout=60000, wait_until="domcontentloaded")
            except Exception as e:
                logger.warning("JobOKO: timeout/error loading page %d sal=%s exp=%s: %s", pages, salary_range, exp_level, e)
                break
            await page.wait_for_timeout(random.randint(900, 2000))
            continue

        did_append = await click_load_more_append(page, item_selector)
        if not did_append:
            break

        load_more_clicks += 1
        pages += 1
        await page.wait_for_timeout(random.randint(900, 1800))


def _find_existing_collection_by_name(client: MongoClient, collection_name: str):
    try:
        db_names = client.list_database_names()
    except Exception:
        return None

    for db_name in db_names:
        try:
            db = client[db_name]
            if collection_name in db.list_collection_names():
                return db[collection_name]
        except Exception:
            continue
    return None


async def main():
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI not found. Please set it in .env before running joboko.py")

    mongo = MongoClient(MONGO_URI)
    try:
        mongo.admin.command("ping")
    except ServerSelectionTimeoutError as exc:
        raise RuntimeError(
            "Cannot connect to MongoDB. Check MONGO_URI and ensure MongoDB/Atlas is reachable."
        ) from exc

    db = mongo[DB_NAME]
    col = db[COLLECTION_NAME]

    try:
        col.create_index("link", unique=True)
    except OperationFailure as exc:
        msg = str(exc)
        if "already using 500 collections" in msg or "cannot create a new collection" in msg:
            fallback = _find_existing_collection_by_name(mongo, COLLECTION_NAME)
            if fallback is not None:
                logger.warning(
                    "MongoDB Atlas dat gioi han collection; dung collection co san: %s.%s",
                    fallback.database.name, fallback.name,
                )
                col = fallback
                try:
                    col.create_index("link", unique=True)
                except Exception:
                    pass
            else:
                raise SystemExit(
                    "MongoDB Atlas da dung du collections, khong tao duoc collection moi."
                )
        else:
            raise

    global_seen = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=80)
        context = await browser.new_context(
            locale="vi-VN",
            timezone_id="Asia/Ho_Chi_Minh",
            viewport={"width": 1366, "height": 768},
        )
        page = await context.new_page()

        try:
            for sal in SALARY_RANGES:
                sal = sal.strip()
                if not sal:
                    continue

                for exp in EXP_LEVELS:
                    exp = exp.strip()
                    if not exp:
                        continue
                    await crawl_salary_range(page, col, INDUSTRY_ID, sal, exp, global_seen)
        finally:
            await browser.close()
            mongo.close()


if __name__ == "__main__":
    asyncio.run(main())