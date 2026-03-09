import asyncio
import logging
import os
import random
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

BASE_URL = "https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257?sort=new&type_keyword=1&category_family=r257&saturday_status=0"
MAX_PAGES = 100
CHALLENGE_WAIT_SECONDS = 60
MAX_NO_NEW_PAGES = 2

logger = logging.getLogger(__name__)


def _strip_accents(text: str) -> str:
	if not text:
		return ""
	return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def _clean_link(link: str) -> str:
	if not link:
		return ""
	return link.split("?")[0].split("#")[0].strip()


def has_quantitative_salary(salary_raw: str) -> bool:
	if not salary_raw:
		return False
	normalized = _strip_accents(salary_raw).lower().strip()
	return "thoa thuan" not in normalized and "canh tranh" not in normalized


def build_page_url(page_num: int) -> str:
	if page_num == 1:
		return BASE_URL
	return (
		"https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257"
		f"?sort=new&type_keyword=1&page={page_num}&category_family=r257&saturday_status=0"
	)


async def detect_challenge(page: Page) -> bool:
	html = (await page.content()).lower()
	return (
		"checking your browser" in html
		or "verify you are human" in html
		or "xac minh" in _strip_accents(html)
		or "cloudflare" in html
	)


async def wait_auto(reason: str, seconds: int = CHALLENGE_WAIT_SECONDS) -> None:
	logger.warning("Security challenge detected at: %s", reason)
	logger.warning("Waiting %ss for auto-resolve or manual verification...", seconds)
	await asyncio.sleep(seconds)


async def parse_listing(page: Page) -> List[Dict]:
	for _ in range(4):
		await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
		await page.wait_for_timeout(1000)

	try:
		await page.wait_for_selector(".job-item-search-result[data-job-id]", timeout=30000)
	except Exception:
		return []

	return await page.evaluate(
		"""
		() => {
			const cards = Array.from(document.querySelectorAll('.job-item-search-result[data-job-id]'));
			const text = (el) => el ? ((el.textContent || el.getAttribute('title') || '').trim()) : '';

			return cards.map((card) => {
				const a = card.querySelector('h3.title a[href]');
				if (!a) return null;
				return {
					title: text(a),
					company: text(card.querySelector('.company-name')),
					salary_raw: text(card.querySelector('.salary')),
					location_text: text(card.querySelector('.city-text')),
					link: a.href || ''
				};
			}).filter(Boolean);
		}
		"""
	)


async def main() -> None:
	client = get_mongo_client()
	collection = get_raw_collection(client)
	ensure_raw_collection_index(collection)

	async with async_playwright() as pw:
		browser = await pw.chromium.launch(headless=False, slow_mo=50)
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
		empty_page_count = 0
		no_new_data_pages = 0
		seen_links = set()

		for page_num in range(1, MAX_PAGES + 1):
			url = build_page_url(page_num)
			logger.info("TopCV page %s: %s", page_num, url)

			try:
				await page.goto(url, timeout=60000, wait_until="domcontentloaded")
				await page.wait_for_timeout(2000)
			except Exception as exc:
				logger.warning("Load error on TopCV page %s: %s", page_num, exc)
				break

			if await detect_challenge(page):
				await wait_auto(url)
				if await detect_challenge(page):
					logger.warning("Challenge still active after waiting. Skip page %s.", page_num)
					continue

			items = await parse_listing(page)
			if not items:
				empty_page_count += 1
				if empty_page_count >= 2:
					break
				continue
			empty_page_count = 0

			jobs_batch = []
			new_links_count = 0
			for item in items:
				link = _clean_link(item.get("link", ""))
				if not link:
					continue
				if link in seen_links:
					continue
				seen_links.add(link)
				new_links_count += 1

				salary_raw = item.get("salary_raw", "")
				jobs_batch.append(
					{
						"title": item.get("title", ""),
						"company": item.get("company", ""),
						"salary_raw": salary_raw,
						"location_text": item.get("location_text", ""),
						"link": link,
						"industry_id": "257",
						"source": "TopCV",
						"has_quantitative_salary": has_quantitative_salary(salary_raw),
					}
				)

			bulk_upsert_jobs(collection, jobs_batch)
			if new_links_count == 0:
				no_new_data_pages += 1
				if no_new_data_pages >= MAX_NO_NEW_PAGES:
					logger.info("Stop TopCV: %s consecutive pages without new links.", MAX_NO_NEW_PAGES)
					break
			else:
				no_new_data_pages = 0
			logger.info("TopCV page %s -> %s new jobs", page_num, new_links_count)
			await page.wait_for_timeout(random.randint(3000, 8000))

		await context.close()
		await browser.close()

	client.close()


if __name__ == "__main__":
	asyncio.run(main())
