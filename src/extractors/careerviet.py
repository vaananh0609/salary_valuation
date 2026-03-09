import asyncio
import logging
import os
import random
import sys
import unicodedata
from typing import Dict, List
from urllib.parse import urljoin

from playwright.async_api import Page, async_playwright

if __package__ in (None, ""):
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)

from src.config.db_settings import ensure_raw_collection_index, get_mongo_client, get_raw_collection
from src.extractors.mongo_upsert import bulk_upsert_jobs

TARGET_URLS = [
	"https://careerviet.vn/viec-lam/cntt-phan-mem-c1-sortdv-vi.html",
	"https://careerviet.vn/viec-lam/cntt-phan-cung-mang-c63-sortdv-vi.html",
]
MAX_PAGES = 50
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
	return "canh tranh" not in normalized and "thoa thuan" not in normalized


def build_page_url(seed_url: str, page_num: int) -> str:
	if page_num == 1:
		return seed_url
	return seed_url.replace("-vi.html", f"-trang-{page_num}-vi.html")


async def parse_listing(page: Page, seed_url: str) -> List[Dict]:
	for _ in range(4):
		await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
		await page.wait_for_timeout(1000)

	try:
		await page.wait_for_selector(".job-item a.job_link[href]", timeout=20000)
	except Exception:
		return []

	items = await page.evaluate(
		r"""
		() => {
			const links = Array.from(document.querySelectorAll('.job-item a.job_link[href]'));
			const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();

			return links.map((a) => {
				const card = a.closest('.job-item');
				if (!card) return null;
				const salaryNode = card.querySelector('.salary p');
				const salaryText = clean(salaryNode ? salaryNode.textContent : '').replace(/^(luong|l\u01b0\u01a1ng)\s*:\s*/i, '').trim();

				return {
					title: clean(a.getAttribute('title') || a.textContent || ''),
					company: clean((card.querySelector('a.company-name') || {}).textContent || ''),
					salary_raw: salaryText,
					location_text: clean((card.querySelector('.location ul li') || {}).textContent || ''),
					link: a.getAttribute('href') || a.href || ''
				};
			}).filter(Boolean);
		}
		"""
	)

	parsed = []
	for item in items:
		link = item.get("link", "")
		if link.startswith("/"):
			link = urljoin(seed_url, link)
		item["link"] = _clean_link(link)
		parsed.append(item)
	return parsed


async def main() -> None:
	client = get_mongo_client()
	collection = get_raw_collection(client)
	ensure_raw_collection_index(collection)

	async with async_playwright() as pw:
		browser = await pw.chromium.launch(headless=True)
		page = await browser.new_page()

		for seed_url in TARGET_URLS:
			industry_id = "1" if "-c1-" in seed_url else "63"
			seen_links = set()
			no_new_data_pages = 0

			for page_num in range(1, MAX_PAGES + 1):
				url = build_page_url(seed_url, page_num)
				logger.info("CareerViet industry %s page %s: %s", industry_id, page_num, url)

				try:
					await page.goto(url, timeout=60000)
				except Exception as exc:
					logger.warning("Load error on CareerViet industry %s page %s: %s", industry_id, page_num, exc)
					break

				items = await parse_listing(page, seed_url)
				if not items:
					break

				jobs_batch = []
				new_links_count = 0
				for item in items:
					link = item.get("link", "")
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
							"industry_id": industry_id,
							"source": "CareerViet",
							"has_quantitative_salary": has_quantitative_salary(salary_raw),
						}
					)

				bulk_upsert_jobs(collection, jobs_batch)
				if new_links_count == 0:
					no_new_data_pages += 1
					if no_new_data_pages >= MAX_NO_NEW_PAGES:
						logger.info(
							"Stop CareerViet industry %s: %s consecutive pages without new links.",
							industry_id,
							MAX_NO_NEW_PAGES,
						)
						break
				else:
					no_new_data_pages = 0

				logger.info("CareerViet industry %s page %s -> %s new jobs", industry_id, page_num, new_links_count)
				await page.wait_for_timeout(random.randint(1500, 3500))

		await browser.close()

	client.close()


if __name__ == "__main__":
	asyncio.run(main())
