from datetime import datetime, timezone
import logging
from typing import Dict, List

from pymongo import UpdateOne

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def bulk_upsert_jobs(collection, jobs_batch: List[Dict]) -> None:
	"""
	Thuc thi co che ghi du lieu theo lo vao MongoDB.
	"""
	if not jobs_batch:
		logger.info("Khong co du lieu moi trong lo hien tai.")
		return

	operations = []
	current_time = datetime.now(timezone.utc)

	for job in jobs_batch:
		update_op = UpdateOne(
			{"link": job["link"]},
			{
				# Cap nhat thong tin moi va moc thoi gian vua nhin thay
				"$set": {
					**job,
					"last_seen_at": current_time,
				},
				# Chi ghi nhan thoi diem thu thap lan dau khi ban ghi duoc tao moi
				"$setOnInsert": {
					"first_crawled_at": current_time,
				},
			},
			upsert=True,
		)
		operations.append(update_op)

	try:
		result = collection.bulk_write(operations, ordered=False)
		logger.info(
			f"Da xu ly lo {len(jobs_batch)} ban ghi. "
			f"Chen moi: {result.upserted_count} | Cap nhat: {result.modified_count}."
		)
	except Exception as e:
		logger.error(f"Loi dinh tuyen khi nap du lieu theo lo: {e}")
