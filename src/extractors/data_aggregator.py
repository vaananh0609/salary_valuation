import asyncio
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

# Allow running this file directly (python path/to/data_aggregator.py)
# while keeping package-style imports for normal module execution.
if __package__ in (None, ""):
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)

from src.extractors import careerviet, joboko, topcv, vietnamworks

LOG_DIR = os.path.join(project_root if __package__ in (None, "") else os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "data_aggregator.log")

_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
_file_handler.setFormatter(_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
	root_logger.addHandler(_stream_handler)
	root_logger.addHandler(_file_handler)
else:
	# Ensure file logging is enabled even when handlers are pre-configured.
	handler_types = {type(h) for h in root_logger.handlers}
	if RotatingFileHandler not in handler_types:
		root_logger.addHandler(_file_handler)
	if logging.StreamHandler not in handler_types:
		root_logger.addHandler(_stream_handler)

logger = logging.getLogger(__name__)


async def _run_extractor(name: str, extractor_main) -> None:
	start = time.time()
	logger.info("[%s] start", name)
	await extractor_main()
	logger.info("[%s] done in %.2fs", name, time.time() - start)


async def run_all_extractors() -> None:
	"""Run all surface extractors concurrently."""
	logger.info("Khoi dong tien trinh thu thap du lieu song song da nen tang...")
	logger.info("Log file: %s", LOG_FILE)
	start_time = time.time()

	results = await asyncio.gather(
		_run_extractor("vietnamworks", vietnamworks.main),
		_run_extractor("topcv", topcv.main),
		_run_extractor("careerviet", careerviet.main),
		_run_extractor("joboko", joboko.main),
		return_exceptions=True,
	)

	extractor_names = ["vietnamworks", "topcv", "careerviet", "joboko"]
	for index, result in enumerate(results):
		if isinstance(result, Exception):
			logger.error("Luong trich xuat %s gap su co: %s", extractor_names[index], result)

	elapsed = time.time() - start_time
	logger.info("Hoan tat Pha 1: Trich xuat be mat. Tong thoi gian: %.2f giay.", elapsed)


if __name__ == "__main__":
	asyncio.run(run_all_extractors())
