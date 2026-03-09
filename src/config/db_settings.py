import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Tat ca ket noi duoc doc tu bien moi truong de tranh hard-code thong tin nhay cam.
MONGO_URI = os.getenv("MONGO_URI")
RAW_DB_NAME = os.getenv("RAW_DB_NAME", "recruitment_datalake")
RAW_COLLECTION_NAME = os.getenv("RAW_COLLECTION_NAME", "raw_jobs")

if not MONGO_URI:
    raise RuntimeError("Thieu bien moi truong MONGO_URI. Hay cap nhat file .env truoc khi chay.")

_RAW_INDEX_READY = False


def get_mongo_client() -> MongoClient:
    """Khoi tao va tra ve ket noi den he quan tri co so du lieu MongoDB."""
    return MongoClient(MONGO_URI)


def get_raw_collection(client: MongoClient):
    """
    Truy xuat collection du lieu tho.
    """
    db = client[RAW_DB_NAME]
    return db[RAW_COLLECTION_NAME]


def ensure_raw_collection_index(collection) -> None:
    """Khoi tao unique index cho truong link mot lan moi tien trinh."""
    global _RAW_INDEX_READY
    if _RAW_INDEX_READY:
        return

    try:
        existing_indexes = {idx.get("name") for idx in collection.list_indexes()}
        if "link_1" not in existing_indexes:
            collection.create_index("link", unique=True, name="link_1")
        _RAW_INDEX_READY = True
    except Exception as e:
        raise RuntimeError(f"Loi khoi tao chi muc MongoDB: {e}") from e
