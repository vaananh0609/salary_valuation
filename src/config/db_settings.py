import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Tat ca ket noi duoc doc tu bien moi truong de tranh hard-code thong tin nhay cam.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
RAW_DB_NAME = os.getenv("RAW_DB_NAME", "jobs")
RAW_COLLECTION_NAME = os.getenv("RAW_COLLECTION_NAME", "raw_jobs")


def get_mongo_client() -> MongoClient:
    """Khoi tao va tra ve ket noi den he quan tri co so du lieu MongoDB."""
    return MongoClient(MONGO_URI)


def get_raw_collection(client: MongoClient):
    """
    Truy xuat collection du lieu tho va tu dong thiet lap khoa chong trung lap tuyet doi.
    """
    db = client[RAW_DB_NAME]
    col = db[RAW_COLLECTION_NAME]

    # Thiet lap chi muc duy nhat dua tren duong dan URL de ngan nap ban ghi lap nguyen ban
    try:
        col.create_index("link", unique=True)
    except Exception as e:
        print(f"Loi khoi tao chi muc co so du lieu: {e}")

    return col
