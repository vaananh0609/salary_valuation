import os
from pymongo import MongoClient

# Thiet lap bien ket noi cuc bo dua tren thong so Docker da khoi tao
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:password123@localhost:27017/")
RAW_DB_NAME = os.getenv("DB_NAME", "jobs")
RAW_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "raw_jobs")


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
