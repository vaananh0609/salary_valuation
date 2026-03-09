import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import re
import unicodedata
from datetime import datetime, timedelta
from pymongo import UpdateOne
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from src.config.db_settings import get_mongo_client, get_raw_collection


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_url(url) -> str:
    """Chuẩn hóa URL"""
    if pd.isna(url) or not isinstance(url, str):
        return ""

    url = url.split("?")[0]
    return url.rstrip("/").lower()


def preprocess_text(text) -> str:
    """Chuẩn hóa text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_company_name(company: str) -> str:
    """Chuẩn hóa tên doanh nghiệp"""

    if pd.isna(company) or not isinstance(company, str):
        return ""

    company = company.lower()

    company = re.sub(
        r"\b(công ty|tnhh|cp|cổ phần|tập đoàn|corp|corporation|jsc|co\.,? ltd|ltd)\b",
        "",
        company,
    )

    return preprocess_text(company)


def is_time_overlap(start1, end1, start2, end2) -> bool:
    """Kiểm tra giao thoa thời gian"""

    if pd.isna(start1) or pd.isna(end1) or pd.isna(start2) or pd.isna(end2):
        return False

    return (start1 <= end2) and (start2 <= end1)


def run_deduplication(days_lookback: int = 14) -> list:

    logger.info("Khởi động tiến trình khử trùng lặp bằng Sentence Embedding...")

    client = get_mongo_client()
    raw_collection = get_raw_collection(client)

    cutoff_date = datetime.utcnow() - timedelta(days=days_lookback)

    query = {
        "$or": [
            {"last_seen_at": {"$gte": cutoff_date}},
            {"first_crawled_at": {"$gte": cutoff_date}},
        ]
    }

    cursor = raw_collection.find(query, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        logger.warning("Không có dữ liệu mới.")
        client.close()
        return []

    initial_count = len(df)

    logger.info(f"Nạp {initial_count} bản ghi.")

    df["title"] = df.get("title", "").fillna("")
    df["company"] = df.get("company", "").fillna("")
    df["location_text"] = df.get("location_text", "").fillna("")
    df["link"] = df.get("link", "").apply(normalize_url)

    # Loại duplicate URL
    df.drop_duplicates(subset=["link"], keep="first", inplace=True)

    logger.info(f"Sau lọc URL: {len(df)}")

    df["first_crawled_at"] = pd.to_datetime(df["first_crawled_at"], errors="coerce")
    df["last_seen_at"] = pd.to_datetime(df["last_seen_at"], errors="coerce")

    df["clean_title"] = df["title"].apply(preprocess_text)
    df["clean_company"] = df["company"].apply(normalize_company_name)
    df["clean_location"] = df["location_text"].apply(preprocess_text)

    # ---------------------------
    # LỚP 1: KHỬ TRÙNG LẶP TUYỆT ĐỐI
    # ---------------------------

    df.sort_values(
        by=["clean_title", "clean_company", "clean_location", "last_seen_at"],
        ascending=[True, True, True, False],
        inplace=True,
    )

    df.reset_index(drop=True, inplace=True)

    to_drop_absolute = set()

    grouped_absolute = df.groupby(
        ["clean_title", "clean_company", "clean_location"]
    )

    for _, group in grouped_absolute:

        if len(group) <= 1:
            continue

        indices = group.index.tolist()

        for i in range(len(indices)):

            idx1 = indices[i]

            if idx1 in to_drop_absolute:
                continue

            for j in range(i + 1, len(indices)):

                idx2 = indices[j]

                if is_time_overlap(
                    df.at[idx1, "first_crawled_at"],
                    df.at[idx1, "last_seen_at"],
                    df.at[idx2, "first_crawled_at"],
                    df.at[idx2, "last_seen_at"],
                ):
                    to_drop_absolute.add(idx2)

    df.drop(index=list(to_drop_absolute), inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Sau lớp lọc tuyệt đối: {len(df)}")

    # ---------------------------
    # LỚP 2: KHỬ TRÙNG LẶP TƯƠNG ĐỐI
    # ---------------------------

    df["feature_string"] = df["clean_title"] + " " + df["clean_company"]

    df["_block_key"] = df.apply(
        lambda r: f"{r['clean_company']}||{r['clean_location']}||{r['clean_title'][:20]}"
        if r["clean_company"]
        else f"__anon__||{r['clean_location']}||{r['clean_title'][:20]}",
        axis=1,
    )

    to_drop_relative = set()

    grouped_company = df.groupby("_block_key")

    for block_key, group in grouped_company:

        if len(group) <= 1:
            continue

        indices = group.index.tolist()

        texts = df.loc[indices, "feature_string"].tolist()

        try:

            embeddings = embedding_model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            cosine_sim = cosine_similarity(embeddings)

        except Exception as e:

            logger.warning(f"Lỗi embedding block {block_key}: {e}")

            continue

        threshold = 0.88

        for i in range(len(indices)):

            idx1 = indices[i]

            if idx1 in to_drop_relative:
                continue

            for j in range(i + 1, len(indices)):

                idx2 = indices[j]

                if cosine_sim[i, j] > threshold:

                    if is_time_overlap(
                        df.at[idx1, "first_crawled_at"],
                        df.at[idx1, "last_seen_at"],
                        df.at[idx2, "first_crawled_at"],
                        df.at[idx2, "last_seen_at"],
                    ):
                        to_drop_relative.add(idx2)

    df_cleaned = df.drop(index=list(to_drop_relative))

    final_count = len(df_cleaned)

    logger.info(f"Số bản ghi cuối: {final_count}")
    logger.info(f"Đã loại {initial_count - final_count} duplicate.")

    # ---------------------------
    # LƯU STAGING
    # ---------------------------

    db = client[raw_collection.database.name]

    staging_collection = db["staging_jobs"]

    columns_to_drop = [
        "_block_key",
        "clean_title",
        "clean_company",
        "clean_location",
        "feature_string",
    ]

    staging_records = (
        df_cleaned.drop(columns=columns_to_drop, errors="ignore")
        .to_dict("records")
    )

    if staging_records:

        ops = [
            UpdateOne({"link": r["link"]}, {"$set": r}, upsert=True)
            for r in staging_records
        ]

        batch_size = 1000

        for i in range(0, len(ops), batch_size):

            staging_collection.bulk_write(
                ops[i : i + batch_size],
                ordered=False,
            )

    client.close()

    logger.info("Đã lưu staging thành công.")

    return df_cleaned["link"].tolist()


if __name__ == "__main__":
    run_deduplication()