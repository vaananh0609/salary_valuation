import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import re
import unicodedata
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import UpdateOne
from src.config.db_settings import get_mongo_client, get_raw_collection
import logging

# Cấu hình hệ thống ghi nhật ký
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Danh sách từ dừng (stop-words) tiếng Việt chuyên ngành tuyển dụng
# Hỗ trợ thuật toán TF-IDF loại bỏ nhiễu ngữ pháp, tăng độ phân giải ma trận
VIETNAMESE_STOP_WORDS = [
    "tuyển", "gấp", "nhân", "viên", "chuyên", "thực", "tập", "sinh",
    "mức", "lương", "địa", "điểm", "làm", "việc", "tại", "công", "ty",
    "cổ", "phần", "tnhh", "trách", "nhiệm", "hữu", "hạn", "tập", "đoàn",
    "chi", "nhánh", "kỹ", "sư", "phát", "triển", "lập", "trình", "developer", "engineer"
]

def preprocess_text(text) -> str:
    """Chuẩn hóa chuỗi, loại bỏ dấu và ký tự đặc biệt."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8").lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_time_overlap(start1, end1, start2, end2) -> bool:
    """Kiểm tra sự giao thoa trên trục thời gian của hai chu kỳ tuyển dụng."""
    if pd.isna(start1) or pd.isna(end1) or pd.isna(start2) or pd.isna(end2):
        return False
    return (start1 <= end2) and (start2 <= end1)

def run_deduplication(days_lookback: int = 14) -> list:
    logger.info("Khởi động tiến trình khử trùng lặp tối ưu hóa (Incremental & Blocking)...")

    client = get_mongo_client()
    raw_collection = get_raw_collection(client)

    # Truy vấn tăng dần bằng Index nhằm giới hạn mức tiêu thụ bộ nhớ
    cutoff_date = datetime.utcnow() - timedelta(days=days_lookback)
    query = {"last_seen_at": {"$gte": cutoff_date}}

    cursor = raw_collection.find(query, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        logger.warning("Không có dữ liệu mới để xử lý.")
        client.close()
        return []

    initial_count = len(df)
    logger.info(f"Đã nạp {initial_count} bản ghi từ cơ sở dữ liệu để xử lý.")

    # Xử lý an toàn các trường dữ liệu khuyết thiếu để tránh lỗi dừng đột ngột
    df.fillna("", inplace=True)

    # Ép kiểu dữ liệu mốc thời gian
    df['first_crawled_at'] = pd.to_datetime(df['first_crawled_at'])
    df['last_seen_at'] = pd.to_datetime(df['last_seen_at'])

    # Tiền xử lý ngôn ngữ
    df['clean_title'] = df['title'].apply(preprocess_text)
    df['clean_company'] = df['company'].apply(preprocess_text)
    df['clean_location'] = df['location_text'].apply(preprocess_text)

    # Loại bỏ trùng lặp nguyên bản ban đầu dựa trên đường dẫn URL
    df.drop_duplicates(
        subset=["clean_title", "clean_company", "clean_location", "link"],
        keep="first",
        inplace=True,
    )

    # ---------------------------
    # LỚP 1: KHỬ TRÙNG LẶP TUYỆT ĐỐI
    # ---------------------------
    # Sắp xếp last_seen_at giảm dần (False) để idx1 luôn là bản ghi mới nhất
    df.sort_values(
        by=['clean_title', 'clean_company', 'clean_location', 'last_seen_at'],
        ascending=[True, True, True, False],
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    to_drop_absolute = set()
    grouped_absolute = df.groupby(['clean_title', 'clean_company', 'clean_location'])

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
                # Nếu giao thoa thời gian -> loại bỏ bản ghi idx2 (bản ghi cũ hơn)
                if is_time_overlap(
                    df.at[idx1, 'first_crawled_at'], df.at[idx1, 'last_seen_at'],
                    df.at[idx2, 'first_crawled_at'], df.at[idx2, 'last_seen_at']
                ):
                    to_drop_absolute.add(idx2)

    df.drop(index=list(to_drop_absolute), inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Số bản ghi sau lớp lọc tuyệt đối: {len(df)}")

    # ---------------------------
    # LỚP 2: KHỬ TRÙNG LẶP TƯƠNG ĐỐI
    # ---------------------------
    df['feature_string'] = df['clean_title'] + " " + df['clean_company'] + " " + df['clean_location']

    # Xây dựng khóa phân nhóm (Block Key) an toàn
    # Sử dụng toàn vẹn chuỗi tên công ty và địa điểm, tránh việc cắt xén ký tự gây phân nhóm sai
    df["_block_key"] = df.apply(
        lambda r: f"{r['clean_company']}||{r['clean_location']}"
        if r["clean_company"]
        else f"__anon__||{r['clean_location']}||{r.get('industry_id', '')}",
        axis=1,
    )

    to_drop_relative = set()
    
    # Khởi tạo mô hình lượng hóa văn bản với từ điển tiếng Việt
    vectorizer = TfidfVectorizer(
        stop_words=VIETNAMESE_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=1
    )

    grouped_company = df.groupby('_block_key')

    for block_key, group in grouped_company:
        if len(group) <= 1:
            continue

        indices = group.index.tolist()
        texts = df.loc[indices, 'feature_string']

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            cosine_sim = cosine_similarity(tfidf_matrix)
        except ValueError:
            # Xử lý trường hợp ngoại lệ khi văn bản chỉ chứa toàn từ dừng (stop_words)
            continue

        threshold = 0.85
        for i in range(len(indices)):
            idx1 = indices[i]
            if idx1 in to_drop_relative:
                continue
            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                if cosine_sim[i, j] > threshold:
                    # Bổ sung điều kiện đối khớp chu kỳ tuyển dụng
                    if is_time_overlap(
                        df.at[idx1, 'first_crawled_at'], df.at[idx1, 'last_seen_at'],
                        df.at[idx2, 'first_crawled_at'], df.at[idx2, 'last_seen_at']
                    ):
                        to_drop_relative.add(idx2)

    df_cleaned = df.drop(index=list(to_drop_relative))
    final_count = len(df_cleaned)
    logger.info(f"Số bản ghi tinh khiết cuối cùng: {final_count}")
    logger.info(f"Đã loại bỏ tổng cộng: {initial_count - final_count} bản ghi trùng lặp.")

    # ---------------------------
    # LƯU TRỮ VÀO VÙNG STAGING
    # ---------------------------
    db = client[raw_collection.database.name]
    staging_collection = db["staging_jobs"]

    # Loại bỏ trường _block_key mang tính tạm thời trước khi đưa vào cơ sở dữ liệu
    staging_records = df_cleaned.drop(columns=["_block_key"], errors="ignore").to_dict("records")

    if staging_records:
        # Cập nhật theo lô sử dụng toán tử Upsert
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
    logger.info("Đã đồng bộ dữ liệu tinh chế vào vùng Staging thành công.")

    # Trả về danh sách các đường dẫn phục vụ cho quá trình thu thập chuyên sâu
    return df_cleaned["link"].tolist()

if __name__ == "__main__":
    run_deduplication()