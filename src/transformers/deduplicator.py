import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import re
import unicodedata
from datetime import datetime, timedelta
from pymongo import UpdateOne
from rapidfuzz import process, fuzz
import logging

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    LAYER2_AVAILABLE = True
except ImportError:
    LAYER2_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("SentenceTransformer not available, Layer 2 will be skipped")

from src.config.db_settings import get_mongo_client, get_raw_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dynamic canonical companies list (built during processing)
canonical_companies = []
company_dict = {}


# ========================================
# TEXT NORMALIZATION & COMPANY NORMALIZATION
# ========================================

def remove_accents(text):
    """Remove accents from unicode text (e.g., Việt Nam → Viet Nam)"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def preprocess_text(text: str) -> str:
    """Simple text preprocessing for title and location: lowercase + remove special chars"""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = remove_accents(text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Comprehensive text normalization: lowercase + remove accents + clean company prefixes.
    Used for company name normalization.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove accents
    text = remove_accents(text)
    
    # Remove special characters, keep only alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove common company legal form prefixes
    prefixes = [
        "cong ty", "tnhh", "co phan", "tap doan",
        "company", "corp", "corporation",
        "ltd", "llc", "inc", "group", "jsc",
        "co", "lld", "vn"
    ]
    
    for prefix in prefixes:
        text = re.sub(r'\b' + prefix + r'\b', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def normalize_company(company: str) -> str:
    """
    Normalize company name using fuzzy matching with RapidFuzz.
    Returns canonical form. Builds canonical list dynamically during processing.
    
    Example:
        "Công ty TNHH Samsung Electronics Việt Nam" → "samsung electronics vietnam"
        "Samsung Electronics Vietnam" → "samsung electronics vietnam"
    """
    global canonical_companies, company_dict
    
    if not isinstance(company, str) or not company.strip():
        return ""
    
    clean = normalize_text(company)
    
    if not clean:
        return ""
    
    # Check if already in company_dict (fast lookup)
    if clean in company_dict:
        return company_dict[clean]
    
    # First company
    if not canonical_companies:
        canonical_companies.append(clean)
        company_dict[clean] = clean
        return clean
    
    # Fuzzy match with existing canonical companies
    match, score, idx = process.extractOne(
        clean,
        canonical_companies,
        scorer=fuzz.token_sort_ratio
    )
    
    # Threshold 90 = very similar companies
    if score >= 90:
        company_dict[clean] = match
        return match
    else:
        # New canonical form found
        canonical_companies.append(clean)
        company_dict[clean] = clean
        return clean


def normalize_url(url: str) -> str:
    """Normalize URL: remove query params"""
    if not isinstance(url, str):
        return ""
    return url.split("?")[0].rstrip("/").lower()


def is_time_overlap(start1, end1, start2, end2) -> bool:
    """Check time overlap between two job posting periods"""
    if pd.isna(start1) or pd.isna(end1) or pd.isna(start2) or pd.isna(end2):
        return False
    return (start1 <= end2) and (start2 <= end1)


def run_deduplication(days_lookback: int = 14) -> list:
    """
    Main deduplication pipeline:
    1. Absolute dedup: exact title+company+location+time
    2. Relative dedup: semantic similarity with Sentence Transformers
    """
    logger.info("=" * 80)
    logger.info("Starting deduplication pipeline (Sentence Embedding + Company Normalization)")
    logger.info("=" * 80)

    client = get_mongo_client()
    raw_collection = get_raw_collection(client)

    # Incremental query
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
        logger.warning("No new data to process.")
        client.close()
        return []

    initial_count = len(df)
    logger.info(f"Loaded {initial_count} records")

    # Fill missing values
    df.fillna("", inplace=True)

    # Normalize timestamp
    df['first_crawled_at'] = pd.to_datetime(df['first_crawled_at'], errors='coerce')
    df['last_seen_at'] = pd.to_datetime(df['last_seen_at'], errors='coerce')

    # Normalize link
    df['link'] = df.get('link', '').apply(normalize_url)
    df.drop_duplicates(subset=['link'], keep='first', inplace=True)
    logger.info(f"After URL dedup: {len(df)}")

    # Text normalization
    df['clean_title'] = df['title'].apply(preprocess_text)
    df['clean_company'] = df['company'].apply(normalize_company)
    df['clean_location'] = df['location_text'].apply(preprocess_text)

    # ===================================
    # LAYER 1: ABSOLUTE DEDUPLICATION
    # ===================================
    logger.info("\n[Layer 1] Absolute deduplication (title+company+location)...")
    
    df.sort_values(
        by=['clean_title', 'clean_company', 'clean_location', 'last_seen_at'],
        ascending=[True, True, True, False],
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    to_drop_absolute = set()
    for _, group in df.groupby(['clean_title', 'clean_company', 'clean_location']):
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
                    df.at[idx1, 'first_crawled_at'], df.at[idx1, 'last_seen_at'],
                    df.at[idx2, 'first_crawled_at'], df.at[idx2, 'last_seen_at']
                ):
                    to_drop_absolute.add(idx2)

    df.drop(index=list(to_drop_absolute), inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"After layer 1: {len(df)}")

    # ===================================
    # LAYER 2: RELATIVE DEDUPLICATION
    # ===================================
    to_drop_relative = set()
    
    if LAYER2_AVAILABLE:
        logger.info("\n[Layer 2] Relative dedup (Sentence Embedding + Cosine Similarity)...")
        
        df['feature_string'] = df['clean_title'] + " " + df['clean_company']
        df['_block_key'] = df.apply(
            lambda r: f"{r['clean_company']}||{r['clean_location']}" 
            if r['clean_company'] else f"__anon__||{r['clean_location']}",
            axis=1
        )

        try:
            vectorizer_global = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

            for block_key, group in df.groupby('_block_key'):
                if len(group) <= 1:
                    continue

                indices = group.index.tolist()
                texts = df.loc[indices, 'feature_string'].tolist()

                try:
                    embeddings = vectorizer_global.encode(
                        texts,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    cosine_sim = cosine_similarity(embeddings)

                    threshold = 0.88
                    for i in range(len(indices)):
                        idx1 = indices[i]
                        if idx1 in to_drop_relative:
                            continue
                        for j in range(i + 1, len(indices)):
                            idx2 = indices[j]
                            if cosine_sim[i, j] > threshold:
                                if is_time_overlap(
                                    df.at[idx1, 'first_crawled_at'], df.at[idx1, 'last_seen_at'],
                                    df.at[idx2, 'first_crawled_at'], df.at[idx2, 'last_seen_at']
                                ):
                                    to_drop_relative.add(idx2)
                except Exception as e:
                    logger.warning(f"Error in block {block_key}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Layer 2 skipped: {e}")
    else:
        logger.info("\n[Layer 2] Skipped (SentenceTransformer not available)")

    df_cleaned = df.drop(index=list(to_drop_relative))
    final_count = len(df_cleaned)
    
    logger.info(f"After layer 2: {final_count}")
    logger.info(f"Total duplicates removed: {initial_count - final_count}")

    # ===================================
    # SAVE TO STAGING
    # ===================================
    logger.info("\n[Staging] Saving to MongoDB...")
    
    db = client[raw_collection.database.name]
    staging_collection = db["staging_jobs"]

    # Prepare records with company normalization
    columns_to_drop = ['_block_key', 'clean_title', 'clean_company', 'clean_location', 'feature_string']
    staging_records = []
    
    for _, row in df_cleaned.iterrows():
        record = row.drop(columns_to_drop, errors='ignore').to_dict()
        # Add company normalization fields
        record['company_raw'] = record.get('company', '')
        record['company_normalized'] = row.get('clean_company', '')
        staging_records.append(record)

    if staging_records:
        ops = [
            UpdateOne({"link": r["link"]}, {"$set": r}, upsert=True)
            for r in staging_records
        ]
        
        batch_size = 1000
        for i in range(0, len(ops), batch_size):
            staging_collection.bulk_write(ops[i:i+batch_size], ordered=False)

        # Log sample
        logger.info(f"\nSample records (first 2):")
        for idx, rec in enumerate(staging_records[:2]):
            logger.info(f"  [{idx}] company_raw='{rec.get('company_raw', 'N/A')}' → company_normalized='{rec.get('company_normalized', 'N/A')}'")

    # Log company dictionary for reference
    logger.info(f"\n[Company Mapping] Built {len(company_dict)} canonical forms:")
    for raw_form, canonical_form in sorted(company_dict.items())[:5]:
        if raw_form != canonical_form:
            logger.info(f"  '{raw_form}' → '{canonical_form}'")

    client.close()
    logger.info("\nDeduplication complete!")
    logger.info("=" * 80)

    return df_cleaned['link'].tolist()


def deduplicate_jobs(days_lookback: int = 14) -> list:
    """
    Wrapper function for backward compatibility.
    Calls run_deduplication with the specified lookback period.
    """
    return run_deduplication(days_lookback=days_lookback)


if __name__ == "__main__":
    run_deduplication()
