import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load company dataset
def load_company_dataset():
    try:
        logger.info("Loading company dataset from HuggingFace...")
        dataset = load_dataset(
            "ThunderDrag/Vietnam-Stock-Symbols-and-Metadata",
            split="train"
        )
        df = dataset.to_pandas()
        companies = df["name"].dropna().unique().tolist()
        logger.info(f"Loaded {len(companies)} companies")
        return companies
    except Exception as e:
        logger.warning(f"Cannot load company dataset: {e}")
        return []

company_list = load_company_dataset()

# Build FAISS index
if company_list:
    logger.info("Building FAISS index...")
    company_embeddings = embedding_model.encode(
        company_list,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    dim = company_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(company_embeddings.astype(np.float32))
    logger.info(f"FAISS index built with {len(company_list)} companies")
else:
    faiss_index = None

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = str(text).lower().strip()
    # Remove special chars but keep letters/numbers/space
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_abbreviation(company: str) -> str:
    """
    Extract abbreviation from company name.
    Examples:
    - "Công ty Cổ phần Fpt" → "fpt"
    - "Samsung Electronics Vietnam" → "sev"
    - "Viettel" → "viettel"
    """
    if not isinstance(company, str):
        return ""
    
    company_clean = preprocess_text(company)
    words = company_clean.split()
    
    # If only 1 word, return as is
    if len(words) == 1:
        return words[0][:10]  # Truncate to 10 chars max
    
    # Try to extract first letters of main words (skip common suffixes)
    stop_words = {"co", "ltd", "inc", "corp", "inc", "jsc", "llc"}
    main_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if main_words:
        # Take first letter of each main word (up to 5 letters)
        abbrev = ''.join(w[0] for w in main_words[:5])
        return abbrev
    
    # Fallback: first 5 chars
    return company_clean[:5]

def normalize_company_name(company: str) -> str:
    """Normalize company using FAISS"""
    if not isinstance(company, str):
        return ""
    
    company_clean = preprocess_text(company)
    
    if not faiss_index or not company_list:
        return company_clean
    
    try:
        emb = embedding_model.encode(
            [company_clean],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        scores, indices = faiss_index.search(emb.astype(np.float32), 1)
        
        best_score = scores[0][0]
        best_idx = indices[0][0]
        
        threshold = 0.85
        if best_score >= threshold:
            normalized = preprocess_text(company_list[best_idx])
            return normalized
    except Exception as e:
        logger.debug(f"Error: {e}")
    
    return company_clean

def test_company_normalization():
    """Test company normalization with various examples"""
    
    test_cases = [
        "Công ty TNHH Samsung Electronics Việt Nam",
        "Samsung Electronics Vietnam",
        "Công ty Cổ phần FPT",
        "FPT Software",
        "Công ty Cổ phần CMC",
        "CMC Corporation",
        "Viettel",
        "Công ty Viettel",
        "ShinhanBank Vietnam",
        "Microsoft Vietnam Ltd",
        "Google Vietnam LLC",
    ]
    
    logger.info("=" * 100)
    logger.info("COMPANY NORMALIZATION TEST")
    logger.info("=" * 100)
    
    results = []
    
    for company_raw in test_cases:
        normalized = normalize_company_name(company_raw)
        abbreviation = extract_abbreviation(normalized)
        
        results.append({
            'raw': company_raw,
            'normalized': normalized,
            'abbreviation': abbreviation
        })
        
        logger.info(f"\nRaw:        {company_raw}")
        logger.info(f"Normalized: {normalized}")
        logger.info(f"Abbrev:     {abbreviation}")
    
    # Print as table
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 100)
    
    df_results = pd.DataFrame(results)
    logger.info("\n" + df_results.to_string(index=False))
    
    return df_results

if __name__ == "__main__":
    test_company_normalization()
