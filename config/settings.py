"""
Configuration settings for Persian Web Internal Link Recommendation System.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EXPORTS_DIR = DATA_DIR / "exports"
SAMPLES_DIR = DATA_DIR / "samples"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, EXPORTS_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Crawler Settings
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_REQUEST_TIMEOUT = 12  # seconds
DEFAULT_MAX_URLS = 50
DEFAULT_CRAWL_DELAY = 0.5  # seconds
DEFAULT_MAX_RETRIES = 3

DEFAULT_EXCLUDE_PATTERNS = [
    r"/tag/",
    r"/category/",
    r"/author/",
    r"/page/",
    r"/cart/",
    r"/checkout/",
    r"/wp-content/",
    r"/wp-json/",
    r"\.jpg$",
    r"\.png$",
    r"\.pdf$",
]

# Similarity & Recommendation Defaults
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.35
DEFAULT_HYBRID_ALPHA = 0.7  # 0.7 * semantic + 0.3 * tfidf

# Similarity Category Thresholds
SIMILARITY_CATEGORIES = {
    "Very High": 0.75,
    "High": 0.55,
    "Medium": 0.35,
    "Low": 0.0,
}

# Similarity Category Persian Labels
SIMILARITY_CATEGORIES_PERSIAN = {
    "Very High": "خیلی بالا",
    "High": "بالا",
    "Medium": "متوسط",
    "Low": "پایین",
}

# NLP & Sentence Transformers Settings
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_CHUNK_WORDS = 250  # Word limit per chunk for sentence transformer embedding

# TF-IDF Defaults
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 0.95
DEFAULT_MAX_FEATURES = 5000

# Orphan / Weak Link Threshold
DEFAULT_WEAK_LINK_THRESHOLD = 2

# Evaluation Defaults
EVALUATION_LABELS = {
    "RELEVANT": "مرتبط",
    "PARTIALLY_RELEVANT": "نیمه‌مرتبط",
    "NOT_RELEVANT": "نامرتبط",
}
