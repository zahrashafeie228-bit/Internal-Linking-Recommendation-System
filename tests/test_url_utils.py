"""
Unit tests for URL normalization and pattern filtering.
"""

from src.utils.url_utils import normalize_url, matches_exclude_patterns, matches_include_patterns
from src.utils.validators import is_valid_url, is_same_domain


def test_normalize_url():
    base = "https://example.com/blog/article-1"
    
    # Relative URL resolution
    assert normalize_url("../page-2", base) == "https://example.com/page-2"
    
    # Fragment stripping & scheme/netloc lowercase
    assert normalize_url("https://EXAMPLE.COM/test/#section") == "https://example.com/test/"
    
    # Query parameter stripping (utm_)
    url_with_utm = "https://example.com/product?utm_source=google&id=12"
    normalized = normalize_url(url_with_utm)
    assert "utm_source" not in normalized
    assert "id=12" in normalized


def test_exclude_patterns():
    patterns = [r"/category/", r"/tag/", r"\.pdf$"]
    assert matches_exclude_patterns("https://example.com/category/news", patterns) is True
    assert matches_exclude_patterns("https://example.com/tag/seo", patterns) is True
    assert matches_exclude_patterns("https://example.com/file.pdf", patterns) is True
    assert matches_exclude_patterns("https://example.com/article/persian-nlp", patterns) is False


def test_validators():
    valid, err = is_valid_url("https://google.com")
    assert valid is True
    
    valid, err = is_valid_url("ftp://google.com")
    assert valid is False
    
    valid, err = is_valid_url("http://127.0.0.1")
    assert valid is False
    
    assert is_same_domain("https://example.com/p1", "http://www.example.com/p2") is True
    assert is_same_domain("https://example.com", "https://another-domain.com") is False
