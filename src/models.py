"""
Data models for internal link recommendation system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class PageData:
    """Represents a crawled web page."""
    page_id: int
    url: str
    title: str = ""
    meta_description: str = ""
    h1: str = ""
    raw_text: str = ""
    clean_text: str = ""
    word_count: int = 0
    status_code: int = 0
    crawl_status: str = "pending"  # "success", "failed", "skipped"
    canonical_url: str = ""
    crawl_error: Optional[str] = None
    headings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page data to dictionary."""
        return {
            "page_id": self.page_id,
            "url": self.url,
            "title": self.title,
            "meta_description": self.meta_description,
            "h1": self.h1,
            "word_count": self.word_count,
            "status_code": self.status_code,
            "crawl_status": self.crawl_status,
            "canonical_url": self.canonical_url,
            "crawl_error": self.crawl_error or "",
        }


@dataclass
class InternalLinkData:
    """Represents an existing internal link between two pages."""
    source_url: str
    target_url: str
    anchor_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "target_url": self.target_url,
            "anchor_text": self.anchor_text,
        }


@dataclass
class RecommendationData:
    """Represents an internal link recommendation."""
    source_url: str
    source_title: str
    target_url: str
    target_title: str
    tfidf_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    similarity_score: float = 0.0
    method: str = "tfidf"  # "tfidf", "semantic", "hybrid"
    existing_link: bool = False
    rank: int = 1
    similarity_category: str = "Low"
    common_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "target_title": self.target_title,
            "target_url": self.target_url,
            "similarity_score": round(self.similarity_score, 4),
            "tfidf_score": round(self.tfidf_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "hybrid_score": round(self.hybrid_score, 4),
            "method": self.method,
            "existing_link": self.existing_link,
            "similarity_category": self.similarity_category,
            "common_terms": ", ".join(self.common_terms),
        }


@dataclass
class EvaluationData:
    """Represents manual evaluation of a recommendation."""
    source_url: str
    target_url: str
    method: str
    label: str  # "Relevant", "Partially Relevant", "Not Relevant"
    numeric_score: float  # 1.0, 0.5, 0.0
    reviewed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "target_url": self.target_url,
            "method": self.method,
            "label": self.label,
            "numeric_score": self.numeric_score,
            "reviewed_at": self.reviewed_at,
        }


@dataclass
class ExperimentConfig:
    """Configuration record for academic reproducibility."""
    sitemap_url: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    max_urls: int = 50
    selected_method: str = "hybrid"
    embedding_model: str = ""
    tfidf_ngram_range: str = "(1, 2)"
    tfidf_max_features: int = 5000
    hybrid_alpha: float = 0.7
    similarity_threshold: float = 0.35
    top_k: int = 5
    discovered_urls_count: int = 0
    crawled_pages_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sitemap_url": self.sitemap_url,
            "timestamp": self.timestamp,
            "max_urls": self.max_urls,
            "selected_method": self.selected_method,
            "embedding_model": self.embedding_model,
            "tfidf_ngram_range": self.tfidf_ngram_range,
            "tfidf_max_features": self.tfidf_max_features,
            "hybrid_alpha": self.hybrid_alpha,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "discovered_urls_count": self.discovered_urls_count,
            "crawled_pages_count": self.crawled_pages_count,
        }
