"""
Unit tests for similarity models, recommendation engine, self-link exclusion, and orphan detection.
"""

import numpy as np
from src.analysis.orphan_detector import OrphanDetector
from src.analysis.recommendation_engine import RecommendationEngine
from src.models import InternalLinkData
from src.nlp.tfidf_model import TFIDFSimilarityModel


def test_tfidf_and_recommendation_pipeline():
    sample_pages = [
        {
            "url": "https://example.com/seo-guide",
            "title": "راهنمای سئو سایت و لینک سازی",
            "clean_text": "سئو سایت شامل بهینه‌سازی موتورهای جستجو و لینک‌سازی داخلی است.",
            "word_count": 10,
        },
        {
            "url": "https://example.com/internal-links",
            "title": "استراتژی لینک سازی داخلی",
            "clean_text": "لینک سازی داخلی به بهبود رتبه صفحات در گوگل و انتقال اعتبار کمک می‌کند.",
            "word_count": 12,
        },
        {
            "url": "https://example.com/recipe-cake",
            "title": "دستور پخت کیک شکلاتی",
            "clean_text": "برای تهیه کیک شکلاتی خانگی به آرد و شکر و خامه نیاز دارید.",
            "word_count": 11,
        },
    ]

    # Fit TF-IDF model
    tfidf_model = TFIDFSimilarityModel()
    sim_matrix = tfidf_model.fit_transform(sample_pages)

    assert sim_matrix.shape == (3, 3)
    # Self-links must be 0
    assert np.all(np.diag(sim_matrix) == 0.0)

    # SEO guide and internal links should have higher similarity than cake recipe
    assert sim_matrix[0, 1] > sim_matrix[0, 2]

    # Generate recommendations
    recs = RecommendationEngine.generate_recommendations(
        pages=sample_pages,
        tfidf_matrix=sim_matrix,
        semantic_matrix=None,
        hybrid_matrix=None,
        existing_links_set=set(),
        method="tfidf",
        top_k=2,
        similarity_threshold=0.01,
        exclude_existing_links=True,
    )

    assert len(recs) > 0
    # Verify no self link recommendations exist
    for r in recs:
        assert r.source_url != r.target_url


def test_orphan_detector():
    pages = [
        {"url": "https://example.com/p1", "title": "Page 1", "word_count": 100},
        {"url": "https://example.com/p2", "title": "Page 2", "word_count": 120},
        {"url": "https://example.com/p3", "title": "Page 3", "word_count": 90},
    ]

    # Only p1 links to p2
    existing_links = [
        InternalLinkData(source_url="https://example.com/p1", target_url="https://example.com/p2")
    ]

    analysis = OrphanDetector.analyze_link_counts(pages, existing_links, weak_threshold=1)

    # p1 and p3 have 0 incoming links -> orphans
    # p2 has 1 incoming link -> not orphan
    assert len(analysis["orphans"]) == 2
    orphan_urls = [p["url"] for p in analysis["orphans"]]
    assert "https://example.com/p3" in orphan_urls
    assert "https://example.com/p1" in orphan_urls
    assert "https://example.com/p2" not in orphan_urls
