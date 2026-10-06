"""
Orphan and weakly-linked page detector based on crawled internal link structure.
"""

from typing import Dict, List, Set, Tuple
from config.settings import DEFAULT_WEAK_LINK_THRESHOLD
from src.models import InternalLinkData
from src.utils.helpers import setup_logger

logger = setup_logger("orphan_detector")


class OrphanDetector:
    """Analyzes incoming/outgoing internal link structure to identify orphans and weakly-linked pages."""

    @staticmethod
    def analyze_link_counts(
        pages: List[Dict[str, str]],
        existing_links: List[InternalLinkData],
        weak_threshold: int = DEFAULT_WEAK_LINK_THRESHOLD,
    ) -> Dict[str, any]:
        """
        Analyze link metrics for each page.
        
        Returns:
            Dict with page_metrics list, orphans list, weakly_linked list, summary counts.
        """
        if not pages:
            return {
                "page_metrics": [],
                "orphans": [],
                "weakly_linked": [],
                "total_orphans": 0,
                "total_weakly_linked": 0,
            }

        url_to_page = {p["url"]: p for p in pages}
        
        # Initialize link counters
        incoming_counts: Dict[str, int] = {p["url"]: 0 for p in pages}
        outgoing_counts: Dict[str, int] = {p["url"]: 0 for p in pages}

        for link in existing_links:
            src = link.source_url
            tgt = link.target_url

            if src in outgoing_counts:
                outgoing_counts[src] += 1
            if tgt in incoming_counts:
                incoming_counts[tgt] += 1

        page_metrics = []
        orphans = []
        weakly_linked = []

        for p in pages:
            url = p["url"]
            in_count = incoming_counts.get(url, 0)
            out_count = outgoing_counts.get(url, 0)
            
            is_orphan = (in_count == 0)
            is_weak = (0 < in_count <= weak_threshold)

            metric = {
                "url": url,
                "title": p.get("title", "") or url,
                "word_count": p.get("word_count", 0),
                "incoming_links": in_count,
                "outgoing_links": out_count,
                "is_orphan": is_orphan,
                "is_weakly_linked": is_weak,
                "status": "صفحه یتیم (بدون ورودی)" if is_orphan else ("لینک‌دهی ضعیف" if is_weak else "مناسب"),
            }
            page_metrics.append(metric)

            if is_orphan:
                orphans.append(metric)
            elif is_weak:
                weakly_linked.append(metric)

        return {
            "page_metrics": page_metrics,
            "orphans": orphans,
            "weakly_linked": weakly_linked,
            "total_orphans": len(orphans),
            "total_weakly_linked": len(weakly_linked),
        }
