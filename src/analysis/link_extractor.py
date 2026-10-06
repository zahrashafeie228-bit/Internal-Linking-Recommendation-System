"""
Extraction and normalization of existing internal links from main article body text only.
"""

import re
from typing import Dict, List, Set, Tuple
from bs4 import BeautifulSoup, Comment

from src.models import InternalLinkData
from src.utils.url_utils import normalize_url, extract_domain
from src.utils.validators import is_same_domain
from src.utils.helpers import setup_logger

logger = setup_logger("link_extractor")


class LinkExtractor:
    """Extracts existing internal hyperlinks strictly from page main article content."""

    @staticmethod
    def extract_internal_links(
        html_content: str, source_url: str, base_domain_url: str
    ) -> List[InternalLinkData]:
        """
        Extract internal links ONLY from the main content body (ignoring header, footer, menus, sidebars).
        """
        if not html_content or not source_url:
            return []

        soup = BeautifulSoup(html_content, "lxml")

        # 1. Decompose non-body boilerplate elements
        for element in soup.find_all([
            "script", "style", "nav", "footer", "header", "aside",
            "form", "noscript", "iframe", "svg", "button"
        ]):
            element.decompose()

        # Remove HTML comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Decompose common sidebars, menus, and footer widgets by selector
        unwanted_selectors = [
            ".sidebar", ".widget", ".comments", "#comments",
            ".nav", ".menu", ".footer", "#footer", ".header", "#header",
            ".related-posts", ".social-share", ".popup", ".modal",
            ".author-box", ".site-header", ".site-footer", ".elementor-location-header",
            ".elementor-location-footer"
        ]
        for selector in unwanted_selectors:
            for el in soup.select(selector):
                el.decompose()

        # 2. Prioritize main article body element
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|post-body|entry-content|article|main", re.I))
            or soup.find("body")
            or soup
        )

        links: List[InternalLinkData] = []
        seen_targets: Set[str] = set()
        normalized_source = normalize_url(source_url)

        # 3. Extract links ONLY inside main content body
        for a_tag in main_content.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
                continue

            target_url = normalize_url(href, base_url=source_url)
            if not target_url:
                continue

            # Must belong to same domain
            if not is_same_domain(target_url, base_domain_url):
                continue

            # Ignore self-pointing links
            if target_url == normalized_source:
                continue

            if target_url in seen_targets:
                continue

            anchor_text = a_tag.get_text(separator=" ", strip=True)

            seen_targets.add(target_url)
            links.append(
                InternalLinkData(
                    source_url=normalized_source,
                    target_url=target_url,
                    anchor_text=anchor_text,
                )
            )

        return links
