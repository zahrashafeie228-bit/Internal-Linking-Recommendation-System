"""
DOM parsing and clean main-body text extraction using BeautifulSoup4.
"""

import re
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup, Comment, Tag


class ContentExtractor:
    """Extracts clean body text, metadata, headings, and internal links from page HTML."""

    @staticmethod
    def extract(html_content: str, page_url: str) -> Dict[str, any]:
        """
        Extract page metadata, main textual content, and headings from HTML safely.
        """
        if not html_content or not html_content.strip():
            return {
                "title": "",
                "meta_description": "",
                "h1": "",
                "headings": [],
                "raw_text": "",
                "clean_text": "",
                "word_count": 0,
                "canonical_url": "",
            }

        soup = BeautifulSoup(html_content, "lxml")

        # 1. Title
        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

        # 2. Meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
        if not meta_tag:
            meta_tag = soup.find("meta", attrs={"property": re.compile(r"og:description", re.I)})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

        # 3. Canonical URL
        canonical_url = ""
        canonical_tag = soup.find("link", attrs={"rel": re.compile(r"canonical", re.I)})
        if canonical_tag and canonical_tag.get("href"):
            canonical_url = canonical_tag["href"].strip()

        # 4. Extract Headings
        headings: List[str] = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            text = h.get_text().strip()
            if text and text not in headings:
                headings.append(text)

        h1 = ""
        h1_tag = soup.find("h1")
        if h1_tag:
            h1 = h1_tag.get_text().strip()

        # Save raw body text
        raw_text = soup.body.get_text(separator=" ", strip=True) if soup.body else soup.get_text(separator=" ", strip=True)

        # 5. Safe element removal (Remove scripts, styles, forms, header, footer, nav ONLY)
        for tag_name in ["script", "style", "noscript", "iframe", "svg", "form", "nav", "footer"]:
            for el in soup.find_all(tag_name):
                el.decompose()

        # Remove HTML comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Target main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|post|entry|main|body", re.I))
            or soup.find("body")
            or soup
        )

        # Extract text paragraphs
        clean_text_paragraphs = []
        for p in main_content.find_all(["p", "h1", "h2", "h3", "h4", "li", "span", "div"]):
            if p.name == "div" and p.find_all(["p", "div"]):
                continue
            text = p.get_text(separator=" ", strip=True)
            if text and len(text) > 5:
                clean_text_paragraphs.append(text)

        clean_text = " ".join(clean_text_paragraphs)
        if not clean_text.strip() or len(clean_text.split()) < 5:
            clean_text = main_content.get_text(separator=" ", strip=True)

        if not clean_text.strip() or len(clean_text.split()) < 5:
            clean_text = raw_text

        # Clean spaces
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        raw_text = re.sub(r"\s+", " ", raw_text).strip()

        word_count = len(clean_text.split())

        return {
            "title": title or page_url,
            "meta_description": meta_desc,
            "h1": h1,
            "headings": headings,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "word_count": word_count,
            "canonical_url": canonical_url,
        }
