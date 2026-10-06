"""
Sitemap XML and Sitemap Index parser with recursive nested sitemap support.
"""

import gzip
import io
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple
import requests
from bs4 import BeautifulSoup

from src.utils.url_utils import (
    normalize_url,
    matches_exclude_patterns,
    matches_include_patterns,
    extract_domain,
)
from src.utils.validators import is_valid_url
from src.utils.helpers import setup_logger

logger = setup_logger("sitemap_parser")


def is_sitemap_url(url: str) -> bool:
    """Check if a URL is a sitemap XML index or sub-sitemap rather than a web page."""
    if not url:
        return False
    url_lower = url.lower()
    path = url_lower.split("?")[0].split("#")[0]
    
    if "sitemap" in path or path.endswith(".xml") or path.endswith(".xml.gz"):
        return True
    return False


class SitemapParser:
    """Parser for standard XML sitemaps, sitemap indexes, and nested sitemaps."""

    def __init__(
        self,
        sitemap_url: str,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        timeout: int = 15,
        max_urls: int = 100,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.sitemap_url = sitemap_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_urls = max_urls
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        
        self.visited_sitemaps: Set[str] = set()
        self.discovered_urls: List[Dict[str, str]] = []
        self.seen_urls: Set[str] = set()

    def parse(self) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Parse sitemap URL recursively.
        
        Returns:
            (urls_list, error_messages)
        """
        errors: List[str] = []

        is_valid, err = is_valid_url(self.sitemap_url)
        if not is_valid:
            return [], [f"آدرس سایت‌مپ نامعتبر است: {err}"]

        self._process_sitemap(self.sitemap_url, errors)

        logger.info(f"Discovered {len(self.discovered_urls)} eligible URLs from sitemap.")
        return self.discovered_urls[: self.max_urls], errors

    def _fetch_content(self, url: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch raw content from URL, handling redirects and timeouts."""
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()

            content = response.content
            # Check if gzipped content
            if url.endswith(".gz") or response.headers.get("Content-Type", "").endswith("gzip"):
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
                        content = gz.read()
                except Exception:
                    pass

            return content, None
        except requests.exceptions.Timeout:
            return None, f"پاسخگویی سایت‌مپ {url} به دلیل TimeOut متوقف شد."
        except requests.exceptions.HTTPError as e:
            return None, f"خطای HTTP ({response.status_code}) در دریافت سایت‌مپ {url}"
        except Exception as e:
            return None, f"خطا در ارتباط با سایت‌مپ {url}: {str(e)}"

    def _process_sitemap(self, url: str, errors: List[str]):
        """Recursively process a single sitemap or sitemap index."""
        normalized_sitemap_url = normalize_url(url)
        if not normalized_sitemap_url or normalized_sitemap_url in self.visited_sitemaps:
            return
        
        if len(self.seen_urls) >= self.max_urls:
            return

        self.visited_sitemaps.add(normalized_sitemap_url)

        content, err = self._fetch_content(url)
        if err:
            errors.append(err)
            return

        if not content:
            errors.append(f"محتوای سایت‌مپ {url} خالی است.")
            return

        # Attempt to parse XML / HTML content
        try:
            soup = BeautifulSoup(content, "lxml-xml")
            
            # 1. Check <sitemap> tags for nested sitemaps (Rank Math, Yoast, WordPress)
            sitemap_tags = soup.find_all("sitemap")
            if sitemap_tags:
                for sm in sitemap_tags:
                    loc = sm.find("loc")
                    if loc and loc.text:
                        nested_url = loc.text.strip()
                        if len(self.seen_urls) >= self.max_urls:
                            break
                        self._process_sitemap(nested_url, errors)
                return

            # 2. Check <url> tags for webpage URLs
            url_tags = soup.find_all("url")
            if url_tags:
                for u in url_tags:
                    if len(self.seen_urls) >= self.max_urls:
                        break

                    loc = u.find("loc")
                    if loc and loc.text:
                        page_url = loc.text.strip()
                        normalized_page_url = normalize_url(page_url)

                        if not normalized_page_url or normalized_page_url in self.seen_urls:
                            continue

                        # If this URL is actually a sub-sitemap (e.g., post-sitemap.xml), recurse into it!
                        if is_sitemap_url(normalized_page_url) and normalized_page_url != normalized_sitemap_url:
                            self._process_sitemap(normalized_page_url, errors)
                            continue

                        # Filter by include/exclude rules
                        if matches_exclude_patterns(normalized_page_url, self.exclude_patterns):
                            continue
                        if not matches_include_patterns(normalized_page_url, self.include_patterns):
                            continue

                        lastmod_node = u.find("lastmod")
                        lastmod = lastmod_node.text.strip() if lastmod_node and lastmod_node.text else ""

                        self.seen_urls.add(normalized_page_url)
                        self.discovered_urls.append({
                            "url": normalized_page_url,
                            "lastmod": lastmod,
                        })
                return

            # 3. Fallback: Parse any <loc> or <a> tags
            loc_tags = soup.find_all(["loc", "a"])
            for loc in loc_tags:
                if len(self.seen_urls) >= self.max_urls:
                    break
                loc_text = (loc.get("href") or loc.text or "").strip()
                if not loc_text or loc_text.startswith("#") or loc_text.startswith("javascript:"):
                    continue

                normalized_loc = normalize_url(loc_text)
                if not normalized_loc:
                    continue

                if is_sitemap_url(normalized_loc) and normalized_loc != normalized_sitemap_url:
                    self._process_sitemap(normalized_loc, errors)
                else:
                    if normalized_loc not in self.seen_urls:
                        if matches_exclude_patterns(normalized_loc, self.exclude_patterns):
                            continue
                        if not matches_include_patterns(normalized_loc, self.include_patterns):
                            continue

                        self.seen_urls.add(normalized_loc)
                        self.discovered_urls.append({"url": normalized_loc, "lastmod": ""})

        except Exception as e:
            errors.append(f"خطا در پردازش XML سایت‌مپ {url}: {str(e)}")
