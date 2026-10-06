"""
URL manipulation, normalization, and path matching helpers.
"""

import re
from urllib.parse import urlparse, urljoin, urlunparse


def normalize_url(url: str, base_url: str = "") -> str:
    """
    Normalize a URL by:
    1. Resolving relative paths against base_url if provided.
    2. Removing fragment (#hash).
    3. Standardizing scheme & netloc to lowercase.
    4. Removing trailing slash (unless it's root '/').
    5. Stripping unnecessary default ports.
    """
    if not url:
        return ""

    url = url.strip()

    # Handle relative URLs
    if base_url and not (url.startswith("http://") or url.startswith("https://")):
        url = urljoin(base_url, url)

    try:
        parsed = urlparse(url)

        # Ignore non-HTTP links (mailto:, tel:, javascript:, etc.)
        if parsed.scheme not in ("http", "https"):
            return ""

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        path = parsed.path or "/"

        # Standardize multiple slashes, but preserve trailing slash for WordPress canonical caching
        path = re.sub(r"/+", "/", path)

        # Standardize query parameters (keep query string sorted)
        query = parsed.query
        if query:
            # Drop tracking queries like utm_*
            query_params = [
                q for q in query.split("&")
                if not q.startswith(("utm_", "fbclid=", "gclid="))
            ]
            query = "&".join(sorted(query_params))

        # Omit fragments entirely
        fragment = ""

        normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
        return normalized

    except Exception:
        return ""


def extract_domain(url: str) -> str:
    """Extract domain from URL (e.g., https://example.com/page -> example.com)."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc
    except Exception:
        return ""


def matches_exclude_patterns(url: str, exclude_patterns: list) -> bool:
    """Check if URL matches any regex pattern in exclude list."""
    if not url or not exclude_patterns:
        return False

    for pattern in exclude_patterns:
        try:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        except re.error:
            # Invalid regex pattern, skip
            continue
    return False


def matches_include_patterns(url: str, include_patterns: list) -> bool:
    """Check if URL matches any regex pattern in include list. Returns True if list is empty."""
    if not include_patterns:
        return True
    
    for pattern in include_patterns:
        try:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False
