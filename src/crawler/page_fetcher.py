"""
Web page fetcher with retry logic, rate limiting, and HTTP status verification.
"""

import time
from typing import Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.utils.helpers import setup_logger
from src.utils.validators import is_valid_url, is_same_domain

logger = setup_logger("page_fetcher")


class PageFetcher:
    """Fetches individual HTML pages cleanly and safely."""

    def __init__(
        self,
        base_domain_url: str,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        timeout: int = 12,
        crawl_delay: float = 0.5,
        max_retries: int = 3,
    ):
        self.base_domain_url = base_domain_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.crawl_delay = crawl_delay
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        })

        retries = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504, 429],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch(self, url: str) -> Tuple[Optional[str], int, Optional[str]]:
        """
        Fetch HTML content from a URL.
        
        Returns:
            (html_content, status_code, error_message)
        """
        is_valid, err = is_valid_url(url)
        if not is_valid:
            return None, 0, f"URL نامعتبر: {err}"

        if not is_same_domain(url, self.base_domain_url):
            return None, 0, "خزش دامنه خارج از سایت اصلی مجاز نمی‌باشد."

        # Respect crawl delay
        if self.crawl_delay > 0:
            time.sleep(self.crawl_delay)

        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            status_code = response.status_code

            if status_code != 200:
                return None, status_code, f"کد وضعیت HTTP {status_code}"

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return None, status_code, f"نوع محتوا غیر از HTML است: {content_type}"

            # Retry with trailing slash if WordPress cache served empty stub (< 200 bytes)
            if len(response.content) < 200 and not url.endswith("/") and not any(url.endswith(ext) for ext in [".xml", ".html", ".php", ".png", ".jpg"]):
                url_slash = url + "/"
                try:
                    response_slash = self.session.get(url_slash, timeout=self.timeout, allow_redirects=True)
                    if response_slash.status_code == 200 and len(response_slash.content) > len(response.content):
                        response = response_slash
                except Exception:
                    pass

            # Proper UTF-8 decoding for Persian web pages to prevent Mojibake
            try:
                html_text = response.content.decode("utf-8")
            except UnicodeDecodeError:
                encoding = response.encoding or response.apparent_encoding or "utf-8"
                html_text = response.content.decode(encoding, errors="replace")

            return html_text, status_code, None

        except requests.exceptions.Timeout:
            return None, 408, "خطای زمان‌پاسخ‌دهی (Timeout) در خزش صفحه"
        except requests.exceptions.ConnectionError:
            return None, 503, "خطای عدم امکان برقراری ارتباط با سرور"
        except Exception as e:
            return None, 500, f"خطای پیش‌بینی نشده در خزش: {str(e)}"
