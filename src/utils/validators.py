"""
URL and input validation utilities.
"""

import ipaddress
import re
from urllib.parse import urlparse
from typing import Tuple


def is_valid_url(url: str, allow_private: bool = False) -> Tuple[bool, str]:
    """
    Validate that a URL is well-formed, uses http/https, and avoids prohibited local/file targets.
    
    Returns:
        (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "آدرس URL خالی است."
    
    url_trimmed = url.strip()
    if not (url_trimmed.startswith("http://") or url_trimmed.startswith("https://")):
        return False, "فقط پروتکل‌های http و https مجاز هستند."
    
    try:
        parsed = urlparse(url_trimmed)
        hostname = parsed.hostname
        
        if not hostname:
            return False, "نام دامنه در آدرس URL نامعتبر است."
            
        if not allow_private:
            # Check for localhost / loopback
            if hostname.lower() in ["localhost", "127.0.0.1", "::1"]:
                return False, "دسترسی به localhost مجاز نمی‌باشد."
                
            # Check IP address targets
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False, "دسترسی به آدرس‌های شبکه داخلی (Private Network) مجاز نمی‌باشد."
            except ValueError:
                # Hostname is a domain name, not an IP address
                pass
                
        return True, ""
    except Exception as e:
        return False, f"خطا در ساختار URL: {str(e)}"


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same base domain (ignoring 'www.' and subdomains)."""
    try:
        n1 = urlparse(url1).netloc.lower().replace("www.", "")
        n2 = urlparse(url2).netloc.lower().replace("www.", "")
        if not n1 or not n2:
            return False
        if n1 == n2:
            return True
        parts1 = n1.split(".")
        parts2 = n2.split(".")
        if len(parts1) >= 2 and len(parts2) >= 2:
            base1 = ".".join(parts1[-2:])
            base2 = ".".join(parts2[-2:])
            return base1 == base2
        return False
    except Exception:
        return False
