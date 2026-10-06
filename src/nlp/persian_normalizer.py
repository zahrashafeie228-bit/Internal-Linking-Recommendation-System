"""
Persian character and text normalizer with Hazm integration and pure-Python fallback.
"""

import html
import re
from typing import Optional

try:
    from hazm import Normalizer as HazmNormalizer
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False


class PersianNormalizer:
    """Normalizes Persian characters, zero-width spaces, numbers, and HTML entities."""

    def __init__(
        self,
        character_refinement: bool = True,
        punctuation_spacing: bool = True,
        remove_extra_spaces: bool = True,
        use_hazm: bool = True,
    ):
        self.character_refinement = character_refinement
        self.punctuation_spacing = punctuation_spacing
        self.remove_extra_spaces = remove_extra_spaces
        
        self.hazm_normalizer = None
        if HAZM_AVAILABLE and use_hazm:
            try:
                self.hazm_normalizer = HazmNormalizer()
            except Exception:
                self.hazm_normalizer = None

    def normalize(self, text: str) -> str:
        """
        Normalize Persian text.
        
        Steps:
        1. Decode HTML entities.
        2. Standardize Arabic characters (ك -> ک, ي -> ی, ۀ -> ه, etc.).
        3. Standardize zero-width spaces (ZWNJS \u200c).
        4. Normalize Arabic/English digits if needed.
        5. Apply Hazm normalizer if available.
        6. Clean up white spaces.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Unescape HTML entities
        text = html.unescape(text)

        # 2. Arabic to Persian character mapping
        arabic_to_persian = {
            "ي": "ی",
            "ى": "ی",
            "ئ": "ی",
            "ك": "ک",
            "كب": "کب",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا",
            "آ": "آ",
            "ٱ": "ا",
            "ۀ": "ه",
            "ە": "ه",
            "‌": "\u200c",  # standard ZWNJS
        }
        for ar, fa in arabic_to_persian.items():
            text = text.replace(ar, fa)

        # 3. Clean duplicate ZWNJS or ZWNJS at word edges
        text = re.sub(r"[\u200c]{2,}", "\u200c", text)
        text = re.sub(r"\s+\u200c+|\u200c+\s+", " ", text)

        # 4. Use Hazm normalizer if available
        if self.hazm_normalizer:
            try:
                text = self.hazm_normalizer.normalize(text)
            except Exception:
                pass

        # 5. Clean up extra whitespace & linebreaks
        if self.remove_extra_spaces:
            text = re.sub(r"[\r\n\t]+", " ", text)
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def normalize_digits_to_english(text: str) -> str:
        """Convert Persian and Arabic digits to English digits."""
        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        english_digits = "0123456789"
        
        table = str.maketrans(persian_digits + arabic_digits, english_digits * 2)
        return text.translate(table)
