"""
Persian text preprocessor, tokenizer, stopword remover, and lemmatizer.
"""

import re
import string
from typing import List, Set, Optional

from src.nlp.persian_normalizer import PersianNormalizer
from src.utils.helpers import setup_logger

logger = setup_logger("text_preprocessor")

try:
    from hazm import WordTokenizer, Lemmatizer, stopwords_list
    HAZM_NLP_AVAILABLE = True
except ImportError:
    HAZM_NLP_AVAILABLE = False


# Default Persian Stopwords fallback list
PERSIAN_STOPWORDS = {
    "از", "به", "با", "در", "برای", "که", "این", "آن", "است", "بود",
    "شد", "را", "یک", "می", "نمود", "گردید", "خود", "نیز", "تا",
    "بر", "هم", "یا", "اما", "ولی", "اگر", "چون", "باید", "شاید",
    "پس", "چند", "همه", "هیچ", "هر", "بین", "روی", "تحت", "طی",
    "برخی", "بسیار", "دیگر", "توسط", "درباره", "جهت", "وی", "آنها",
    "ایشان", "ما", "شما", "من", "او", "آنان", "حتی", "چرا", "چه",
    "کدام", "کی", "کجا", "چگونه", "بی", "بدون", "طبق", "بنابر",
    "بر اساس", "داشته", "داشت", "دارد", "کرده", "کرد", "کند", "خواهند",
    "بوده", "هستند", "باشند", "می‌شود", "می‌کند", "می‌باشد", "می‌دهد",
    "سایت", "صفحه", "کاربر", "کلیک", "پست", "مطلب"
}

if HAZM_NLP_AVAILABLE:
    try:
        hazm_stops = set(stopwords_list())
        PERSIAN_STOPWORDS.update(hazm_stops)
    except Exception:
        pass


class TextPreprocessor:
    """Full text preprocessing pipeline for Persian web content."""

    def __init__(
        self,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        remove_digits: bool = False,
        apply_lemmatization: bool = True,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        self.normalizer = PersianNormalizer()
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.apply_lemmatization = apply_lemmatization

        self.stopwords = set(PERSIAN_STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        self.tokenizer = None
        self.lemmatizer = None
        if HAZM_NLP_AVAILABLE:
            try:
                self.tokenizer = WordTokenizer()
            except Exception:
                self.tokenizer = None

            if apply_lemmatization:
                try:
                    self.lemmatizer = Lemmatizer()
                except Exception:
                    self.lemmatizer = None

    def preprocess_tokens(self, text: str) -> List[str]:
        """
        Preprocess text and return clean list of tokens.
        """
        if not text or not isinstance(text, str):
            return []

        # 1. Normalize
        normalized_text = self.normalizer.normalize(text)

        # 2. Convert digits if remove_digits is requested
        if self.remove_digits:
            normalized_text = self.normalizer.normalize_digits_to_english(normalized_text)
            normalized_text = re.sub(r"\d+", "", normalized_text)

        # 3. Remove Punctuation
        if self.remove_punctuation:
            persian_punctuation = "،؛؟«»٪" + string.punctuation
            table = str.maketrans("", "", persian_punctuation)
            normalized_text = normalized_text.translate(table)

        # 4. Tokenize
        if self.tokenizer:
            try:
                tokens = self.tokenizer.tokenize(normalized_text)
            except Exception:
                tokens = normalized_text.split()
        else:
            tokens = normalized_text.split()

        clean_tokens = []
        for token in tokens:
            token = token.strip()
            if not token or len(token) < 2:
                continue

            # Remove stopwords
            if self.remove_stopwords and token in self.stopwords:
                continue

            # Lemmatize
            if self.lemmatizer:
                try:
                    token = self.lemmatizer.lemmatize(token).split("#")[0]
                except Exception:
                    pass

            if token and len(token) >= 2 and token not in self.stopwords:
                clean_tokens.append(token)

        return clean_tokens

    def preprocess_text(self, text: str) -> str:
        """Return preprocessed text as a space-separated string."""
        tokens = self.preprocess_tokens(text)
        return " ".join(tokens)
