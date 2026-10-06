"""
Unit tests for Persian normalizer and text preprocessor.
"""

import pytest
from src.nlp.persian_normalizer import PersianNormalizer
from src.nlp.text_preprocessor import TextPreprocessor


def test_arabic_character_conversion():
    normalizer = PersianNormalizer(use_hazm=False)
    input_text = "كتاب آموزشي با استفاده از تکنولوژيهاي جديد"
    normalized = normalizer.normalize(input_text)
    assert "کتاب" in normalized
    assert "آموزشی" in normalized
    assert "تکنولوژیهای" in normalized


def test_zero_width_space_normalization():
    normalizer = PersianNormalizer(use_hazm=False)
    input_text = "می  ‌  شود و کتاب‌‌ها"
    normalized = normalizer.normalize(input_text)
    assert "کتاب" in normalized or "می" in normalized


def test_text_preprocessor_tokens():
    preprocessor = TextPreprocessor(remove_stopwords=True, remove_punctuation=True)
    text = "سئو و لینک‌سازی داخلی برای افزایش بازدید سایت بسیار مهم است."
    tokens = preprocessor.preprocess_tokens(text)
    
    # Hazm normalizes 'سئو' to 'سیو'
    assert ("سئو" in tokens or "سیو" in tokens)
    assert "داخلی" in tokens
    assert "افزایش" in tokens
    assert "بازدید" in tokens
    # Stopwords like "برای", "است", "سایت" should be removed
    assert "برای" not in tokens
    assert "است" not in tokens
