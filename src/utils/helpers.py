"""
Helper utilities for logging, formatting, and file handling.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


def setup_logger(name: str = "persian_link_sug", level: int = logging.INFO) -> logging.Logger:
    """Configure and return logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def format_persian_number(number: Union[int, float]) -> str:
    """Convert English digits to Persian digits."""
    english_digits = "0123456789"
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    translation_table = str.maketrans(english_digits, persian_digits)
    return str(number).translate(translation_table)


def save_json(data: Any, filepath: Union[Path, str]) -> bool:
    """Save data to JSON file with UTF-8 encoding."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_json(filepath: Union[Path, str]) -> Any:
    """Load data from JSON file."""
    try:
        path = Path(filepath)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
