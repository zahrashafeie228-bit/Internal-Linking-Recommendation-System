"""
Evaluation manager for recording and loading manual relevance judgments.
"""

from typing import Dict, List, Optional
from config.settings import CACHE_DIR
from src.models import EvaluationData
from src.utils.helpers import load_json, save_json, setup_logger

logger = setup_logger("evaluator")


class EvaluationStore:
    """Manages persistent manual evaluation labels in JSON storage."""

    def __init__(self, cache_file: str = "evaluations.json"):
        self.file_path = CACHE_DIR / cache_file
        self.evaluations: Dict[str, EvaluationData] = {}
        self.load()

    def _make_key(self, source_url: str, target_url: str, method: str) -> str:
        return f"{source_url}|||{target_url}|||{method}"

    def add_evaluation(
        self,
        source_url: str,
        target_url: str,
        method: str,
        label: str,
    ):
        """Record or update manual evaluation label."""
        numeric_score = 1.0 if label == "Relevant" else (0.5 if label == "Partially Relevant" else 0.0)
        
        data = EvaluationData(
            source_url=source_url,
            target_url=target_url,
            method=method,
            label=label,
            numeric_score=numeric_score,
        )
        
        key = self._make_key(source_url, target_url, method)
        self.evaluations[key] = data
        self.save()

    def get_evaluation(
        self, source_url: str, target_url: str, method: str
    ) -> Optional[EvaluationData]:
        key = self._make_key(source_url, target_url, method)
        return self.evaluations.get(key)

    def get_all(self) -> List[EvaluationData]:
        return list(self.evaluations.values())

    def save(self):
        dict_data = [e.to_dict() for e in self.evaluations.values()]
        save_json(dict_data, self.file_path)

    def load(self):
        raw = load_json(self.file_path)
        if raw and isinstance(raw, list):
            self.evaluations.clear()
            for item in raw:
                ev = EvaluationData(
                    source_url=item.get("source_url", ""),
                    target_url=item.get("target_url", ""),
                    method=item.get("method", ""),
                    label=item.get("label", "Not Relevant"),
                    numeric_score=float(item.get("numeric_score", 0.0)),
                    reviewed_at=item.get("reviewed_at", ""),
                )
                key = self._make_key(ev.source_url, ev.target_url, ev.method)
                self.evaluations[key] = ev
