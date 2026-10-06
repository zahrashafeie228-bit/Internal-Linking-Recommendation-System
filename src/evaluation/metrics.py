"""
Evaluation metrics calculation for recommendation quality (Precision@K, review counts).
"""

from typing import Dict, List
from src.models import EvaluationData, RecommendationData


def compute_precision_at_k(
    recommendations: List[RecommendationData],
    evaluations: List[EvaluationData],
    k: int = 5,
    partial_is_relevant: bool = True,
) -> Dict[str, float]:
    """
    Compute Precision@K for recommendations evaluated by user.
    
    Precision@K = (Count of Relevant items in top K evaluated) / (Count of evaluated items in top K)
    
    Returns dictionary mapping method -> precision_at_k score.
    """
    # Build evaluation lookup map: (source_url, target_url, method) -> score
    eval_map: Dict[tuple, float] = {}
    for ev in evaluations:
        key = (ev.source_url, ev.target_url, ev.method)
        score = ev.numeric_score
        if not partial_is_relevant and ev.label == "Partially Relevant":
            score = 0.0
        eval_map[key] = score

    method_totals: Dict[str, float] = {}
    method_counts: Dict[str, int] = {}

    for rec in recommendations:
        if rec.rank > k:
            continue

        key = (rec.source_url, rec.target_url, rec.method)
        if key in eval_map:
            score = eval_map[key]
            method = rec.method
            method_totals[method] = method_totals.get(method, 0.0) + score
            method_counts[method] = method_counts.get(method, 0) + 1

    precision_scores: Dict[str, float] = {}
    for method in ["tfidf", "semantic", "hybrid"]:
        count = method_counts.get(method, 0)
        total_score = method_totals.get(method, 0.0)
        if count > 0:
            precision_scores[method] = round(total_score / count, 4)
        else:
            precision_scores[method] = 0.0

    return precision_scores


def compute_evaluation_summary(evaluations: List[EvaluationData]) -> Dict[str, any]:
    """Calculate aggregate counts of evaluated recommendations."""
    total = len(evaluations)
    relevant = sum(1 for e in evaluations if e.label == "Relevant")
    partially_relevant = sum(1 for e in evaluations if e.label == "Partially Relevant")
    not_relevant = sum(1 for e in evaluations if e.label == "Not Relevant")

    by_method: Dict[str, Dict[str, int]] = {}
    for e in evaluations:
        m = e.method
        if m not in by_method:
            by_method[m] = {"total": 0, "relevant": 0, "partially_relevant": 0, "not_relevant": 0}
        by_method[m]["total"] += 1
        if e.label == "Relevant":
            by_method[m]["relevant"] += 1
        elif e.label == "Partially Relevant":
            by_method[m]["partially_relevant"] += 1
        else:
            by_method[m]["not_relevant"] += 1

    return {
        "total_reviews": total,
        "relevant_count": relevant,
        "partially_relevant_count": partially_relevant,
        "not_relevant_count": not_relevant,
        "by_method": by_method,
    }
