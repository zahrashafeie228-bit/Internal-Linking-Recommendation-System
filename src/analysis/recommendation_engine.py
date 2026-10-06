"""
Core recommendation engine for generating Top-K internal link recommendations.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from config.settings import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TOP_K, SIMILARITY_CATEGORIES
from src.models import RecommendationData, PageData
from src.nlp.tfidf_model import TFIDFSimilarityModel
from src.utils.helpers import setup_logger

logger = setup_logger("recommendation_engine")


def categorize_similarity(score: float) -> str:
    """Categorize numerical similarity score into readable level."""
    if score >= SIMILARITY_CATEGORIES["Very High"]:
        return "Very High"
    elif score >= SIMILARITY_CATEGORIES["High"]:
        return "High"
    elif score >= SIMILARITY_CATEGORIES["Medium"]:
        return "Medium"
    else:
        return "Low"


class RecommendationEngine:
    """Generates internal link recommendations from similarity matrices and existing link graphs."""

    @staticmethod
    def generate_recommendations(
        pages: List[Dict[str, str]],
        tfidf_matrix: np.ndarray,
        semantic_matrix: Optional[np.ndarray],
        hybrid_matrix: Optional[np.ndarray],
        existing_links_set: Set[Tuple[str, str]],
        method: str = "hybrid",  # "tfidf", "semantic", "hybrid"
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        exclude_existing_links: bool = True,
        tfidf_model: Optional[TFIDFSimilarityModel] = None,
    ) -> List[RecommendationData]:
        """
        Generate ranked internal link recommendations.
        
        Args:
            pages: List of page dictionary objects.
            tfidf_matrix: N x N matrix of TF-IDF similarities.
            semantic_matrix: N x N matrix of Semantic similarities.
            hybrid_matrix: N x N matrix of Hybrid similarities.
            existing_links_set: Set of (source_url, target_url) tuples for already present links.
            method: Selected scoring method ("tfidf", "semantic", "hybrid").
            top_k: Number of recommendations per source page.
            similarity_threshold: Minimum similarity score cutoff.
            exclude_existing_links: Whether to skip existing links.
            tfidf_model: Reference to fitted TF-IDF model for common term explainability.
            
        Returns:
            List of RecommendationData objects.
        """
        if not pages or tfidf_matrix is None or tfidf_matrix.size == 0:
            return []

        # Choose primary matrix based on method
        if method == "semantic" and semantic_matrix is not None and semantic_matrix.size > 0:
            primary_matrix = semantic_matrix
        elif method == "hybrid" and hybrid_matrix is not None and hybrid_matrix.size > 0:
            primary_matrix = hybrid_matrix
        else:
            primary_matrix = tfidf_matrix
            method = "tfidf"

        n_pages = len(pages)
        url_to_page = {p["url"]: p for p in pages}
        recommendations: List[RecommendationData] = []

        for i in range(n_pages):
            source_page = pages[i]
            source_url = source_page["url"]
            source_title = source_page.get("title", "") or source_url

            # Get similarity scores for this source page
            scores = primary_matrix[i]
            
            # Sort target indices by similarity score descending
            sorted_indices = np.argsort(scores)[::-1]

            rank = 1
            for j in sorted_indices:
                if rank > top_k:
                    break

                target_page = pages[j]
                target_url = target_page["url"]

                # Exclude self links
                if i == j or source_url == target_url:
                    continue

                sim_score = float(scores[j])

                # Threshold filtering
                if sim_score < similarity_threshold:
                    continue

                # Existing link status check
                is_existing = (source_url, target_url) in existing_links_set

                if exclude_existing_links and is_existing:
                    continue

                target_title = target_page.get("title", "") or target_url

                tf_score = float(tfidf_matrix[i, j]) if tfidf_matrix is not None and i < tfidf_matrix.shape[0] and j < tfidf_matrix.shape[1] else 0.0
                sem_score = float(semantic_matrix[i, j]) if semantic_matrix is not None and i < semantic_matrix.shape[0] and j < semantic_matrix.shape[1] else 0.0
                hyb_score = float(hybrid_matrix[i, j]) if hybrid_matrix is not None and i < hybrid_matrix.shape[0] and j < hybrid_matrix.shape[1] else 0.0

                # Extract common keywords for explainability if TF-IDF model is available
                common_terms = []
                if tfidf_model:
                    common_terms = tfidf_model.get_common_keywords(i, j, top_n=4)

                cat = categorize_similarity(sim_score)

                recommendation = RecommendationData(
                    source_url=source_url,
                    source_title=source_title,
                    target_url=target_url,
                    target_title=target_title,
                    tfidf_score=tf_score,
                    semantic_score=sem_score,
                    hybrid_score=hyb_score,
                    similarity_score=sim_score,
                    method=method,
                    existing_link=is_existing,
                    rank=rank,
                    similarity_category=cat,
                    common_terms=common_terms,
                )
                recommendations.append(recommendation)
                rank += 1

        return recommendations
