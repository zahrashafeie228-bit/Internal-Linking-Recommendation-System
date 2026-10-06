"""
Hybrid similarity calculator combining TF-IDF and Semantic embeddings.
"""

from typing import Optional, Tuple
import numpy as np

from config.settings import DEFAULT_HYBRID_ALPHA
from src.utils.helpers import setup_logger

logger = setup_logger("similarity")


def compute_hybrid_similarity_matrix(
    tfidf_matrix: np.ndarray,
    semantic_matrix: Optional[np.ndarray],
    alpha: float = DEFAULT_HYBRID_ALPHA,
) -> np.ndarray:
    """
    Compute hybrid similarity matrix:
        hybrid = alpha * semantic + (1 - alpha) * tfidf
        
    If semantic_matrix is None, returns tfidf_matrix.
    """
    if tfidf_matrix is None or tfidf_matrix.size == 0:
        if semantic_matrix is not None:
            return semantic_matrix
        return np.array([])

    if semantic_matrix is None or semantic_matrix.size == 0:
        logger.warning("Semantic similarity matrix unavailable, using TF-IDF matrix.")
        return tfidf_matrix

    alpha = max(0.0, min(1.0, alpha))
    hybrid_matrix = (alpha * semantic_matrix) + ((1.0 - alpha) * tfidf_matrix)

    # Set diagonal to 0
    np.fill_diagonal(hybrid_matrix, 0.0)

    return hybrid_matrix
