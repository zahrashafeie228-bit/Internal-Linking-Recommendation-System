"""
TF-IDF vectorizer model for Persian web pages.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp.text_preprocessor import TextPreprocessor
from src.utils.helpers import setup_logger

logger = setup_logger("tfidf_model")


class TFIDFSimilarityModel:
    """Computes TF-IDF representations and pairwise cosine similarity for web pages."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        max_features: int = 5000,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.preprocessor = preprocessor or TextPreprocessor()

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.urls: List[str] = []

    def fit_transform(self, pages: List[Dict[str, str]]) -> np.ndarray:
        """
        Fit vectorizer and transform page texts into TF-IDF matrix.
        
        Args:
            pages: List of dicts containing 'url', 'title', and 'clean_text'.
            
        Returns:
            Pairwise Cosine Similarity Matrix (N x N)
        """
        if not pages:
            return np.array([])

        self.urls = [p["url"] for p in pages]
        
        # Prepare corpus: combine title (weighted 2x) and clean_text
        corpus = []
        for p in pages:
            title = p.get("title", "")
            text = p.get("clean_text", "")
            combined = f"{title} {title} {text}"
            corpus.append(combined)

        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocessor.preprocess_tokens,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df if len(pages) > 2 else 1.0,
            max_features=self.max_features,
            token_pattern=None,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Calculate cosine similarity matrix
        sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Set diagonal to 0 to prevent self-links
        np.fill_diagonal(sim_matrix, 0.0)

        return sim_matrix

    def get_common_keywords(self, idx1: int, idx2: int, top_n: int = 4) -> List[str]:
        """
        Find top shared high-weight TF-IDF terms between page idx1 and page idx2.
        Used for explainability.
        """
        if self.tfidf_matrix is None or idx1 >= self.tfidf_matrix.shape[0] or idx2 >= self.tfidf_matrix.shape[0]:
            return []

        try:
            vec1 = self.tfidf_matrix[idx1].toarray().flatten()
            vec2 = self.tfidf_matrix[idx2].toarray().flatten()

            # Element-wise product of TF-IDF weights gives term contribution to similarity
            shared_weights = vec1 * vec2
            top_indices = np.argsort(shared_weights)[::-1][:top_n]

            common = []
            for idx in top_indices:
                if shared_weights[idx] > 0 and idx < len(self.feature_names):
                    common.append(self.feature_names[idx])
            return common
        except Exception as e:
            logger.warning(f"Error computing common keywords: {e}")
            return []
