"""
Sentence-Transformers multilingual semantic embedding model with document chunking and caching.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import CACHE_DIR, DEFAULT_SEMANTIC_MODEL, MAX_CHUNK_WORDS
from src.utils.helpers import setup_logger

logger = setup_logger("semantic_model")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticSimilarityModel:
    """Computes semantic embeddings for Persian documents using multilingual SentenceTransformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        max_chunk_words: int = MAX_CHUNK_WORDS,
        cache_dir: Path = CACHE_DIR,
    ):
        self.model_name = model_name
        self.max_chunk_words = max_chunk_words
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[SentenceTransformer] = None

    def _load_model(self) -> Tuple[bool, str]:
        """Lazy load sentence transformer model."""
        if self.model is not None:
            return True, ""

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False, "کتابخانه sentence-transformers نصب نشده است."

        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            return True, ""
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            return False, f"خطا در بارگذاری مدل {self.model_name}: {str(e)}"

    def _chunk_text(self, text: str) -> List[str]:
        """Split long document text into smaller chunks of max_chunk_words."""
        words = text.split()
        if not words:
            return []

        if len(words) <= self.max_chunk_words:
            return [" ".join(words)]

        chunks = []
        for i in range(0, len(words), self.max_chunk_words):
            chunk = " ".join(words[i : i + self.max_chunk_words])
            chunks.append(chunk)
        return chunks

    def _get_cache_key(self, pages: List[Dict[str, str]]) -> str:
        """Generate MD5 hash key for caching embeddings of a document set."""
        content_bytes = self.model_name.encode("utf-8")
        for p in pages:
            url = p.get("url", "")
            clean_text = p.get("clean_text", "")
            content_bytes += f"{url}:{len(clean_text)}".encode("utf-8")
        return hashlib.md5(content_bytes).hexdigest()

    def get_document_embeddings(
        self, pages: List[Dict[str, str]], use_cache: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Generate document-level embeddings for all pages (with chunk mean pooling).
        
        Returns:
            (embeddings_matrix, error_message)
        """
        if not pages:
            return None, "لیست صفحات برای تحلیل معنایی خالی است."

        # Check disk cache
        cache_key = self._get_cache_key(pages)
        cache_path = self.cache_dir / f"embeddings_{cache_key}.joblib"

        if use_cache and cache_path.exists():
            try:
                logger.info(f"Loading cached embeddings from {cache_path}")
                embeddings = joblib.load(cache_path)
                return embeddings, None
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")

        # Load model
        success, err = self._load_model()
        if not success:
            return None, err

        embeddings_list = []

        for p in pages:
            title = p.get("title", "")
            text = p.get("clean_text", "")
            full_content = f"{title}. {text}".strip()

            chunks = self._chunk_text(full_content)
            if not chunks:
                # Zero vector fallback if document is completely empty
                dim = self.model.get_sentence_embedding_dimension()
                embeddings_list.append(np.zeros(dim))
                continue

            try:
                # Compute embeddings for all chunks of the document
                chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True)
                
                # Combine chunk embeddings using Mean Pooling
                doc_embedding = np.mean(chunk_embeddings, axis=0)
                
                # Normalize embedding to unit length
                norm = np.linalg.norm(doc_embedding)
                if norm > 0:
                    doc_embedding = doc_embedding / norm

                embeddings_list.append(doc_embedding)
            except Exception as e:
                logger.error(f"Error embedding URL {p.get('url')}: {e}")
                dim = self.model.get_sentence_embedding_dimension()
                embeddings_list.append(np.zeros(dim))

        embeddings_matrix = np.array(embeddings_list)

        # Save to cache
        if use_cache:
            try:
                joblib.dump(embeddings_matrix, cache_path)
            except Exception as e:
                logger.warning(f"Failed to save embeddings cache: {e}")

        return embeddings_matrix, None

    def compute_similarity_matrix(
        self, pages: List[Dict[str, str]], use_cache: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Compute pairwise cosine similarity matrix between document embeddings.
        
        Returns:
            (similarity_matrix, error_message)
        """
        embeddings, err = self.get_document_embeddings(pages, use_cache=use_cache)
        if err or embeddings is None:
            return None, err

        sim_matrix = cosine_similarity(embeddings, embeddings)

        # Set diagonal to 0 to prevent self-links
        np.fill_diagonal(sim_matrix, 0.0)

        return sim_matrix, None
