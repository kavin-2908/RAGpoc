"""
Embedding Service — Wraps sentence-transformers for vector generation.

This service is responsible for converting text into dense vector embeddings.
It uses the 'sentence-transformers/all-MiniLM-L6-v2' model, which is small,
fast, and produces 384-dimensional vectors.

It exposes two main methods:
- embed_documents(texts): Used during upload/indexing (batch processing).
- embed_query(text): Used during chat (single string).
"""

import logging
from typing import Union

from django.conf import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service to generate embeddings using sentence-transformers.
    
    The model is loaded lazily upon the first request to avoid
    slow startup times for the Django server.
    """

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None

    def _get_model(self) -> SentenceTransformer:
        """
        Lazy load the sentence-transformers model.
        
        This prevents the model from being loaded into memory when Django
        starts up, which speeds up server boot time. It only loads when
        the first embedding is requested.
        """
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of strings (e.g., chunks from a document).
            
        Returns:
            List of embedding vectors (each vector is a list of floats).
        """
        if not texts:
            return []

        model = self._get_model()
        
        logger.debug("Generating embeddings for %d chunks", len(texts))
        
        # model.encode returns a numpy array, we convert it to a standard
        # Python list of lists for easy compatibility with Django ORM/pgvector.
        embeddings = model.encode(texts, show_progress_bar=False)
        
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single search query.
        
        Args:
            text: The search query string.
            
        Returns:
            A single embedding vector (list of floats).
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty query")
            
        model = self._get_model()
        
        logger.debug("Generating embedding for query")
        
        # encode() normally returns a 2D array if given a list,
        # but passing a single string returns a 1D array.
        embedding = model.encode(text, show_progress_bar=False)
        
        return embedding.tolist()
