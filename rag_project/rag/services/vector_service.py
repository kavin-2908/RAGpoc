"""
Vector Service — Manages database operations for vectors.

This service abstracts away the Django ORM and pgvector specifics.
It is responsible for:
1. Storing documents and their chunks (with embeddings) into PostgreSQL.
2. Performing cosine similarity searches to find relevant chunks.

It is the ONLY service that should import from rag.models.
"""

import logging

from django.db import transaction
from pgvector.django import CosineDistance

from rag.models import Chunk, Document

logger = logging.getLogger(__name__)


class VectorService:
    """
    Handles database operations for documents, chunks, and vector search.
    """

    @transaction.atomic
    def store_document(
        self,
        filename: str,
        file_type: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> Document:
        """
        Save a document and all its chunks/embeddings to the database.

        Uses a database transaction (@transaction.atomic) so that if anything
        fails, the entire operation rolls back (preventing orphaned chunks).
        Uses bulk_create to efficiently insert all chunks in a single query.

        Args:
            filename: Original filename.
            file_type: File extension.
            chunks: List of text strings.
            embeddings: List of 384-dimensional vectors.

        Returns:
            The created Document instance.

        Raises:
            ValueError: If the number of chunks and embeddings don't match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        # 1. Create the parent Document record
        document = Document.objects.create(
            filename=filename,
            file_type=file_type,
        )

        # 2. Prepare Chunk objects in memory
        chunk_objects = []
        for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_objects.append(
                Chunk(
                    document=document,
                    chunk_index=i,
                    content=content,
                    embedding=embedding,
                )
            )

        # 3. Bulk insert all chunks efficiently
        Chunk.objects.bulk_create(chunk_objects)

        logger.info(
            "Stored document id=%d with %d chunks",
            document.id,
            len(chunk_objects),
        )

        return document

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Find the most relevant chunks using cosine similarity.

        Uses pgvector's CosineDistance function.
        In pgvector, distance is (1 - cosine_similarity).
        Smaller distance = more similar.

        Args:
            query_embedding: The 384-dimensional vector of the search query.
            top_k: Number of results to return.

        Returns:
            List of the top_k most similar Chunk objects, ordered by relevance.
        """
        logger.debug("Performing vector similarity search (top_k=%d)", top_k)

        # The pgvector CosineDistance function calculates the distance between
        # the stored 'embedding' field and the provided 'query_embedding'.
        # We order by this distance (ascending) to get the most similar first.
        results = (
            Chunk.objects.annotate(
                distance=CosineDistance('embedding', query_embedding)
            )
            .order_by('distance')[:top_k]
        )

        # Convert queryset to a list to ensure the query executes immediately
        result_list = list(results)
        
        logger.info("Found %d relevant chunks", len(result_list))
        return result_list
