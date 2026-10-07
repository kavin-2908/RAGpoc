"""
Text utilities for the RAG pipeline.

Contains functions for processing and splitting text before embedding.
"""

import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split a large text into smaller overlapping chunks.

    Uses a simple character-based sliding window approach.
    Overlap ensures that context is preserved across chunk boundaries
    so that a sentence cut in half still makes sense in the next chunk.

    Args:
        text: The full text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    # Clean the text: remove excessive whitespace and newlines
    text = " ".join(text.split())

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    # Sliding window logic
    while start < text_length:
        # Calculate the end index for the current chunk
        end = start + chunk_size

        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)

        # Move the start index forward for the next chunk
        # The step size is (chunk_size - overlap)
        start += (chunk_size - overlap)

        # If the remaining text is smaller than the overlap,
        # we're done. Otherwise we'd just create a tiny chunk
        # that is entirely contained in the previous chunk.
        if start >= text_length:
             break

    logger.debug(
        "Chunked text of length %d into %d chunks (size=%d, overlap=%d)",
        text_length, len(chunks), chunk_size, overlap
    )

    return chunks
