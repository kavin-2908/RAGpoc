"""
Document Service — Orchestrates the document upload pipeline.

Flow:
    1. Determine file type from filename.
    2. Save the file to the uploads/ directory.
    3. Pick the correct loader based on file type.
    4. Extract text from the file.
    5. Split text into overlapping chunks.
    6. Generate embeddings for each chunk.
    7. Store the document, chunks, and embeddings in PostgreSQL.

This service coordinates loaders, text_utils, embedding_service,
and vector_service. It contains no DRF imports — pure Python.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from django.conf import settings

from rag.loaders.csv_loader import CSVLoader
from rag.loaders.docx_loader import DOCXLoader
from rag.loaders.excel_loader import ExcelLoader
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.text_loader import TextLoader
from rag.services.embedding_service import EmbeddingService
from rag.services.vector_service import VectorService
from rag.utils.text_utils import chunk_text

logger = logging.getLogger(__name__)

# Maps file extensions to their loader classes
LOADER_MAP = {
    'pdf': PDFLoader,
    'docx': DOCXLoader,
    'txt': TextLoader,
    'csv': CSVLoader,
    'xlsx': ExcelLoader,
}


class DocumentService:
    """
    Orchestrates the full document upload pipeline.

    Usage:
        service = DocumentService()
        service.process_document(file=uploaded_file, filename="notes.pdf")
    """

    def __init__(self):
        self.upload_dir: Path = settings.UPLOAD_DIR
        self.chunk_size: int = settings.CHUNK_SIZE
        self.chunk_overlap: int = settings.CHUNK_OVERLAP
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()

    def process_document(self, file, filename: str) -> dict:
        """
        Process an uploaded document through the full RAG pipeline.

        Args:
            file: The uploaded file object (Django InMemoryUploadedFile).
            filename: Original filename of the uploaded file.

        Returns:
            dict with 'document_id' and 'chunks_created' count.

        Raises:
            ValueError: If file type is unsupported, text is empty,
                        or the file is corrupted.
        """
        # Step 1: Determine file type
        file_type = self._get_file_type(filename)
        logger.info("Processing document: %s (type: %s)", filename, file_type)

        # Step 2: Save file to disk
        file_path = self._save_file(file, filename)
        logger.info("File saved to: %s", file_path)

        try:
            # Step 3: Extract text using the appropriate loader
            text = self._extract_text(file_path, file_type)
            if not text or not text.strip():
                raise ValueError(
                    f"No text could be extracted from '{filename}'. "
                    "The file may be empty or contain only images."
                )
            logger.info(
                "Extracted %d characters from %s", len(text), filename
            )

            # Step 4: Split text into chunks
            chunks = chunk_text(
                text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            if not chunks:
                raise ValueError(
                    f"Text from '{filename}' could not be split into chunks."
                )
            logger.info("Created %d chunks from %s", len(chunks), filename)

            # Step 5: Generate embeddings for all chunks
            embeddings = self.embedding_service.embed_documents(chunks)
            logger.info(
                "Generated %d embeddings for %s", len(embeddings), filename
            )

            # Step 6: Store document + chunks + embeddings in PostgreSQL
            document = self.vector_service.store_document(
                filename=filename,
                file_type=file_type,
                chunks=chunks,
                embeddings=embeddings,
            )
            logger.info(
                "Stored document '%s' with %d chunks (id=%s)",
                filename,
                len(chunks),
                document.id,
            )

            return {
                "document_id": document.id,
                "chunks_created": len(chunks),
            }

        except Exception:
            # If processing fails after saving, clean up the file
            self._delete_file(file_path)
            raise

    def _get_file_type(self, filename: str) -> str:
        """
        Extract and validate the file extension.

        Args:
            filename: The original filename.

        Returns:
            Lowercase file extension (e.g., 'pdf').

        Raises:
            ValueError: If the extension is not supported.
        """
        if '.' not in filename:
            raise ValueError(f"Filename '{filename}' has no extension.")

        extension = filename.rsplit('.', 1)[-1].lower()

        if extension not in LOADER_MAP:
            raise ValueError(
                f"Unsupported file type: '.{extension}'. "
                f"Supported types: {', '.join(sorted(LOADER_MAP.keys()))}."
            )
        return extension

    def _save_file(self, file, filename: str) -> Path:
        """
        Save the uploaded file to the uploads/ directory.

        If a file with the same name already exists, it gets overwritten.

        Args:
            file: Django uploaded file object.
            filename: Name to save the file as.

        Returns:
            Path to the saved file.
        """
        file_path = self.upload_dir / filename

        with open(file_path, 'wb') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        return file_path

    def _extract_text(self, file_path: Path, file_type: str) -> str:
        """
        Extract text from a file using the appropriate loader.

        Args:
            file_path: Path to the saved file.
            file_type: File extension (e.g., 'pdf').

        Returns:
            Extracted text as a string.

        Raises:
            ValueError: If the loader fails (e.g., corrupted file).
        """
        loader_class = LOADER_MAP[file_type]
        loader = loader_class()

        try:
            text = loader.load(str(file_path))
            return text
        except Exception as e:
            raise ValueError(
                f"Failed to extract text from '{file_path.name}': {e}"
            ) from e

    def _delete_file(self, file_path: Path) -> None:
        """
        Delete a file from disk. Logs a warning if deletion fails.

        Args:
            file_path: Path to the file to delete.
        """
        try:
            if file_path.exists():
                os.remove(file_path)
                logger.info("Cleaned up file: %s", file_path)
        except OSError as e:
            logger.warning("Failed to clean up file %s: %s", file_path, e)
