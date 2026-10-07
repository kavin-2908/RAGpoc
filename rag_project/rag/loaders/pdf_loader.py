"""
PDF Loader — Extracts text from PDF files using PyMuPDF.

PyMuPDF (imported as 'fitz') is a fast, lightweight PDF library.
It extracts text page-by-page using the getText() / get_text() method,
which returns the raw text content of each page.

Why PyMuPDF over PyPDF2?
- Faster text extraction.
- Better handling of complex PDF layouts.
- More reliable with edge-case PDFs.
"""

import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFLoader:
    """Extracts text from PDF files."""

    def load(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Opens the PDF, iterates through every page, and concatenates
        the extracted text with newlines between pages.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            Extracted text as a single string.

        Raises:
            Exception: If the PDF is corrupted or cannot be opened.
        """
        logger.info("Loading PDF: %s", file_path)

        text_parts: list[str] = []

        # fitz.open() opens the PDF document
        # We use a context manager to ensure the file is closed properly
        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc):
                # get_text() extracts the plain text content of the page
                page_text = page.get_text()

                if page_text.strip():
                    text_parts.append(page_text)

                logger.debug(
                    "Page %d: extracted %d characters",
                    page_number + 1,
                    len(page_text),
                )

        full_text = "\n".join(text_parts)
        logger.info(
            "PDF loaded: %d pages, %d characters total",
            len(text_parts),
            len(full_text),
        )

        return full_text
