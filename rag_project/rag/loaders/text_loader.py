"""
Text Loader — Reads plain text files.

Uses Python's built-in open() function. No external dependencies.

Tries UTF-8 encoding first (covers most text files), then falls back
to latin-1 (which never raises a decoding error since every byte
is a valid latin-1 character).
"""

import logging

logger = logging.getLogger(__name__)


class TextLoader:
    """Extracts text from plain .txt files."""

    def load(self, file_path: str) -> str:
        """
        Read the entire contents of a text file.

        Attempts UTF-8 first. If that fails (UnicodeDecodeError),
        falls back to latin-1 encoding.

        Args:
            file_path: Absolute path to the text file.

        Returns:
            File contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read.
        """
        logger.info("Loading TXT: %s", file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decoding failed for %s, falling back to latin-1",
                file_path,
            )
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()

        logger.info("TXT loaded: %d characters", len(text))
        return text
