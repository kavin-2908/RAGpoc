"""
CSV Loader — Extracts text from CSV files.

Uses Python's built-in csv module. No external dependencies.

Strategy:
- Reads all rows from the CSV file.
- If a header row exists, it's included as the first line.
- Each row is joined into a comma-separated string.
- All rows are joined with newlines.

This produces a readable text representation that works well
for chunking and embedding.
"""

import csv
import logging

logger = logging.getLogger(__name__)


class CSVLoader:
    """Extracts text from CSV files."""

    def load(self, file_path: str) -> str:
        """
        Read a CSV file and convert it to a text representation.

        Each row becomes a line of text with cells separated by commas.
        This makes the data readable for both chunking and the LLM.

        Args:
            file_path: Absolute path to the CSV file.

        Returns:
            Text representation of the CSV data.

        Raises:
            FileNotFoundError: If the file does not exist.
            csv.Error: If the CSV is malformed.
        """
        logger.info("Loading CSV: %s", file_path)

        rows: list[str] = []

        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)

            for row in reader:
                # Join each cell in the row with ', ' for readability
                # Filter out empty cells to keep output clean
                cell_values = [cell.strip() for cell in row if cell.strip()]

                if cell_values:
                    rows.append(", ".join(cell_values))

        full_text = "\n".join(rows)
        logger.info(
            "CSV loaded: %d rows, %d characters total",
            len(rows),
            len(full_text),
        )

        return full_text
