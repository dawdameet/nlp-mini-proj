"""
Comment ingestion from CSV files.

This is the primary data source for development and evaluation.
The Instagram API adapter will be added later as a separate module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Comment:
    """A single comment with metadata."""
    text: str
    comment_id: str = ""
    username: str = ""               # redacted in exports
    timestamp: str = ""
    source: str = "csv"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CSVCommentLoader:
    """
    Load comments from a CSV file.

    Expected CSV columns (flexible):
    - Required: 'text' or 'comment' or 'comment_text' or 'body'
    - Optional: 'id', 'username', 'timestamp', 'date'
    """

    # Column name aliases for the text field
    TEXT_COLUMNS = {"text", "comment", "comment_text", "body", "content", "message"}
    ID_COLUMNS = {"id", "comment_id", "cid"}
    USER_COLUMNS = {"username", "user", "author", "commenter"}
    TIME_COLUMNS = {"timestamp", "date", "time", "created_at", "datetime"}

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
        if not self.filepath.suffix.lower() == ".csv":
            raise ValueError(f"Expected a .csv file, got: {self.filepath.suffix}")

    def load(self, k: int | None = None) -> list[Comment]:
        """
        Load up to k comments from the CSV.

        Parameters
        ----------
        k : int, optional
            Maximum number of comments to load. None = load all.

        Returns
        -------
        list[Comment]
            List of Comment objects.

        Raises
        ------
        ValueError
            If no text column can be identified.
        """
        comments: list[Comment] = []

        with open(self.filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = {h.lower().strip(): h for h in (reader.fieldnames or [])}

            text_col = self._find_column(headers, self.TEXT_COLUMNS)
            if text_col is None:
                raise ValueError(
                    f"Cannot find a text column in {self.filepath}. "
                    f"Expected one of: {', '.join(sorted(self.TEXT_COLUMNS))}. "
                    f"Found columns: {', '.join(headers.values())}"
                )

            id_col = self._find_column(headers, self.ID_COLUMNS)
            user_col = self._find_column(headers, self.USER_COLUMNS)
            time_col = self._find_column(headers, self.TIME_COLUMNS)

            for i, row in enumerate(reader):
                if k is not None and len(comments) >= k:
                    break

                text = row.get(text_col, "").strip()
                if not text:
                    continue

                comments.append(Comment(
                    text=text,
                    comment_id=row.get(id_col, str(i)) if id_col else str(i),
                    username=row.get(user_col, "") if user_col else "",
                    timestamp=row.get(time_col, "") if time_col else "",
                    source="csv",
                ))

        return comments

    @staticmethod
    def _find_column(
        headers: dict[str, str], aliases: set[str]
    ) -> Optional[str]:
        """Find the actual column name matching any alias."""
        for alias in aliases:
            if alias in headers:
                return headers[alias]
        return None

    @property
    def info(self) -> dict:
        """Return metadata about the CSV file."""
        with open(self.filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            row_count = sum(1 for _ in reader)

        return {
            "filepath": str(self.filepath),
            "columns": columns,
            "row_count": row_count,
        }
