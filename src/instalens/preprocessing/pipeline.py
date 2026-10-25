"""
End-to-end preprocessing pipeline.

Chains normalizer → contraction expansion → tokenizer and returns
a structured object ready for the scoring engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .normalizer import normalize_text
from .contractions import expand_contractions
from .tokenizer import tokenize, TokenizedComment


@dataclass
class PreprocessedComment:
    """Bundles original text, normalized form, and tokenization result."""
    original: str
    normalized: str
    tokenized: TokenizedComment


def preprocess_comment(text: str) -> PreprocessedComment:
    """
    Run the full preprocessing pipeline on a single comment.

    Steps (order matters):
    1. Normalize Unicode, URLs, mentions, hashtags, repeated chars/punct.
    2. Expand contractions ("can't" → "can not").
    3. Tokenize with clause boundary and emoji detection.

    Parameters
    ----------
    text : str
        Raw comment text as retrieved from the data source.

    Returns
    -------
    PreprocessedComment
        Structured result with original text preserved.
    """
    normalized = normalize_text(text)
    normalized = expand_contractions(normalized)
    tokenized = tokenize(normalized)

    return PreprocessedComment(
        original=text,
        normalized=normalized,
        tokenized=tokenized,
    )
