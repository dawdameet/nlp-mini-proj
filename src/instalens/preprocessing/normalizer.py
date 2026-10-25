"""
Text normalization for social-media comments.

Handles Unicode normalization, whitespace cleanup, URL/mention stripping,
and emoji preservation. Designed to be non-destructive — retains signals
that matter for sentiment (emojis, punctuation patterns, negation words).
"""

import re
import unicodedata


# ── Regex patterns (compiled once) ──────────────────────────────────────────

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
_MENTION_RE = re.compile(r"@[\w.]+")
_HASHTAG_SPLIT_RE = re.compile(r"#(\w+)")
_REPEATED_PUNCT_RE = re.compile(r"([!?.])\1{2,}")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """Normalize to NFC form, strip zero-width and control characters."""
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width chars and most C0/C1 control chars, but keep \n \t
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060\ufeff]", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space; strip edges."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def handle_urls(text: str) -> str:
    """Replace URLs with a placeholder token."""
    return _URL_RE.sub(" <URL> ", text)


def handle_mentions(text: str) -> str:
    """Replace @mentions with a placeholder token."""
    return _MENTION_RE.sub(" <USER> ", text)


def handle_hashtags(text: str) -> str:
    """Strip the '#' but keep the word (e.g. #love → love)."""
    return _HASHTAG_SPLIT_RE.sub(r" \1 ", text)


def normalize_repeated_punct(text: str) -> str:
    """Collapse e.g. '!!!!' → '!!' (keep two to signal emphasis)."""
    return _REPEATED_PUNCT_RE.sub(r"\1\1", text)


def normalize_repeated_chars(text: str) -> str:
    """Collapse e.g. 'soooooo' → 'soo' (keep two for elongation signal)."""
    return _REPEATED_CHAR_RE.sub(r"\1\1", text)


def normalize_text(text: str) -> str:
    """
    Full normalization pipeline.

    Order matters — URLs/mentions must be handled before punctuation
    normalization, and whitespace cleanup comes last.

    Parameters
    ----------
    text : str
        Raw comment text.

    Returns
    -------
    str
        Normalized text with emojis, meaningful punctuation, and
        negation words preserved.
    """
    if not text or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = handle_urls(text)
    text = handle_mentions(text)
    text = handle_hashtags(text)
    text = normalize_repeated_punct(text)
    text = normalize_repeated_chars(text)
    text = normalize_whitespace(text)

    return text
