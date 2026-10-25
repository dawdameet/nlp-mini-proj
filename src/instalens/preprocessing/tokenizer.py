"""
Tokenizer tuned for social-media comments.

Uses NLTK's TweetTokenizer as a base (handles emojis, emoticons,
hashtags gracefully) and wraps it with clause-boundary detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nltk.tokenize import TweetTokenizer

_tweet_tok = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)

# Clause boundaries — conjunctions / punctuation that reset negation scope
CLAUSE_BOUNDARY_TOKENS = frozenset({
    "but", "however", "although", "though", "yet", "nevertheless",
    "whereas", "while", "except", "instead",
    ",", ";", ":", ".", "!", "?", "—", "–", "...",
})

# Contrast conjunctions — the clause *after* these typically carries
# the dominant sentiment ("not bad, but actually amazing")
CONTRAST_CONJUNCTIONS = frozenset({
    "but", "however", "although", "though", "yet",
    "nevertheless", "whereas", "except", "instead",
})


@dataclass
class Token:
    """A single token with positional and structural metadata."""
    text: str
    index: int                          # position in the flat token list
    clause_index: int = 0              # which clause this token belongs to
    is_clause_boundary: bool = False
    is_emoji: bool = False
    is_punctuation: bool = False
    is_negation: bool = False          # set later by scoring engine
    is_intensifier: bool = False       # set later by scoring engine


@dataclass
class TokenizedComment:
    """Result of tokenizing a single comment."""
    tokens: list[Token] = field(default_factory=list)
    clauses: list[list[Token]] = field(default_factory=list)
    has_contrast: bool = False
    contrast_index: int | None = None  # clause index after contrast word


# Simple emoji regex (covers most common emoji ranges)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero width joiner
    "]+",
    flags=re.UNICODE,
)

_PUNCT_RE = re.compile(r"^[^\w\s]+$")


def _is_emoji(token_text: str) -> bool:
    return bool(_EMOJI_RE.fullmatch(token_text))


def _is_punct(token_text: str) -> bool:
    return bool(_PUNCT_RE.match(token_text))


def tokenize(text: str) -> TokenizedComment:
    """
    Tokenize a normalized comment string.

    Returns a TokenizedComment with flat token list, clause segmentation,
    and contrast detection.
    """
    raw_tokens = _tweet_tok.tokenize(text)

    tokens: list[Token] = []
    clauses: list[list[Token]] = [[]]
    clause_idx = 0
    has_contrast = False
    contrast_index: int | None = None

    for i, raw in enumerate(raw_tokens):
        is_boundary = raw in CLAUSE_BOUNDARY_TOKENS
        is_contrast = raw in CONTRAST_CONJUNCTIONS
        emoji = _is_emoji(raw)
        punct = _is_punct(raw) and not emoji

        tok = Token(
            text=raw,
            index=i,
            clause_index=clause_idx,
            is_clause_boundary=is_boundary,
            is_emoji=emoji,
            is_punctuation=punct,
        )
        tokens.append(tok)
        clauses[clause_idx].append(tok)

        if is_boundary:
            clause_idx += 1
            clauses.append([])
            if is_contrast:
                has_contrast = True
                contrast_index = clause_idx

    # Remove trailing empty clause
    if clauses and not clauses[-1]:
        clauses.pop()

    return TokenizedComment(
        tokens=tokens,
        clauses=clauses,
        has_contrast=has_contrast,
        contrast_index=contrast_index,
    )
