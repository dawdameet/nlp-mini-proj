"""
Sentiment lexicon — loading, querying, and management.

Supports multiple sources: SentiWordNet, manual social-media slang,
emoji polarity, and phrase-level entries. Each entry tracks provenance
and validation status as required by the PRD.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class PolarityDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EntrySource(str, Enum):
    SENTIWORDNET = "sentiwordnet"
    MANUAL_SLANG = "manual_slang"
    EMOJI = "emoji"
    PHRASE = "phrase"
    EMBEDDING_EXPANSION = "embedding_expansion"


class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


@dataclass
class LexiconEntry:
    """
    A single entry in the sentiment lexicon.

    Fields match PRD §6.2: token/phrase, polarity, strength, source,
    validation status, optional notes.
    """
    token: str
    polarity: PolarityDirection
    strength: float                     # 0.0 – 1.0
    source: EntrySource
    status: ValidationStatus = ValidationStatus.VALIDATED
    notes: str = ""
    is_phrase: bool = False             # multi-word entry?


class SentimentLexicon:
    """
    Composite sentiment lexicon.

    Provides O(1) single-token lookup and phrase matching over a
    combined vocabulary built from multiple sources.
    """

    def __init__(self) -> None:
        self._entries: dict[str, LexiconEntry] = {}
        self._phrases: dict[str, LexiconEntry] = {}
        self._stats: dict[str, int] = {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "phrases": 0,
        }

    # ── Loading ─────────────────────────────────────────────────────────

    def load_from_json(self, path: str | Path) -> int:
        """
        Load entries from a JSON file.

        Expected format: list of objects with keys matching LexiconEntry fields.
        Returns number of entries loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        count = 0
        for item in data:
            entry = LexiconEntry(
                token=item["token"].lower(),
                polarity=PolarityDirection(item["polarity"]),
                strength=float(item["strength"]),
                source=EntrySource(item["source"]),
                status=ValidationStatus(item.get("status", "validated")),
                notes=item.get("notes", ""),
                is_phrase=item.get("is_phrase", False),
            )
            self.add_entry(entry)
            count += 1

        return count

    def add_entry(self, entry: LexiconEntry) -> None:
        """Add a single entry to the lexicon."""
        key = entry.token.lower()
        if entry.is_phrase:
            self._phrases[key] = entry
            self._stats["phrases"] += 1
        else:
            self._entries[key] = entry

        self._stats["total"] += 1
        self._stats[entry.polarity.value] += 1

    # ── Querying ────────────────────────────────────────────────────────

    def lookup(self, token: str) -> Optional[LexiconEntry]:
        """Look up a single token (case-insensitive)."""
        return self._entries.get(token.lower())

    def lookup_phrase(self, phrase: str) -> Optional[LexiconEntry]:
        """Look up a multi-word phrase (case-insensitive)."""
        return self._phrases.get(phrase.lower())

    def find_phrases_in_tokens(self, tokens: list[str]) -> list[tuple[int, int, LexiconEntry]]:
        """
        Scan a token list for phrase matches.

        Returns list of (start_idx, end_idx, entry) tuples.
        Greedy longest-match-first.
        """
        matches = []
        max_phrase_len = max(
            (len(p.split()) for p in self._phrases), default=0
        )
        i = 0
        while i < len(tokens):
            matched = False
            for length in range(min(max_phrase_len, len(tokens) - i), 0, -1):
                candidate = " ".join(tokens[i : i + length])
                entry = self._phrases.get(candidate.lower())
                if entry and entry.status == ValidationStatus.VALIDATED:
                    matches.append((i, i + length, entry))
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1
        return matches

    def has_token(self, token: str) -> bool:
        return token.lower() in self._entries

    @property
    def size(self) -> int:
        return self._stats["total"]

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def export_json(self, path: str | Path) -> None:
        """Export the full lexicon to JSON."""
        all_entries = list(self._entries.values()) + list(self._phrases.values())
        with open(path, "w") as f:
            json.dump([asdict(e) for e in all_entries], f, indent=2, default=str)

    def __contains__(self, token: str) -> bool:
        return self.has_token(token)

    def __len__(self) -> int:
        return self.size


# ── Default lexicon builder ─────────────────────────────────────────────────

def build_default_lexicon(data_dir: str | Path | None = None) -> SentimentLexicon:
    """
    Build the default InstaLens lexicon from bundled data files.

    Loads in order:
    1. Base lexicon (SentiWordNet-derived + manual curation)
    2. Emoji lexicon
    3. Slang lexicon
    4. Phrase lexicon

    Falls back to the built-in minimal lexicon if no data files exist.
    """
    lex = SentimentLexicon()

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "lexicons"
    else:
        data_dir = Path(data_dir)

    for filename in ["base_lexicon.json", "emoji_lexicon.json",
                     "slang_lexicon.json", "phrase_lexicon.json"]:
        filepath = data_dir / filename
        if filepath.exists():
            lex.load_from_json(filepath)

    # If nothing loaded, build a minimal built-in lexicon
    if lex.size == 0:
        _add_builtin_entries(lex)

    return lex


def _add_builtin_entries(lex: SentimentLexicon) -> None:
    """Hardcoded minimal lexicon for development / fallback."""

    positive_words = {
        "good": 0.6, "great": 0.8, "amazing": 0.9, "awesome": 0.85,
        "excellent": 0.9, "wonderful": 0.85, "fantastic": 0.9,
        "love": 0.8, "loved": 0.8, "loving": 0.75,
        "beautiful": 0.8, "brilliant": 0.85, "perfect": 0.9,
        "best": 0.85, "happy": 0.7, "glad": 0.6,
        "nice": 0.5, "cool": 0.5, "fire": 0.7,
        "lit": 0.7, "dope": 0.7, "sick": 0.6,
        "blessed": 0.7, "goat": 0.8, "queen": 0.6, "king": 0.6,
        "stunning": 0.85, "gorgeous": 0.8, "incredible": 0.85,
        "insane": 0.7, "legendary": 0.85, "iconic": 0.8,
        "obsessed": 0.7, "slay": 0.75, "slaying": 0.75,
        "periodt": 0.6, "vibe": 0.5, "vibes": 0.5,
        "talented": 0.8, "inspiring": 0.75, "proud": 0.7,
        "cute": 0.6, "adorable": 0.7, "precious": 0.65,
        "masterpiece": 0.9, "phenomenal": 0.9, "outstanding": 0.85,
        "superb": 0.85, "magnificent": 0.9, "marvelous": 0.85,
        "terrific": 0.8, "fabulous": 0.8, "glorious": 0.8,
        "delightful": 0.75, "impressive": 0.75, "remarkable": 0.8,
        "exceptional": 0.85, "splendid": 0.8, "elegant": 0.7,
        "graceful": 0.7, "charming": 0.65, "radiant": 0.75,
        "goated": 0.8, "wholesome": 0.75, "underrated": 0.7,
        "clean": 0.6, "heat": 0.65, "banger": 0.75, "slaps": 0.7,
        "w": 0.65, "bop": 0.6,
    }

    negative_words = {
        "bad": 0.6, "terrible": 0.85, "horrible": 0.85, "awful": 0.8,
        "worst": 0.9, "hate": 0.85, "hated": 0.85, "ugly": 0.7,
        "disgusting": 0.85, "pathetic": 0.8, "trash": 0.75,
        "garbage": 0.75, "cringe": 0.6, "cringy": 0.6,
        "boring": 0.6, "lame": 0.55, "mid": 0.4,
        "disappointed": 0.7, "disappointing": 0.7, "overrated": 0.6,
        "fake": 0.6, "fraud": 0.75, "scam": 0.8,
        "toxic": 0.7, "annoying": 0.6, "irritating": 0.65,
        "stupid": 0.65, "dumb": 0.6, "idiot": 0.7,
        "fool": 0.6, "clown": 0.6, "joke": 0.5,
        "wack": 0.6, "whack": 0.6, "sus": 0.4,
        "flop": 0.65, "flopped": 0.65, "fail": 0.65,
        "failed": 0.65, "failure": 0.7, "disaster": 0.8,
        "catastrophe": 0.85, "atrocious": 0.9, "abysmal": 0.85,
        "dreadful": 0.8, "horrendous": 0.85, "ghastly": 0.8,
        "appalling": 0.85, "repulsive": 0.8, "vile": 0.8,
        "wretched": 0.75, "miserable": 0.75, "deplorable": 0.8,
        "offensive": 0.7, "unacceptable": 0.7, "mediocre": 0.5,
        "inferior": 0.6, "lousy": 0.65, "pitiful": 0.7,
        "shameful": 0.7, "disgraceful": 0.8, "worthless": 0.8,
        "useless": 0.7, "pointless": 0.6, "rubbish": 0.7,
    }

    neutral_words = {
        "ok": 0.1, "okay": 0.1, "fine": 0.15, "alright": 0.1,
        "average": 0.1, "decent": 0.2, "normal": 0.05, "regular": 0.05,
        "standard": 0.05, "moderate": 0.1, "fair": 0.15,
        "reasonable": 0.15, "adequate": 0.1, "acceptable": 0.15,
        "ordinary": 0.05, "typical": 0.05, "common": 0.05,
        "usual": 0.05, "basic": 0.1,
    }

    for word, strength in positive_words.items():
        lex.add_entry(LexiconEntry(
            token=word, polarity=PolarityDirection.POSITIVE,
            strength=strength, source=EntrySource.MANUAL_SLANG,
        ))

    for word, strength in negative_words.items():
        lex.add_entry(LexiconEntry(
            token=word, polarity=PolarityDirection.NEGATIVE,
            strength=strength, source=EntrySource.MANUAL_SLANG,
        ))

    for word, strength in neutral_words.items():
        lex.add_entry(LexiconEntry(
            token=word, polarity=PolarityDirection.NEUTRAL,
            strength=strength, source=EntrySource.MANUAL_SLANG,
        ))

    # ── Emoji entries ───────────────────────────────────────────────────
    emoji_sentiment = {
        "❤️": ("positive", 0.8), "❤": ("positive", 0.8),
        "😍": ("positive", 0.85),
        "🔥": ("positive", 0.7), "💯": ("positive", 0.75),
        "👏": ("positive", 0.6), "🙌": ("positive", 0.7),
        "😂": ("positive", 0.5), "🤣": ("positive", 0.55),
        "😊": ("positive", 0.65), "🥰": ("positive", 0.8),
        "💕": ("positive", 0.75), "✨": ("positive", 0.6),
        "🎉": ("positive", 0.7), "👑": ("positive", 0.65),
        "💪": ("positive", 0.6), "🙏": ("positive", 0.5),
        "😭": ("positive", 0.4),  # often used positively in social media
        "💀": ("positive", 0.3),  # "I'm dead" = funny
        "😢": ("negative", 0.6), "😡": ("negative", 0.75),
        "🤮": ("negative", 0.8), "👎": ("negative", 0.7),
        "💔": ("negative", 0.7), "😤": ("negative", 0.6),
        "🙄": ("negative", 0.5), "😒": ("negative", 0.55),
        "🤡": ("negative", 0.6), "🗑️": ("negative", 0.7), "🗑": ("negative", 0.7),
        "😐": ("neutral", 0.1), "🤷": ("neutral", 0.1),
    }

    for emoji, (pol, strength) in emoji_sentiment.items():
        lex.add_entry(LexiconEntry(
            token=emoji, polarity=PolarityDirection(pol),
            strength=strength, source=EntrySource.EMOJI,
        ))

    # ── Phrase entries ──────────────────────────────────────────────────
    phrase_sentiment = {
        "not bad": ("positive", 0.4),
        "not good": ("negative", 0.4),
        "no way": ("negative", 0.3),
        "on point": ("positive", 0.7),
        "on fire": ("positive", 0.75),
        "shut up": ("negative", 0.5),
        "give up": ("negative", 0.5),
        "let down": ("negative", 0.6),
        "rip off": ("negative", 0.7),
        "sold out": ("neutral", 0.1),
        "well done": ("positive", 0.7),
        "top notch": ("positive", 0.8),
        "hands down": ("positive", 0.6),
        "no doubt": ("positive", 0.5),
        "next level": ("positive", 0.75),
        "out of this world": ("positive", 0.85),
        "piece of art": ("positive", 0.8),
        "waste of time": ("negative", 0.75),
        "waste of money": ("negative", 0.8),
        "could not care less": ("negative", 0.6),
        "over the top": ("negative", 0.4),
        "so good": ("positive", 0.75),
        "so bad": ("negative", 0.75),
        "too good": ("positive", 0.7),
        "too bad": ("negative", 0.6),
        "bro cooked": ("positive", 0.8),
        "cooked with": ("positive", 0.75),
        "made my day": ("positive", 0.85),
        "on repeat": ("positive", 0.7),
        "fell off": ("negative", 0.65),
        "w reel": ("positive", 0.75),
        "l take": ("negative", 0.7),
    }

    for phrase, (pol, strength) in phrase_sentiment.items():
        lex.add_entry(LexiconEntry(
            token=phrase, polarity=PolarityDirection(pol),
            strength=strength, source=EntrySource.PHRASE,
            is_phrase=True,
        ))
