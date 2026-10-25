"""
Sentiment scoring engine.

Implements the deterministic, rule-based scoring described in PRD §6.4:
  S(c) = Σ polarity contributions, modified by negation, intensification,
          downtoning, phrase matches, clause weighting, and emoji evidence.

Produces an explainable SentimentResult for every comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..lexicon.lexicon import (
    SentimentLexicon,
    LexiconEntry,
    PolarityDirection,
    ValidationStatus,
)
from ..preprocessing.pipeline import PreprocessedComment
from ..preprocessing.tokenizer import TokenizedComment, Token


# ── Classification labels ───────────────────────────────────────────────────

class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    UNCERTAIN = "Uncertain"


# ── Evidence tracking ───────────────────────────────────────────────────────

@dataclass
class ScoringEvidence:
    """A single piece of scoring evidence for explainability."""
    token: str
    raw_polarity: str
    raw_strength: float
    modifier: str               # e.g. "negated", "intensified", "downtoned", "phrase"
    final_contribution: float   # the actual value added to the score
    source: str                 # lexicon source tag


@dataclass
class SentimentResult:
    """Complete sentiment analysis result for a single comment."""
    original_text: str
    normalized_text: str
    label: SentimentLabel
    score: float                       # aggregate polarity score
    positive_score: float              # sum of positive contributions
    negative_score: float              # sum of negative contributions (as positive magnitude)
    confidence: float                  # 0.0 – 1.0
    evidence: list[ScoringEvidence] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        """Human-readable explanation of the classification."""
        lines = []
        for ev in self.evidence:
            mod_str = f" [{ev.modifier}]" if ev.modifier != "none" else ""
            sign = "+" if ev.final_contribution >= 0 else ""
            lines.append(
                f'  "{ev.token}": {ev.raw_polarity} ({ev.raw_strength:.2f})'
                f"{mod_str} → {sign}{ev.final_contribution:.3f}"
            )
        if self.uncertainty_flags:
            lines.append(f"  ⚠ Flags: {', '.join(self.uncertainty_flags)}")
        lines.append(f"  Score: {self.score:+.3f} → {self.label.value}")
        return "\n".join(lines)


# ── Scorer configuration ────────────────────────────────────────────────────

@dataclass
class ScorerConfig:
    """Tunable thresholds and weights. Versioned alongside the lexicon."""
    positive_threshold: float = 0.15
    negative_threshold: float = -0.15
    mixed_threshold: float = 0.10       # min evidence on both sides
    min_evidence_tokens: int = 1        # below this → Uncertain
    negation_multiplier: float = -0.8   # flip + weaken slightly
    intensifier_multiplier: float = 1.5
    downtoner_multiplier: float = 0.5
    contrast_weight: float = 1.5        # boost clause after "but"
    emoji_weight: float = 0.8           # relative to word evidence
    confidence_scale: float = 2.0       # logistic scaling


# ── Word lists for modifiers ────────────────────────────────────────────────

NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "without", "hardly", "barely",
    "scarcely", "seldom", "rarely", "lack", "lacking",
})

INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "so": 1.4, "extremely": 1.7,
    "absolutely": 1.6, "totally": 1.5, "completely": 1.5,
    "utterly": 1.6, "incredibly": 1.6, "insanely": 1.6,
    "super": 1.5, "hella": 1.5, "mad": 1.4, "crazy": 1.4,
    "freaking": 1.4, "freakin": 1.4, "frickin": 1.4, "damn": 1.4,
    "bloody": 1.4, "most": 1.3, "highly": 1.4, "deeply": 1.4,
    "remarkably": 1.4, "exceptionally": 1.5, "extraordinarily": 1.6,
    "genuinely": 1.3, "truly": 1.4, "seriously": 1.4,
    "ridiculously": 1.5, "unbelievably": 1.6, "impossibly": 1.5,
    "wildly": 1.4, "immensely": 1.5, "enormously": 1.5,
}

DOWNTONERS = {
    "slightly": 0.5, "somewhat": 0.6, "fairly": 0.7,
    "rather": 0.7, "quite": 0.8, "a bit": 0.5, "a little": 0.5,
    "kind of": 0.5, "sort of": 0.5, "mildly": 0.5,
    "partially": 0.6, "marginally": 0.4, "hardly": 0.3,
    "barely": 0.3, "just": 0.7, "only": 0.6,
}

# Negation scope: how many tokens after a negation word are affected
NEGATION_SCOPE = 3


class SentimentScorer:
    """
    Rule-based sentiment scoring engine.

    For each comment:
    1. Check for phrase matches first (they take priority).
    2. Score individual tokens with modifier detection.
    3. Weight clauses (boost post-contrast clause).
    4. Aggregate and classify.
    """

    def __init__(
        self,
        lexicon: SentimentLexicon,
        config: ScorerConfig | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.config = config or ScorerConfig()

    def score(self, comment: PreprocessedComment) -> SentimentResult:
        """
        Score a single preprocessed comment.

        Returns a fully explainable SentimentResult.
        """
        tc = comment.tokenized
        token_texts = [t.text for t in tc.tokens]
        evidence: list[ScoringEvidence] = []
        uncertainty_flags: list[str] = []

        # Track which token positions are consumed by phrase matches
        consumed: set[int] = set()

        # ── Phase 1: Phrase matching ────────────────────────────────────
        phrase_matches = self.lexicon.find_phrases_in_tokens(token_texts)
        for start, end, entry in phrase_matches:
            contribution = self._polarity_sign(entry.polarity) * entry.strength
            # Apply clause weighting if phrase is in post-contrast clause
            if tc.has_contrast and tc.contrast_index is not None:
                phrase_clause = tc.tokens[start].clause_index
                if phrase_clause >= tc.contrast_index:
                    contribution *= self.config.contrast_weight

            evidence.append(ScoringEvidence(
                token=entry.token,
                raw_polarity=entry.polarity.value,
                raw_strength=entry.strength,
                modifier="phrase",
                final_contribution=contribution,
                source=entry.source.value,
            ))
            for idx in range(start, end):
                consumed.add(idx)

        # ── Phase 2: Token-level scoring ────────────────────────────────
        negation_remaining = 0   # tokens remaining in negation scope

        for i, tok in enumerate(tc.tokens):
            if i in consumed:
                continue
            if tok.is_clause_boundary:
                negation_remaining = 0  # reset negation at clause boundary
                continue
            if tok.is_punctuation and not tok.is_emoji:
                continue

            text = tok.text.lower()

            # Check for negation trigger
            if text in NEGATION_WORDS:
                negation_remaining = NEGATION_SCOPE
                continue

            # Check for intensifier/downtoner (affects next sentiment word)
            if text in INTENSIFIERS or text in DOWNTONERS:
                # Don't consume — we check these when scoring the next word
                continue

            # Look up in lexicon
            entry = self.lexicon.lookup(text)
            if entry is None or entry.status != ValidationStatus.VALIDATED:
                if negation_remaining > 0:
                    negation_remaining -= 1
                continue

            # Base contribution
            contribution = self._polarity_sign(entry.polarity) * entry.strength
            modifier = "none"

            # Apply emoji weight
            if tok.is_emoji:
                contribution *= self.config.emoji_weight
                modifier = "emoji"

            # Check for preceding intensifier/downtoner
            if i > 0:
                prev_text = tc.tokens[i - 1].text.lower()
                if prev_text in INTENSIFIERS:
                    mult = INTENSIFIERS[prev_text]
                    contribution *= mult
                    modifier = f"intensified (×{mult})"
                elif prev_text in DOWNTONERS:
                    mult = DOWNTONERS[prev_text]
                    contribution *= mult
                    modifier = f"downtoned (×{mult})"

            # Apply negation
            if negation_remaining > 0:
                contribution *= self.config.negation_multiplier
                modifier = "negated" if modifier == "none" else f"{modifier}, negated"
                negation_remaining -= 1

            # Apply contrast clause weighting
            if tc.has_contrast and tc.contrast_index is not None:
                if tok.clause_index >= tc.contrast_index:
                    contribution *= self.config.contrast_weight
                    modifier = (
                        "contrast-boosted"
                        if modifier == "none"
                        else f"{modifier}, contrast-boosted"
                    )

            evidence.append(ScoringEvidence(
                token=text,
                raw_polarity=entry.polarity.value,
                raw_strength=entry.strength,
                modifier=modifier,
                final_contribution=contribution,
                source=entry.source.value,
            ))

            if negation_remaining > 0:
                negation_remaining -= 1

        # ── Phase 3: Aggregation and classification ─────────────────────
        total_score = sum(ev.final_contribution for ev in evidence)
        pos_score = sum(ev.final_contribution for ev in evidence if ev.final_contribution > 0)
        neg_score = sum(abs(ev.final_contribution) for ev in evidence if ev.final_contribution < 0)

        # Uncertainty checks
        if len(evidence) < self.config.min_evidence_tokens:
            uncertainty_flags.append("insufficient_evidence")

        if not token_texts:
            uncertainty_flags.append("empty_comment")

        # Classify
        label = self._classify(total_score, pos_score, neg_score,
                               len(evidence), uncertainty_flags)

        # Confidence: logistic-ish scaling of evidence strength
        if evidence:
            avg_strength = sum(abs(ev.final_contribution) for ev in evidence) / len(evidence)
            confidence = min(1.0, avg_strength * self.config.confidence_scale * (len(evidence) ** 0.3))
        else:
            confidence = 0.0

        return SentimentResult(
            original_text=comment.original,
            normalized_text=comment.normalized,
            label=label,
            score=total_score,
            positive_score=pos_score,
            negative_score=neg_score,
            confidence=confidence,
            evidence=evidence,
            uncertainty_flags=uncertainty_flags,
        )

    def _classify(
        self,
        score: float,
        pos_score: float,
        neg_score: float,
        evidence_count: int,
        flags: list[str],
    ) -> SentimentLabel:
        """Apply classification rules per PRD §6.4."""
        if "insufficient_evidence" in flags or "empty_comment" in flags:
            return SentimentLabel.UNCERTAIN

        # Mixed: substantial evidence on both sides
        if (pos_score >= self.config.mixed_threshold
                and neg_score >= self.config.mixed_threshold):
            return SentimentLabel.MIXED

        if score > self.config.positive_threshold:
            return SentimentLabel.POSITIVE
        elif score < self.config.negative_threshold:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL

    @staticmethod
    def _polarity_sign(polarity: PolarityDirection) -> float:
        if polarity == PolarityDirection.POSITIVE:
            return 1.0
        elif polarity == PolarityDirection.NEGATIVE:
            return -1.0
        return 0.0
