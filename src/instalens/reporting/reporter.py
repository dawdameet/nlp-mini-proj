"""
Report generation — aggregate statistics, formatted output, and CSV export.

Covers PRD §5 Reporting requirements:
- Comment counts (requested, retrieved, analyzed)
- Sentiment distribution
- Individual classifications with explanations
- Confidence/uncertainty indicators
- Top lexical signals
- Downloadable CSV
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

from ..scoring.engine import SentimentResult, SentimentLabel


@dataclass
class AggregateReport:
    """Aggregate sentiment analysis report."""
    # Counts
    comments_requested: int
    comments_retrieved: int
    comments_analyzed: int
    comments_abstained: int

    # Distribution
    distribution: dict[str, int]       # label → count
    distribution_pct: dict[str, float] # label → percentage

    # Top signals
    top_positive_signals: list[tuple[str, float]]  # (token, total_contribution)
    top_negative_signals: list[tuple[str, float]]

    # Individual results
    results: list[SentimentResult]

    # Metadata
    timestamp: str = ""
    source: str = ""
    config_summary: dict = field(default_factory=dict)

    # Overall sentiment summary
    overall_sentiment: str = "Neutral"
    overall_score: float = 0.0
    net_sentiment: float = 0.0
    sentiment_verdict: str = ""


class ReportGenerator:
    """Generates aggregate reports from a list of SentimentResults."""

    def __init__(
        self,
        results: list[SentimentResult],
        comments_requested: int = 0,
        comments_retrieved: int = 0,
        source: str = "",
    ) -> None:
        self.results = results
        self.comments_requested = comments_requested
        self.comments_retrieved = comments_retrieved
        self.source = source

    def generate(self) -> AggregateReport:
        """Build the aggregate report."""
        analyzed = [r for r in self.results if r.label != SentimentLabel.UNCERTAIN]
        abstained = [r for r in self.results if r.label == SentimentLabel.UNCERTAIN]

        # Distribution
        dist = Counter(r.label.value for r in self.results)
        total = len(self.results) or 1
        dist_pct = {label: (count / total) * 100 for label, count in dist.items()}

        # Top signals
        pos_signals: Counter = Counter()
        neg_signals: Counter = Counter()
        for r in self.results:
            for ev in r.evidence:
                if ev.final_contribution > 0:
                    pos_signals[ev.token] += ev.final_contribution
                elif ev.final_contribution < 0:
                    neg_signals[ev.token] += abs(ev.final_contribution)

        # Compute overall sentiment & verdict
        total_analyzed = len(analyzed)
        if total_analyzed == 0:
            overall_sentiment = "Uncertain"
            overall_score = 0.0
            net_sentiment = 0.0
            sentiment_verdict = "Insufficient evidence to determine sentiment."
        else:
            pos_count = dist.get("Positive", 0)
            neg_count = dist.get("Negative", 0)
            neu_count = dist.get("Neutral", 0)
            mix_count = dist.get("Mixed", 0)

            pos_pct = dist_pct.get("Positive", 0.0)
            neg_pct = dist_pct.get("Negative", 0.0)
            neu_pct = dist_pct.get("Neutral", 0.0)
            mix_pct = dist_pct.get("Mixed", 0.0)

            overall_score = sum(r.score for r in analyzed) / total_analyzed
            net_sentiment = (pos_count - neg_count) / total_analyzed

            if pos_pct >= 50.0 or (pos_pct >= 40.0 and pos_pct > neg_pct * 1.5):
                overall_sentiment = "Positive"
                sentiment_verdict = f"Audience reaction is overwhelmingly positive ({pos_pct:.1f}% favorable)."
            elif neg_pct >= 50.0 or (neg_pct >= 40.0 and neg_pct > pos_pct * 1.5):
                overall_sentiment = "Negative"
                sentiment_verdict = f"Audience reaction is predominantly negative/critical ({neg_pct:.1f}% unfavorable)."
            elif mix_pct >= 30.0 or (pos_pct >= 25.0 and neg_pct >= 25.0):
                overall_sentiment = "Mixed"
                sentiment_verdict = f"Polarized audience reaction — balanced mix of positive ({pos_pct:.1f}%) and negative ({neg_pct:.1f}%) opinions."
            elif neu_pct >= 50.0 or (neu_pct > pos_pct and neu_pct > neg_pct):
                overall_sentiment = "Neutral"
                sentiment_verdict = f"Predominantly neutral / conversational responses ({neu_pct:.1f}% neutral)."
            else:
                majority = max(
                    [("Positive", pos_count), ("Negative", neg_count), ("Neutral", neu_count), ("Mixed", mix_count)],
                    key=lambda x: x[1]
                )[0]
                overall_sentiment = majority
                sentiment_verdict = f"Leaning {majority.lower()} based on prevailing comment distribution."

        return AggregateReport(
            comments_requested=self.comments_requested,
            comments_retrieved=self.comments_retrieved,
            comments_analyzed=len(analyzed),
            comments_abstained=len(abstained),
            distribution=dict(dist),
            distribution_pct=dist_pct,
            top_positive_signals=pos_signals.most_common(15),
            top_negative_signals=neg_signals.most_common(15),
            results=self.results,
            timestamp=datetime.now().isoformat(),
            source=self.source,
            overall_sentiment=overall_sentiment,
            overall_score=overall_score,
            net_sentiment=net_sentiment,
            sentiment_verdict=sentiment_verdict,
        )

    def export_csv(self, path: str | Path) -> Path:
        """
        Export individual results to a CSV file.

        Columns: comment_id, original_text, normalized_text, label,
                 score, positive_score, negative_score, confidence,
                 evidence_summary, uncertainty_flags
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index", "original_text", "normalized_text", "label",
                "score", "positive_score", "negative_score", "confidence",
                "evidence_summary", "uncertainty_flags",
            ])
            for i, r in enumerate(self.results):
                ev_summary = "; ".join(
                    f"{ev.token}({ev.modifier})={ev.final_contribution:+.3f}"
                    for ev in r.evidence
                )
                writer.writerow([
                    i,
                    r.original_text,
                    r.normalized_text,
                    r.label.value,
                    f"{r.score:.4f}",
                    f"{r.positive_score:.4f}",
                    f"{r.negative_score:.4f}",
                    f"{r.confidence:.4f}",
                    ev_summary,
                    "|".join(r.uncertainty_flags),
                ])

        return path

    def format_summary(self, report: AggregateReport | None = None) -> str:
        """Format a human-readable summary for CLI output."""
        if report is None:
            report = self.generate()

        icon_map = {
            "Positive": "🟢",
            "Negative": "🔴",
            "Neutral": "⚪",
            "Mixed": "🟡",
            "Uncertain": "❓",
        }
        icon = icon_map.get(report.overall_sentiment, "📊")

        lines = [
            "",
            "═" * 60,
            "  InstaLens — Sentiment Analysis Report",
            "═" * 60,
            "",
            f"  Source:    {report.source or 'N/A'}",
            f"  Time:      {report.timestamp}",
            "",
            f"  🎯 OVERALL SENTIMENT:  {icon} {report.overall_sentiment.upper()}",
            f"  Verdict:   {report.sentiment_verdict}",
            f"             Net Polarity: {report.net_sentiment:+.2f} | Average Score: {report.overall_score:+.2f}",
            "",
            "  ── Comment Counts ──────────────────────",
            f"  Requested:  {report.comments_requested}",
            f"  Retrieved:  {report.comments_retrieved}",
            f"  Analyzed:   {report.comments_analyzed}",
            f"  Abstained:  {report.comments_abstained}  (Uncertain)",
            "",
            "  ── Sentiment Distribution ──────────────",
        ]

        # Distribution bar chart
        max_count = max(report.distribution.values()) if report.distribution else 1
        label_order = ["Positive", "Negative", "Neutral", "Mixed", "Uncertain"]
        for label in label_order:
            count = report.distribution.get(label, 0)
            pct = report.distribution_pct.get(label, 0.0)
            bar_len = int((count / max_count) * 25) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"  {label:<12} {bar:<25} {count:>4} ({pct:5.1f}%)")

        # Top signals
        if report.top_positive_signals:
            lines.append("")
            lines.append("  ── Top Positive Signals ────────────────")
            for token, contrib in report.top_positive_signals[:10]:
                lines.append(f"    {token:<20} +{contrib:.3f}")

        if report.top_negative_signals:
            lines.append("")
            lines.append("  ── Top Negative Signals ────────────────")
            for token, contrib in report.top_negative_signals[:10]:
                lines.append(f"    {token:<20} -{contrib:.3f}")

        lines.append("")
        lines.append("═" * 60)

        return "\n".join(lines)

    def format_individual(
        self, max_display: int = 20, show_all: bool = False
    ) -> str:
        """Format individual comment results for CLI display."""
        lines = [
            "",
            "  ── Individual Results ──────────────────",
            "",
        ]

        results_to_show = self.results if show_all else self.results[:max_display]

        for i, r in enumerate(results_to_show):
            label_icon = {
                SentimentLabel.POSITIVE: "🟢",
                SentimentLabel.NEGATIVE: "🔴",
                SentimentLabel.NEUTRAL: "⚪",
                SentimentLabel.MIXED: "🟡",
                SentimentLabel.UNCERTAIN: "❓",
            }.get(r.label, "?")

            # Truncate long comments
            text = r.original_text
            if len(text) > 80:
                text = text[:77] + "..."

            lines.append(f"  {i+1:>3}. {label_icon} [{r.label.value:<10}] {text}")
            lines.append(f"       Score: {r.score:+.3f}  Confidence: {r.confidence:.2f}")

            if r.evidence:
                ev_str = ", ".join(
                    f'"{ev.token}"' for ev in r.evidence[:5]
                )
                if len(r.evidence) > 5:
                    ev_str += f" (+{len(r.evidence) - 5} more)"
                lines.append(f"       Evidence: {ev_str}")

            if r.uncertainty_flags:
                lines.append(f"       ⚠ {', '.join(r.uncertainty_flags)}")

            lines.append("")

        if not show_all and len(self.results) > max_display:
            lines.append(
                f"  ... and {len(self.results) - max_display} more. "
                f"Use --show-all or export CSV for full results."
            )

        return "\n".join(lines)
