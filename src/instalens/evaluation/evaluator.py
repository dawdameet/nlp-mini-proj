"""
Evaluation framework — metrics computation, baseline comparison, and error analysis.

Implements PRD §11 Evaluation Plan:
- Macro-F1, per-class precision/recall
- Confusion matrix
- Abstention rate, lexical coverage
- Baseline comparison (A/B/C/D)
- Error categorization
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..scoring.engine import SentimentResult, SentimentLabel


LABEL_ORDER = ["Positive", "Negative", "Neutral", "Mixed", "Uncertain"]


@dataclass
class ClassMetrics:
    """Precision, recall, and F1 for a single class."""
    label: str
    precision: float
    recall: float
    f1: float
    support: int       # number of true instances


@dataclass
class EvaluationReport:
    """Complete evaluation results."""
    total: int
    macro_f1: float
    class_metrics: list[ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    abstention_rate: float
    lexical_coverage: float
    error_categories: dict[str, int]
    baseline_name: str = ""


class Evaluator:
    """
    Evaluate sentiment predictions against human annotations.

    Parameters
    ----------
    predictions : list[SentimentResult]
        Model outputs.
    gold_labels : list[str]
        Human-annotated labels (same order as predictions).
    baseline_name : str
        Name of this baseline for comparison.
    """

    def __init__(
        self,
        predictions: list[SentimentResult],
        gold_labels: list[str],
        baseline_name: str = "Baseline",
    ) -> None:
        if len(predictions) != len(gold_labels):
            raise ValueError(
                f"Predictions ({len(predictions)}) and gold labels "
                f"({len(gold_labels)}) must have the same length."
            )
        self.predictions = predictions
        self.gold_labels = [self._normalize_label(g) for g in gold_labels]
        self.baseline_name = baseline_name

    def evaluate(self) -> EvaluationReport:
        """Run full evaluation and return metrics."""
        pred_labels = [r.label.value for r in self.predictions]
        total = len(pred_labels)

        # Confusion matrix
        labels = LABEL_ORDER
        cm: dict[str, dict[str, int]] = {
            true: {pred: 0 for pred in labels} for true in labels
        }
        for gold, pred in zip(self.gold_labels, pred_labels):
            if gold in cm and pred in cm[gold]:
                cm[gold][pred] += 1

        # Per-class metrics
        class_metrics = []
        f1_scores = []
        for label in labels:
            tp = cm[label][label]
            fp = sum(cm[other][label] for other in labels if other != label)
            fn = sum(cm[label][other] for other in labels if other != label)
            support = sum(cm[label].values())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            class_metrics.append(ClassMetrics(
                label=label,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            ))
            if support > 0:
                f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # Abstention rate
        uncertain_count = sum(1 for p in pred_labels if p == "Uncertain")
        abstention_rate = uncertain_count / total if total > 0 else 0.0

        # Lexical coverage: fraction of comments with at least one evidence token
        covered = sum(1 for r in self.predictions if len(r.evidence) > 0)
        lexical_coverage = covered / total if total > 0 else 0.0

        # Error categorization
        error_cats = self._categorize_errors(pred_labels)

        return EvaluationReport(
            total=total,
            macro_f1=macro_f1,
            class_metrics=class_metrics,
            confusion_matrix=cm,
            abstention_rate=abstention_rate,
            lexical_coverage=lexical_coverage,
            error_categories=error_cats,
            baseline_name=self.baseline_name,
        )

    def _categorize_errors(self, pred_labels: list[str]) -> dict[str, int]:
        """Categorize prediction errors."""
        cats: Counter = Counter()
        for gold, pred, result in zip(self.gold_labels, pred_labels, self.predictions):
            if gold == pred:
                continue
            # Polarity flip
            if ({gold, pred} == {"Positive", "Negative"}):
                cats["polarity_flip"] += 1
            # Missed sentiment (predicted Neutral/Uncertain for Pos/Neg)
            elif gold in ("Positive", "Negative") and pred in ("Neutral", "Uncertain"):
                cats["missed_sentiment"] += 1
            # False sentiment (predicted Pos/Neg for Neutral)
            elif gold in ("Neutral", "Uncertain") and pred in ("Positive", "Negative"):
                cats["false_sentiment"] += 1
            # Mixed confusion
            elif "Mixed" in (gold, pred):
                cats["mixed_confusion"] += 1
            else:
                cats["other"] += 1
        return dict(cats)

    def format_report(self, report: EvaluationReport | None = None) -> str:
        """Format evaluation results for CLI display."""
        if report is None:
            report = self.evaluate()

        lines = [
            "",
            "═" * 60,
            f"  Evaluation Report: {report.baseline_name}",
            "═" * 60,
            "",
            f"  Total samples:     {report.total}",
            f"  Macro-F1:          {report.macro_f1:.4f}",
            f"  Abstention rate:   {report.abstention_rate:.2%}",
            f"  Lexical coverage:  {report.lexical_coverage:.2%}",
            "",
            "  ── Per-Class Metrics ───────────────────",
            f"  {'Label':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}",
            "  " + "─" * 44,
        ]

        for cm in report.class_metrics:
            lines.append(
                f"  {cm.label:<12} {cm.precision:>8.4f} {cm.recall:>8.4f} "
                f"{cm.f1:>8.4f} {cm.support:>8}"
            )

        # Confusion matrix
        labels = LABEL_ORDER
        lines.append("")
        lines.append("  ── Confusion Matrix ────────────────────")
        header = "  " + f"{'True↓ Pred→':<12}" + "".join(f"{l[:5]:>7}" for l in labels)
        lines.append(header)
        lines.append("  " + "─" * (12 + 7 * len(labels)))
        for true_label in labels:
            row = f"  {true_label:<12}"
            for pred_label in labels:
                count = report.confusion_matrix.get(true_label, {}).get(pred_label, 0)
                row += f"{count:>7}"
            lines.append(row)

        # Error categories
        if report.error_categories:
            lines.append("")
            lines.append("  ── Error Categories ────────────────────")
            for cat, count in sorted(report.error_categories.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat:<25} {count:>5}")

        lines.append("")
        lines.append("═" * 60)

        return "\n".join(lines)

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize label string to match SentimentLabel values."""
        mapping = {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "mixed": "Mixed",
            "uncertain": "Uncertain",
            "pos": "Positive",
            "neg": "Negative",
            "neu": "Neutral",
            "mix": "Mixed",
            "unc": "Uncertain",
        }
        return mapping.get(label.lower().strip(), label.strip())
