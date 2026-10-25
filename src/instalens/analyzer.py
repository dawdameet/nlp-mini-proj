"""
InstaLens — Analyzer orchestrator.

Wires together the full pipeline:
  ingestion → preprocessing → scoring → reporting

This is the portable core that both the CLI and (future) FastAPI backend use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .preprocessing.pipeline import preprocess_comment, PreprocessedComment
from .scoring.engine import SentimentScorer, ScorerConfig, SentimentResult
from .lexicon.lexicon import SentimentLexicon, build_default_lexicon
from .ingestion.csv_loader import CSVCommentLoader, Comment
from .reporting.reporter import ReportGenerator, AggregateReport


class Analyzer:
    """
    High-level analyzer that orchestrates the InstaLens pipeline.

    Usage:
        analyzer = Analyzer()
        report = analyzer.analyze_csv("comments.csv", k=100)
        print(analyzer.format_report(report))
    """

    def __init__(
        self,
        lexicon: SentimentLexicon | None = None,
        config: ScorerConfig | None = None,
    ) -> None:
        self.lexicon = lexicon or build_default_lexicon()
        self.config = config or ScorerConfig()
        self.scorer = SentimentScorer(self.lexicon, self.config)

    def analyze_comments(
        self,
        comments: list[Comment],
        comments_requested: int | None = None,
    ) -> tuple[list[SentimentResult], AggregateReport]:
        """
        Analyze a list of Comment objects.

        Returns (individual_results, aggregate_report).
        """
        results: list[SentimentResult] = []

        for comment in comments:
            preprocessed = preprocess_comment(comment.text)
            result = self.scorer.score(preprocessed)
            results.append(result)

        gen = ReportGenerator(
            results=results,
            comments_requested=comments_requested or len(comments),
            comments_retrieved=len(comments),
            source="comments",
        )
        report = gen.generate()

        return results, report

    def analyze_csv(
        self,
        filepath: str | Path,
        k: int | None = None,
    ) -> tuple[list[SentimentResult], AggregateReport]:
        """
        Analyze comments from a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Path to CSV file.
        k : int, optional
            Maximum number of comments to analyze.

        Returns
        -------
        (results, report) tuple
        """
        loader = CSVCommentLoader(filepath)
        comments = loader.load(k=k)

        results: list[SentimentResult] = []
        errors: list[str] = []

        for comment in comments:
            try:
                preprocessed = preprocess_comment(comment.text)
                result = self.scorer.score(preprocessed)
                results.append(result)
            except Exception as e:
                errors.append(f"Error processing comment {comment.comment_id}: {e}")
                # Create an error result instead of silently skipping
                results.append(_make_error_result(comment.text, str(e)))

        gen = ReportGenerator(
            results=results,
            comments_requested=k or len(comments),
            comments_retrieved=len(comments),
            source=str(filepath),
        )
        report = gen.generate()

        if errors:
            print(f"\n⚠ {len(errors)} processing errors occurred.")
            for err in errors[:5]:
                print(f"  • {err}")

        return results, report

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze a single text string. Convenience method."""
        preprocessed = preprocess_comment(text)
        return self.scorer.score(preprocessed)

    def format_report(self, report: AggregateReport) -> str:
        """Format an aggregate report for display."""
        gen = ReportGenerator(report.results)
        return gen.format_summary(report)

    def format_individual(
        self, results: list[SentimentResult], max_display: int = 20
    ) -> str:
        """Format individual results for display."""
        gen = ReportGenerator(results)
        return gen.format_individual(max_display=max_display)

    def export_csv(
        self,
        results: list[SentimentResult],
        output_path: str | Path,
    ) -> Path:
        """Export results to CSV."""
        gen = ReportGenerator(results)
        return gen.export_csv(output_path)


def _make_error_result(text: str, error_msg: str) -> SentimentResult:
    """Create an Uncertain result for a comment that failed processing."""
    from .scoring.engine import SentimentLabel
    return SentimentResult(
        original_text=text,
        normalized_text="",
        label=SentimentLabel.UNCERTAIN,
        score=0.0,
        positive_score=0.0,
        negative_score=0.0,
        confidence=0.0,
        evidence=[],
        uncertainty_flags=[f"processing_error: {error_msg}"],
    )
