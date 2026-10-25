"""Tests for the InstaLens NLP pipeline."""

import sys
from pathlib import Path

import pytest

# ════════════════════════════════════════════════════════════════════════════
# Preprocessing tests
# ════════════════════════════════════════════════════════════════════════════

class TestNormalizer:
    """Test text normalization."""

    def test_empty_input(self):
        from instalens.preprocessing.normalizer import normalize_text
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_url_replacement(self):
        from instalens.preprocessing.normalizer import normalize_text
        result = normalize_text("Check this https://example.com/page")
        assert "<URL>" in result
        assert "https://" not in result

    def test_mention_replacement(self):
        from instalens.preprocessing.normalizer import normalize_text
        result = normalize_text("Thanks @john_doe for this!")
        assert "<USER>" in result
        assert "@john_doe" not in result

    def test_hashtag_handling(self):
        from instalens.preprocessing.normalizer import normalize_text
        result = normalize_text("#love this #amazing post")
        assert "love" in result
        assert "amazing" in result
        assert "#" not in result

    def test_repeated_punctuation(self):
        from instalens.preprocessing.normalizer import normalize_text
        result = normalize_text("This is amazing!!!!")
        assert "!!" in result
        assert "!!!!" not in result

    def test_repeated_characters(self):
        from instalens.preprocessing.normalizer import normalize_text
        result = normalize_text("sooooooo good")
        assert "soo" in result
        assert "soooo" not in result

    def test_emoji_preservation(self):
        from instalens.preprocessing.normalizer import normalize_text
        result = normalize_text("Love this ❤️🔥")
        assert "❤️" in result or "❤" in result
        assert "🔥" in result

    def test_whitespace_normalization(self):
        from instalens.preprocessing.normalizer import normalize_text
        result = normalize_text("  too   many    spaces  ")
        assert result == "too many spaces"


class TestContractions:
    """Test contraction expansion."""

    def test_basic_contractions(self):
        from instalens.preprocessing.contractions import expand_contractions
        assert "can not" in expand_contractions("I can't believe it")
        assert "do not" in expand_contractions("I don't like it")
        assert "is not" in expand_contractions("It isn't good")

    def test_informal_contractions(self):
        from instalens.preprocessing.contractions import expand_contractions
        assert "going to" in expand_contractions("gonna do it")
        assert "want to" in expand_contractions("wanna go")
        assert "do not know" in expand_contractions("I dunno")

    def test_case_preservation(self):
        from instalens.preprocessing.contractions import expand_contractions
        result = expand_contractions("Can't stop")
        assert result.startswith("Can not") or result.startswith("can not")


class TestTokenizer:
    """Test tokenization and clause detection."""

    def test_basic_tokenization(self):
        from instalens.preprocessing.tokenizer import tokenize
        result = tokenize("this is a test")
        assert len(result.tokens) == 4
        assert result.tokens[0].text == "this"

    def test_clause_detection(self):
        from instalens.preprocessing.tokenizer import tokenize
        result = tokenize("not bad , but actually amazing")
        assert result.has_contrast is True
        assert len(result.clauses) >= 2

    def test_emoji_detection(self):
        from instalens.preprocessing.tokenizer import tokenize
        result = tokenize("love this 🔥")
        emoji_tokens = [t for t in result.tokens if t.is_emoji]
        assert len(emoji_tokens) >= 1


class TestPipeline:
    """Test the full preprocessing pipeline."""

    def test_full_pipeline(self):
        from instalens.preprocessing.pipeline import preprocess_comment
        result = preprocess_comment("I can't believe how amazing this is!!! @user 🔥")
        assert result.original == "I can't believe how amazing this is!!! @user 🔥"
        assert "can not" in result.normalized
        assert "<USER>" in result.normalized
        assert len(result.tokenized.tokens) > 0


# ════════════════════════════════════════════════════════════════════════════
# Lexicon tests
# ════════════════════════════════════════════════════════════════════════════

class TestLexicon:
    """Test sentiment lexicon."""

    def test_default_lexicon_loads(self):
        from instalens.lexicon.lexicon import build_default_lexicon
        lex = build_default_lexicon()
        assert lex.size > 0

    def test_positive_lookup(self):
        from instalens.lexicon.lexicon import build_default_lexicon
        lex = build_default_lexicon()
        entry = lex.lookup("amazing")
        assert entry is not None
        assert entry.polarity.value == "positive"

    def test_negative_lookup(self):
        from instalens.lexicon.lexicon import build_default_lexicon
        lex = build_default_lexicon()
        entry = lex.lookup("terrible")
        assert entry is not None
        assert entry.polarity.value == "negative"

    def test_phrase_lookup(self):
        from instalens.lexicon.lexicon import build_default_lexicon
        lex = build_default_lexicon()
        entry = lex.lookup_phrase("not bad")
        assert entry is not None

    def test_phrase_matching(self):
        from instalens.lexicon.lexicon import build_default_lexicon
        lex = build_default_lexicon()
        tokens = ["this", "is", "not", "bad", "at", "all"]
        matches = lex.find_phrases_in_tokens(tokens)
        assert len(matches) > 0

    def test_case_insensitive(self):
        from instalens.lexicon.lexicon import build_default_lexicon
        lex = build_default_lexicon()
        assert lex.lookup("AMAZING") is not None
        assert lex.lookup("Amazing") is not None

    def test_emoji_in_lexicon(self):
        from instalens.lexicon.lexicon import build_default_lexicon
        lex = build_default_lexicon()
        entry = lex.lookup("🔥")
        assert entry is not None
        assert entry.polarity.value == "positive"


# ════════════════════════════════════════════════════════════════════════════
# Scoring tests
# ════════════════════════════════════════════════════════════════════════════

class TestScoring:
    """Test sentiment scoring engine."""

    def _analyze(self, text: str):
        from instalens.analyzer import Analyzer
        analyzer = Analyzer()
        return analyzer.analyze_text(text)

    def test_positive_comment(self):
        result = self._analyze("This is absolutely amazing and beautiful!")
        assert result.label.value == "Positive"
        assert result.score > 0

    def test_negative_comment(self):
        result = self._analyze("This is terrible and horrible")
        assert result.label.value == "Negative"
        assert result.score < 0

    def test_neutral_comment(self):
        result = self._analyze("What time is it?")
        # Should be Neutral or Uncertain (no sentiment words)
        assert result.label.value in ("Neutral", "Uncertain")

    def test_negation(self):
        result_positive = self._analyze("This is good")
        result_negated = self._analyze("This is not good")
        # Negation should flip or reduce the score
        assert result_negated.score < result_positive.score

    def test_intensifier(self):
        result_base = self._analyze("This is good")
        result_intense = self._analyze("This is very good")
        # Intensifier should increase magnitude
        assert abs(result_intense.score) >= abs(result_base.score)

    def test_contrast_clause(self):
        result = self._analyze("It's bad, but actually amazing")
        # Post-contrast clause ("amazing") should carry more weight
        assert result.score > 0

    def test_mixed_sentiment(self):
        result = self._analyze("Love the concept, hate the execution")
        # Should detect both positive and negative evidence
        assert result.positive_score > 0
        assert result.negative_score > 0

    def test_empty_comment(self):
        result = self._analyze("")
        assert result.label.value == "Uncertain"

    def test_evidence_generated(self):
        result = self._analyze("This is amazing!")
        assert len(result.evidence) > 0

    def test_explanation_string(self):
        result = self._analyze("This is amazing!")
        assert len(result.explanation) > 0
        assert "amazing" in result.explanation.lower()

    def test_emoji_scoring(self):
        result = self._analyze("🔥🔥🔥")
        assert result.score > 0

    def test_every_comment_gets_result(self):
        """PRD: Every analyzed comment receives a result or explicit abstention."""
        texts = [
            "Amazing!", "Terrible!", "ok", "",
            "asdfghjkl", "love it but hate the color",
        ]
        for text in texts:
            result = self._analyze(text)
            assert result.label is not None
            assert result.label.value in (
                "Positive", "Negative", "Neutral", "Mixed", "Uncertain"
            )


# ════════════════════════════════════════════════════════════════════════════
# Ingestion tests
# ════════════════════════════════════════════════════════════════════════════

class TestCSVLoader:
    """Test CSV comment loading."""

    def test_load_sample(self):
        from instalens.ingestion.csv_loader import CSVCommentLoader
        sample = Path(__file__).parent.parent / "data" / "sample" / "sample_comments.csv"
        if sample.exists():
            loader = CSVCommentLoader(sample)
            comments = loader.load()
            assert len(comments) > 0
            assert all(c.text for c in comments)

    def test_load_with_k(self):
        from instalens.ingestion.csv_loader import CSVCommentLoader
        sample = Path(__file__).parent.parent / "data" / "sample" / "sample_comments.csv"
        if sample.exists():
            loader = CSVCommentLoader(sample)
            comments = loader.load(k=5)
            assert len(comments) <= 5

    def test_file_not_found(self):
        from instalens.ingestion.csv_loader import CSVCommentLoader
        with pytest.raises(FileNotFoundError):
            CSVCommentLoader("/nonexistent/file.csv")


class TestURLValidator:
    """Test Instagram URL validation."""

    def test_valid_post_url(self):
        from instalens.ingestion.validator import validate_url
        result = validate_url("https://www.instagram.com/p/ABC12345678/")
        assert result["valid"] is True
        assert result["shortcode"] == "ABC12345678"

    def test_valid_reel_url(self):
        from instalens.ingestion.validator import validate_url
        result = validate_url("https://instagram.com/reel/ABC12345678/")
        assert result["valid"] is True
        assert result["media_type"] == "reel"

    def test_invalid_url(self):
        from instalens.ingestion.validator import validate_url
        result = validate_url("https://google.com")
        assert result["valid"] is False

    def test_empty_url(self):
        from instalens.ingestion.validator import validate_url
        result = validate_url("")
        assert result["valid"] is False


# ════════════════════════════════════════════════════════════════════════════
# Integration tests
# ════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_csv_analysis_pipeline(self):
        """Test the full CSV → analysis → report pipeline."""
        from instalens.analyzer import Analyzer
        sample = Path(__file__).parent.parent / "data" / "sample" / "sample_comments.csv"
        if not sample.exists():
            pytest.skip("Sample data not found")

        analyzer = Analyzer()
        results, report = analyzer.analyze_csv(sample, k=20)

        assert len(results) == 20
        assert report.comments_retrieved == 20
        assert report.comments_analyzed + report.comments_abstained == 20
        assert sum(report.distribution.values()) == 20

    def test_report_formatting(self):
        """Test that report formatting doesn't crash."""
        from instalens.analyzer import Analyzer
        from instalens.reporting.reporter import ReportGenerator

        analyzer = Analyzer()
        results = [analyzer.analyze_text(t) for t in [
            "Amazing!", "Terrible!", "okay", "love it but hate it"
        ]]

        gen = ReportGenerator(results, comments_requested=4,
                              comments_retrieved=4, source="test")
        report = gen.generate()
        summary = gen.format_summary(report)
        individual = gen.format_individual()

        assert "InstaLens" in summary
        assert len(summary) > 100
        assert len(individual) > 50

    def test_evaluation_pipeline(self):
        """Test evaluation against gold labels."""
        from instalens.analyzer import Analyzer
        from instalens.evaluation.evaluator import Evaluator

        texts = ["Amazing!", "Terrible!", "okay", "love but hate"]
        gold = ["Positive", "Negative", "Neutral", "Mixed"]

        analyzer = Analyzer()
        results = [analyzer.analyze_text(t) for t in texts]

        evaluator = Evaluator(results, gold, baseline_name="Test")
        report = evaluator.evaluate()

        assert report.total == 4
        assert 0.0 <= report.macro_f1 <= 1.0
        assert 0.0 <= report.abstention_rate <= 1.0
        formatted = evaluator.format_report(report)
        assert "Macro-F1" in formatted

    def test_aggregate_consistency(self):
        """PRD: Aggregate counts are internally consistent."""
        from instalens.analyzer import Analyzer
        sample = Path(__file__).parent.parent / "data" / "sample" / "sample_comments.csv"
        if not sample.exists():
            pytest.skip("Sample data not found")

        analyzer = Analyzer()
        results, report = analyzer.analyze_csv(sample)

        total_from_dist = sum(report.distribution.values())
        assert total_from_dist == len(results)
        assert report.comments_analyzed + report.comments_abstained == len(results)


# ════════════════════════════════════════════════════════════════════════════
# Instagram Ingestion & Live Fetching Tests
# ════════════════════════════════════════════════════════════════════════════

class TestInstagramIngestion:
    """Test Instagram URL validation and live comment fetcher."""

    def test_reel_url_validation(self):
        from instalens.ingestion.validator import validate_url

        # Standard reel URL
        v1 = validate_url("https://www.instagram.com/reel/DFwQQfbSH4Z/")
        assert v1["valid"] is True
        assert v1["shortcode"] == "DFwQQfbSH4Z"
        assert v1["media_type"] == "reel"

        # Reel URL with query parameters
        v2 = validate_url("https://www.instagram.com/reel/DFwQQfbSH4Z/?igsh=MWQ1ZGUxMzBkMA==")
        assert v2["valid"] is True
        assert v2["shortcode"] == "DFwQQfbSH4Z"

        # Plural "reels" URL format
        v3 = validate_url("https://instagram.com/reels/DFwQQfbSH4Z")
        assert v3["valid"] is True
        assert v3["shortcode"] == "DFwQQfbSH4Z"
        assert v3["media_type"] == "reel"

        # Bare shortcode
        v4 = validate_url("DFwQQfbSH4Z")
        assert v4["valid"] is True
        assert v4["shortcode"] == "DFwQQfbSH4Z"

        # Invalid URL
        v5 = validate_url("https://example.com/not-instagram")
        assert v5["valid"] is False

    def test_instagram_fetcher_demo_mode(self):
        from instalens.ingestion.instagram import InstagramFetcher
        from instalens.analyzer import Analyzer

        fetcher = InstagramFetcher()
        result = fetcher.fetch("https://www.instagram.com/reel/DFwQQfbSH4Z/", k=15, demo=True)

        assert result.error is None
        assert result.post_shortcode == "DFwQQfbSH4Z"
        assert len(result.comments) == 15
        assert result.comments_retrieved == 15
        assert result.comments[0].text is not None

        # Pass retrieved comments through the analyzer
        analyzer = Analyzer()
        results, report = analyzer.analyze_comments(result.comments)

        assert len(results) == 15
        assert report.overall_sentiment in ["Positive", "Negative", "Neutral", "Mixed", "Uncertain"]
        assert len(report.sentiment_verdict) > 0
        assert report.comments_analyzed + report.comments_abstained == 15

    def test_overall_sentiment_calculation(self):
        from instalens.reporting.reporter import ReportGenerator
        from instalens.scoring.engine import SentimentResult, SentimentLabel

        def make_res(label, score):
            return SentimentResult(
                original_text="", normalized_text="", label=label,
                score=score, positive_score=max(0, score), negative_score=abs(min(0, score)),
                confidence=1.0, evidence=[], uncertainty_flags=[],
            )

        # Strongly positive batch
        pos_results = [make_res(SentimentLabel.POSITIVE, 0.8) for _ in range(8)] + \
                      [make_res(SentimentLabel.NEGATIVE, -0.5) for _ in range(2)]
        rep1 = ReportGenerator(pos_results).generate()
        assert rep1.overall_sentiment == "Positive"
        assert rep1.net_sentiment > 0

        # Strongly negative batch
        neg_results = [make_res(SentimentLabel.NEGATIVE, -0.8) for _ in range(7)] + \
                      [make_res(SentimentLabel.POSITIVE, 0.5) for _ in range(1)]
        rep2 = ReportGenerator(neg_results).generate()
        assert rep2.overall_sentiment == "Negative"
        assert rep2.net_sentiment < 0

