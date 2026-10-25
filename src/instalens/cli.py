"""
InstaLens CLI — Command-line interface for sentiment analysis.

Commands:
  analyze   — Analyze comments from a CSV file or single text
  evaluate  — Evaluate predictions against gold annotations
  expand    — Explore GloVe embeddings for vocabulary expansion
  info      — Show lexicon statistics and configuration
"""

import sys
import os
from pathlib import Path

import click

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instalens.analyzer import Analyzer
from instalens.scoring.engine import ScorerConfig, SentimentLabel
from instalens.lexicon.lexicon import build_default_lexicon
from instalens.reporting.reporter import ReportGenerator
from instalens.evaluation.evaluator import Evaluator
from instalens.ingestion.csv_loader import CSVCommentLoader
from instalens.ingestion.validator import validate_url
from instalens.ingestion.instagram import InstagramFetcher
from instalens.embeddings.glove import GloVeEmbeddings


@click.group()
@click.version_option(version="1.0.0", prog_name="InstaLens")
def cli():
    """
    InstaLens — Instagram Comment Sentiment Analyzer

    A lexicon-driven NLP pipeline for explainable sentiment analysis
    of social media comments.
    """
    pass


# ════════════════════════════════════════════════════════════════════════════
# analyze command
# ════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("source")
@click.option("-k", "--top-k", type=int, default=None,
              help="Max number of comments to analyze (default: all)")
@click.option("-o", "--output", type=click.Path(), default=None,
              help="Export results to CSV at this path")
@click.option("--show-all", is_flag=True,
              help="Show all individual results (not just top 20)")
@click.option("--show-evidence", is_flag=True,
              help="Show detailed scoring evidence for each comment")
@click.option("--positive-threshold", type=float, default=0.15,
              help="Score threshold for Positive classification")
@click.option("--negative-threshold", type=float, default=-0.15,
              help="Score threshold for Negative classification")
@click.option("--quiet", "-q", is_flag=True,
              help="Only show the summary, not individual results")
@click.option("--session-id", "-s", default=None, envvar="INSTAGRAM_SESSION_ID",
              help="Instagram sessionid cookie from browser (bypasses login/2FA)")
@click.option("--login", nargs=2, type=str, default=None, metavar="USER PASS",
              help="Instagram username and password for authenticated access")
@click.option("--demo", is_flag=True,
              help="Use realistic simulated reel comments (offline demo mode)")
@click.option("--apify-token", default=None, envvar="APIFY_API_TOKEN",
              help="Apify API token for reliable cloud scraping (free at https://apify.com)")
def analyze(source, top_k, output, show_all, show_evidence,
            positive_threshold, negative_threshold, quiet, session_id, login, demo, apify_token):
    """
    Analyze sentiment of comments.

    SOURCE can be:
      - An Instagram post/reel URL (fetches comments live)
      - A path to a CSV file containing comments
      - A quoted text string to analyze a single comment
    """
    config = ScorerConfig(
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )
    analyzer = Analyzer(config=config)

    # Determine source type
    source_path = Path(source)

    if source_path.exists() and source_path.suffix.lower() == ".csv":
        # CSV file analysis
        click.echo(f"\n  📂 Loading comments from: {source_path.name}")
        try:
            results, report = analyzer.analyze_csv(source_path, k=top_k)
        except Exception as e:
            click.echo(f"\n  ❌ Error: {e}", err=True)
            sys.exit(1)

        click.echo(f"  ✓ Loaded {len(results)} comments")

        # Print summary
        gen = ReportGenerator(
            results, comments_requested=top_k or len(results),
            comments_retrieved=len(results), source=str(source_path),
        )
        click.echo(gen.format_summary(report))

        # Print individual results
        if not quiet:
            max_display = len(results) if show_all else 20
            click.echo(gen.format_individual(max_display=max_display,
                                             show_all=show_all))

        # Show detailed evidence
        if show_evidence:
            click.echo("\n  ── Detailed Evidence ───────────────────\n")
            for i, r in enumerate(results):
                click.echo(f"  Comment {i+1}: \"{r.original_text[:60]}...\"")
                click.echo(r.explanation)
                click.echo()

        # Export
        if output:
            out_path = gen.export_csv(output)
            click.echo(f"\n  📄 Results exported to: {out_path}")

    elif source_path.exists():
        click.echo(f"\n  ❌ Unsupported file format: {source_path.suffix}", err=True)
        click.echo("     Expected a .csv file.", err=True)
        sys.exit(1)

    else:
        # Check if it's an Instagram URL
        url_check = validate_url(source)
        if url_check["valid"]:
            # ── Instagram live fetch ────────────────────────────────
            k = top_k or 100
            click.echo(f"\n  📸 Instagram URL detected")
            click.echo(f"  Shortcode: {url_check['shortcode']}")
            click.echo(f"  Fetching top {k} comments...\n")

            try:
                if apify_token or os.environ.get("APIFY_API_TOKEN"):
                    from instalens.ingestion.apify_fetcher import ApifyInstagramFetcher
                    fetcher = ApifyInstagramFetcher(api_token=apify_token)
                    fetch_result = fetcher.fetch(source, k=k)
                else:
                    login_user = login[0] if login else None
                    login_pass = login[1] if login else None
                    fetcher = InstagramFetcher(
                        session_id=session_id,
                        login_user=login_user,
                        login_pass=login_pass,
                    )
                    fetch_result = fetcher.fetch(source, k=k, demo=demo)
            except ValueError as e:
                click.echo(f"  ❌ Authentication error: {e}", err=True)
                sys.exit(1)
            except Exception as e:
                click.echo(f"  ❌ Fetch error: {e}", err=True)
                sys.exit(1)

            if fetch_result.error:
                click.echo(f"  ❌ {fetch_result.error}", err=True)
                sys.exit(1)

            # Show post metadata
            click.echo(f"  ── Post Info ───────────────────────────")
            if fetch_result.post_caption:
                caption_display = fetch_result.post_caption[:100]
                if len(fetch_result.post_caption) > 100:
                    caption_display += "..."
                click.echo(f"  Caption:  {caption_display}")
            click.echo(f"  Likes:    {fetch_result.post_likes:,}")
            click.echo(f"  Comments: {fetch_result.post_comments_count:,} total")
            click.echo(f"  Fetched:  {fetch_result.comments_retrieved}")

            if fetch_result.warnings:
                for w in fetch_result.warnings:
                    click.echo(f"  ⚠ {w}")

            if not fetch_result.comments:
                click.echo("\n  No comments to analyze.")
                sys.exit(0)

            # Run analysis
            click.echo(f"\n  🔍 Analyzing {len(fetch_result.comments)} comments...")
            results, report = analyzer.analyze_comments(
                fetch_result.comments,
                comments_requested=k,
            )

            # Update report source
            gen = ReportGenerator(
                results,
                comments_requested=k,
                comments_retrieved=fetch_result.comments_retrieved,
                source=f"instagram.com/p/{fetch_result.post_shortcode}",
            )
            report = gen.generate()
            click.echo(gen.format_summary(report))

            if not quiet:
                max_display = len(results) if show_all else 20
                click.echo(gen.format_individual(
                    max_display=max_display, show_all=show_all
                ))

            if show_evidence:
                click.echo("\n  ── Detailed Evidence ───────────────────\n")
                for i, r in enumerate(results):
                    text_preview = r.original_text[:60]
                    click.echo(f'  Comment {i+1}: "{text_preview}..."')
                    click.echo(r.explanation)
                    click.echo()

            if output:
                out_path = gen.export_csv(output)
                click.echo(f"\n  📄 Results exported to: {out_path}")

            sys.exit(0)

        # Guard: Check if source looks like a URL that failed validation
        cleaned_source = source.strip().replace("\\", "")
        if cleaned_source.startswith(("http://", "https://")) or "instagram.com" in cleaned_source.lower():
            click.echo(f"\n  ❌ Invalid Instagram URL: {source}", err=True)
            if url_check.get("error"):
                click.echo(f"     {url_check['error']}", err=True)
            sys.exit(1)

        # Single text analysis
        click.echo(f'\n  📝 Analyzing single comment...\n')
        result = analyzer.analyze_text(source)

        label_icon = {
            SentimentLabel.POSITIVE: "🟢",
            SentimentLabel.NEGATIVE: "🔴",
            SentimentLabel.NEUTRAL: "⚪",
            SentimentLabel.MIXED: "🟡",
            SentimentLabel.UNCERTAIN: "❓",
        }.get(result.label, "?")

        click.echo(f'  Text: "{source}"')
        click.echo(f"  {label_icon} Label: {result.label.value}")
        click.echo(f"  Score: {result.score:+.3f}")
        click.echo(f"  Confidence: {result.confidence:.2f}")
        click.echo(f"\n  Evidence:")
        click.echo(result.explanation)
        click.echo()


# ════════════════════════════════════════════════════════════════════════════
# evaluate command
# ════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("predictions_csv", type=click.Path(exists=True))
@click.option("--gold-column", default="gold_label",
              help="Column name for gold/annotated labels")
@click.option("--text-column", default="text",
              help="Column name for comment text")
@click.option("--baseline-name", default="Baseline",
              help="Name for this baseline configuration")
def evaluate(predictions_csv, gold_column, text_column, baseline_name):
    """
    Evaluate sentiment predictions against gold annotations.

    PREDICTIONS_CSV must contain both comment text and gold labels.
    """
    import csv as csv_module

    filepath = Path(predictions_csv)
    click.echo(f"\n  📊 Evaluating against: {filepath.name}")

    # Load gold data
    with open(filepath, encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        headers = reader.fieldnames or []

        if gold_column not in headers:
            click.echo(f"\n  ❌ Gold label column '{gold_column}' not found.", err=True)
            click.echo(f"  Available columns: {', '.join(headers)}", err=True)
            sys.exit(1)

        text_col = None
        for alias in ["text", "comment", "comment_text", "body", "content"]:
            if alias in headers:
                text_col = alias
                break
        if text_col is None:
            text_col = text_column
            if text_col not in headers:
                click.echo(f"\n  ❌ Text column not found.", err=True)
                sys.exit(1)

        rows = list(reader)

    # Run analysis
    analyzer = Analyzer()
    gold_labels = []
    results = []

    for row in rows:
        text = row.get(text_col, "").strip()
        gold = row.get(gold_column, "").strip()
        if not text or not gold:
            continue
        gold_labels.append(gold)
        results.append(analyzer.analyze_text(text))

    click.echo(f"  ✓ Analyzed {len(results)} annotated comments")

    # Evaluate
    evaluator = Evaluator(results, gold_labels, baseline_name=baseline_name)
    report = evaluator.evaluate()
    click.echo(evaluator.format_report(report))


# ════════════════════════════════════════════════════════════════════════════
# expand command (GloVe vocabulary expansion)
# ════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("glove_path", type=click.Path(exists=True))
@click.option("--word", "-w", multiple=True,
              help="Word(s) to find neighbors for")
@click.option("--top-k", "-k", type=int, default=20,
              help="Number of neighbors to show")
@click.option("--similarity", "-s", type=float, default=0.5,
              help="Minimum similarity threshold")
@click.option("--expand", is_flag=True,
              help="Run full vocabulary expansion with default seeds")
def expand(glove_path, word, top_k, similarity, expand_vocab):
    """
    Explore GloVe embeddings and expand vocabulary.

    GLOVE_PATH is the path to a GloVe vectors file (e.g. glove.twitter.27B.100d.txt).
    """
    click.echo(f"\n  📦 Loading GloVe embeddings from: {glove_path}")
    click.echo("  (This may take a minute...)\n")

    glove = GloVeEmbeddings()
    count = glove.load(glove_path)
    click.echo(f"  ✓ Loaded {count:,} vectors ({glove.dimension}d)\n")

    if word:
        for w in word:
            if not glove.has_word(w):
                click.echo(f"  ❌ '{w}' not in vocabulary\n")
                continue

            neighbors = glove.nearest_neighbors(w, top_k=top_k)
            click.echo(f"  Nearest neighbors for '{w}':")
            for neighbor, sim in neighbors:
                bar_len = int(sim * 30)
                bar = "█" * bar_len
                click.echo(f"    {neighbor:<20} {bar} {sim:.4f}")
            click.echo()

    if expand_vocab:
        click.echo("  ── Vocabulary Expansion ───────────────────\n")

        pos_seeds = ["love", "amazing", "beautiful", "perfect", "great",
                     "awesome", "excellent", "wonderful", "fantastic", "brilliant"]
        neg_seeds = ["hate", "terrible", "horrible", "awful", "worst",
                     "disgusting", "ugly", "pathetic", "trash", "garbage"]

        candidates = glove.expand_vocabulary(
            pos_seeds, neg_seeds,
            top_k=top_k, similarity_threshold=similarity,
        )

        click.echo(f"  Positive candidates ({len(candidates['positive_candidates'])}):")
        for w, sim, seed in candidates["positive_candidates"][:20]:
            click.echo(f"    {w:<20} sim={sim:.4f}  (from '{seed}')")

        click.echo(f"\n  Negative candidates ({len(candidates['negative_candidates'])}):")
        for w, sim, seed in candidates["negative_candidates"][:20]:
            click.echo(f"    {w:<20} sim={sim:.4f}  (from '{seed}')")

        if candidates["ambiguous"]:
            click.echo(f"\n  ⚠ Ambiguous (appeared in both): {', '.join(candidates['ambiguous'][:15])}")

        click.echo(
            "\n  ℹ These are CANDIDATES for manual review. "
            "They have NOT been added to the lexicon."
        )


# ════════════════════════════════════════════════════════════════════════════
# info command
# ════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--lexicon-path", type=click.Path(), default=None,
              help="Path to a custom lexicon JSON file")
def info(lexicon_path):
    """Show lexicon statistics and system configuration."""
    if lexicon_path:
        from instalens.lexicon.lexicon import SentimentLexicon
        lex = SentimentLexicon()
        count = lex.load_from_json(lexicon_path)
        click.echo(f"\n  Custom lexicon: {lexicon_path}")
        click.echo(f"  Entries loaded: {count}")
    else:
        lex = build_default_lexicon()
        click.echo(f"\n  Using built-in lexicon")

    stats = lex.stats
    click.echo(f"\n  ── Lexicon Statistics ──────────────────")
    click.echo(f"  Total entries:    {stats['total']}")
    click.echo(f"  Positive:         {stats['positive']}")
    click.echo(f"  Negative:         {stats['negative']}")
    click.echo(f"  Neutral:          {stats['neutral']}")
    click.echo(f"  Phrases:          {stats['phrases']}")

    click.echo(f"\n  ── Scorer Configuration ───────────────")
    config = ScorerConfig()
    click.echo(f"  Positive threshold:    {config.positive_threshold}")
    click.echo(f"  Negative threshold:    {config.negative_threshold}")
    click.echo(f"  Mixed threshold:       {config.mixed_threshold}")
    click.echo(f"  Negation multiplier:   {config.negation_multiplier}")
    click.echo(f"  Intensifier mult:      {config.intensifier_multiplier}")
    click.echo(f"  Downtoner mult:        {config.downtoner_multiplier}")
    click.echo(f"  Contrast weight:       {config.contrast_weight}")
    click.echo(f"  Emoji weight:          {config.emoji_weight}")
    click.echo()


if __name__ == "__main__":
    cli()
