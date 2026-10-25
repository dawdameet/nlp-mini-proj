# InstaLens — Instagram Comment Sentiment Analyzer

A lexicon-driven NLP pipeline for explainable sentiment analysis of social media comments.

## Quick Start

```bash
# Activate the virtual environment
source ~/venv/bin/activate

# Install in development mode
cd ~/college/nlp
pip install -e .

# Download required NLTK data
python -c "import nltk; nltk.download('punkt_tab')"

# Analyze an Instagram reel (Live via Apify cloud scraper)
instalens analyze "https://www.instagram.com/reel/DXcJ6qNATsP/" -k 50 --apify-token "YOUR_APIFY_TOKEN"
# Or export APIFY_API_TOKEN in your shell:
# export APIFY_API_TOKEN="YOUR_APIFY_TOKEN"
# instalens analyze "https://www.instagram.com/reel/DXcJ6qNATsP/" -k 50

# Analyze an Instagram reel (Demo / Offline presentation mode)
instalens analyze "https://www.instagram.com/reel/DXcJ6qNATsP/" -k 20 --demo

# Analyze a single comment
instalens analyze "This is absolutely amazing! Love it 🔥"

# Analyze comments from CSV
instalens analyze data/sample/sample_comments.csv

# Analyze with export
instalens analyze data/sample/sample_comments.csv -o results.csv --show-all

# Evaluate against gold annotations
instalens evaluate data/sample/sample_comments.csv --gold-column gold_label

# Show lexicon info
instalens info
```

## Project Structure

```
src/instalens/
├── __init__.py              # Package root
├── analyzer.py              # Pipeline orchestrator (portable core)
├── cli.py                   # Click-based CLI
├── preprocessing/
│   ├── normalizer.py        # Unicode, URL, mention, emoji normalization
│   ├── contractions.py      # Contraction expansion (can't → can not)
│   ├── tokenizer.py         # TweetTokenizer + clause segmentation
│   └── pipeline.py          # End-to-end preprocessing chain
├── lexicon/
│   └── lexicon.py           # Sentiment lexicon (multi-source, phrase matching)
├── scoring/
│   └── engine.py            # Rule-based scorer (negation, intensifiers, contrast)
├── reporting/
│   └── reporter.py          # Aggregate stats, CSV export, formatted output
├── evaluation/
│   └── evaluator.py         # Metrics: macro-F1, confusion matrix, error analysis
└── embeddings/
    └── glove.py             # GloVe loader + vocabulary expansion utility

data/
├── sample/                  # Sample datasets for development
├── lexicons/                # JSON lexicon files (base, emoji, slang, phrase)
└── glove/                   # GloVe embedding files (download separately)

tests/                       # pytest test suite
```

## NLP Pipeline

1. **Preprocessing**: Unicode normalize → URL/mention handling → contraction expansion → tokenize with clause boundaries
2. **Phrase Matching**: Greedy longest-match against phrase lexicon ("not bad", "waste of time")
3. **Token Scoring**: Lexicon lookup → negation scope (3 tokens) → intensifier/downtoner → emoji weighting
4. **Clause Weighting**: Contrast conjunctions ("but", "however") boost the subsequent clause
5. **Classification**: Positive / Negative / Neutral / Mixed / Uncertain with configurable thresholds
6. **Explainability**: Every result includes full evidence chain and scoring breakdown

## GloVe Embeddings (Optional)

Download Stanford GloVe Twitter embeddings for vocabulary expansion:
```bash
cd data/glove
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```

Then use the expand command:
```bash
instalens expand data/glove/glove.twitter.27B.100d.txt -w love -w hate
instalens expand data/glove/glove.twitter.27B.100d.txt --expand
```

## Architecture

The backend is designed to be **portable** — the `Analyzer` class can be used directly from Python:

```python
from instalens.analyzer import Analyzer

analyzer = Analyzer()
result = analyzer.analyze_text("This is amazing!")
print(result.label, result.score, result.explanation)
```

This makes future integration with FastAPI + Next.js straightforward.
