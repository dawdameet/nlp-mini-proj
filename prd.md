ignore the bullshitty timeline and complete the product. Make it CLI based for now (but keep the backend portable to nextjs NOT NOW BUT LATER )

InstaLens
Product Requirements Document (PRD)
Version: 1.0
Status: Proposed
Project Type: College NLP Project
Primary Domain: Natural Language Processing
Target: Functional, demonstrable MVP
1. Product Overview
InstaLens is a web application that analyzes the sentiment expressed in comments on an Instagram post.
A user submits a supported Instagram post URL and specifies the number of comments to analyze. The application retrieves accessible comments, processes them using a lexicon-driven NLP pipeline augmented with pretrained Twitter embeddings, and produces an explainable sentiment report.
The application focuses on linguistic processing, domain vocabulary adaptation, and interpretable sentiment analysis.
2. Problem Statement
Manually interpreting hundreds of comments is tedious and subjective.
Generic sentiment systems can struggle with informal social-media language, slang, abbreviations, emojis, negation, and mixed sentiment.
InstaLens aims to provide a lightweight tool for understanding the overall sentiment expressed in an accessible sample of comments.
3. Objectives
Accept supported Instagram post URLs.
Analyze a configurable top-k sample of comments.
Classify comments as Positive, Negative, Neutral, Mixed, or Uncertain.
Produce an aggregate sentiment distribution.
Explain classifications using lexical evidence and fired linguistic rules.
Investigate whether embedding-assisted lexicon expansion improves sentiment analysis.
Success criteria
End-to-end analysis works on supported inputs.
The sentiment pipeline is independently testable.
Every analyzed comment receives a result or an explicit abstention.
Aggregate counts are internally consistent.
A reproducible evaluation is performed against human annotations.
The project can be demonstrated using a saved sample dataset even if live Instagram retrieval is unavailable.
4. Target Users
Primary: College students and project evaluators.
Secondary: Creators or users curious about reactions to a supported post.
This project is not intended for enterprise brand monitoring, automated moderation, or psychological profiling.
5. MVP Scope
P0: Must have
Input
Instagram post URL.
Configurable k, default 100 comments.
Validation of URL and supported media types.
Comment ingestion
Extract comments through an authorized, supported source.
Store comment text, source identifier, and available metadata where permitted.
Clearly indicate how comments were selected.
Handle inaccessible posts, private content, rate limits, and empty results.
NLP processing
Text normalization.
Tokenization.
Contraction handling.
Lexicon-based polarity scoring.
Negation handling.
Intensifier and downtoner handling.
Basic punctuation and emoji processing.
Basic contrast clause handling.
Sentiment classification.
Explainable score breakdown.
Reporting
Number of comments requested, retrieved, and analyzed.
Sentiment distribution.
Individual comment classifications.
Confidence/uncertainty indicators.
Top positive and negative lexical signals.
Downloadable CSV report.
Evaluation
Human-annotated test dataset.
Baseline comparison.
Metrics and error analysis.
P1: Stretch goals
Aspect-based sentiment using manually defined aspect lexicons.
Word and phrase frequency explorer.
Embedding neighborhood explorer.
Hinglish vocabulary support.
Language detection.
Temporal sentiment comparisons, if timestamps are available and permitted.
Explicitly out of scope
Deep learning sentiment classifiers.
LLM-based sentiment inference.
User profiling.
Sentiment prediction of individual commenters.
Automatic moderation or deletion.
Guaranteed analysis of arbitrary public Instagram URLs.
Real-time continuous monitoring.
Production-scale distributed processing.
6. NLP Design
6.1 Pretrained embeddings
Use Stanford GloVe Twitter embeddings as the initial semantic resource.
Recommended starting model: glove.twitter.27B.100d
The Twitter release is based on 2 billion tweets and 27 billion tokens, with approximately 1.2 million vocabulary entries. It is available in several dimensions.
Reference: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
The embeddings are used for vocabulary exploration and lexicon expansion, NOT direct sentiment classification.
6.2 Sentiment lexicon
Construct a sentiment lexicon using:
SentiWordNet as a source of lexical sentiment evidence.
Manually curated social-media slang.
Emoji polarity entries.
Validated phrase-level entries.
Embedding-assisted candidate discovery.
Candidate entries must be reviewed before being added to the production lexicon.
Each lexicon entry should support:
Token or phrase
Polarity
Strength
Source
Validation status
Optional usage notes
6.3 Text preprocessing
Processing pipeline:
Preserve original comment text.
Normalize Unicode and whitespace.
Handle URLs and mentions.
Tokenize while preserving meaningful emojis, hashtags, and contractions.
Detect punctuation and clause boundaries.
Apply phrase matching.
Identify sentiment-bearing terms.
Avoid destructive preprocessing such as removing all punctuation, emojis, or negation words.
6.4 Sentiment scoring
For a comment c:
S(c) = Σ polarity contributions
A contribution may be modified by:
Negation scope
Intensification
Downtoning
Phrase-level matches
Clause weighting
Emoji evidence
Use deterministic scoring rules with configurable thresholds.
Classification:
Positive: score exceeds positive threshold.
Negative: score falls below negative threshold.
Neutral: sufficient evidence indicates neither positive nor negative sentiment.
Mixed: meaningful positive and negative evidence coexist.
Uncertain: insufficient evidence or unsupported language.
Mixed sentiment takes precedence when substantial evidence exists on both sides.
Uncertain is distinct from Neutral. A comment such as an unsupported-language sentence should not automatically be treated as neutral.
6.5 Embedding-assisted vocabulary expansion
Procedure:
Define positive and negative seed terms.
Retrieve nearest neighbors using cosine similarity.
Collect candidate terms.
Rank candidates using similarity-based heuristics.
Manually validate candidate polarity.
Add validated terms to the lexicon.
Compare the expanded lexicon against the baseline.
Embedding similarity is not sentiment probability. Antonyms, contextual ambiguity, and unrelated terms may appear among candidates.
6.6 Explainability
For each result, show:
Detected sentiment-bearing terms.
Phrase matches.
Negation or intensity adjustments.
Final heuristic score.
Classification threshold.
Any uncertainty flags.
Example:
Comment: "not bad, actually amazing"
Explanation:
"not bad": positive-ish phrase evidence.
"amazing": strong positive evidence.
Overall: Positive.
This is an explanation of the system's rules, not proof of the author's true emotional state.
7. Data Strategy
Development data
Use:
Existing sentiment lexicons.
Publicly available datasets with suitable usage terms.
Synthetic edge cases for unit tests.
A small manually curated Instagram-style vocabulary.
Evaluation dataset
Initial target: 600 manually annotated comments.
Suggested annotation labels:
Positive
Negative
Neutral
Mixed
Uncertain / insufficient context
Annotate using written guidelines. Double-annotate a subset and resolve disagreements.
Split data before tuning the lexicon or thresholds. Keep the final test set untouched during development.
Where feasible, stratify by sentiment class and include varied comment lengths, emojis, slang, and mixed sentiment.
If real comments are used, minimize personal data, redact usernames, and follow source terms and applicable privacy requirements.
8. Functional Requirements
FR-01: The application must validate supported post URLs.
FR-02: The application must retrieve comments through an authorized integration or accept an approved dataset upload.
FR-03: The user must be able to specify k.
FR-04: The application must report actual retrieval counts, not assume k comments were retrieved.
FR-05: Each comment must receive a sentiment result or abstention.
FR-06: Each result must include an explanation of the scoring evidence.
FR-07: The application must generate aggregate statistics.
FR-08: The user must be able to export the report.
FR-09: Failures must produce actionable messages.
FR-10: The system must distinguish retrieval errors from NLP processing errors.
9. Non-Functional Requirements
Local development should be possible on an ordinary student laptop.
The NLP engine must run without GPU acceleration.
Analysis of a saved dataset should be reproducible with a fixed configuration.
Lexicons and thresholds should be versioned.
No passwords, session cookies, or unnecessary account credentials should be stored.
Comments should not be retained longer than necessary.
The application should support basic concurrent request handling.
The NLP pipeline should be modular and independently testable.
10. Technical Architecture
Frontend: Next.js
Backend: Python + FastAPI
NLP: NLTK, custom rules, SentiWordNet, GloVe vectors, NumPy
Storage: SQLite
Visualization: Recharts or Chart.js
Testing: pytest
Logical components
URL validation
Comment source adapter
Comment normalization
Tokenizer and phrase matcher
Sentiment lexicon
Scoring engine
Aggregator
Report generator
The comment source adapter and NLP engine must be independent.
This enables development using CSV data without waiting for Instagram API integration.
11. Evaluation Plan
Baselines
Baseline A: Generic lexicon + basic token scoring.
Baseline B: Generic lexicon + negation, intensifiers, and phrase rules.
Baseline C: Baseline B + social-media vocabulary.
Baseline D: Baseline C + embedding-assisted lexicon expansion.
Metrics
Macro-F1
Per-class precision and recall
Confusion matrix
Abstention rate
Lexical coverage
Human annotation agreement
Error categories
Report both overall performance and performance on challenging subsets.
Do not claim improvement unless observed on held-out evaluation data.
Initial project targets
All functional requirements pass their tests.
Every output is reproducible under the same lexicon and configuration.
The system produces a complete report for a supported sample dataset.
The evaluation includes macro-F1, class-wise results, and documented failure cases.
No arbitrary accuracy guarantee is established before a baseline is measured.
12. Risks and Mitigations
Instagram access restrictions
Mitigation: Develop against CSV fixtures and treat live retrieval as a separate integration milestone.
Slang ambiguity
Mitigation: Curated phrase lexicon and uncertainty handling.
Embedding noise
Mitigation: Human review of candidates and separate evaluation of expansion.
Sarcasm
Mitigation: Document as a known limitation; do not claim reliable detection.
Class imbalance
Mitigation: Macro-F1, class-aware evaluation, and annotation guidelines.
Privacy and data handling
Mitigation: Minimal collection, redaction, retention limits, and source-policy compliance.
13. Milestones
Week 1: Requirements, architecture, dataset selection.
Week 2: NLP preprocessing and lexicon baseline.
Week 3: Negation, intensifiers, phrases, and scoring engine.
Week 4: GloVe integration and vocabulary exploration.
Week 5: Dataset annotation and evaluation.
Week 6: Backend, reporting, and frontend integration.
Week 7: Testing, error analysis, and improvements.
Week 8: Final demo, documentation, and presentation.
14. Deliverables
Working web application.
Independent NLP library.
Versioned sentiment lexicon.
Embedding-assisted vocabulary expansion utility.
Evaluation dataset and annotation guidelines, subject to data permissions.
Evaluation report.
README and setup instructions.
Demonstration video or live demo.
Project presentation.
15. Definition of Done
The project is complete when a user can:
Supply a supported post URL or sample dataset.
Select the comment sample size.
Run analysis.
View aggregate sentiment and individual explanations.
Export results.
Understand what the system could not reliably classify.
The final presentation should demonstrate the product, explain the NLP methodology, compare baselines, and discuss limitations honestly.