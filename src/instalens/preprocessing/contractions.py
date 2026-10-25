"""
Contraction expansion for informal / social-media English.

Includes standard English contractions plus common internet shorthand.
Applied *before* tokenization so that "can't" → "can not" feeds the
negation detector cleanly.
"""

import re

# ── Contraction map ─────────────────────────────────────────────────────────

CONTRACTIONS: dict[str, str] = {
    # Standard English
    "ain't": "is not",
    "aren't": "are not",
    "can't": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",

    # Internet / social media shorthand
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "imma": "i am going to",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "tryna": "trying to",
}

# Build a regex that matches any contraction (longest-first to avoid partial matches)
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """
    Replace contractions with their expanded forms.

    Preserves the original casing style (all-lower stays lower;
    title-case input gets title-case output).
    """
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        key = word.lower()
        expanded = CONTRACTIONS.get(key, word)
        # Preserve title case
        if word[0].isupper():
            expanded = expanded.capitalize()
        return expanded

    return _CONTRACTION_RE.sub(_replace, text)
