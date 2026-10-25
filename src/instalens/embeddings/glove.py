"""
GloVe embedding loader and vocabulary expansion utility.

Implements PRD §6.1 and §6.5:
- Load Stanford GloVe Twitter embeddings (glove.twitter.27B.100d)
- Cosine similarity search for nearest neighbors
- Vocabulary expansion pipeline: seed → neighbors → candidates → review
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


class GloVeEmbeddings:
    """
    Load and query GloVe embeddings.

    Optimized for the Twitter 27B model but works with any GloVe format.
    """

    def __init__(self) -> None:
        self._word2idx: dict[str, int] = {}
        self._idx2word: list[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._dim: int = 0
        self._loaded: bool = False

    def load(self, filepath: str | Path) -> int:
        """
        Load GloVe vectors from a text file.

        Expected format: one line per word, space-separated:
            word 0.123 -0.456 ...

        Parameters
        ----------
        filepath : str or Path
            Path to the GloVe .txt file.

        Returns
        -------
        int
            Number of vectors loaded.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GloVe file not found: {filepath}")

        words = []
        vectors = []

        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                parts = line.rstrip().split(" ")
                if len(parts) < 3:
                    continue
                word = parts[0]
                try:
                    vec = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                words.append(word)
                vectors.append(vec)

        if not vectors:
            raise ValueError(f"No valid vectors found in {filepath}")

        self._dim = len(vectors[0])
        self._idx2word = words
        self._word2idx = {w: i for i, w in enumerate(words)}
        self._vectors = np.array(vectors, dtype=np.float32)

        # Normalize vectors for faster cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = self._vectors / norms

        self._loaded = True
        return len(words)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get the normalized embedding vector for a word."""
        idx = self._word2idx.get(word.lower())
        if idx is None:
            return None
        return self._vectors[idx]

    def has_word(self, word: str) -> bool:
        return word.lower() in self._word2idx

    def cosine_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Compute cosine similarity between two words."""
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        if v1 is None or v2 is None:
            return None
        # Vectors are pre-normalized, so dot product = cosine similarity
        return float(np.dot(v1, v2))

    def nearest_neighbors(
        self,
        word: str,
        top_k: int = 20,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Find the top-k nearest neighbors by cosine similarity.

        Parameters
        ----------
        word : str
            Query word.
        top_k : int
            Number of neighbors to return.
        exclude : set of str, optional
            Words to exclude from results.

        Returns
        -------
        list of (word, similarity) tuples, sorted by similarity descending.
        """
        self._check_loaded()
        vec = self.get_vector(word)
        if vec is None:
            return []

        # Dot product with all vectors (pre-normalized → cosine sim)
        similarities = self._vectors @ vec
        top_indices = np.argsort(similarities)[::-1]

        exclude = exclude or set()
        exclude.add(word.lower())

        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            w = self._idx2word[idx]
            if w not in exclude:
                results.append((w, float(similarities[idx])))

        return results

    def expand_vocabulary(
        self,
        seed_positive: list[str],
        seed_negative: list[str],
        top_k: int = 30,
        similarity_threshold: float = 0.5,
    ) -> dict[str, list[tuple[str, float, str]]]:
        """
        Vocabulary expansion using seed terms (PRD §6.5).

        Procedure:
        1. For each seed term, retrieve nearest neighbors.
        2. Filter by similarity threshold.
        3. Rank candidates.
        4. Return candidates for manual review (NOT auto-added).

        Parameters
        ----------
        seed_positive : list of str
            Positive seed terms.
        seed_negative : list of str
            Negative seed terms.
        top_k : int
            Neighbors per seed.
        similarity_threshold : float
            Minimum similarity to include.

        Returns
        -------
        dict with keys "positive_candidates" and "negative_candidates",
        each a list of (word, max_similarity, closest_seed) tuples.
        """
        self._check_loaded()

        pos_candidates: dict[str, tuple[float, str]] = {}
        neg_candidates: dict[str, tuple[float, str]] = {}

        all_seeds = set(s.lower() for s in seed_positive + seed_negative)

        for seed in seed_positive:
            neighbors = self.nearest_neighbors(seed, top_k=top_k, exclude=all_seeds)
            for word, sim in neighbors:
                if sim >= similarity_threshold:
                    if word not in pos_candidates or sim > pos_candidates[word][0]:
                        pos_candidates[word] = (sim, seed)

        for seed in seed_negative:
            neighbors = self.nearest_neighbors(seed, top_k=top_k, exclude=all_seeds)
            for word, sim in neighbors:
                if sim >= similarity_threshold:
                    if word not in neg_candidates or sim > neg_candidates[word][0]:
                        neg_candidates[word] = (sim, seed)

        # Remove words that appear in both (ambiguous)
        ambiguous = set(pos_candidates.keys()) & set(neg_candidates.keys())

        return {
            "positive_candidates": sorted(
                [(w, sim, seed) for w, (sim, seed) in pos_candidates.items()
                 if w not in ambiguous],
                key=lambda x: -x[1],
            ),
            "negative_candidates": sorted(
                [(w, sim, seed) for w, (sim, seed) in neg_candidates.items()
                 if w not in ambiguous],
                key=lambda x: -x[1],
            ),
            "ambiguous": sorted(ambiguous),
        }

    @property
    def vocab_size(self) -> int:
        return len(self._word2idx)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "GloVe embeddings not loaded. Call .load(filepath) first."
            )

    def __contains__(self, word: str) -> bool:
        return self.has_word(word)

    def __len__(self) -> int:
        return self.vocab_size
