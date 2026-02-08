"""Translation Retriever (TR-TRY) for Akkadian translation.

Builds a TF-IDF-based translation memory from training data and retrieves
the most similar training translation for a given query transliteration.

Based on the approach from top Kaggle notebooks that combine ByT5 neural
translation with rule-based translation retrieval.

Usage:
    from translation_retriever import TranslationRetriever

    retriever = TranslationRetriever("data/train.csv")
    translation, score = retriever.retrieve("KIŠIB ma-nu-ba-lúm-a-šur")
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_akkadian(text: str) -> str:
    """Normalize Akkadian transliteration for retrieval matching."""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    x = str(text).lower().strip()

    # Remove subscript/superscript digits for homophone folding
    x = re.sub(r"[₀-₉⁰-⁹0-9]", "", x)

    # Normalize determinatives: {d}, {m}, {ki}, etc.
    x = re.sub(r"\{([^}]+)\}", r" DET_\1 ", x)

    # Parenthetical determinatives: (d), (ki)
    x = re.sub(r"\(([a-z]{1,6})\)", r" DET_\1 ", x)

    # Remove editorial brackets but keep content
    x = x.replace("[", " ").replace("]", " ")
    x = x.replace("<", " ").replace(">", " ")
    x = x.replace("<<", " ").replace(">>", " ")

    # Remove apostrophe-like signs
    x = re.sub(r"[ʾʿˀˁ']", "", x)

    # Gap markers
    x = x.replace("…", " GAP ").replace("...", " GAP ")
    x = re.sub(r"\b[xX]{1,3}\b", " GAP ", x)

    # Dot separators (e.g. KÙ.AN)
    x = x.replace(".", " ")

    # Plus sign in sign writing
    x = x.replace("+", "")

    # Unicode NFKD normalization (remove diacritics)
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))

    # Keep only ASCII alphanumeric, hyphens, underscores, spaces
    x = re.sub(r"[^a-z0-9\-_ ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _split_src_transliteration(doc: str) -> list[str]:
    """Split transliteration doc into line-like segments."""
    if pd.isna(doc):
        return []
    doc = str(doc)
    # Line-based split
    parts = re.split(r"\n+", doc)
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]
    # Fallback: if no newlines, split on " ; " or long spaces
    if len(parts) <= 1:
        parts = re.split(r"\s{2,}|;\s+", re.sub(r"\s+", " ", doc).strip())
        parts = [p.strip() for p in parts if p.strip()]
    return parts


def _split_tgt_english(doc: str) -> list[str]:
    """Split English translation into sentences."""
    if pd.isna(doc):
        return []
    doc = re.sub(r"\s+", " ", str(doc)).strip()
    if not doc:
        return []
    sents = re.split(r"(?<=[.!?])\s+", doc)
    return [s.strip() for s in sents if s.strip()]


def _align_monotone(
    src_units: list[str],
    tgt_units: list[str],
    max_src_merge: int = 2,
    max_tgt_merge: int = 2,
) -> list[tuple[str, str]]:
    """Monotone greedy alignment by matching length proportions."""
    import math as _math

    i = j = 0
    pairs: list[tuple[str, str]] = []
    while i < len(src_units) and j < len(tgt_units):
        best = None
        for si in range(1, max_src_merge + 1):
            for tj in range(1, max_tgt_merge + 1):
                if i + si <= len(src_units) and j + tj <= len(tgt_units):
                    src_chunk = " ".join(src_units[i : i + si])
                    tgt_chunk = " ".join(tgt_units[j : j + tj])
                    ls = max(1, len(src_chunk.split()))
                    lt = max(1, len(tgt_chunk.split()))
                    score = abs(_math.log(ls / lt))
                    if best is None or score < best[0]:
                        best = (score, si, tj, src_chunk, tgt_chunk)
        if best is None:
            break
        _, si, tj, s_chunk, t_chunk = best
        pairs.append((s_chunk, t_chunk))
        i += si
        j += tj

    if pairs:
        if i < len(src_units):
            pairs[-1] = (
                pairs[-1][0] + " " + " ".join(src_units[i:]),
                pairs[-1][1],
            )
        if j < len(tgt_units):
            pairs[-1] = (
                pairs[-1][0],
                pairs[-1][1] + " " + " ".join(tgt_units[j:]),
            )
    return pairs


def build_sentence_pairs(train_path: str | Path) -> pd.DataFrame:
    """Build sentence-level parallel pairs from document-level train.csv."""
    df = pd.read_csv(train_path)
    df = df.dropna(subset=["transliteration", "translation"])

    all_pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        src_lines = _split_src_transliteration(row["transliteration"])
        tgt_sents = _split_tgt_english(row["translation"])

        if not src_lines or not tgt_sents:
            # Keep as single pair if can't split
            all_pairs.append(
                (str(row["transliteration"]), str(row["translation"]))
            )
            continue

        doc_pairs = _align_monotone(src_lines, tgt_sents)
        if doc_pairs:
            all_pairs.extend(doc_pairs)
        else:
            all_pairs.append(
                (str(row["transliteration"]), str(row["translation"]))
            )

    result = pd.DataFrame(all_pairs, columns=["src", "tgt"])
    result = result[
        (result["src"].str.len() > 0) & (result["tgt"].str.len() > 0)
    ].drop_duplicates()
    return result


class TranslationRetriever:
    """TF-IDF-based translation retrieval from training corpus."""

    def __init__(
        self,
        train_path: str | Path,
        w_char: float = 0.75,
        w_word: float = 0.20,
        w_seq: float = 0.05,
        top_k: int = 60,
        rerank_k: int = 10,
        len_penalty_power: float = 0.3,
        min_accept_score: float = 0.10,
        use_sentence_pairs: bool = False,
    ) -> None:
        self.w_char = w_char
        self.w_word = w_word
        self.w_seq = w_seq
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.len_penalty_power = len_penalty_power
        self.min_accept_score = min_accept_score

        if use_sentence_pairs:
            self._load_sentence_pairs(Path(train_path))
        else:
            self._load_and_index(Path(train_path))

    def _load_sentence_pairs(self, train_path: Path) -> None:
        """Load training data, split into sentence pairs, build index."""
        pairs_df = build_sentence_pairs(train_path)
        self.src_raw = pairs_df["src"].tolist()
        self.tgt_raw = pairs_df["tgt"].tolist()
        self.src_norm = [normalize_akkadian(s) for s in self.src_raw]
        self._build_index()

    def _load_and_index(self, train_path: Path) -> None:
        """Load training data and build TF-IDF indices."""
        df = pd.read_csv(train_path)
        df = df.dropna(subset=["transliteration", "translation"])

        self.src_raw = df["transliteration"].tolist()
        self.tgt_raw = df["translation"].tolist()
        self.src_norm = [normalize_akkadian(s) for s in self.src_raw]
        self._build_index()

    def _build_index(self) -> None:
        """Build TF-IDF indices from loaded data."""

        # Char n-gram TF-IDF (3-6 grams)
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 6)
        )
        self.X_char = self.char_vec.fit_transform(self.src_norm)

        # Word n-gram TF-IDF (1-2 grams)
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2)
        )
        self.X_word = self.word_vec.fit_transform(self.src_norm)

        # Pre-compute source lengths for length penalty
        self.src_lengths = np.array(
            [len(s.split()) for s in self.src_norm], dtype=np.float32
        )

    def _length_penalty(
        self, query_len: int, cand_lengths: np.ndarray
    ) -> np.ndarray:
        """Compute length penalty based on word count ratio."""
        ratio = cand_lengths / max(1, query_len)
        return np.exp(
            -np.abs(np.log(ratio + 1e-5)) * self.len_penalty_power
        )

    def retrieve(self, query: str) -> tuple[str, float]:
        """Retrieve the best matching translation for a query.

        Args:
            query: Raw Akkadian transliteration string.

        Returns:
            Tuple of (translation, confidence_score).
            If no good match found, returns ("", 0.0).
        """
        q_norm = normalize_akkadian(query)
        if not q_norm:
            return "", 0.0

        # TF-IDF similarity
        q_char = self.char_vec.transform([q_norm])
        q_word = self.word_vec.transform([q_norm])

        sc = (q_char @ self.X_char.T).toarray()[0]
        sw = (q_word @ self.X_word.T).toarray()[0]

        combined = self.w_char * sc + self.w_word * sw

        # Top-K with length penalty
        k = min(self.top_k, len(self.src_norm))
        cand_idx = np.argpartition(-combined, k)[:k]
        lps = self._length_penalty(
            len(q_norm.split()), self.src_lengths[cand_idx]
        )
        final_scores = combined[cand_idx] * lps

        # Rerank top candidates with SequenceMatcher
        rerank_idx = cand_idx[np.argsort(-final_scores)[: self.rerank_k]]

        best_score = -1.0
        best_idx = -1

        for idx in rerank_idx:
            seq_score = SequenceMatcher(
                None, q_norm, self.src_norm[idx]
            ).ratio()
            total = final_scores[
                np.where(cand_idx == idx)[0][0]
            ] + self.w_seq * seq_score

            if total > best_score:
                best_score = total
                best_idx = idx

        if best_score > self.min_accept_score and best_idx >= 0:
            return self.tgt_raw[best_idx], best_score
        return "", 0.0

    def retrieve_batch(
        self, queries: list[str]
    ) -> list[tuple[str, float]]:
        """Retrieve translations for a batch of queries."""
        return [self.retrieve(q) for q in queries]


if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    retriever = TranslationRetriever(train_path)
    print(f"Indexed {len(retriever.src_raw)} training pairs")

    test_queries = [
        "KIŠIB ma-nu-ba-lúm-a-šur",
        "a-na",
        "1 ma-na KÙ.BABBAR",
    ]
    for q in test_queries:
        trans, score = retriever.retrieve(q)
        print(f"\n  Query: {q}")
        print(f"  Score: {score:.4f}")
        print(f"  Translation: {trans[:100]}...")
