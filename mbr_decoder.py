"""Minimum Bayes Risk (MBR) Decoding for Akkadian translation.

Selects the candidate that maximizes expected utility (chrF++) across
all other candidates, rather than picking the highest-probability output.

Usage:
    from mbr_decoder import MBRDecoder

    decoder = MBRDecoder(utility="chrf")
    best, score, details = decoder.decode(candidates)

Integration with ByT5 inference:
    outputs = model.generate(num_return_sequences=N, num_beams=N, ...)
    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    best, score, details = decoder.decode(candidates)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Self-contained chrF++ implementation (no sacrebleu dependency)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> Counter:
    """Extract character n-grams from text."""
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _word_ngrams(text: str, n: int) -> Counter:
    """Extract word n-grams from text."""
    words = text.split()
    return Counter(
        tuple(words[i:i + n]) for i in range(len(words) - n + 1)
    )


def sentence_chrf(
    hypothesis: str,
    reference: str,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """Compute sentence-level chrF++ score.

    Args:
        hypothesis: Candidate translation.
        reference: Reference translation.
        char_order: Maximum character n-gram order (default 6).
        word_order: Maximum word n-gram order (default 2 for chrF++).
        beta: F-score beta parameter (default 2.0, recall-weighted).

    Returns:
        chrF++ score in [0.0, 100.0].
    """
    if not hypothesis and not reference:
        return 100.0
    if not hypothesis or not reference:
        return 0.0

    total_f = 0.0
    count = 0

    # Character n-grams
    for n in range(1, char_order + 1):
        hyp_ngrams = _char_ngrams(hypothesis, n)
        ref_ngrams = _char_ngrams(reference, n)

        if not hyp_ngrams or not ref_ngrams:
            total_f += 0.0
            count += 1
            continue

        common = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = common / hyp_total if hyp_total > 0 else 0.0
        recall = common / ref_total if ref_total > 0 else 0.0

        if precision + recall > 0:
            f = (1 + beta ** 2) * precision * recall / (
                beta ** 2 * precision + recall
            )
        else:
            f = 0.0

        total_f += f
        count += 1

    # Word n-grams (the ++ part)
    for n in range(1, word_order + 1):
        hyp_ngrams = _word_ngrams(hypothesis, n)
        ref_ngrams = _word_ngrams(reference, n)

        if not hyp_ngrams or not ref_ngrams:
            total_f += 0.0
            count += 1
            continue

        common = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = common / hyp_total if hyp_total > 0 else 0.0
        recall = common / ref_total if ref_total > 0 else 0.0

        if precision + recall > 0:
            f = (1 + beta ** 2) * precision * recall / (
                beta ** 2 * precision + recall
            )
        else:
            f = 0.0

        total_f += f
        count += 1

    return (total_f / count * 100.0) if count > 0 else 0.0


def sentence_bleu(hypothesis: str, reference: str, max_order: int = 4) -> float:
    """Compute sentence-level BLEU score (smoothed).

    Uses +1 smoothing (add-one) for robustness on short sentences.

    Args:
        hypothesis: Candidate translation.
        reference: Reference translation.
        max_order: Maximum n-gram order (default 4).

    Returns:
        BLEU score in [0.0, 100.0].
    """
    if not hypothesis or not reference:
        return 0.0

    hyp_words = hypothesis.split()
    ref_words = reference.split()

    if not hyp_words or not ref_words:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1.0 - len(ref_words) / len(hyp_words)))

    log_avg = 0.0
    for n in range(1, max_order + 1):
        hyp_ngrams = Counter(
            tuple(hyp_words[i:i + n]) for i in range(len(hyp_words) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_words[i:i + n]) for i in range(len(ref_words) - n + 1)
        )

        clipped = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())

        # Add-one smoothing
        precision = (clipped + 1) / (total + 1)
        log_avg += math.log(precision) / max_order

    return bp * math.exp(log_avg) * 100.0


def geo_mean_utility(hypothesis: str, reference: str) -> float:
    """Compute sqrt(BLEU * chrF++) - matches the competition metric."""
    bleu = sentence_bleu(hypothesis, reference)
    chrf = sentence_chrf(hypothesis, reference)

    if bleu <= 0 or chrf <= 0:
        return 0.0
    return math.sqrt(bleu * chrf)


# ---------------------------------------------------------------------------
# MBR Decoder
# ---------------------------------------------------------------------------

# Pre-defined utility functions
UTILITY_FUNCTIONS: dict[str, Callable[[str, str], float]] = {
    "chrf": sentence_chrf,
    "bleu": sentence_bleu,
    "geo_mean": geo_mean_utility,
}


class MBRDecoder:
    """Minimum Bayes Risk decoder for translation candidates.

    Given N candidates, selects the one that maximizes expected utility
    across all other candidates (used as pseudo-references).

    Args:
        utility: Utility function name ("chrf", "bleu", "geo_mean")
                 or a callable(hypothesis, reference) -> float.
        exclude_self: Whether to exclude self-comparison (default True).
    """

    def __init__(
        self,
        utility: str | Callable[[str, str], float] = "chrf",
        exclude_self: bool = True,
    ) -> None:
        if isinstance(utility, str):
            if utility not in UTILITY_FUNCTIONS:
                raise ValueError(
                    f"Unknown utility '{utility}'. "
                    f"Choose from: {list(UTILITY_FUNCTIONS.keys())}"
                )
            self.utility_fn = UTILITY_FUNCTIONS[utility]
            self.utility_name = utility
        else:
            self.utility_fn = utility
            self.utility_name = "custom"

        self.exclude_self = exclude_self

    def decode(
        self,
        candidates: list[str],
        weights: Optional[list[float]] = None,
    ) -> tuple[str, float, dict]:
        """Select the best candidate via MBR decoding.

        Args:
            candidates: List of translation candidates.
            weights: Optional weights for each pseudo-reference
                     (e.g., model scores). If None, uniform weights.

        Returns:
            Tuple of (best_candidate, mbr_score, details_dict).
        """
        n = len(candidates)

        if n == 0:
            return "", 0.0, {}
        if n == 1:
            return candidates[0], 100.0, {"mbr_scores": [100.0]}

        # Deduplicate while preserving order (for efficiency)
        unique_candidates, candidate_map = self._deduplicate(candidates)

        # Compute pairwise utility matrix
        utility_matrix = self._compute_utility_matrix(unique_candidates)

        # Compute MBR scores (expected utility)
        if weights is not None:
            norm_weights = self._normalize_weights(weights, candidate_map)
        else:
            norm_weights = None

        mbr_scores = self._compute_mbr_scores(
            utility_matrix, unique_candidates, norm_weights
        )

        # Map back to original candidates
        original_scores = [mbr_scores[candidate_map[i]] for i in range(n)]

        # Select best
        best_unique_idx = max(range(len(unique_candidates)), key=lambda i: mbr_scores[i])
        best_original_idx = next(
            i for i in range(n) if candidate_map[i] == best_unique_idx
        )

        return candidates[best_original_idx], original_scores[best_original_idx], {
            "mbr_scores": original_scores,
            "best_idx": best_original_idx,
            "utility": self.utility_name,
            "n_candidates": n,
            "n_unique": len(unique_candidates),
            "utility_matrix": utility_matrix,
        }

    def _deduplicate(
        self, candidates: list[str]
    ) -> tuple[list[str], list[int]]:
        """Remove duplicate candidates, returning unique list and index map."""
        seen: dict[str, int] = {}
        unique: list[str] = []
        mapping: list[int] = []

        for cand in candidates:
            if cand not in seen:
                seen[cand] = len(unique)
                unique.append(cand)
            mapping.append(seen[cand])

        return unique, mapping

    def _compute_utility_matrix(
        self, candidates: list[str]
    ) -> list[list[float]]:
        """Compute NxN pairwise utility matrix."""
        n = len(candidates)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j and self.exclude_self:
                    continue
                matrix[i][j] = self.utility_fn(candidates[i], candidates[j])

        return matrix

    def _normalize_weights(
        self, weights: list[float], candidate_map: list[int]
    ) -> list[float]:
        """Aggregate and normalize weights for unique candidates."""
        n_unique = max(candidate_map) + 1
        agg_weights = [0.0] * n_unique

        for i, w in enumerate(weights):
            agg_weights[candidate_map[i]] += w

        total = sum(agg_weights)
        if total > 0:
            return [w / total for w in agg_weights]
        return [1.0 / n_unique] * n_unique

    def _compute_mbr_scores(
        self,
        utility_matrix: list[list[float]],
        candidates: list[str],
        weights: Optional[list[float]],
    ) -> list[float]:
        """Compute expected utility for each candidate."""
        n = len(candidates)
        scores = []

        for i in range(n):
            expected = 0.0
            denom = 0.0

            for j in range(n):
                if i == j and self.exclude_self:
                    continue

                w = weights[j] if weights else 1.0
                expected += w * utility_matrix[i][j]
                denom += w

            scores.append(expected / denom if denom > 0 else 0.0)

        return scores


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def mbr_decode(
    candidates: list[str],
    utility: str = "chrf",
    weights: Optional[list[float]] = None,
) -> tuple[str, float, dict]:
    """One-shot MBR decoding (functional API).

    Args:
        candidates: Translation candidates from beam search.
        utility: "chrf" (default), "bleu", or "geo_mean".
        weights: Optional model scores for weighted MBR.

    Returns:
        (best_candidate, mbr_score, details)
    """
    return MBRDecoder(utility=utility).decode(candidates, weights)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Simulated beam search candidates (typical Akkadian translation)
    test_candidates = [
        "Seal of Mannum-balum-Assur, son of Silli-Adad.",
        "Seal of Mannum-balum-Assur son of Silli-Adad.",
        "The seal of Mannum-balum-Assur, son of Silli-Adad.",
        "Seal: Mannum-balum-Assur, son of Silli-Adad.",
        "Seal of Mannum, son of Adad.",
        "The seal belongs to Mannum-balum-Assur.",
        "Cylinder seal of Mannum-balum-Assur, son of Silli-Adad.",
        "Seal of Mannum-balum-Assur, the son of Silli-Adad.",
    ]

    print("=" * 60)
    print("  MBR Decoding Test")
    print("=" * 60)

    for utility_name in ["chrf", "bleu", "geo_mean"]:
        decoder = MBRDecoder(utility=utility_name)
        best, score, details = decoder.decode(test_candidates)

        print(f"\n--- Utility: {utility_name} ---")
        print(f"  Best: {best}")
        print(f"  MBR Score: {score:.2f}")
        print(f"  Best IDX: {details['best_idx']}")
        print(f"  Unique candidates: {details['n_unique']}")
        print(f"  All MBR scores:")
        for i, (cand, s) in enumerate(
            zip(test_candidates, details["mbr_scores"])
        ):
            marker = " <<< BEST" if i == details["best_idx"] else ""
            print(f"    [{i}] {s:6.2f} | {cand}{marker}")

    # Quick benchmark
    print(f"\n--- Performance ---")
    import time
    start = time.perf_counter()
    for _ in range(100):
        mbr_decode(test_candidates, utility="chrf")
    elapsed = time.perf_counter() - start
    print(f"  100 iterations (8 candidates, chrF++): {elapsed:.3f}s")
    print(f"  Per decode: {elapsed / 100 * 1000:.1f}ms")
