"""Unified preprocessing and postprocessing for Akkadian translation.

Consolidates logic from all three competition notebooks into reusable,
testable classes with individually toggleable processing steps.

Usage:
    from preprocessing import AkkadianPreprocessor, AkkadianPostprocessor

    pre = AkkadianPreprocessor()
    post = AkkadianPostprocessor(remove_brackets=True, remove_big_gap=True)

    clean_input = pre.preprocess("KIŠIB {d}UTU-ba-ni ...")
    clean_output = post.postprocess(raw_translation)
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_PREFIX = "translate Akkadian to English: "

_SUBSCRIPT_TABLE = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_TABLE = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

_FRACTION_MAP: dict[str, str] = {
    "1/2": "\u00bd",   # ½
    "1/3": "\u2153",   # ⅓
    "2/3": "\u2154",   # ⅔
    "1/4": "\u00bc",   # ¼
    "3/4": "\u00be",   # ¾
    "1/5": "\u2155",   # ⅕
    "1/6": "\u2159",   # ⅙
    "5/6": "\u215a",   # ⅚
    "1/8": "\u215b",   # ⅛
}

_DECIMAL_FRACTION_PATTERNS: list[tuple[str, str]] = [
    (r"\.5\b", " \u00bd"),
    (r"\.25\b", " \u00bc"),
    (r"\.75\b", " \u00be"),
    (r"\.33+\d*\b", " \u2153"),
    (r"\.66+\d*\b", " \u2154"),
]

_BAD_OUTPUT_CHARS = '!?()"—\u2013<>\u2308\u230b\u230a[]+\u02be/;'

_SHORT_INPUT_MAP: dict[str, str] = {
    "a-na": "To",
    "um-ma": "saying:",
    "IGI": "Witnesses:",
    "KIŠIB": "Seal of",
    "ma-na": "mina of silver",
    "šu-ma": "If he does not pay",
    "i-na": "In",
    "ITU.KAM": "Month:",
    "iš-tù": "From the",
    "KÙ.BABBAR": "silver",
    "GÍN": "shekels of silver",
    "ú-ṣa-áb": "he will add interest",
    "li-mu-um": "Eponymy of",
    "ù": "Also,",
    "ša": "of",
    "lá": "",
    "URUDU": "copper",
    "x": "",
    "\u2026": "",
    "a-ma-kam": "Here",
    "en-um-a-šur": "Ennum-A\u0161\u0161ur",
    "kà-ru-um": "The colony",
}

_BROKEN_TEXT_PATTERN = re.compile(r"^[\[\(]?\.+[\]\)]?$|^x+$|^\u2026$")

_DETERMINATIVE_PATTERN = re.compile(
    r"\{(?:d|f|m|ki|kur|uru|lu2?|na4|gi[s\u0161]|mul|u[d\u0161]|an|i[d\u0161]|tug2?|ku[s\u0161])\}",
    re.IGNORECASE,
)

_GRAMMAR_ANNOTATION_PATTERN = re.compile(
    r"\((?:fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)",
    re.IGNORECASE,
)

_BRACKET_CONTENT_PATTERN = re.compile(r"[\(\[][^\)\]]*[\)\]]")


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class AkkadianPreprocessor:
    """Cleans Akkadian transliteration text before model input.

    Processing steps (in order):
    1. Normalize gaps (damaged/missing text markers)
    2. Normalize determinatives ({d}, {ki}, etc.)
    3. Normalize subscript and superscript digits
    4. Remove modern scholarly symbols
    5. Collapse whitespace
    6. Truncate overly long inputs
    7. Add task prefix
    """

    def __init__(
        self,
        max_length: int = 800,
        prefix: str = TASK_PREFIX,
        add_prefix: bool = True,
    ) -> None:
        self.max_length = max_length
        self.prefix = prefix
        self.add_prefix = add_prefix

    def preprocess(self, text: str) -> str:
        """Run the full preprocessing pipeline.

        Args:
            text: Raw transliteration string.

        Returns:
            Cleaned string ready for tokenization.
        """
        if not isinstance(text, str) or not text.strip():
            return self.prefix if self.add_prefix else ""

        result = text
        result = self.normalize_gaps(result)
        result = self.normalize_determinatives(result)
        result = self.normalize_subscripts(result)
        result = self.normalize_superscripts(result)
        result = self.remove_scholarly_symbols(result)
        result = self.collapse_whitespace(result)
        result = self.truncate(result)

        if self.add_prefix:
            result = self.prefix + result

        return result

    @staticmethod
    def normalize_gaps(text: str) -> str:
        """Standardize gap/lacuna notation to <big_gap> and <gap>."""
        result = re.sub(r"\[\s*\.\s*\.\s*\.\s*\]", "<big_gap>", text)
        result = re.sub(r"\(\s*\.\s*\.\s*\.\s*\)", "<big_gap>", result)
        result = re.sub(r"\.{5,}", "<big_gap>", result)
        result = re.sub(r"\u2026{2,}", "<big_gap>", result)
        result = re.sub(r"\.{3,4}", "<big_gap>", result)
        result = re.sub(r"\u2026", "<big_gap>", result)
        result = re.sub(r"\[x+\]", "<gap>", result)
        result = re.sub(r"xx+", "<gap>", result)
        result = re.sub(r"\s+x\s+", " <gap> ", result)
        result = re.sub(r"<gap>(\s*<gap>)+", "<gap>", result)
        result = re.sub(r"<big_gap>(\s*<big_gap>)+", "<big_gap>", result)
        return result

    @staticmethod
    def normalize_determinatives(text: str) -> str:
        """Remove or normalize Akkadian determinatives like {d}, {ki}."""
        return _DETERMINATIVE_PATTERN.sub("", text)

    @staticmethod
    def normalize_subscripts(text: str) -> str:
        """Convert Unicode subscript digits to regular digits."""
        return text.translate(_SUBSCRIPT_TABLE)

    @staticmethod
    def normalize_superscripts(text: str) -> str:
        """Convert Unicode superscript digits to regular digits."""
        return text.translate(_SUPERSCRIPT_TABLE)

    @staticmethod
    def remove_scholarly_symbols(text: str) -> str:
        """Remove modern editorial markers (half-brackets, etc.)."""
        return text.translate(str.maketrans("", "", "\u2308\u2309\u230a\u230b\u00b0"))

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """Collapse multiple spaces into one and strip."""
        return re.sub(r"\s+", " ", text).strip()

    def truncate(self, text: str) -> str:
        """Truncate text exceeding max_length, appending <big_gap>."""
        if len(text) <= self.max_length:
            return text
        return text[: self.max_length] + " <big_gap>"


# ---------------------------------------------------------------------------
# Postprocessor
# ---------------------------------------------------------------------------

class AkkadianPostprocessor:
    """Cleans model output for final submission.

    Processing steps (in order):
    1. Transliterate special Akkadian chars (h/H)
    2. Normalize subscripts/superscripts
    3. Normalize gap tokens in output
    4. Remove grammar annotations
    5. Remove bracket content (optional, improves LB)
    6. Remove special characters
    7. Convert fractions
    8. Remove repeated words/phrases
    9. Remove <big_gap> tokens (optional)
    10. Final whitespace cleanup
    """

    def __init__(
        self,
        remove_brackets: bool = True,
        remove_big_gap: bool = True,
    ) -> None:
        self.remove_brackets = remove_brackets
        self.remove_big_gap = remove_big_gap

    def postprocess(self, text: str) -> str:
        """Run the full postprocessing pipeline.

        Args:
            text: Raw model output string.

        Returns:
            Cleaned translation string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        result = text
        result = self.transliterate_special_chars(result)
        result = self.normalize_subscripts(result)
        result = self.normalize_superscripts(result)
        result = self.normalize_output_gaps(result)
        result = self.remove_grammar_annotations(result)

        if self.remove_brackets:
            result = self.remove_bracket_content(result)

        result = self.remove_special_chars(result)
        result = self.convert_fractions(result)
        result = self.remove_decimal_fractions(result)
        result = self.fix_zero_fraction(result)
        result = self.remove_repeated_phrases(result)

        if self.remove_big_gap:
            result = self.strip_big_gap(result)

        result = self.truncate_to_complete_sentence(result)
        result = self.final_cleanup(result)
        return result

    @staticmethod
    def transliterate_special_chars(text: str) -> str:
        """Convert Akkadian-specific characters: h -> h, H -> H."""
        return text.replace("\u1e2b", "h").replace("\u1e2a", "H")

    @staticmethod
    def normalize_subscripts(text: str) -> str:
        """Convert Unicode subscript digits to regular digits."""
        return text.translate(_SUBSCRIPT_TABLE)

    @staticmethod
    def normalize_superscripts(text: str) -> str:
        """Convert Unicode superscript digits to regular digits."""
        return text.translate(_SUPERSCRIPT_TABLE)

    @staticmethod
    def normalize_output_gaps(text: str) -> str:
        """Normalize gap markers in model output."""
        result = re.sub(r"\[x\]|\(x\)|\bx\b", "<gap>", text, flags=re.IGNORECASE)
        result = re.sub(r"\.{3,}|\u2026|\[\.+\]", "<big_gap>", result)
        result = re.sub(r"<gap>\s*<gap>", " <big_gap> ", result)
        result = re.sub(r"<big_gap>\s*<big_gap>", " <big_gap> ", result)
        return result

    @staticmethod
    def remove_grammar_annotations(text: str) -> str:
        """Remove grammatical annotations like (fem), (plur)."""
        return _GRAMMAR_ANNOTATION_PATTERN.sub("", text)

    @staticmethod
    def remove_bracket_content(text: str) -> str:
        """Remove all text within () and [] brackets.

        Confirmed to improve LB score per competition discussion.
        Gap tokens are preserved before removal.
        """
        result = text.replace("<gap>", "\x00GAP\x00")
        result = result.replace("<big_gap>", "\x00BIG\x00")
        result = _BRACKET_CONTENT_PATTERN.sub("", result)
        result = result.replace("\x00GAP\x00", "<gap>")
        result = result.replace("\x00BIG\x00", "<big_gap>")
        return result

    @staticmethod
    def remove_special_chars(text: str) -> str:
        """Remove unwanted punctuation and symbols, preserving gap tokens."""
        result = text.replace("<gap>", "\x00GAP\x00")
        result = result.replace("<big_gap>", "\x00BIG\x00")
        result = result.translate(str.maketrans("", "", _BAD_OUTPUT_CHARS))
        result = result.replace("\x00GAP\x00", " <gap> ")
        result = result.replace("\x00BIG\x00", " <big_gap> ")
        return result

    @staticmethod
    def convert_fractions(text: str) -> str:
        """Convert ASCII fraction notation to Unicode fraction symbols."""
        result = text
        for frac, symbol in _FRACTION_MAP.items():
            result = result.replace(frac, symbol)
        return result

    @staticmethod
    def remove_decimal_fractions(text: str) -> str:
        """Convert decimal fractions (0.5 etc.) to Unicode symbols."""
        result = text
        for pattern, replacement in _DECIMAL_FRACTION_PATTERNS:
            result = re.sub(r"(\d+)" + pattern, r"\1" + replacement, result)
            result = re.sub(r"\b0" + pattern, replacement.strip(), result)
        return result

    @staticmethod
    def fix_zero_fraction(text: str) -> str:
        """Remove leading '0' before Unicode fraction symbols.

        Converts patterns like '0 ⅓' → '⅓', '0 ½' → '½'.
        """
        return re.sub(r"\b0\s+([\u00bc\u00bd\u00be\u2153-\u215e])", r"\1", text)

    @staticmethod
    def remove_repeated_phrases(text: str) -> str:
        """Remove consecutive duplicate words, phrases, and subtle repetitions."""
        result = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text)
        for n in range(4, 1, -1):
            pattern = (
                r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
            )
            result = re.sub(pattern, r"\1", result)

        result = AkkadianPostprocessor._remove_ngram_loops(result)
        return result

    @staticmethod
    def _remove_ngram_loops(text: str, min_ngram: int = 3, max_ngram: int = 8) -> str:
        """Detect and remove looping n-gram patterns.

        Catches subtle repetitions like "the king of the land the king of the land"
        where slight word variation may exist between repetitions.
        """
        words = text.split()
        if len(words) < min_ngram * 2:
            return text

        for n in range(max_ngram, min_ngram - 1, -1):
            i = 0
            cleaned = []
            while i < len(words):
                if i + 2 * n <= len(words):
                    chunk = words[i : i + n]
                    next_chunk = words[i + n : i + 2 * n]
                    if chunk == next_chunk:
                        cleaned.extend(chunk)
                        i += n
                        while i + n <= len(words) and words[i : i + n] == chunk:
                            i += n
                        continue
                cleaned.append(words[i])
                i += 1
            words = cleaned

        return " ".join(words)

    @staticmethod
    def strip_big_gap(text: str) -> str:
        """Remove <big_gap> tokens from the output."""
        return " ".join(text.replace("<big_gap>", "").split())

    @staticmethod
    def truncate_to_complete_sentence(text: str) -> str:
        """Remove trailing incomplete sentence fragments.

        If the output was cut off mid-sentence (at max_new_tokens),
        truncate to the last sentence-ending punctuation or comma.
        Only applies if the text is long enough that truncation is likely.
        """
        if len(text) < 200:
            return text
        last_period = text.rfind(".")
        last_comma = text.rfind(",")
        cut = max(last_period, last_comma)
        if cut > len(text) * 0.5:
            return text[:cut].rstrip()
        return text

    @staticmethod
    def cap_output_length(text: str, input_text: str) -> str:
        """Cap output length based on input transliteration length.

        Uses a linear model: max_chars = 0.5 * input_len + 30.
        Truncates at word boundary to avoid splitting words.
        """
        input_len = len(input_text)
        max_len = int(input_len * 0.5 + 30)
        if len(text) <= max_len:
            return text
        words = text.split()
        result: list[str] = []
        length = 0
        for w in words:
            if length + len(w) + 1 > max_len and length > 0:
                break
            result.append(w)
            length += len(w) + 1
        return " ".join(result)

    @staticmethod
    def get_short_input_translation(raw_input: str) -> str | None:
        """Return a dictionary translation for 1-token inputs.

        Returns None if no match found (fall through to model output).
        """
        tokens = raw_input.strip().split()
        if len(tokens) != 1:
            return None
        return _SHORT_INPUT_MAP.get(tokens[0])

    @staticmethod
    def is_broken_text(raw_input: str) -> bool:
        """Detect heavily damaged transliterations that can't be translated."""
        text = str(raw_input).strip()
        if text.startswith("[..."):
            return True
        tokens = text.split()
        if not tokens:
            return True
        gap_tokens = sum(1 for t in tokens if _BROKEN_TEXT_PATTERN.match(t))
        return gap_tokens / len(tokens) > 0.4 and len(tokens) < 20

    @staticmethod
    def final_cleanup(text: str) -> str:
        """Collapse whitespace and strip leading/trailing dashes."""
        return re.sub(r"\s+", " ", text).strip().strip("-").strip()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def preprocess(
    text: str,
    max_length: int = 800,
    add_prefix: bool = True,
) -> str:
    """Preprocess a single transliteration string (functional API)."""
    return AkkadianPreprocessor(
        max_length=max_length,
        add_prefix=add_prefix,
    ).preprocess(text)


def postprocess(
    text: str,
    remove_brackets: bool = True,
    remove_big_gap: bool = True,
) -> str:
    """Postprocess a single translation string (functional API)."""
    return AkkadianPostprocessor(
        remove_brackets=remove_brackets,
        remove_big_gap=remove_big_gap,
    ).postprocess(text)


def preprocess_batch(
    texts: list[str],
    max_length: int = 800,
    add_prefix: bool = True,
) -> list[str]:
    """Preprocess a list of transliteration strings."""
    processor = AkkadianPreprocessor(
        max_length=max_length,
        add_prefix=add_prefix,
    )
    return [processor.preprocess(t) for t in texts]


def postprocess_batch(
    texts: list[str],
    remove_brackets: bool = True,
    remove_big_gap: bool = True,
) -> list[str]:
    """Postprocess a list of translation strings."""
    processor = AkkadianPostprocessor(
        remove_brackets=remove_brackets,
        remove_big_gap=remove_big_gap,
    )
    return [processor.postprocess(t) for t in texts]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    pre = AkkadianPreprocessor()
    post = AkkadianPostprocessor(remove_brackets=True, remove_big_gap=True)

    test_inputs = [
        "KISIB {d}UTU-ba-ni DUMU e-na-su-en2",
        "1 ma-na KU.BABBAR ... sa-ru-pa-am",
        "{m}a-sur-i-di [...] ana {ki}ka-ni-is",
        "",
        None,
    ]

    print("=== Preprocessor ===")
    for inp in test_inputs:
        out = pre.preprocess(inp)
        display = repr(inp)[:50]
        print(f"  {display:50s} -> {out}")

    test_outputs = [
        "Seal of Samas-bani, son of Enasuen2",
        "1 mina of refined silver halved (fem.) price",
        "Assur-idi [...] to Kanes (the city) the the tablet",
        "He said: I will give 0.5 silver",
        "",
    ]

    print("\n=== Postprocessor ===")
    for out in test_outputs:
        cleaned = post.postprocess(out)
        display = out[:50]
        print(f"  {display:50s} -> {cleaned}")
