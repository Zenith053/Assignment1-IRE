"""Tokenisation shared by both languages.

BM25 itself is language-agnostic; only the character class and the stopword
list differ between English and Danish. Both are selected from config, so the
retrieval code never branches on dataset or language.
"""

from __future__ import annotations

import re

# \w with re.UNICODE already covers Danish ae/oe/aa, so one pattern serves both.
# Keep intra-word apostrophes out: "don't" -> "don", "t" is worse than "dont".
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Deliberately short lists. Aggressive stopword removal hurts BM25 on short
# documents: titles are ~10 tokens, and IDF already down-weights common words.
STOPWORDS = {
    "en": {
        "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
        "with", "about", "into", "to", "from", "in", "on", "off", "over",
        "under", "is", "are", "was", "were", "be", "been", "being", "am",
        "do", "does", "did", "has", "have", "had", "it", "its", "this",
        "that", "these", "those", "as", "s", "t",
    },
    "da": {
        "og", "i", "jeg", "det", "at", "en", "den", "til", "er", "som",
        "paa", "de", "med", "han", "af", "for", "ikke", "der", "var", "mig",
        "sig", "men", "et", "har", "om", "vi", "min", "havde", "ham", "hun",
        "nu", "over", "da", "fra", "du", "ud", "sin", "dem", "os", "op",
        "man", "hans", "hvor", "eller", "hvad", "skal", "selv", "her",
        "alle", "vil", "blev", "kunne", "ind", "naar", "vaere", "dog",
    },
}


def tokenize(text: str, language: str = "en", drop_stopwords: bool = True,
             min_length: int = 2) -> list[str]:
    """Lowercase, split on non-letters, and optionally drop stopwords.

    Digits are dropped entirely: in news text they are mostly dates, scores and
    counts, which add index size without helping topical matching.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if min_length > 1:
        tokens = [t for t in tokens if len(t) >= min_length]
    if drop_stopwords:
        stop = STOPWORDS.get(language, STOPWORDS["en"])
        tokens = [t for t in tokens if t not in stop]
    return tokens


def build_document_text(row, fields: list[str]) -> str:
    """Join the configured article fields into one indexable string."""
    parts = [str(row[f]) for f in fields if f in row and row[f]]
    return " ".join(parts)
