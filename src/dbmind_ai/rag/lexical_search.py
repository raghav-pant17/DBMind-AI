"""Deterministic lexical scoring for structured schema retrieval."""

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def tokenize(text: str) -> list[str]:
    """Split text into normalized lexical tokens."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


def lexical_score(query: str, document: str) -> float:
    """Score query-document overlap using deterministic token matching."""

    query_terms = Counter(tokenize(query))
    document_terms = Counter(tokenize(document))
    if not query_terms or not document_terms:
        return 0.0
    return sum(min(count, document_terms[term]) for term, count in query_terms.items())
