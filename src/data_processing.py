"""
data_processing.py
------------------
Handles loading, cleaning, and tokenizing the campus support knowledge base
for use in Word2Vec and GloVe-based semantic retrieval.
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Return a list of lowercase alphabetic tokens from *text*."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def safe_sentence_tokenize(text: str) -> list[str]:
    """Split *text* into sentences without requiring NLTK data files."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def safe_word_tokenize(text: str) -> list[str]:
    """Tokenize *text* without requiring NLTK data files."""
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    except LookupError:
        return re.findall(r"[a-zA-Z']+", text.lower())


# ---------------------------------------------------------------------------
# Knowledge-base I/O
# ---------------------------------------------------------------------------

def load_knowledge_base(path: str | Path = "data/knowledge_base.json") -> list[dict]:
    """Load the Q-A knowledge base from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.  Defaults to ``data/knowledge_base.json``.

    Returns
    -------
    list[dict]
        Each element is a dict with ``"question"`` and ``"answer"`` keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def build_corpus(knowledge_base: list[dict]) -> list[list[str]]:
    """Convert the knowledge base into a tokenised corpus.

    Each Q-A pair is concatenated and tokenised into a list of word tokens.
    The resulting list-of-lists is the training corpus expected by Gensim's
    Word2Vec.

    Parameters
    ----------
    knowledge_base:
        Output of :func:`load_knowledge_base`.

    Returns
    -------
    list[list[str]]
        Tokenised sentences.
    """
    corpus: list[list[str]] = []
    for item in knowledge_base:
        combined = item["question"] + " " + item["answer"]
        tokens = [t for t in tokenize(combined) if t.isalpha()]
        if tokens:
            corpus.append(tokens)
    return corpus


def build_vocabulary(knowledge_base: list[dict]) -> list[str]:
    """Return a sorted list of unique tokens across the entire knowledge base."""
    all_text = " ".join(
        item["question"] + " " + item["answer"] for item in knowledge_base
    )
    return sorted(set(tokenize(all_text)))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kb = load_knowledge_base()
    corpus = build_corpus(kb)
    vocab = build_vocabulary(kb)

    print(f"Loaded {len(kb)} Q-A pairs.")
    print(f"Corpus sentences : {len(corpus)}")
    print(f"Unique vocabulary: {len(vocab)} tokens")
    print(f"\nFirst corpus sentence: {corpus[0]}")
