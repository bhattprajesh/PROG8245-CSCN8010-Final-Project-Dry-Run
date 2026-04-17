"""
predict.py
----------
Exposes a simple prediction interface for the Campus Support Assistant.

It loads trained artefacts once and answers natural-language campus
questions using GloVe semantic similarity (preferred) with a Word2Vec
fallback when GloVe vectors are unavailable.

Usage (CLI — interactive REPL)
-------------------------------
    python src/predict.py
"""

import sys
from pathlib import Path

import numpy as np

from data_processing import load_knowledge_base, build_vocabulary, tokenize
from train import (
    load_word2vec,
    ensure_glove_50d,
    load_glove_subset,
    EMBEDDING_DIM,
    GLOVE_TXT_PATH,
)
from evaluate import (
    build_kb_vectors_glove,
    build_kb_vectors_w2v,
    retrieve_top_k,
    sentence_vector_glove,
    sentence_vector_w2v,
)

CONFIDENCE_THRESHOLD = 0.25  # answers below this score are flagged as uncertain


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class CampusAssistant:
    """End-to-end inference wrapper for the Campus Support Assistant.

    Parameters
    ----------
    use_glove:
        When *True* (default), GloVe vectors are used for retrieval.
        Falls back to Word2Vec if GloVe embeddings are unavailable.
    top_k:
        Number of candidate answers to retrieve before returning the best.
    """

    def __init__(self, use_glove: bool = True, top_k: int = 3):
        self.top_k = top_k
        self._load_artifacts(use_glove)

    # ------------------------------------------------------------------
    def _load_artifacts(self, use_glove: bool) -> None:
        """Load all required models and pre-compute KB vectors."""
        self.knowledge_base = load_knowledge_base()
        vocab = build_vocabulary(self.knowledge_base)

        # Always load Word2Vec as baseline
        self.w2v = load_word2vec()
        self.kb_vecs_w2v = build_kb_vectors_w2v(self.knowledge_base, self.w2v)

        self.glove_embeddings: dict | None = None
        self.kb_vecs_glove: np.ndarray | None = None

        if use_glove:
            try:
                glove_path = ensure_glove_50d()
                self.glove_embeddings = load_glove_subset(glove_path, vocab)
                self.kb_vecs_glove = build_kb_vectors_glove(
                    self.knowledge_base, self.glove_embeddings
                )
                print("Using GloVe for semantic retrieval.")
            except Exception as exc:
                print(f"GloVe unavailable ({exc}). Falling back to Word2Vec.")

    # ------------------------------------------------------------------
    def _sentence_fn(self, text: str) -> np.ndarray:
        """Return the appropriate sentence vector depending on availability."""
        if self.glove_embeddings is not None:
            return sentence_vector_glove(text, self.glove_embeddings)
        return sentence_vector_w2v(text, self.w2v)

    def _kb_vectors(self) -> np.ndarray:
        if self.kb_vecs_glove is not None:
            return self.kb_vecs_glove
        return self.kb_vecs_w2v

    # ------------------------------------------------------------------
    def answer(self, query: str) -> dict:
        """Answer a campus support question.

        Parameters
        ----------
        query:
            Natural-language question from the user.

        Returns
        -------
        dict
            ``answer``      – best matched answer string
            ``confidence``  – cosine similarity score (0–1)
            ``candidates``  – full ranked list of top-k results
            ``uncertain``   – True if confidence < threshold
        """
        candidates = retrieve_top_k(
            query,
            self.knowledge_base,
            self._kb_vectors(),
            self._sentence_fn,
            top_k=self.top_k,
        )
        best = candidates[0] if candidates else {"answer": "No answer found.", "score": 0.0}
        return {
            "answer": best["answer"],
            "confidence": best["score"],
            "candidates": candidates,
            "uncertain": best["score"] < CONFIDENCE_THRESHOLD,
        }


# ---------------------------------------------------------------------------
# CLI interactive REPL
# ---------------------------------------------------------------------------

def interactive_session(assistant: CampusAssistant) -> None:
    """Run a simple question-answering REPL in the terminal."""
    print("\n🎓 Campus Support Assistant")
    print("   Type your question and press Enter.  Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Assistant: Goodbye! 👋")
            break

        result = assistant.answer(query)
        print(f"\nAssistant: {result['answer']}")
        if result["uncertain"]:
            print("  (Low confidence – please rephrase or contact reception.)")
        print(f"  [confidence: {result['confidence']:.3f}]\n")


if __name__ == "__main__":
    print("Loading models …")
    assistant = CampusAssistant(use_glove=True, top_k=3)
    interactive_session(assistant)
