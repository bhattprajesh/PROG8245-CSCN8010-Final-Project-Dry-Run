"""
evaluate.py
-----------
Evaluates the Campus Support Assistant's retrieval quality using:
  - Cosine-similarity ranking (both Word2Vec & GloVe sentence vectors)
  - Top-k accuracy on a small held-out evaluation set

Usage (CLI)
-----------
    python src/evaluate.py
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_processing import load_knowledge_base, build_vocabulary, tokenize
from train import (
    load_word2vec,
    ensure_glove_50d,
    load_glove_subset,
    EMBEDDING_DIM,
    GLOVE_TXT_PATH,
)

# ---------------------------------------------------------------------------
# Sentence-vector helpers
# ---------------------------------------------------------------------------

def sentence_vector_w2v(
    text: str, model, vector_size: int = 50
) -> np.ndarray:
    """Compute a sentence vector by averaging Word2Vec word vectors."""
    tokens = [t for t in tokenize(text) if t in model.wv]
    if not tokens:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean([model.wv[t] for t in tokens], axis=0).astype(np.float32)


def sentence_vector_glove(
    text: str,
    embedding_lookup: dict[str, np.ndarray],
    embedding_dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """Compute a sentence vector by averaging GloVe word vectors."""
    tokens = [t for t in tokenize(text) if t in embedding_lookup]
    if not tokens:
        return np.zeros(embedding_dim, dtype=np.float32)
    return np.mean([embedding_lookup[t] for t in tokens], axis=0)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def build_kb_vectors_w2v(
    knowledge_base: list[dict], model
) -> np.ndarray:
    """Return a matrix of Word2Vec sentence vectors for the knowledge base."""
    return np.array(
        [
            sentence_vector_w2v(
                item["question"] + " " + item["answer"], model
            )
            for item in knowledge_base
        ]
    )


def build_kb_vectors_glove(
    knowledge_base: list[dict],
    glove_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """Return a matrix of GloVe sentence vectors for the knowledge base."""
    return np.array(
        [
            sentence_vector_glove(
                item["question"] + " " + item["answer"], glove_embeddings
            )
            for item in knowledge_base
        ]
    )


def retrieve_top_k(
    query: str,
    knowledge_base: list[dict],
    kb_vectors: np.ndarray,
    sentence_fn,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve the *top_k* most relevant knowledge-base entries for *query*.

    Parameters
    ----------
    query:
        The user's natural-language question.
    knowledge_base:
        List of Q-A dicts.
    kb_vectors:
        Pre-computed sentence vectors for every KB entry.
    sentence_fn:
        Callable that maps a string → sentence vector (matching kb_vectors).
    top_k:
        Number of results to return.

    Returns
    -------
    list[dict]
        Each dict contains ``score``, ``question``, and ``answer``.
    """
    query_vec = sentence_fn(query).reshape(1, -1)
    scores = cosine_similarity(query_vec, kb_vectors)[0]
    ranked = scores.argsort()[::-1][:top_k]
    return [
        {
            "score": float(scores[i]),
            "question": knowledge_base[i]["question"],
            "answer": knowledge_base[i]["answer"],
        }
        for i in ranked
    ]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

# A small, manually constructed evaluation set:
# Each entry is (query, expected_answer_substring).
EVAL_SET = [
    ("What time does the library open?", "library opens"),
    ("When does the lab close?", "computer lab"),
    ("How do I contact support?", "student services"),
    ("Is there a gym?", "gym"),
    ("parking?", "parking"),
    ("registration last day", "registration"),
    ("book advising appointment", "advising"),
    ("cafeteria schedule", "cafeteria"),
    ("student ID application", "student ID"),
    ("financial aid location", "financial aid"),
]


def top_k_accuracy(
    eval_set: list[tuple[str, str]],
    knowledge_base: list[dict],
    kb_vectors: np.ndarray,
    sentence_fn,
    k: int = 1,
) -> float:
    """Compute top-*k* accuracy on *eval_set*.

    A query is considered correctly answered if the expected substring
    appears in any of the top-*k* retrieved answers.
    """
    hits = 0
    for query, expected_substring in eval_set:
        results = retrieve_top_k(query, knowledge_base, kb_vectors, sentence_fn, top_k=k)
        answers = " ".join(r["answer"].lower() for r in results)
        if expected_substring.lower() in answers:
            hits += 1
    return hits / len(eval_set)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Campus Support Assistant — Evaluation ===\n")

    kb = load_knowledge_base()
    vocab = build_vocabulary(kb)

    # ----- Word2Vec -----
    print("Loading Word2Vec model …")
    w2v = load_word2vec()
    kb_vecs_w2v = build_kb_vectors_w2v(kb, w2v)
    sentence_fn_w2v = lambda q: sentence_vector_w2v(q, w2v)

    for k in (1, 3):
        acc = top_k_accuracy(EVAL_SET, kb, kb_vecs_w2v, sentence_fn_w2v, k=k)
        print(f"Word2Vec Top-{k} accuracy: {acc:.0%}")

    # ----- GloVe -----
    print("\nLoading GloVe embeddings …")
    glove_path = ensure_glove_50d()
    glove_emb = load_glove_subset(glove_path, vocab)
    kb_vecs_glove = build_kb_vectors_glove(kb, glove_emb)
    sentence_fn_glove = lambda q: sentence_vector_glove(q, glove_emb)

    for k in (1, 3):
        acc = top_k_accuracy(EVAL_SET, kb, kb_vecs_glove, sentence_fn_glove, k=k)
        print(f"GloVe   Top-{k} accuracy: {acc:.0%}")

    # ----- Sample retrieval demo -----
    print("\n--- Sample retrieval (Word2Vec) ---")
    sample = "How do I contact the college office?"
    results = retrieve_top_k(sample, kb, kb_vecs_w2v, sentence_fn_w2v, top_k=3)
    print(f"Query: {sample}")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['answer']}")
