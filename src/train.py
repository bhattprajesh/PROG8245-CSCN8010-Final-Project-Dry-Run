"""
train.py
--------
Trains a Word2Vec (Skip-gram) model on the campus-support knowledge base
and optionally downloads / loads pre-trained GloVe 50-d vectors.

Usage (CLI)
-----------
    python src/train.py
"""

import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from data_processing import load_knowledge_base, build_corpus, build_vocabulary

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")
WORD2VEC_PATH = MODELS_DIR / "word2vec_campus.bin"

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = Path("glove.6B")
GLOVE_ZIP_PATH = Path("glove.6B.zip")
GLOVE_TXT_PATH = GLOVE_DIR / "glove.6B.50d.txt"
EMBEDDING_DIM = 50


# ---------------------------------------------------------------------------
# Word2Vec
# ---------------------------------------------------------------------------

def train_word2vec(
    corpus: list[list[str]],
    vector_size: int = 50,
    window: int = 3,
    min_count: int = 1,
    sg: int = 1,
    seed: int = 42,
) -> Word2Vec:
    """Train a Word2Vec model on *corpus* and return it.

    Parameters
    ----------
    corpus:
        Tokenised sentences (list-of-lists of strings).
    vector_size:
        Dimensionality of the word vectors.
    window:
        Maximum distance between the current and predicted word.
    min_count:
        Ignore words with total frequency lower than this.
    sg:
        1 → Skip-gram;  0 → CBOW.
    seed:
        Random seed for reproducibility.
    """
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        seed=seed,
    )
    return model


def save_word2vec(model: Word2Vec, path: str | Path = WORD2VEC_PATH) -> None:
    """Persist a trained Word2Vec model to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"Word2Vec model saved → {path}")


def load_word2vec(path: str | Path = WORD2VEC_PATH) -> Word2Vec:
    """Load a previously saved Word2Vec model from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Word2Vec model not found at {path}. Run train.py first."
        )
    return Word2Vec.load(str(path))


# ---------------------------------------------------------------------------
# GloVe
# ---------------------------------------------------------------------------

def ensure_glove_50d(
    glove_url: str = GLOVE_URL,
    zip_path: Path = GLOVE_ZIP_PATH,
    glove_dir: Path = GLOVE_DIR,
) -> Path:
    """Download and extract ``glove.6B.50d.txt`` if not already present.

    Returns
    -------
    Path
        Path to the extracted ``.txt`` file.
    """
    target_file = glove_dir / "glove.6B.50d.txt"

    if target_file.exists():
        print(f"GloVe file already available: {target_file}")
        return target_file

    glove_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print("Downloading Stanford GloVe 6B embeddings (this may take a while)…")
        urllib.request.urlretrieve(glove_url, zip_path)
        print(f"Download complete: {zip_path}")

    print("Extracting glove.6B.50d.txt…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("glove.6B.50d.txt", path=glove_dir)

    print(f"Extracted: {target_file}")
    return target_file


def load_glove_subset(
    filepath: str | Path, vocab: list[str]
) -> dict[str, np.ndarray]:
    """Load only the GloVe vectors that appear in *vocab*.

    Parameters
    ----------
    filepath:
        Path to the ``glove.6B.50d.txt`` file.
    vocab:
        List of words to load vectors for.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from word → 50-d float32 vector.
    """
    embeddings: dict[str, np.ndarray] = {}
    vocab_set = set(vocab)

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab_set:
                embeddings[word] = np.asarray(parts[1:], dtype=np.float32)

    print(f"Loaded {len(embeddings)} GloVe vectors for the knowledge-base vocabulary.")
    return embeddings


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Campus Support Assistant — Training ===\n")

    kb = load_knowledge_base()
    corpus = build_corpus(kb)
    vocab = build_vocabulary(kb)

    # --- Word2Vec ---
    print("Training Word2Vec (Skip-gram) …")
    w2v_model = train_word2vec(corpus)
    save_word2vec(w2v_model)
    print(f"Vocabulary size: {len(w2v_model.wv)}\n")

    # --- GloVe ---
    print("Fetching GloVe embeddings …")
    glove_path = ensure_glove_50d()
    glove_embeddings = load_glove_subset(glove_path, vocab)
    print(f"\nAll done. Models are stored in '{MODELS_DIR}/'.")
