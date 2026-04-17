# Software Architecture — Campus Support Assistant

## Overview

The Campus Support Assistant is a **Retrieval-Augmented** semantic Q-A system
that maps natural-language campus questions to pre-written answers using word
embeddings and cosine similarity.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py  (Gradio UI)                   │
│                                                              │
│   [User Question] ──► [Model Selector] ──► [Top-K Slider]   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    src/predict.py                            │
│                  CampusAssistant.answer()                    │
└─────┬──────────────────────────────────────────┬────────────┘
      │                                          │
      ▼                                          ▼
┌─────────────┐                        ┌──────────────────┐
│ src/        │                        │ src/             │
│ evaluate.py │                        │ train.py         │
│             │  retrieve_top_k()      │                  │
│  cosine     │◄───────────────────────│  Word2Vec model  │
│  similarity │                        │  GloVe vectors   │
└─────┬───────┘                        └──────┬───────────┘
      │                                       │
      ▼                                       ▼
┌─────────────┐                        ┌──────────────────┐
│ src/        │                        │ data/            │
│ data_       │                        │ knowledge_       │
│ processing  │◄───────────────────────│ base.json        │
│ .py         │  load_knowledge_base() │                  │
└─────────────┘                        └──────────────────┘
                                              │
                                              ▼
                                       ┌──────────────────┐
                                       │ models/          │
                                       │ word2vec_        │
                                       │ campus.bin       │
                                       └──────────────────┘
```

---

## Data Flow

1. **Startup** — `train.py` reads `data/knowledge_base.json`, tokenises each
   Q-A pair, trains a Word2Vec Skip-gram model, and saves it to
   `models/word2vec_campus.bin`.  GloVe 50-d vectors are downloaded from
   Stanford and loaded into memory.

2. **Inference** — When a user submits a question:
   a. `data_processing.tokenize()` cleans the input.
   b. `evaluate.sentence_vector_glove()` (or `…_w2v()`) averages the token
      embeddings into a single 50-d sentence vector.
   c. `evaluate.retrieve_top_k()` computes cosine similarity against all KB
      sentence vectors and returns the ranked list.
   d. The top-1 answer is returned to the UI.

3. **Evaluation** — `evaluate.py` runs the system against a held-out set of
   10 labelled queries and reports Top-1 and Top-3 accuracy.

---

## Technology Stack

| Layer | Technology |
|---|---|
| NLP embeddings | Gensim Word2Vec, Stanford GloVe 6B |
| Similarity search | scikit-learn cosine_similarity |
| Data | JSON (custom knowledge base) |
| Web UI | Gradio 4 |
| Language | Python 3.11 |

---

## Design Decisions

- **Modular scripts** — each concern (data, training, evaluation, prediction)
  lives in its own module, making it easy to swap components independently.
- **GloVe > Word2Vec on small corpora** — the knowledge base has only 10
  sentences, which is too small to train meaningful embeddings from scratch.
  Pre-trained GloVe vectors provide much richer semantic information.
- **Stateless retrieval** — no session state is maintained; each query is
  answered independently, keeping the system simple and scalable.
