"""
app.py
------
Gradio web application for the Campus Support Assistant.

Launch
------
    python app.py

Then open http://localhost:7860 in your browser.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import numpy as np

from data_processing import load_knowledge_base, build_vocabulary
from train import (
    load_word2vec,
    ensure_glove_50d,
    load_glove_subset,
    EMBEDDING_DIM,
)
from evaluate import (
    build_kb_vectors_glove,
    build_kb_vectors_w2v,
    retrieve_top_k,
    sentence_vector_glove,
    sentence_vector_w2v,
)

# ---------------------------------------------------------------------------
# Globals – loaded once at startup
# ---------------------------------------------------------------------------

KB = load_knowledge_base()
VOCAB = build_vocabulary(KB)

W2V = load_word2vec()
KB_VECS_W2V = build_kb_vectors_w2v(KB, W2V)

GLOVE_EMBEDDINGS: dict | None = None
KB_VECS_GLOVE: np.ndarray | None = None

try:
    glove_path = ensure_glove_50d()
    GLOVE_EMBEDDINGS = load_glove_subset(glove_path, VOCAB)
    KB_VECS_GLOVE = build_kb_vectors_glove(KB, GLOVE_EMBEDDINGS)
    print("GloVe embeddings loaded successfully.")
except Exception as exc:
    print(f"GloVe unavailable ({exc}). Word2Vec will be used.")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.25


def get_sentence_fn(model_choice: str):
    """Return the appropriate sentence-vector function."""
    if model_choice == "GloVe" and GLOVE_EMBEDDINGS is not None:
        return lambda text: sentence_vector_glove(text, GLOVE_EMBEDDINGS)
    return lambda text: sentence_vector_w2v(text, W2V)


def get_kb_vectors(model_choice: str) -> np.ndarray:
    if model_choice == "GloVe" and KB_VECS_GLOVE is not None:
        return KB_VECS_GLOVE
    return KB_VECS_W2V


# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

def answer_query(query: str, model_choice: str, top_k: int) -> tuple[str, str]:
    """Main Gradio handler.

    Returns
    -------
    (answer_text, details_text)
    """
    if not query.strip():
        return "Please enter a question.", ""

    sentence_fn = get_sentence_fn(model_choice)
    kb_vecs = get_kb_vectors(model_choice)

    candidates = retrieve_top_k(query, KB, kb_vecs, sentence_fn, top_k=top_k)

    if not candidates:
        return "No answer found in the knowledge base.", ""

    best = candidates[0]
    answer = best["answer"]
    confidence = best["score"]

    uncertain_note = ""
    if confidence < CONFIDENCE_THRESHOLD:
        uncertain_note = "\n\n⚠️ Low confidence – try rephrasing or contact reception."

    details_lines = [f"**Top-{top_k} retrieved matches ({model_choice}):**\n"]
    for i, r in enumerate(candidates, 1):
        details_lines.append(
            f"{i}. [{r['score']:.3f}] **Q:** {r['question']}\n   **A:** {r['answer']}"
        )

    return answer + uncertain_note, "\n\n".join(details_lines)


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Campus Support Assistant") as demo:
    gr.Markdown(
        """
        # 🎓 Campus Support Assistant
        Ask any campus-related question and get an instant answer powered by
        **Word2Vec** or **GloVe** semantic similarity retrieval.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What time does the library open?",
                lines=2,
            )
        with gr.Column(scale=1):
            model_radio = gr.Radio(
                choices=["GloVe", "Word2Vec"],
                value="GloVe" if GLOVE_EMBEDDINGS is not None else "Word2Vec",
                label="Embedding model",
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=5, step=1, value=3, label="Top-K results"
            )

    submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column():
            answer_box = gr.Textbox(label="Answer", lines=3, interactive=False)
        with gr.Column():
            details_box = gr.Markdown(label="Retrieval details")

    gr.Examples(
        examples=[
            ["What time does the library open?", "GloVe", 3],
            ["How do I contact support?", "Word2Vec", 3],
            ["Is there a gym on campus?", "GloVe", 3],
            ["Where is the financial aid office?", "GloVe", 3],
            ["When is the registration deadline?", "Word2Vec", 3],
        ],
        inputs=[query_box, model_radio, top_k_slider],
    )

    submit_btn.click(
        fn=answer_query,
        inputs=[query_box, model_radio, top_k_slider],
        outputs=[answer_box, details_box],
    )

    gr.Markdown(
        """
        ---
        **How it works:**
        1. Your question is converted into a sentence vector by averaging the
           word embeddings (GloVe 50-d or Word2Vec 50-d).
        2. Cosine similarity is computed against every entry in the campus
           knowledge base.
        3. The top-K most similar Q-A pairs are retrieved and the best answer
           is displayed.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
