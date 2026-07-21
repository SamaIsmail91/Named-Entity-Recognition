"""
app.py
======
Professional Gradio web application for the NER system. Lets users pick
which trained architecture to run (LSTM / BiLSTM / BiLSTM+CRF / fine-tuned
Transformer / a quick pretrained Hub demo), type or paste free text, and see
detected entities highlighted in real time with color-coded spans
(PER / ORG / LOC / MISC), plus a live entity summary table and counts.

Run:
    python app.py

The app looks for trained checkpoints in ./checkpoints (produced by
train_scratch.py and train_transformer.py). Any model whose checkpoint is
missing is disabled in the dropdown with a helpful note, EXCEPT the
"Quick Demo" option, which downloads a small public pretrained NER model
from the HuggingFace Hub on first use (requires internet).
"""

import os
import glob
import gradio as gr
import pandas as pd

from inference import (
    ScratchModelPredictor, TransformerPredictor, HubDemoPredictor, iob_to_spans
)

CKPT_DIR = "checkpoints"
ENTITY_COLORS = {
    "PER": "#FF6B6B",
    "ORG": "#4ECDC4",
    "LOC": "#FFD93D",
    "MISC": "#A78BFA",
}
ENTITY_NAMES = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Miscellaneous",
}

EXAMPLES = [
    "Barack Obama met Angela Merkel in Berlin during the G7 Summit last October.",
    "Google announced a new partnership with the United Nations to fight climate change.",
    "Marie Curie won the Nobel Prize for her research conducted in Paris, France.",
    "Lionel Messi helped FC Barcelona win the Champions League in a thrilling final.",
    "NASA's Perseverance rover continues to explore Mount Sharp on Mars.",
]

# --------------------------------------------------------------------- #
# Lazy model registry: only instantiate a predictor the first time it's
# actually requested (keeps startup fast, avoids loading unused models).
# --------------------------------------------------------------------- #
_predictor_cache = {}


def available_models():
    options = {}
    for arch, label in [("lstm", "LSTM (scratch)"),
                         ("bilstm", "BiLSTM (scratch)"),
                         ("bilstm_crf", "BiLSTM + CRF (scratch)")]:
        ckpt = os.path.join(CKPT_DIR, f"{arch}.pt")
        options[label] = ("scratch", ckpt) if os.path.exists(ckpt) else ("missing", ckpt)

    transformer_dir = os.path.join(CKPT_DIR, "transformer")
    if os.path.exists(os.path.join(transformer_dir, "config.json")):
        options["Fine-tuned Transformer (BERT/DistilBERT)"] = ("transformer", transformer_dir)
    else:
        options["Fine-tuned Transformer (BERT/DistilBERT)"] = ("missing", transformer_dir)

    options["Quick Demo (pretrained BERT-NER, downloads from Hub)"] = ("hub_demo", "dslim/bert-base-NER")
    return options


def get_predictor(kind, path):
    key = (kind, path)
    if key in _predictor_cache:
        return _predictor_cache[key]
    if kind == "scratch":
        pred = ScratchModelPredictor(path)
    elif kind == "transformer":
        pred = TransformerPredictor(path)
    elif kind == "hub_demo":
        pred = HubDemoPredictor(path)
    else:
        raise ValueError("Model checkpoint not found. Train it first (see README).")
    _predictor_cache[key] = pred
    return pred


def run_ner(text, model_label):
    models = available_models()
    kind, path = models[model_label]

    if kind == "missing":
        empty = pd.DataFrame(columns=["Entity", "Type", "Count"])
        msg = (f"⚠️ No checkpoint found at `{path}`.\n\n"
               f"Train this model first, e.g.:\n"
               f"`python train_scratch.py --arch {os.path.splitext(os.path.basename(path))[0]}`\n"
               f"or `python train_transformer.py` for the transformer.\n\n"
               f"Meanwhile, try **Quick Demo** in the dropdown for an instant working example.")
        return {"text": text, "entities": []}, empty, msg

    if not text or not text.strip():
        empty = pd.DataFrame(columns=["Entity", "Type", "Count"])
        return {"text": "", "entities": []}, empty, "Type or paste some text above, then click **Analyze**."

    try:
        predictor = get_predictor(kind, path)
        tagged = predictor.predict(text)
    except Exception as e:
        empty = pd.DataFrame(columns=["Entity", "Type", "Count"])
        return {"text": text, "entities": []}, empty, f"❌ Error running model: {e}"

    chunks = iob_to_spans(tagged)

    # HighlightedText wants (text, label) tuples directly.
    highlighted = [(word, etype) for word, etype in chunks]

    # Build entity summary table.
    from collections import Counter
    counts = Counter()
    entity_words = {}
    for word, etype in chunks:
        if etype:
            counts[etype] += 1
            entity_words.setdefault(etype, []).append(word)

    rows = []
    for etype in ["PER", "ORG", "LOC", "MISC"]:
        if counts[etype] > 0:
            rows.append({
                "Type": f"{ENTITY_NAMES[etype]} ({etype})",
                "Entities Found": ", ".join(dict.fromkeys(entity_words[etype])),
                "Count": counts[etype],
            })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Type", "Entities Found", "Count"])

    total = sum(counts.values())
    status = f"✅ Found **{total}** entities across **{len(counts)}** categories using **{model_label}**."

    return highlighted, df, status


# --------------------------------------------------------------------- #
# Advanced custom styling
# --------------------------------------------------------------------- #
CUSTOM_CSS = """
:root {
    --per-color: #FF6B6B; --org-color: #4ECDC4; --loc-color: #FFD93D; --misc-color: #A78BFA;
}
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}
#hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 45%, #6d28d9 100%);
    border-radius: 20px;
    padding: 36px 40px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 40px rgba(76, 29, 149, 0.25);
}
#hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
#hero p {
    font-size: 1.02rem;
    opacity: 0.9;
    margin: 0;
}
.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
    color: #1a1a1a;
}
#input-card, #output-card {
    border-radius: 16px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.05);
}
#analyze-btn {
    background: linear-gradient(135deg, #6d28d9, #4c1d95) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
}
#status-md { font-size: 0.95rem; margin-top: 4px; }
.entity-table table { border-radius: 12px; overflow: hidden; }
footer { display: none !important; }
"""

with gr.Blocks(title="Named Entity Recognition Studio") as demo:

    gr.HTML("""
    <div id="hero">
        <h1>🔍 Named Entity Recognition Studio</h1>
        <p>Compare LSTM, BiLSTM, BiLSTM+CRF, and fine-tuned Transformer architectures on real text —
        entities highlighted live, color-coded by type.</p>
        <div style="margin-top:16px;">
            <span class="legend-chip" style="background:#FF6B6B;">● PERSON</span>
            <span class="legend-chip" style="background:#4ECDC4;">● ORGANIZATION</span>
            <span class="legend-chip" style="background:#FFD93D;">● LOCATION</span>
            <span class="legend-chip" style="background:#A78BFA;">● MISC</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_id="input-card"):
            model_dd = gr.Dropdown(
                choices=list(available_models().keys()),
                value=list(available_models().keys())[-1],
                label="🧠 Model architecture",
                info="Pick which trained NER model to run inference with.",
            )
            text_in = gr.Textbox(
                label="✍️ Input text",
                placeholder="Paste a sentence or paragraph here...",
                lines=6,
            )
            with gr.Row():
                analyze_btn = gr.Button("⚡ Analyze", elem_id="analyze-btn", variant="primary", scale=2)
                clear_btn = gr.ClearButton([text_in], value="Clear", scale=1)

            gr.Examples(examples=EXAMPLES, inputs=text_in, label="Try an example")

    status_md = gr.Markdown(elem_id="status-md")

    with gr.Column(elem_id="output-card"):
        gr.Markdown("### 🎯 Detected Entities")
        highlighted_out = gr.HighlightedText(
            label=None,
            color_map=ENTITY_COLORS,
            show_legend=True,
            combine_adjacent=False,
        )
        gr.Markdown("### 📊 Entity Summary")
        table_out = gr.Dataframe(
            headers=["Type", "Entities Found", "Count"],
            elem_classes=["entity-table"],
            wrap=True,
        )

    analyze_btn.click(run_ner, inputs=[text_in, model_dd],
                       outputs=[highlighted_out, table_out, status_md])
    text_in.submit(run_ner, inputs=[text_in, model_dd],
                    outputs=[highlighted_out, table_out, status_md])

    gr.Markdown(
        "---\n"
        "**Architecture notes:** LSTM sees only left context → BiLSTM adds right context for "
        "richer representations → BiLSTM+CRF adds a learned transition matrix so the model can't "
        "output illegal tag sequences, fixing most entity **boundary** errors → the fine-tuned "
        "Transformer replaces the whole recurrent encoder with contextual subword attention for "
        "the strongest overall accuracy.\n\n"
        "*No checkpoint yet for a model? Train it with `train_scratch.py` / `train_transformer.py`, "
        "or use the Quick Demo option above for an instant working example.*"
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="violet", secondary_hue="teal"))
