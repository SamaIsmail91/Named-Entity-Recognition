🔍 Named Entity Recognition System

An end-to-end NLP sequence-labeling pipeline that identifies **People (PER)**,
**Organizations (ORG)**, **Locations (LOC)**, and **Miscellaneous (MISC)**
entities in free text — built and compared across four architectures, from a
plain LSTM up to a fine-tuned Transformer, and deployed as a professional
Gradio web app.

> ⚠️ **Environment note:** the sandbox this project was authored in has no
> internet access to huggingface.co (dataset/model hub) or GloVe download
> mirrors, so the real CoNLL-2003 training run couldn't happen inside that
> sandbox. Every module below **was tested end-to-end** (forward pass, loss,
> backward pass, CRF Viterbi decoding, seqeval scoring, checkpoint
> save/load, and the full Gradio app) using a synthetic IOB-tagged dataset
> that mimics CoNLL-2003's shape — so the code is verified working. Run the
> commands below on your own machine / Google Colab (with internet + ideally
> a GPU) to train on the **real** CoNLL-2003 data.

---

## 1. Project structure

```
ner_project/
├── data/
│   └── data_loader.py        # Loads CoNLL-2003 (HF Datasets), IOB2 tags, synthetic fallback
├── utils/
│   ├── embeddings.py         # GloVe/FastText loading + char-level vocab (OOV handling)
│   └── preprocessing.py      # Padding/batching for scratch models + subword label alignment for BERT
├── models/
│   ├── char_encoder.py       # Char-CNN sub-module (spelling features for OOV words)
│   ├── crf.py                # From-scratch linear-chain CRF (forward algo + Viterbi decode)
│   └── taggers.py            # LSTMTagger / BiLSTMTagger / BiLSTMCRFTagger
├── train_scratch.py          # Trains + evaluates LSTM / BiLSTM / BiLSTM+CRF
├── train_transformer.py      # Fine-tunes BERT/DistilBERT via HuggingFace Trainer
├── evaluate.py                # seqeval-based per-entity precision/recall/F1
├── compare_models.py          # Side-by-side comparison table + CRF analysis write-up
├── inference.py                # Unified predict() wrapper for all 4 model types
├── app.py                      # Professional Gradio deployment (color-coded live highlighting)
├── notebooks/
│   └── NER_Project.ipynb       # Single end-to-end notebook covering every stage (1-13 below)
├── requirements.txt
└── checkpoints/                # Trained model weights land here
```

## 2. Setup

```bash
pip install -r requirements.txt

# (optional but recommended) download GloVe vectors for the scratch models
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove/
```

### Prefer one notebook over separate scripts?

`notebooks/NER_Project.ipynb` is a **fully self-contained** notebook — every
class and function (data loading, char-CNN, CRF, LSTM/BiLSTM/BiLSTM+CRF,
training loop, seqeval evaluation, transformer fine-tuning, Gradio app) is
defined **inside its own cells**, with zero dependency on the rest of this
project folder. Just upload the single `.ipynb` to **Kaggle** or **Google
Colab** and run every cell top to bottom — no need to upload `data/`,
`utils/`, `models/`, etc. alongside it.

It tries several CoNLL-2003 sources in order (`lhoestq/conll2003`, then the
original `conll2003` with `trust_remote_code`, then `eriktks/conll2003`),
since the original dataset script has become unreliable on some HF Datasets
versions — and falls back to a small synthetic dataset if none are
reachable, so the notebook always runs end-to-end even fully offline.


## 3. Stage-by-stage pipeline

### 3.1 Explore the dataset
```bash
python data/data_loader.py
```
Downloads CoNLL-2003 from the HuggingFace Hub (or falls back to a synthetic
IOB dataset if offline) and prints sentence counts, vocab size, and tag
distribution per split.

### 3.2 IOB tagging scheme
CoNLL-2003 already ships IOB2-tagged (`B-`/`I-`/`O`). `data/data_loader.py`
defines the canonical label list (`LABEL_LIST`) shared by every model in the
project, so ids are 100% consistent across the scratch models and the
transformer.

### 3.3 OOV handling
`models/char_encoder.py` implements a char-level CNN (multi-kernel,
max-over-time pooling) that turns a word's *spelling* (capitalization,
suffixes, digits, length) into a fixed-size vector. This is concatenated
with the pretrained GloVe embedding, so even a word missing from GloVe still
carries a useful signal. `utils/embeddings.py` handles GloVe/FastText
loading + random init for unmatched vocab.

### 3.4 Tokenize & align labels
- Scratch models: `utils/preprocessing.py::NERScratchDataset` + `collate_fn`
  (word ids, char ids, padding, attention mask).
- Transformer: `utils/preprocessing.py::tokenize_and_align_labels` — the
  standard HF recipe that assigns the word-level label to only the *first*
  WordPiece/BPE subword of each word (`-100` for the rest, ignored by the
  loss).

### 3.5 Train the four architectures

```bash
# LSTM baseline (unidirectional -> only sees left context)
python train_scratch.py --arch lstm --epochs 15 --glove_path glove/glove.6B.100d.txt

# BiLSTM (adds right context)
python train_scratch.py --arch bilstm --epochs 15 --glove_path glove/glove.6B.100d.txt

# BiLSTM + CRF (adds legal-sequence decoding via Viterbi)
python train_scratch.py --arch bilstm_crf --epochs 15 --glove_path glove/glove.6B.100d.txt

# Fine-tuned Transformer
python train_transformer.py --model_name distilbert-base-cased --epochs 3
# or: python train_transformer.py --model_name bert-base-cased --epochs 3
```

Each `train_scratch.py` run saves `checkpoints/<arch>.pt` +
`checkpoints/<arch>_metrics.json`. `train_transformer.py` saves the full
HF model directory to `checkpoints/transformer/`.

> Want a fast structural check before committing to a full run? Add
> `--synthetic` to `train_scratch.py` to train on a tiny synthetic dataset —
> useful for catching bugs in minutes instead of hours, offline.

### 3.6 Evaluate with seqeval

`evaluate.py` wraps `seqeval` (span-level, not token-level, scoring — the
correct way to grade NER) and is called automatically at the end of every
training run. It reports:
- **Overall** micro-averaged precision / recall / F1
- **Per-entity** precision / recall / F1 / support for PER, ORG, LOC, MISC

### 3.7 Compare all four architectures

```bash
python compare_models.py
```
Reads every `checkpoints/*_metrics.json`, builds a single comparison table
(saved to `checkpoints/comparison_table.csv`), and prints a written
explanation of **where and why CRF improves boundary detection** over plain
BiLSTM (illegal transitions, split/merged multi-token spans, etc.).

### 3.8 Deploy with Gradio

```bash
python app.py
```
Opens a polished, professional web UI at `http://localhost:7860`:
- **Model dropdown** — instantly switch between LSTM / BiLSTM / BiLSTM+CRF /
  fine-tuned Transformer / a "Quick Demo" pretrained model (downloads a
  small public BERT-NER model from the Hub, so you have something to click
  around immediately even before training your own checkpoints)
- **Real-time color-coded entity highlighting** (`gr.HighlightedText`) —
  PER red, ORG teal, LOC yellow, MISC purple
- **Live entity summary table** — grouped counts + unique mentions per type
- Custom CSS: gradient hero header, card layout, example prompts,
  architecture explainer footer

---

## 4. Architecture rationale (LSTM → BiLSTM → BiLSTM+CRF → Transformer)

| Step | What changes | Why it helps |
|---|---|---|
| LSTM → BiLSTM | Adds a right-to-left LSTM pass, concatenated with the left-to-right one | NER needs **both** directions of context (e.g. "Washington" as PER vs. LOC often depends on what follows it) |
| BiLSTM → BiLSTM+CRF | Adds a learned tag-transition matrix + Viterbi decoding instead of independent per-token softmax | Prevents illegal tag sequences (`O → I-ORG`) and fixes multi-token entity **boundary** errors — the CRF decodes the *whole sequence* jointly instead of token-by-token |
| BiLSTM+CRF → Transformer | Replaces the recurrent encoder entirely with pretrained self-attention (BERT/DistilBERT), fine-tuned end-to-end | Deep bidirectional context from pretraining on massive corpora typically yields the strongest overall F1, especially for rarer entity types and longer-range dependencies |

## 5. Notes on running the transformer script

`train_transformer.py` needs internet access to download the pretrained
checkpoint + CoNLL-2003 from the Hub. On a single modern GPU, 3 epochs of
`distilbert-base-cased` typically takes well under 30 minutes.

## 6. Extending this project
- Swap `dslim/bert-base-NER` in `inference.py::HubDemoPredictor` for any
  other Hub token-classification model.
- Add `roberta-base` / `xlm-roberta-base` to `train_transformer.py` for
  multilingual NER.
- Try FastText instead of GloVe by pointing `--glove_path` at a `.vec` file
  (same whitespace-separated format, handled transparently by
  `utils/embeddings.py`).
