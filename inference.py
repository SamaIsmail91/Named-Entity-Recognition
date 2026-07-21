"""
inference.py
=============
Unified inference wrapper so the Gradio app can call any of the four
trained architectures (LSTM, BiLSTM, BiLSTM+CRF, fine-tuned Transformer)
through one common `predict(text) -> List[(word, label)]` interface.

Also includes a "quick demo" fallback that pulls a public pretrained
NER model from the HuggingFace Hub (`dslim/bert-base-NER`) so the Gradio
app has something to show immediately, even before you've trained your
own checkpoints on CoNLL-2003.
"""

import os
import re
import torch
from typing import List, Tuple

from data.data_loader import LABEL_LIST, ID2LABEL
from utils.embeddings import WordVocab, CharVocab
from models.taggers import build_model

SIMPLE_TOKENIZE_RE = re.compile(r"\w+|[^\w\s]")


def simple_word_tokenize(text: str) -> List[str]:
    """Lightweight whitespace/punctuation tokenizer matching CoNLL-2003 style."""
    return SIMPLE_TOKENIZE_RE.findall(text)


class ScratchModelPredictor:
    """Loads a checkpoint saved by train_scratch.py and runs inference."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.device = device
        self.arch = ckpt["arch"]
        self.label_list = ckpt["label_list"]

        # Rebuild vocabs from the saved itos lists.
        self.word_vocab = WordVocab([[]])  # placeholder, overwritten below
        self.word_vocab.itos = ckpt["word_vocab_itos"]
        self.word_vocab.stoi = {w: i for i, w in enumerate(self.word_vocab.itos)}

        self.char_vocab = CharVocab([[]])
        self.char_vocab.itos = ckpt["char_vocab_itos"]
        self.char_vocab.stoi = {c: i for i, c in enumerate(self.char_vocab.itos)}

        self.model = build_model(
            self.arch,
            vocab_size=len(self.word_vocab), num_tags=len(self.label_list),
            char_vocab_size=len(self.char_vocab),
            word_emb_dim=ckpt["word_emb_dim"], hidden_dim=ckpt["hidden_dim"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def predict(self, text: str) -> List[Tuple[str, str]]:
        tokens = simple_word_tokenize(text)
        if not tokens:
            return []
        word_ids = torch.tensor([self.word_vocab.encode(tokens)], dtype=torch.long).to(self.device)
        char_ids = torch.tensor([self.char_vocab.encode_sentence(tokens)], dtype=torch.long).to(self.device)
        mask = torch.ones(1, len(tokens), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            emissions = self.model(word_ids, char_ids, mask)
            pred_ids = self.model.predict(emissions, mask)[0]

        return list(zip(tokens, [ID2LABEL[i] for i in pred_ids]))


class TransformerPredictor:
    """Loads a fine-tuned HuggingFace token-classification checkpoint."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.device = device

    def predict(self, text: str) -> List[Tuple[str, str]]:
        tokens = simple_word_tokenize(text)
        if not tokens:
            return []
        enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                              truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        pred_ids = logits.argmax(-1)[0].tolist()
        word_ids = enc.word_ids(batch_index=0)

        results, seen = [], set()
        id2label = self.model.config.id2label
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            results.append((tokens[wid], id2label[pid]))
        return results


class HubDemoPredictor:
    """Quick-start fallback: a public pretrained NER model from the HF Hub."""

    def __init__(self, model_name: str = "dslim/bert-base-NER", device: str = "cpu"):
        from transformers import pipeline
        self.pipe = pipeline("ner", model=model_name, aggregation_strategy=None, device=-1)

    def predict(self, text: str) -> List[Tuple[str, str]]:
        tokens = simple_word_tokenize(text)
        results = self.pipe(text)
        # Map char-span predictions back onto our simple word tokens.
        tags = ["O"] * len(tokens)
        offset = 0
        spans = []
        pos = 0
        for tok in tokens:
            start = text.index(tok, pos)
            end = start + len(tok)
            spans.append((start, end))
            pos = end
        for r in results:
            for i, (s, e) in enumerate(spans):
                if s >= r["start"] and e <= r["end"]:
                    label = r["entity"]
                    tags[i] = label if label.startswith(("B-", "I-")) else f"B-{label}"
        return list(zip(tokens, tags))


def iob_to_spans(tagged: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Converts [(token, IOB_tag), ...] into HighlightedText-friendly (text, label) chunks,
    merging consecutive I-/B- tokens of the same entity type into one span, and using
    label=None for 'O' tokens (so Gradio doesn't color them)."""
    chunks = []
    cur_words, cur_type = [], None
    for word, tag in tagged:
        if tag == "O":
            if cur_words:
                chunks.append((" ".join(cur_words), cur_type))
                cur_words, cur_type = [], None
            chunks.append((word, None))
        else:
            etype = tag.split("-")[-1]
            is_begin = tag.startswith("B-") or etype != cur_type
            if is_begin and cur_words:
                chunks.append((" ".join(cur_words), cur_type))
                cur_words = []
            cur_words.append(word)
            cur_type = etype
    if cur_words:
        chunks.append((" ".join(cur_words), cur_type))
    return chunks
