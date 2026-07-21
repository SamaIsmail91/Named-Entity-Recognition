"""
compare_models.py
==================
Loads the saved *_metrics.json files produced by train_scratch.py and
train_transformer.py and builds a single side-by-side comparison table
(overall + per-entity F1) across all four architectures, plus a short
written analysis of where the CRF layer helps.

Usage (after training all 4 models):
    python compare_models.py
"""

import os
import json
import pandas as pd

CKPT_DIR = "checkpoints"

MODEL_FILES = {
    "LSTM":            os.path.join(CKPT_DIR, "lstm_metrics.json"),
    "BiLSTM":          os.path.join(CKPT_DIR, "bilstm_metrics.json"),
    "BiLSTM + CRF":    os.path.join(CKPT_DIR, "bilstm_crf_metrics.json"),
    "Transformer":     os.path.join(CKPT_DIR, "transformer_metrics.json"),
}


def load_metrics():
    results = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"[skip] {name}: no metrics file at {path} (train it first)")
    return results


def build_table(results):
    rows = []
    for name, res in results.items():
        row = {"Model": name,
               "Precision": res["overall"]["precision"],
               "Recall": res["overall"]["recall"],
               "F1": res["overall"]["f1"]}
        for etype, m in res["per_entity"].items():
            row[f"{etype}_F1"] = m["f1"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model").round(4)


CRF_ANALYSIS = """
Where BiLSTM + CRF improves over plain BiLSTM
-----------------------------------------------
A plain BiLSTM classifier makes an INDEPENDENT softmax decision at every
token, so it has no way to enforce that tag sequences stay legal/consistent.
This leads to two common boundary errors:

  1. Illegal transitions, e.g. predicting "O" then jumping straight into
     "I-ORG" with no preceding "B-ORG" -- there was never a valid entity
     start, so it's unclear where the span begins.
  2. Split/merge errors on multi-word entities, e.g. tagging "New" as
     B-LOC but "York" as O, splitting a single two-word location into a
     partial span, or the reverse: merging two adjacent-but-separate
     entities into one span because nothing enforces a B- boundary between
     them.

The CRF layer adds a learned transition-score matrix between adjacent tags
and decodes the *whole sequence* jointly with the Viterbi algorithm (instead
of independent per-token argmax). Sequences like O -> I-ORG receive a large
negative transition score during training, so the model learns to strongly
prefer legal paths (O -> B-ORG -> I-ORG -> O). In practice this shows up as:
  - Fewer partial/incomplete entity spans (higher per-entity RECALL)
  - Fewer spurious single-token entities (higher per-entity PRECISION)
  - The gain is usually largest on multi-token entity types (ORG, MISC)
    since those have the most opportunities for a token-independent
    classifier to break a span midway.
"""


def main():
    results = load_metrics()
    if not results:
        print("No trained models found yet. Train at least one first, e.g.:\n"
              "  python train_scratch.py --arch lstm\n"
              "  python train_scratch.py --arch bilstm\n"
              "  python train_scratch.py --arch bilstm_crf\n"
              "  python train_transformer.py")
        return

    df = build_table(results)
    print("\n=== Architecture comparison (test set) ===\n")
    print(df.to_string())
    df.to_csv(os.path.join(CKPT_DIR, "comparison_table.csv"))
    print(f"\nSaved -> {CKPT_DIR}/comparison_table.csv")
    print(CRF_ANALYSIS)


if __name__ == "__main__":
    main()
