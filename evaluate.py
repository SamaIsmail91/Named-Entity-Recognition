"""
evaluate.py
===========
Shared evaluation utilities based on `seqeval`, which is the standard
library for sequence-labeling evaluation because it scores whole entity
SPANS, not individual tokens (e.g. if a 3-token ORG entity has 2 tokens
tagged correctly and 1 wrong, span-level scoring correctly counts this as
ONE wrong prediction, whereas naive token accuracy would over-credit it).

Produces:
  - overall precision / recall / F1 (micro-averaged across all entity types)
  - per-entity-type precision / recall / F1 for PER, ORG, LOC, MISC
  - a comparison table across all four trained architectures
"""

from typing import List, Dict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


def evaluate_predictions(true_labels: List[List[str]], pred_labels: List[List[str]]) -> Dict:
    """
    true_labels / pred_labels: list of sentences, each a list of IOB2 string
    tags, e.g. [["O", "B-PER", "I-PER"], ["O", "O", "B-LOC"], ...]
    """
    report = classification_report(true_labels, pred_labels, scheme=IOB2, digits=4, output_dict=True)
    overall = {
        "precision": precision_score(true_labels, pred_labels, scheme=IOB2),
        "recall": recall_score(true_labels, pred_labels, scheme=IOB2),
        "f1": f1_score(true_labels, pred_labels, scheme=IOB2),
    }
    per_entity = {
        etype: {
            "precision": float(report.get(etype, {}).get("precision", 0.0)),
            "recall": float(report.get(etype, {}).get("recall", 0.0)),
            "f1": float(report.get(etype, {}).get("f1-score", 0.0)),
            "support": int(report.get(etype, {}).get("support", 0)),
        }
        for etype in ["PER", "ORG", "LOC", "MISC"]
    }
    overall = {k: float(v) for k, v in overall.items()}
    return {"overall": overall, "per_entity": per_entity}


def print_evaluation(results: Dict, model_name: str = ""):
    print(f"\n{'=' * 60}\nEvaluation: {model_name}\n{'=' * 60}")
    o = results["overall"]
    print(f"Overall  -> Precision: {o['precision']:.4f}  Recall: {o['recall']:.4f}  F1: {o['f1']:.4f}")
    print(f"{'-' * 60}")
    print(f"{'Entity':<8}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    for etype, m in results["per_entity"].items():
        print(f"{etype:<8}{m['precision']:>12.4f}{m['recall']:>12.4f}{m['f1']:>12.4f}{m['support']:>12}")


def comparison_table(all_results: Dict[str, Dict]) -> "pandas.DataFrame":
    """
    all_results: {"LSTM": eval_dict, "BiLSTM": eval_dict, "BiLSTM+CRF": eval_dict,
                  "Transformer": eval_dict}
    Returns a pandas DataFrame comparing overall + per-entity F1 across models.
    """
    import pandas as pd
    rows = []
    for model_name, res in all_results.items():
        row = {"Model": model_name,
               "Overall_P": res["overall"]["precision"],
               "Overall_R": res["overall"]["recall"],
               "Overall_F1": res["overall"]["f1"]}
        for etype, m in res["per_entity"].items():
            row[f"{etype}_F1"] = m["f1"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.round(4)
