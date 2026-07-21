"""
train_transformer.py
=====================
Fine-tunes a pretrained transformer (BERT / DistilBERT by default) for
token classification on CoNLL-2003 using HuggingFace `transformers` +
`Trainer`, and reports seqeval per-entity metrics.

Usage:
    python train_transformer.py --model_name distilbert-base-cased --epochs 3
    python train_transformer.py --model_name bert-base-cased --epochs 3

Requires internet access (to download the pretrained checkpoint + tokenizer
from the HuggingFace Hub, and the CoNLL-2003 dataset) and, ideally, a GPU.
"""

import argparse
import numpy as np

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from data.data_loader import load_conll2003, LABEL_LIST, ID2LABEL
from utils.preprocessing import tokenize_and_align_labels


def build_compute_metrics():
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            cur_true, cur_pred = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                cur_true.append(ID2LABEL[l])
                cur_pred.append(ID2LABEL[p])
            true_labels.append(cur_true)
            true_preds.append(cur_pred)

        report = classification_report(true_labels, true_preds, output_dict=True, digits=4)
        result = {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }
        for etype in ["PER", "ORG", "LOC", "MISC"]:
            if etype in report:
                result[f"{etype}_f1"] = report[etype]["f1-score"]
        return result

    return compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-cased",
                     help="e.g. bert-base-cased, distilbert-base-cased, roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", type=str, default="checkpoints/transformer")
    args = ap.parse_args()

    print(f"[data] Loading CoNLL-2003...")
    ds = load_conll2003()

    print(f"[model] Loading tokenizer + model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
    )

    tokenized_ds = DatasetDict({
        split: ds[split].map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=True,
            remove_columns=ds[split].column_names,
        )
        for split in ["train", "validation", "test"]
    })

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()

    print("\n[test] Final evaluation on held-out test set:")
    test_metrics = trainer.evaluate(tokenized_ds["test"])
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"\nSaved fine-tuned model -> {args.out_dir}")


if __name__ == "__main__":
    main()
