"""
train_scratch.py
=================
Trains and evaluates the three "from scratch" architectures (LSTM, BiLSTM,
BiLSTM+CRF) on CoNLL-2003, using pretrained GloVe word embeddings + a
char-CNN for OOV handling, and reports seqeval per-entity metrics.

Usage:
    python train_scratch.py --arch bilstm_crf --epochs 15 --glove_path glove/glove.6B.100d.txt
    python train_scratch.py --arch lstm --epochs 10
    python train_scratch.py --arch bilstm --epochs 10

    # quick structural smoke-test with synthetic data (no internet needed):
    python train_scratch.py --arch bilstm_crf --synthetic --epochs 2
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import load_conll2003, make_synthetic_dataset, LABEL_LIST, ID2LABEL
from utils.embeddings import WordVocab, CharVocab, load_pretrained_embeddings
from utils.preprocessing import NERScratchDataset, collate_fn
from models.taggers import build_model
from evaluate import evaluate_predictions, print_evaluation


def ids_to_labels(id_lists):
    return [[ID2LABEL[i] for i in seq] for seq in id_lists]


def run_epoch(model, loader, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_gold = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False):
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            emissions = model(word_ids, char_ids, mask)
            loss = model.loss(emissions, labels, mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()

            preds = model.predict(emissions, mask)
            for b in range(labels.shape[0]):
                L = mask[b].sum().item()
                gold_seq = labels[b, :L].tolist()
                all_gold.append(gold_seq)
                all_preds.append(preds[b][:L])

    return total_loss / len(loader), all_gold, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["lstm", "bilstm", "bilstm_crf"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--word_emb_dim", type=int, default=100)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--glove_path", type=str, default=None,
                     help="Path to glove.6B.100d.txt (download separately)")
    ap.add_argument("--synthetic", action="store_true",
                     help="Use a small synthetic dataset instead of real CoNLL-2003 (offline smoke-test)")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load data -------------------------------------------------- #
    if args.synthetic:
        print("[data] Using SYNTHETIC dataset (structural smoke-test only).")
        raw = make_synthetic_dataset(n_train=300, n_val=60, n_test=60)
    else:
        print("[data] Loading real CoNLL-2003 from HuggingFace Hub...")
        ds = load_conll2003()
        raw = {split: list(ds[split]) for split in ["train", "validation", "test"]}

    train_sentences = [ex["tokens"] for ex in raw["train"]]
    word_vocab = WordVocab(train_sentences)
    char_vocab = CharVocab(train_sentences)
    emb_matrix = load_pretrained_embeddings(word_vocab, args.glove_path, args.word_emb_dim)

    train_ds = NERScratchDataset(raw["train"], word_vocab, char_vocab)
    val_ds = NERScratchDataset(raw["validation"], word_vocab, char_vocab)
    test_ds = NERScratchDataset(raw["test"], word_vocab, char_vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Build model -------------------------------------------------- #
    model = build_model(
        args.arch,
        vocab_size=len(word_vocab), num_tags=len(LABEL_LIST), char_vocab_size=len(char_vocab),
        word_emb_dim=args.word_emb_dim, hidden_dim=args.hidden_dim,
        pretrained_embeddings=emb_matrix,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[model] {args.arch} | params: {sum(p.numel() for p in model.parameters()):,}")

    best_f1, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_gold, val_preds = run_epoch(model, val_loader, None, device)

        gold_lbls = ids_to_labels(val_gold)
        pred_lbls = ids_to_labels(val_preds)
        val_metrics = evaluate_predictions(gold_lbls, pred_lbls)
        f1 = val_metrics["overall"]["f1"]

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final test evaluation ---------------------------------------- #
    _, test_gold, test_preds = run_epoch(model, test_loader, None, device)
    test_metrics = evaluate_predictions(ids_to_labels(test_gold), ids_to_labels(test_preds))
    print_evaluation(test_metrics, model_name=args.arch.upper())

    # ---- Save checkpoint + vocabs + metrics ---------------------------- #
    ckpt_path = os.path.join(args.out_dir, f"{args.arch}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "arch": args.arch,
        "word_emb_dim": args.word_emb_dim,
        "hidden_dim": args.hidden_dim,
        "word_vocab_itos": word_vocab.itos,
        "char_vocab_itos": char_vocab.itos,
        "label_list": LABEL_LIST,
    }, ckpt_path)
    with open(os.path.join(args.out_dir, f"{args.arch}_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nSaved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
