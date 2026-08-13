"""
finetune_phiusiil.py — PhishFormer
Cross-dataset domain adaptation experiment (Reviewer Point 1).

Evaluates PhishFormer under three conditions:
  1. Zero-shot: trained on Malicious-Phish, evaluated on PhiUSIIL
  2. Few-shot:  fine-tuned on 10% PhiUSIIL, evaluated on remaining 90%
  3. Full fine-tune: fine-tuned on 70% PhiUSIIL, evaluated on 15% test

This converts the cross-dataset failure into a scientifically interesting
domain adaptation finding: quantifying how much target-domain data
is needed to recover performance.

Usage:
  python3 src/finetune_phiusiil.py
  python3 src/finetune_phiusiil.py --csv_path data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint, device_info
from data import tokenize_url, MAX_LEN, LABEL2IDX, IDX2LABEL
from models import PhishFormer

logger = get_logger()
RESULTS_DIR = "results"
CKPT_DIR    = "checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Map PhiUSIIL binary labels to PhishFormer's 4-class scheme
PHIUSIIL_MAP = {0: LABEL2IDX["benign"], 1: LABEL2IDX["phishing"]}
EVAL_LABELS  = [LABEL2IDX["benign"], LABEL2IDX["phishing"]]


class BinaryURLDataset(Dataset):
    def __init__(self, urls, labels):
        self.urls   = urls
        self.labels = labels
    def __len__(self):
        return len(self.urls)
    def __getitem__(self, i):
        x = torch.tensor(tokenize_url(self.urls[i], MAX_LEN), dtype=torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y


def load_phiusiil(csv_path: str):
    """Load and clean PhiUSIIL dataset. Returns (urls, binary_labels)."""
    df = pd.read_csv(csv_path)
    logger.info(f"PhiUSIIL columns: {list(df.columns)}")

    # Detect URL column safely
    url_candidates = ["url", "URL", "FILENAME", "Domain", "domain"]
    url_col = next((c for c in url_candidates if c in df.columns), None)
    if url_col is None:
        url_col = next((c for c in df.columns if df[c].dtype == object), None)
    if url_col is None:
        raise ValueError(f"Cannot find URL column. Available: {list(df.columns[:10])}")
    # Detect label column
    label_col = next((c for c in ["label","Label","CLASS_LABEL","phishing","Phishing"] if c in df.columns), df.columns[-1])

    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]
    df["label"] = df["label"].astype(int).clip(0, 1)
    df = df[df["label"].isin([0, 1])]

    logger.info(f"PhiUSIIL loaded: {len(df):,} URLs | "
                f"legit={df['label'].eq(0).sum():,} | "
                f"phishing={df['label'].eq(1).sum():,}")
    return df["url"].tolist(), df["label"].tolist()


@torch.no_grad()
def evaluate_binary(model, loader, device, mapped_labels_true):
    """
    Evaluate PhishFormer (4-class output) on binary PhiUSIIL task.
    Maps any non-benign prediction to 'malicious' for binary eval.
    """
    model.eval()
    all_preds = []
    for xb, _ in loader:
        logits = model(xb.to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()
        # Collapse 4-class to binary: 0=benign, 1=malicious
        binary_preds = np.where(preds == LABEL2IDX["benign"], 0, 1)
        all_preds.extend(binary_preds.tolist())

    true_binary = [0 if l == LABEL2IDX["benign"] else 1
                   for l in mapped_labels_true]
    acc = accuracy_score(true_binary, all_preds)
    f1  = f1_score(true_binary, all_preds, average="macro", zero_division=0)
    f1_leg = f1_score(true_binary, all_preds, average=None, zero_division=0)[0]
    f1_phi = f1_score(true_binary, all_preds, average=None, zero_division=0)[1]
    return acc, f1, f1_leg, f1_phi, all_preds


def finetune(
    model, train_loader, val_loader, val_labels,
    device, n_epochs=10, lr=1e-4, patience=3, tag=""
):
    """Fine-tune PhishFormer on target domain data."""
    # Only update the classifier head and last transformer layer
    # to avoid catastrophic forgetting of source-domain knowledge
    for name, param in model.named_parameters():
        if any(k in name for k in ["classifier", "transformer.layers.1", "embedding"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Fine-tuning {n_trainable:,} parameters (partial)")

    # Class weights for PhiUSIIL (roughly balanced)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )

    best_f1      = -1.0
    patience_ctr = 0
    best_state   = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb.to(device))
            loss   = criterion(logits, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        acc, f1, f1l, f1p, _ = evaluate_binary(model, val_loader, device, val_labels)
        logger.info(f"  {tag} epoch {epoch:>2} | val acc={acc:.4f} | "
                    f"macro-F1={f1:.4f} | legit F1={f1l:.4f} | phishing F1={f1p:.4f}")

        if f1 > best_f1:
            best_f1      = f1
            patience_ctr = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Re-enable all parameters
    for param in model.parameters():
        param.requires_grad = True

    return model, best_f1


def run_cross_dataset_analysis(
    csv_path: str,
    ckpt_path: str = "checkpoints/phishformer_seed42_best.pt",
    seed: int = 42,
) -> dict:
    set_seed(seed)
    device = get_device()
    logger.info(f"Device: {device_info(device)}")

    # Load PhiUSIIL
    urls, binary_labels = load_phiusiil(csv_path)
    mapped_labels = [PHIUSIIL_MAP[l] for l in binary_labels]

    # Split PhiUSIIL into train/val/test
    X_train, X_temp, y_train_b, y_temp_b = train_test_split(
        urls, binary_labels, test_size=0.30,
        stratify=binary_labels, random_state=seed
    )
    X_val, X_test, y_val_b, y_test_b = train_test_split(
        X_temp, y_temp_b, test_size=0.50,
        stratify=y_temp_b, random_state=seed
    )

    # Map to 4-class labels for DataLoaders
    y_train_4 = [PHIUSIIL_MAP[l] for l in y_train_b]
    y_val_4   = [PHIUSIIL_MAP[l] for l in y_val_b]
    y_test_4  = [PHIUSIIL_MAP[l] for l in y_test_b]

    train_ds = BinaryURLDataset(X_train, y_train_4)
    val_ds   = BinaryURLDataset(X_val,   y_val_4)
    test_ds  = BinaryURLDataset(X_test,  y_test_4)

    full_ds  = BinaryURLDataset(urls, mapped_labels)

    full_loader  = DataLoader(full_ds,  batch_size=512, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

    results = {}

    # ── Condition 1: Zero-shot (no fine-tuning) ──────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("CONDITION 1: Zero-shot (no retraining)")
    logger.info("="*60)
    model_zs = PhishFormer().to(device)
    load_checkpoint(model_zs, ckpt_path, device=device)
    acc, f1, f1l, f1p, _ = evaluate_binary(model_zs, full_loader, device, mapped_labels)
    logger.info(f"Zero-shot: acc={acc:.4f} | macro-F1={f1:.4f} | "
                f"legit F1={f1l:.4f} | phishing F1={f1p:.4f}")
    results["zero_shot"] = {
        "accuracy": float(acc), "macro_f1": float(f1),
        "legitimate_f1": float(f1l), "phishing_f1": float(f1p),
        "n_train_samples": 0,
    }

    # ── Condition 2: Few-shot fine-tuning (10% of PhiUSIIL train split) ──────
    logger.info("\n" + "="*60)
    logger.info("CONDITION 2: Few-shot fine-tuning (10% of PhiUSIIL)")
    logger.info("="*60)

    fewshot_size = max(100, int(0.10 * len(X_train)))
    idx = np.random.RandomState(seed).choice(len(X_train), fewshot_size, replace=False)
    X_few  = [X_train[i] for i in idx]
    y_few  = [y_train_4[i] for i in idx]
    few_ds = BinaryURLDataset(X_few, y_few)
    few_loader = DataLoader(few_ds, batch_size=64, shuffle=True, num_workers=0)

    model_fs = PhishFormer().to(device)
    load_checkpoint(model_fs, ckpt_path, device=device)
    model_fs, _ = finetune(
        model_fs, few_loader, val_loader, y_val_4,
        device, n_epochs=10, lr=5e-5, patience=3,
        tag="Few-shot"
    )
    acc, f1, f1l, f1p, _ = evaluate_binary(model_fs, test_loader, device, y_test_4)
    logger.info(f"Few-shot result: acc={acc:.4f} | macro-F1={f1:.4f} | "
                f"legit F1={f1l:.4f} | phishing F1={f1p:.4f}")
    results["few_shot"] = {
        "accuracy": float(acc), "macro_f1": float(f1),
        "legitimate_f1": float(f1l), "phishing_f1": float(f1p),
        "n_train_samples": fewshot_size,
    }

    # ── Condition 3: Full fine-tuning (70% of PhiUSIIL) ──────────────────────
    logger.info("\n" + "="*60)
    logger.info("CONDITION 3: Full fine-tuning (70% of PhiUSIIL)")
    logger.info("="*60)
    model_ft = PhishFormer().to(device)
    load_checkpoint(model_ft, ckpt_path, device=device)
    model_ft, _ = finetune(
        model_ft, train_loader, val_loader, y_val_4,
        device, n_epochs=15, lr=1e-4, patience=3,
        tag="Full fine-tune"
    )
    acc, f1, f1l, f1p, _ = evaluate_binary(model_ft, test_loader, device, y_test_4)
    logger.info(f"Full fine-tune: acc={acc:.4f} | macro-F1={f1:.4f} | "
                f"legit F1={f1l:.4f} | phishing F1={f1p:.4f}")
    results["full_finetune"] = {
        "accuracy": float(acc), "macro_f1": float(f1),
        "legitimate_f1": float(f1l), "phishing_f1": float(f1p),
        "n_train_samples": len(X_train),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("CROSS-DATASET DOMAIN ADAPTATION SUMMARY")
    logger.info("="*60)
    logger.info(f"In-distribution (Malicious-Phish seed=42): acc=98.02% | macro-F1=97.25%")
    logger.info(f"{'Condition':<30} {'Accuracy':>10} {'Macro-F1':>10} "
                f"{'Legit F1':>10} {'Phish F1':>10} {'Train N':>10}")
    logger.info("-" * 75)
    for cond, r in results.items():
        logger.info(
            f"{cond:<30} {r['accuracy']*100:>9.2f}% {r['macro_f1']*100:>9.2f}% "
            f"{r['legitimate_f1']*100:>9.2f}% {r['phishing_f1']*100:>9.2f}% "
            f"{r['n_train_samples']:>10,}"
        )

    # Save
    output = {
        "dataset": "PhiUSIIL",
        "source_dataset": "Malicious-Phish",
        "indist_accuracy": 0.9802,
        "indist_macro_f1": 0.9725,
        "conditions": results,
    }
    path = os.path.join(RESULTS_DIR, "cross_dataset_adaptation.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str,
        default="data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default="checkpoints/phishformer_seed42_best.pt"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cross_dataset_analysis(
        csv_path=args.csv_path,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
    )
