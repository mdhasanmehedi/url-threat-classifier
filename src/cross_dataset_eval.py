"""
cross_dataset_eval.py — PhishFormer
Cross-dataset generalisation test using PhiUSIIL dataset.

Takes the trained PhishFormer checkpoint (seed=42, no retraining)
and evaluates it on PhiUSIIL URLs, mapping binary labels to
PhishFormer's 4-class scheme (phishing→2, legitimate→0).

Reports accuracy, F1, and confidence drop vs in-distribution performance.

Usage:
  python3 src/cross_dataset_eval.py
  python3 src/cross_dataset_eval.py --csv_path data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint, device_info
from data import tokenize_url, MAX_LEN, PAD_IDX, LABEL2IDX, IDX2LABEL
from models import PhishFormer

logger = get_logger()
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Label mapping ─────────────────────────────────────────────────────────────
# PhiUSIIL uses binary labels: 1=phishing, 0=legitimate
# We map these to PhishFormer's 4-class scheme:
#   0 (legitimate) → 0 (benign)
#   1 (phishing)   → 2 (phishing)
# We then evaluate only on these two classes.

PHIUSIIL_MAP = {
    0: LABEL2IDX["benign"],    # legitimate → benign (class 0)
    1: LABEL2IDX["phishing"],  # phishing   → phishing (class 2)
}

EVAL_CLASSES = [LABEL2IDX["benign"], LABEL2IDX["phishing"]]  # [0, 2]
EVAL_NAMES   = ["benign", "phishing"]


def load_phiusiil(csv_path: str):
    """
    Load PhiUSIIL dataset and detect its column structure.
    Handles multiple known column naming conventions.
    Returns (urls, binary_labels) where binary_labels are 0/1.
    """
    logger.info(f"Loading PhiUSIIL from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Columns found: {list(df.columns)}")
    logger.info(f"Shape: {df.shape}")

    # ── Detect URL column ─────────────────────────────────────────────────────
    url_col = None
    for candidate in ["url", "URL", "FILENAME", "Domain", "domain"]:
        if candidate in df.columns:
            url_col = candidate
            break
    if url_col is None:
        # Try first string column
        for col in df.columns:
            if df[col].dtype == object:
                url_col = col
                logger.warning(f"URL column not found by name — using first string column: '{col}'")
                break
    assert url_col is not None, f"Cannot find URL column in {list(df.columns)}"
    logger.info(f"Using URL column: '{url_col}'")

    # ── Detect label column ───────────────────────────────────────────────────
    label_col = None
    for candidate in ["label", "Label", "CLASS_LABEL", "class", "phishing", "Phishing"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        # Use last column as label (common convention)
        label_col = df.columns[-1]
        logger.warning(f"Label column not found by name — using last column: '{label_col}'")
    logger.info(f"Using label column: '{label_col}'")

    # ── Clean and validate ────────────────────────────────────────────────────
    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]

    # Normalise labels to 0/1
    unique_labels = df["label"].unique()
    logger.info(f"Unique label values: {unique_labels}")

    # Handle string labels
    if df["label"].dtype == object:
        label_map = {}
        for v in unique_labels:
            v_lower = str(v).lower().strip()
            if v_lower in ["phishing", "1", "malicious", "bad", "yes"]:
                label_map[v] = 1
            else:
                label_map[v] = 0
        df["label"] = df["label"].map(label_map)
        logger.info(f"Mapped string labels: {label_map}")
    else:
        # Ensure binary 0/1
        df["label"] = df["label"].astype(int)
        # If labels are not 0/1, normalise
        if set(df["label"].unique()) - {0, 1}:
            min_l = df["label"].min()
            df["label"] = (df["label"] - min_l).clip(0, 1)

    df = df[df["label"].isin([0, 1])]

    urls   = df["url"].astype(str).tolist()
    labels = df["label"].tolist()

    logger.info(f"Total URLs after cleaning: {len(urls):,}")
    logger.info(f"  Legitimate (0): {labels.count(0):,} ({100*labels.count(0)/len(labels):.1f}%)")
    logger.info(f"  Phishing   (1): {labels.count(1):,} ({100*labels.count(1)/len(labels):.1f}%)")

    return urls, labels


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    urls: list,
    device: torch.device,
    batch_size: int = 512,
) -> tuple:
    """
    Run PhishFormer inference on a list of URLs.
    Returns (predicted_4class_labels, confidence_scores).
    """
    model.eval()
    all_preds   = []
    all_confs   = []

    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        tokens = torch.tensor(
            [tokenize_url(u, MAX_LEN) for u in batch_urls],
            dtype=torch.long
        ).to(device)

        logits = model(tokens)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        # Confidence = max probability for the predicted class
        confs = probs.max(dim=1).values

        all_preds.extend(preds.cpu().numpy().tolist())
        all_confs.extend(confs.cpu().numpy().tolist())

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Processed {min(i+batch_size, len(urls)):,}/{len(urls):,} URLs")

    return np.array(all_preds), np.array(all_confs)


def evaluate_cross_dataset(
    csv_path: str,
    ckpt_path: str = "checkpoints/phishformer_seed42_best.pt",
    seed: int = 42,
) -> dict:
    """
    Main cross-dataset evaluation function.
    """
    set_seed(seed)
    device = get_device()
    logger.info(f"Device: {device_info(device)}")

    # ── Load PhiUSIIL ─────────────────────────────────────────────────────────
    urls, binary_labels = load_phiusiil(csv_path)

    # Map binary labels to PhishFormer's 4-class scheme
    mapped_labels = [PHIUSIIL_MAP[l] for l in binary_labels]
    # mapped_labels contains 0 (benign) or 2 (phishing)

    # ── Load trained PhishFormer ──────────────────────────────────────────────
    model = PhishFormer().to(device)
    if os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path, device=device)
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.error(f"Checkpoint not found at {ckpt_path}")
        logger.error("Run train.py first to generate the checkpoint.")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    logger.info(f"Running inference on {len(urls):,} PhiUSIIL URLs...")
    preds_4class, confs = run_inference(model, urls, device)

    # ── Evaluate only on benign/phishing predictions ──────────────────────────
    # For URLs that the model predicts as defacement or malware,
    # we count as "malicious" (pooled with phishing for this binary evaluation)
    # BUT also report what fraction get predicted as defacement/malware
    # (out-of-distribution behaviour).

    true_labels = np.array(mapped_labels)  # 0 or 2

    # Strict evaluation: only benign(0) vs phishing(2)
    # Treat any non-benign prediction as "malicious" for binary eval
    preds_binary = np.where(preds_4class == LABEL2IDX["benign"], 0, 1)
    true_binary  = np.where(true_labels  == LABEL2IDX["benign"], 0, 1)

    binary_acc = accuracy_score(true_binary, preds_binary)
    binary_f1  = f1_score(true_binary, preds_binary, average="macro", zero_division=0)
    binary_f1_per_class = f1_score(
        true_binary, preds_binary, average=None, zero_division=0
    ).tolist()

    binary_report = classification_report(
        true_binary, preds_binary,
        target_names=["legitimate", "phishing"],
        digits=4,
    )
    cm = confusion_matrix(true_binary, preds_binary)

    # Out-of-distribution predictions (model predicts defacement or malware
    # for URLs that are actually phishing or legitimate — these are
    # distribution-shift artefacts)
    ood_mask  = ~np.isin(preds_4class, [LABEL2IDX["benign"], LABEL2IDX["phishing"]])
    ood_count = ood_mask.sum()
    ood_pct   = 100 * ood_count / len(preds_4class)

    # Prediction distribution
    pred_dist = {
        IDX2LABEL[i]: int((preds_4class == i).sum())
        for i in range(4)
    }

    # Performance gap vs in-distribution
    # In-distribution (seed=42 test set): 98.06% acc, 97.28% macro-F1
    indist_acc = 0.9806294878961173
    indist_f1  = 0.9728166112328396
    acc_drop   = indist_acc - binary_acc
    f1_drop    = indist_f1  - binary_f1

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-DATASET EVALUATION RESULTS — PhiUSIIL")
    logger.info("=" * 60)
    logger.info(f"Dataset          : PhiUSIIL ({len(urls):,} URLs)")
    logger.info(f"Checkpoint       : {ckpt_path} (seed=42, no retraining)")
    logger.info(f"")
    logger.info(f"Binary accuracy  : {binary_acc:.4f} ({binary_acc*100:.2f}%)")
    logger.info(f"Binary macro-F1  : {binary_f1:.4f} ({binary_f1*100:.2f}%)")
    logger.info(f"  Legitimate F1  : {binary_f1_per_class[0]:.4f}")
    logger.info(f"  Phishing F1    : {binary_f1_per_class[1]:.4f}")
    logger.info(f"")
    logger.info(f"In-distribution (Malicious-Phish test set, seed=42):")
    logger.info(f"  Accuracy       : {indist_acc:.4f} ({indist_acc*100:.2f}%)")
    logger.info(f"  Macro-F1       : {indist_f1:.4f} ({indist_f1*100:.2f}%)")
    logger.info(f"")
    logger.info(f"Performance drop vs in-distribution:")
    logger.info(f"  Accuracy drop  : {acc_drop:.4f} ({acc_drop*100:.2f} pp)")
    logger.info(f"  Macro-F1 drop  : {f1_drop:.4f} ({f1_drop*100:.2f} pp)")
    logger.info(f"")
    logger.info(f"Out-of-distribution predictions (defacement/malware):")
    logger.info(f"  Count          : {ood_count:,} ({ood_pct:.1f}%)")
    logger.info(f"")
    logger.info(f"Full prediction distribution on PhiUSIIL:")
    for cls, count in pred_dist.items():
        logger.info(f"  {cls:>12s}: {count:>7,} ({100*count/len(preds_4class):.1f}%)")
    logger.info(f"")
    logger.info(f"Confusion Matrix (legitimate / phishing):")
    logger.info(f"\n{cm}")
    logger.info(f"\nClassification Report:\n{binary_report}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "dataset":              "PhiUSIIL",
        "n_urls":               len(urls),
        "checkpoint":           ckpt_path,
        "binary_accuracy":      float(binary_acc),
        "binary_macro_f1":      float(binary_f1),
        "per_class_f1": {
            "legitimate": float(binary_f1_per_class[0]),
            "phishing":   float(binary_f1_per_class[1]),
        },
        "indist_accuracy":      indist_acc,
        "indist_macro_f1":      indist_f1,
        "accuracy_drop_pp":     float(acc_drop * 100),
        "f1_drop_pp":           float(f1_drop * 100),
        "ood_predictions":      int(ood_count),
        "ood_pct":              float(ood_pct),
        "prediction_dist":      pred_dist,
        "confusion_matrix":     cm.tolist(),
        "classification_report": binary_report,
    }

    path = os.path.join(RESULTS_DIR, "cross_dataset_phiusiil.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results saved to {path}")

    # ── Paper-ready summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PAPER-READY SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"PhishFormer (trained on Malicious-Phish, evaluated on PhiUSIIL "
        f"without retraining): {binary_acc*100:.2f}% accuracy, "
        f"{binary_f1*100:.2f}% macro-F1. "
        f"Performance drop vs in-distribution: "
        f"{acc_drop*100:.2f} pp accuracy, {f1_drop*100:.2f} pp macro-F1. "
        f"Out-of-distribution class predictions (defacement/malware): "
        f"{ood_pct:.1f}% of PhiUSIIL URLs."
    )

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str,
        default="data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
        help="Path to PhiUSIIL CSV file"
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default="checkpoints/phishformer_seed42_best.pt",
        help="Path to trained PhishFormer checkpoint"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_cross_dataset(
        csv_path=args.csv_path,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
    )
