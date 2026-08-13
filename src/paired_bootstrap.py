"""
paired_bootstrap.py -- PhishFormer
Computes paired-bootstrap 95% confidence intervals for the macro-F1
difference between PhishFormer and each baseline (CNN-only, LSTM, BiLSTM),
using the seed=42 checkpoints and test split -- matching how Table 7 and
Section 6.1 already anchor single-checkpoint comparisons to seed=42.

Method: paired bootstrap. In each of N_BOOT iterations, the same set of
resampled test-set indices (drawn with replacement) is used to recompute
macro-F1 for BOTH models being compared. Because both models are scored
on the identical resampled instances each time, per-instance sampling
noise that both models share cancels out, giving a tighter CI than
resampling each model independently. The 95% CI is the [2.5, 97.5]
percentile of the N_BOOT differences (PhishFormer macro-F1 minus
baseline macro-F1).

Usage:
    python3 src/paired_bootstrap.py
"""

import os
import sys
import json

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint
from data import load_and_split, get_dataloaders, NUM_CLASSES
from models import get_model

logger = get_logger()

SEED = 42  # matches Table 7 / Section 6.1 anchor checkpoint
CKPT_DIR = "checkpoints"
N_BOOT = 10000
BASELINES = ["cnn", "lstm", "bilstm"]  # compared against PhishFormer
RESULTS_DIR = "results"


def get_predictions(model_name: str, seed: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Load a model's checkpoint and return (predictions, true_labels) on the test set."""
    set_seed(seed)
    train_ds, val_ds, test_ds, _ = load_and_split("data/raw/malicious_phish.csv", seed=seed)
    _, _, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=512)

    model = get_model(model_name).to(device)
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_seed{seed}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    load_checkpoint(model, ckpt_path, device=device)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(p)
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


def paired_bootstrap_ci(preds_a, preds_b, labels, n_boot=N_BOOT, seed=SEED):
    """
    Returns (mean_diff, ci_low, ci_high) for macro-F1(a) - macro-F1(b),
    using paired bootstrap resampling of the shared test set.
    """
    assert len(preds_a) == len(preds_b) == len(labels), "Prediction arrays must be aligned"
    n = len(labels)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        f1_a = f1_score(labels[idx], preds_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(labels[idx], preds_b[idx], average="macro", zero_division=0)
        diffs[i] = f1_a - f1_b

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(ci_low), float(ci_high)


if __name__ == "__main__":
    device = get_device()
    logger.info(f"Paired bootstrap (seed={SEED}, N_BOOT={N_BOOT}) on device: {device}")

    logger.info("Loading PhishFormer predictions...")
    pf_preds, pf_labels = get_predictions("phishformer", SEED, device)

    results = {}
    for baseline in BASELINES:
        logger.info(f"Loading {baseline} predictions...")
        b_preds, b_labels = get_predictions(baseline, SEED, device)
        assert np.array_equal(pf_labels, b_labels), (
            f"Label mismatch between PhishFormer and {baseline} -- "
            f"test sets are not aligned (check seed/split consistency)."
        )

        logger.info(f"Running {N_BOOT} paired bootstrap iterations: PhishFormer vs {baseline}...")
        mean_diff, ci_low, ci_high = paired_bootstrap_ci(pf_preds, b_preds, pf_labels)
        results[baseline] = {"mean_diff": mean_diff, "ci_95": [ci_low, ci_high]}
        logger.info(
            f"  PhishFormer vs {baseline}: mean diff={mean_diff*100:+.2f}pp, "
            f"95% CI=[{ci_low*100:+.2f}, {ci_high*100:+.2f}]pp, "
            f"{'RESOLVED (excludes 0)' if ci_low > 0 or ci_high < 0 else 'NOT resolved (includes 0)'}"
        )

    out_path = os.path.join(RESULTS_DIR, "paired_bootstrap_ci.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "n_boot": N_BOOT, "results": results}, f, indent=2)
    logger.info(f"Saved to {out_path}")
