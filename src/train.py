"""
train.py — PhishFormer
Full training pipeline:
  - Weighted cross-entropy loss (handles class imbalance)
  - Early stopping on validation macro-F1
  - Checkpoint saving (best model only)
  - Per-epoch logging of loss, accuracy, macro-F1
  - Multi-seed support for statistical significance (Section 6.6)
  - Inference latency benchmarking (CPU + MPS) for Section 5.3

Usage:
  # Train PhishFormer with default settings
  python3 src/train.py --model phishformer

  # Train all models sequentially
  python3 src/train.py --model all

  # Train with multiple seeds for statistical significance
  python3 src/train.py --model phishformer --seeds 42 123 456

  # Run latency benchmark after training
  python3 src/train.py --model phishformer --benchmark
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, device_info, save_checkpoint, load_checkpoint
from data import load_and_split, get_dataloaders, IDX2LABEL, NUM_CLASSES
from models import get_model, RandomForestBaseline

logger = get_logger()

# ── Default hyperparameters (Section 4.7) ────────────────────────────────────
DEFAULTS = {
    "batch_size":    512,
    "lr":            1e-3,
    "epochs":        30,       # max epochs; early stopping will typically trigger earlier
    "patience":      5,        # early stopping patience on val macro-F1
    "weight_decay":  1e-4,
    "csv_path":      "data/raw/malicious_phish.csv",
    "checkpoint_dir":"checkpoints",
    "results_dir":   "results",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core training and evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """Run one full training epoch. Returns dict of loss and accuracy."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping — stabilises transformer training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataloader. Returns loss, accuracy, macro-F1."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    n = len(all_labels)
    return {
        "loss":          total_loss / n,
        "accuracy":      accuracy_score(all_labels, all_preds),
        "macro_f1":      f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class_f1":  f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        "preds":         all_preds,
        "labels":        all_labels,
    }


def train_model(
    model_name: str,
    seed: int = 42,
    cfg: dict = None,
    device: torch.device = None,
) -> dict:
    """
    Full training run for one model with one seed.
    Returns a results dict containing all metrics.
    """
    if cfg is None:
        cfg = DEFAULTS.copy()
    if device is None:
        device = get_device()

    set_seed(seed)
    logger.info(f"{'='*60}")
    logger.info(f"Model: {model_name.upper()} | Seed: {seed} | Device: {device_info(device)}")
    logger.info(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, class_weights = load_and_split(
        cfg["csv_path"], seed=seed
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=cfg["batch_size"]
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(model_name).to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    # ── Loss: weighted cross-entropy ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    # Cosine annealing: smoothly decays LR over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    # ── Early stopping setup ──────────────────────────────────────────────────
    best_val_f1  = -1.0
    patience_ctr = 0
    best_epoch   = 0
    ckpt_path    = os.path.join(
        cfg["checkpoint_dir"], f"{model_name}_seed{seed}_best.pt"
    )
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    history = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch:>2}/{cfg['epochs']} | "
            f"Train loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}  F1: {train_metrics['macro_f1']:.4f} | "
            f"Val   loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}  F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {lr_now:.2e}"
        )

        history.append({
            "epoch":         epoch,
            "train_loss":    train_metrics["loss"],
            "train_acc":     train_metrics["accuracy"],
            "train_f1":      train_metrics["macro_f1"],
            "val_loss":      val_metrics["loss"],
            "val_acc":       val_metrics["accuracy"],
            "val_f1":        val_metrics["macro_f1"],
        })

        # Save best checkpoint
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1  = val_metrics["macro_f1"]
            best_epoch   = epoch
            patience_ctr = 0
            save_checkpoint(model, optimizer, epoch, best_val_f1, ckpt_path)
            logger.info(f"  ✓ New best val macro-F1: {best_val_f1:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                logger.info(f"Early stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    logger.info(f"Best epoch: {best_epoch} | Best val macro-F1: {best_val_f1:.4f}")

    # ── Load best checkpoint and evaluate on test set ─────────────────────────
    load_checkpoint(model, ckpt_path, device=device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info(f"\n{'─'*50}")
    logger.info(f"TEST RESULTS — {model_name.upper()} (seed {seed})")
    logger.info(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro-F1 : {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Per-class F1:")
    for i, f1 in enumerate(test_metrics["per_class_f1"]):
        logger.info(f"    {IDX2LABEL[i]:>12s}: {f1:.4f}")

    # Full classification report
    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=[IDX2LABEL[i] for i in range(NUM_CLASSES)],
        digits=4,
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "model":         model_name,
        "seed":          seed,
        "best_epoch":    best_epoch,
        "best_val_f1":   best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "per_class_f1":  test_metrics["per_class_f1"],
        "confusion_matrix": cm.tolist(),
        "history":       history,
        "report":        report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest (sklearn — separate pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(seed: int = 42, cfg: dict = None) -> dict:
    """Train and evaluate the Random Forest baseline."""
    if cfg is None:
        cfg = DEFAULTS.copy()

    set_seed(seed)
    logger.info(f"{'='*60}")
    logger.info(f"Model: RANDOM FOREST | Seed: {seed}")
    logger.info(f"{'='*60}")

    train_ds, val_ds, test_ds, _ = load_and_split(cfg["csv_path"], seed=seed)

    # Extract raw URLs and labels from datasets
    train_urls   = [train_ds.urls[i] for i in range(len(train_ds))]
    train_labels = train_ds.labels
    test_urls    = [test_ds.urls[i]  for i in range(len(test_ds))]
    test_labels  = test_ds.labels

    logger.info("Extracting features and training Random Forest...")
    t0 = time.time()
    rf = RandomForestBaseline(seed=seed)
    rf.fit(train_urls, train_labels)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    preds = rf.predict(test_urls)
    acc   = accuracy_score(test_labels, preds)
    f1    = f1_score(test_labels, preds, average="macro", zero_division=0)
    pcf1  = f1_score(test_labels, preds, average=None, zero_division=0).tolist()
    cm    = confusion_matrix(test_labels, preds)

    report = classification_report(
        test_labels, preds,
        target_names=[IDX2LABEL[i] for i in range(NUM_CLASSES)],
        digits=4,
    )
    logger.info(f"TEST Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "model":            "random_forest",
        "seed":             seed,
        "test_accuracy":    acc,
        "test_macro_f1":    f1,
        "per_class_f1":     pcf1,
        "confusion_matrix": cm.tolist(),
        "report":           report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Latency benchmarking (Section 5.3 / deployment claim)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_latency(
    model_name: str,
    ckpt_path: str,
    n_urls: int = 1000,
    n_warmup: int = 50,
) -> dict:
    """
    Measure per-URL inference latency on both CPU and MPS.
    Reports mean ± std in milliseconds over n_urls individual URL passes.
    Used to support the 'real-time deployment' claim in Section 1.
    """
    from data import tokenize_url, MAX_LEN

    # Use a representative mix of URL lengths
    sample_urls = [
        "https://paypal.com-secure-login.phishing.xyz/account?token=abc123",
        "https://google.com",
        "http://malware-distribution-c2-server.ru/payload.exe?id=99",
        "https://legitimate-bank.com/login",
        "http://192.168.1.1/admin",
    ] * (n_urls // 5 + 1)
    sample_urls = sample_urls[:n_urls]

    tokens = torch.tensor(
        [tokenize_url(u, MAX_LEN) for u in sample_urls],
        dtype=torch.long,
    )  # (n_urls, MAX_LEN)

    results = {}

    for device_name in ["cpu", "mps"]:
        if device_name == "mps" and not torch.backends.mps.is_available():
            logger.info("MPS not available — skipping MPS latency benchmark")
            continue

        device = torch.device(device_name)
        model  = get_model(model_name).to(device)

        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path, device=device)
            logger.info(f"Loaded checkpoint from {ckpt_path}")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path} — using untrained model for shape check only")

        model.eval()
        latencies = []

        with torch.no_grad():
            # Warmup
            for i in range(n_warmup):
                x = tokens[i % len(tokens)].unsqueeze(0).to(device)
                _ = model(x)
            if device_name == "mps":
                torch.mps.synchronize()

            # Timed runs — single URL at a time (real deployment scenario)
            for i in range(n_urls):
                x = tokens[i].unsqueeze(0).to(device)   # (1, MAX_LEN)
                t0 = time.perf_counter()
                _ = model(x)
                if device_name == "mps":
                    torch.mps.synchronize()   # ensure GPU op is complete
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)   # convert to ms

        mean_ms = np.mean(latencies)
        std_ms  = np.std(latencies)
        p95_ms  = np.percentile(latencies, 95)

        logger.info(
            f"Latency [{device_name.upper():>3s}] | "
            f"mean: {mean_ms:.3f} ms | std: {std_ms:.3f} ms | p95: {p95_ms:.3f} ms"
        )
        results[device_name] = {
            "mean_ms": mean_ms,
            "std_ms":  std_ms,
            "p95_ms":  p95_ms,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed runner (for statistical significance, Section 6.6)
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed(
    model_name: str,
    seeds: list,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    Train the same model with multiple seeds and compute
    mean ± std across seeds for all key metrics.
    Reports results suitable for Table 6 (statistical significance).
    """
    all_results = []

    for seed in seeds:
        if model_name == "random_forest":
            result = train_random_forest(seed=seed, cfg=cfg)
        else:
            result = train_model(model_name, seed=seed, cfg=cfg, device=device)
        all_results.append(result)

    accs  = [r["test_accuracy"] for r in all_results]
    f1s   = [r["test_macro_f1"] for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"MULTI-SEED SUMMARY — {model_name.upper()} | Seeds: {seeds}")
    logger.info(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"  Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"{'='*60}\n")

    return {
        "model":          model_name,
        "seeds":          seeds,
        "accuracy_mean":  np.mean(accs),
        "accuracy_std":   np.std(accs),
        "macro_f1_mean":  np.mean(f1s),
        "macro_f1_std":   np.std(f1s),
        "per_seed":       all_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PhishFormer training pipeline")
    parser.add_argument(
        "--model", type=str, default="phishformer",
        help="Model to train: phishformer | cnn | transformer | lstm | bilstm | random_forest | all"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Random seeds for training (e.g. --seeds 42 123 456)"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULTS["epochs"],
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULTS["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULTS["lr"],
        help="Learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULTS["patience"],
        help="Early stopping patience (epochs without val F1 improvement)"
    )
    parser.add_argument(
        "--csv_path", type=str, default=DEFAULTS["csv_path"],
        help="Path to malicious_phish.csv"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference latency benchmark after training"
    )
    parser.add_argument(
        "--cpu_only", action="store_true",
        help="Force CPU (useful for debugging)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = {
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "epochs":         args.epochs,
        "patience":       args.patience,
        "weight_decay":   DEFAULTS["weight_decay"],
        "csv_path":       args.csv_path,
        "checkpoint_dir": DEFAULTS["checkpoint_dir"],
        "results_dir":    DEFAULTS["results_dir"],
    }

    device = get_device(prefer_gpu=not args.cpu_only)
    logger.info(f"Using device: {device_info(device)}")

    # Which models to train
    all_dl_models = ["phishformer", "cnn", "transformer", "lstm", "bilstm"]
    if args.model == "all":
        models_to_run = all_dl_models + ["random_forest"]
    else:
        models_to_run = [args.model]

    all_results = {}

    for model_name in models_to_run:
        if model_name == "random_forest":
            if len(args.seeds) > 1:
                result = run_multi_seed("random_forest", args.seeds, cfg, device)
            else:
                result = train_random_forest(seed=args.seeds[0], cfg=cfg)
        else:
            if len(args.seeds) > 1:
                result = run_multi_seed(model_name, args.seeds, cfg, device)
            else:
                result = train_model(model_name, seed=args.seeds[0], cfg=cfg, device=device)

        all_results[model_name] = result

        # Save after each model in case of interruption
        save_results(
            result,
            os.path.join(cfg["results_dir"], f"{model_name}_seed{'_'.join(map(str, args.seeds))}.json"),
        )

        # Latency benchmark if requested
        if args.benchmark and model_name != "random_forest":
            ckpt_path = os.path.join(
                cfg["checkpoint_dir"], f"{model_name}_seed{args.seeds[0]}_best.pt"
            )
            latency = benchmark_latency(model_name, ckpt_path)
            all_results[f"{model_name}_latency"] = latency
            save_results(
                latency,
                os.path.join(cfg["results_dir"], f"{model_name}_latency.json"),
            )

    # Save combined results
    save_results(
        all_results,
        os.path.join(cfg["results_dir"], "all_results.json"),
    )

    logger.info("Training pipeline complete.")
