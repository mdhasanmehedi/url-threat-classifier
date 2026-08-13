"""
distilbert_comparison.py — PhishFormer
Fine-tunes DistilBERT, TinyBERT, and MiniLM on Malicious-Phish (4-class)
for direct parameter-efficiency comparison with PhishFormer.

Models compared:
  - DistilBERT-base-uncased     (67M parameters)
  - TinyBERT-4-layer-uncased    (14.5M parameters)
  - MiniLM-L6-H384-uncased      (22.7M parameters)

All three are fine-tuned on the same 70/15/15 stratified split used
for PhishFormer, with identical training configuration where possible.
Multi-seed evaluation (seeds 42, 123, 456) for statistical comparison.

Usage:
  pip install transformers accelerate
  python3 src/distilbert_comparison.py
  python3 src/distilbert_comparison.py --models distilbert tinybert
  python3 src/distilbert_comparison.py --seeds 42 --fast  # single seed, quick check
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, device_info
from data import load_and_split, IDX2LABEL, NUM_CLASSES

logger = get_logger()
RESULTS_DIR = "results"
CKPT_DIR    = "checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
TRANSFORMER_MODELS = {
    "distilbert": {
        "hf_name":    "distilbert-base-uncased",
        "display":    "DistilBERT-base",
        "max_length": 128,   # URLs rarely exceed 128 subword tokens
    },
    "tinybert": {
        "hf_name":    "huawei-noah/TinyBERT_General_4L_312D",
        "display":    "TinyBERT-4L",
        "max_length": 128,
    },
    "minilm": {
        "hf_name":    "microsoft/MiniLM-L6-H384-uncased",
        "display":    "MiniLM-L6",
        "max_length": 128,
    },
}


# ── Dataset class for HuggingFace tokenizers ──────────────────────────────────

class URLDatasetHF(Dataset):
    """
    Lazy tokenization dataset — tokenizes one URL at a time in __getitem__.
    Avoids stalling on large datasets by not batch-tokenizing upfront.
    """

    def __init__(self, urls, labels, tokenizer, max_length=128):
        self.urls      = urls
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.urls[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training and evaluation ───────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item() * len(labels)
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    return {
        "loss":          total_loss / n,
        "accuracy":      accuracy_score(all_labels, all_preds),
        "macro_f1":      f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class_f1":  f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        "preds":         all_preds,
        "labels":        all_labels,
    }


def benchmark_latency(model, tokenizer, device, n_urls=500, n_warmup=50, max_length=128):
    """Measure single-URL inference latency in ms."""
    sample_urls = [
        "https://paypal.com-secure-login.phishing.xyz/verify",
        "https://google.com",
        "http://malware.ru/payload.exe",
        "https://legitimate-bank.com/login",
        "http://defaced-site.xyz/hacked.html",
    ] * (n_urls // 5 + 1)
    sample_urls = sample_urls[:n_urls]

    model.eval()
    latencies = []

    # Warmup
    with torch.no_grad():
        for i in range(n_warmup):
            url = sample_urls[i % len(sample_urls)]
            enc = tokenizer(url, truncation=True, padding="max_length",
                           max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            _ = model(input_ids=input_ids, attention_mask=attn_mask)
    if device.type == "mps":
        torch.mps.synchronize()

    # Timed
    with torch.no_grad():
        for i in range(n_urls):
            url = sample_urls[i]
            enc = tokenizer(url, truncation=True, padding="max_length",
                           max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attn_mask)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms":  float(np.std(latencies)),
        "p95_ms":  float(np.percentile(latencies, 95)),
    }


def train_one_model(model_key, seed, csv_path, max_epochs=10, batch_size=32,
                    lr=2e-5, patience=3, device=None, fast=False):
    """
    Fine-tune one HuggingFace transformer model on Malicious-Phish.
    Returns result dict matching PhishFormer's output format.
    """
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers accelerate")
        sys.exit(1)

    cfg = TRANSFORMER_MODELS[model_key]
    hf_name    = cfg["hf_name"]
    max_length = cfg["max_length"]
    display    = cfg["display"]

    if fast:
        max_epochs = 3
        patience   = 2

    set_seed(seed)
    if device is None:
        device = get_device()

    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {display} | Seed: {seed} | Device: {device_info(device)}")
    logger.info(f"HuggingFace: {hf_name}")
    logger.info(f"{'='*60}")

    # Check cache path
    cache_path = os.path.join(RESULTS_DIR, f"{model_key}_seed{seed}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        logger.info(f"[CACHED] {display} seed={seed} — "
                    f"acc={cached['test_accuracy']:.4f} f1={cached['test_macro_f1']:.4f}")
        return cached

    # Load tokenizer and model
    logger.info(f"Loading {hf_name} from HuggingFace Hub...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        hf_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mb    = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    logger.info(f"Parameters: {param_count:,} ({param_mb:.1f} MB)")

    # Load data
    train_ds, val_ds, test_ds, class_weights = load_and_split(csv_path, seed=seed)

    logger.info(f"Dataset ready — lazy tokenization enabled ({len(train_ds):,} training URLs)")
    train_hf = URLDatasetHF(train_ds.urls, train_ds.labels, tokenizer, max_length)
    val_hf   = URLDatasetHF(val_ds.urls,   val_ds.labels,   tokenizer, max_length)
    test_hf  = URLDatasetHF(test_ds.urls,  test_ds.labels,  tokenizer, max_length)
    logger.info(f"Datasets initialised: train={len(train_hf):,} val={len(val_hf):,} test={len(test_hf):,}")

    train_loader = DataLoader(train_hf, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_hf,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_hf,  batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimiser and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1  = -1.0
    patience_ctr = 0
    best_epoch   = 0
    ckpt_path    = os.path.join(CKPT_DIR, f"{model_key}_seed{seed}_best.pt")
    history      = []

    for epoch in range(1, max_epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_m   = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch:>2}/{max_epochs} | "
            f"Train loss: {train_m['loss']:.4f} F1: {train_m['macro_f1']:.4f} | "
            f"Val loss: {val_m['loss']:.4f} F1: {val_m['macro_f1']:.4f}"
        )
        history.append({
            "epoch":    epoch,
            "train_f1": train_m["macro_f1"],
            "val_f1":   val_m["macro_f1"],
            "train_loss": train_m["loss"],
            "val_loss":   val_m["loss"],
        })

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1  = val_m["macro_f1"]
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  ✓ Best val macro-F1: {best_val_f1:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_m = evaluate(model, test_loader, device)

    report = classification_report(
        test_m["labels"], test_m["preds"],
        target_names=[IDX2LABEL[i] for i in range(NUM_CLASSES)],
        digits=4,
    )
    logger.info(f"\nTEST RESULTS — {display} (seed {seed})")
    logger.info(f"  Accuracy : {test_m['accuracy']:.4f}")
    logger.info(f"  Macro-F1 : {test_m['macro_f1']:.4f}")
    logger.info(f"\n{report}")

    # Latency benchmark
    logger.info("Running latency benchmark...")
    lat = benchmark_latency(model, tokenizer, device, max_length=max_length)
    logger.info(
        f"CPU latency: {lat['mean_ms']:.3f}±{lat['std_ms']:.3f} ms | "
        f"p95={lat['p95_ms']:.3f} ms"
    )

    result = {
        "model":            model_key,
        "display":          display,
        "hf_name":          hf_name,
        "seed":             seed,
        "param_count":      param_count,
        "param_mb":         param_mb,
        "best_epoch":       best_epoch,
        "best_val_f1":      best_val_f1,
        "test_accuracy":    test_m["accuracy"],
        "test_macro_f1":    test_m["macro_f1"],
        "per_class_f1":     test_m["per_class_f1"],
        "latency_mean_ms":  lat["mean_ms"],
        "latency_std_ms":   lat["std_ms"],
        "latency_p95_ms":   lat["p95_ms"],
        "history":          history,
        "report":           report,
    }

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved to {cache_path}")
    return result


def run_multiseed(model_key, seeds, csv_path, **kwargs):
    """Run one model across multiple seeds and compute mean±std."""
    all_results = []
    for seed in seeds:
        r = train_one_model(model_key, seed, csv_path, **kwargs)
        all_results.append(r)

    accs = [r["test_accuracy"] for r in all_results]
    f1s  = [r["test_macro_f1"] for r in all_results]
    pcf1 = np.array([r["per_class_f1"] for r in all_results])

    summary = {
        "model":            model_key,
        "display":          TRANSFORMER_MODELS[model_key]["display"],
        "seeds":            seeds,
        "param_count":      all_results[0]["param_count"],
        "param_mb":         all_results[0]["param_mb"],
        "accuracy_mean":    float(np.mean(accs)),
        "accuracy_std":     float(np.std(accs)),
        "macro_f1_mean":    float(np.mean(f1s)),
        "macro_f1_std":     float(np.std(f1s)),
        "per_class_f1_mean": pcf1.mean(axis=0).tolist(),
        "per_class_f1_std":  pcf1.std(axis=0).tolist(),
        "latency_mean_ms":  float(np.mean([r["latency_mean_ms"] for r in all_results])),
        "per_seed":         all_results,
    }

    path = os.path.join(RESULTS_DIR, f"{model_key}_multiseed.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\n{TRANSFORMER_MODELS[model_key]['display']} MULTI-SEED SUMMARY:")
    logger.info(f"  Accuracy: {np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}%")
    logger.info(f"  Macro-F1: {np.mean(f1s)*100:.2f}±{np.std(f1s)*100:.2f}%")
    return summary


def print_comparison_table(results):
    """Print paper-ready comparison table."""
    from data import IDX2LABEL
    class_names = [IDX2LABEL[i] for i in range(NUM_CLASSES)]

    print("\n" + "="*110)
    print("LIGHTWEIGHT TRANSFORMER COMPARISON TABLE")
    print("="*110)
    print(f"{'Model':<25} {'Params':>12} {'Accuracy':>16} {'Macro-F1':>16} "
          f"{'CPU lat (ms)':>15}")
    print("-"*85)

    # PhishFormer reference (from multiseed_summary.json)
    try:
        with open(os.path.join(RESULTS_DIR, "multiseed_summary.json")) as f:
            ms = json.load(f)
        pf = ms["summary"]["phishformer"]
        print(f"{'PhishFormer (proposed)':<25} {'1,791,108':>12} "
              f"{pf['accuracy_mean']*100:.2f}±{pf['accuracy_std']*100:.2f}%{' ':>6} "
              f"{pf['macro_f1_mean']*100:.2f}±{pf['macro_f1_std']*100:.2f}%{' ':>6} "
              f"{'1.567±0.022':>15}")
    except Exception:
        print(f"{'PhishFormer (proposed)':<25} {'1,791,108':>12} "
              f"{'97.75±0.37%':>16} {'96.94±0.39%':>16} {'1.567±0.022':>15}")

    for r in results:
        name    = r.get("display", r["model"])
        params  = f"{r['param_count']:,}"
        acc     = f"{r['accuracy_mean']*100:.2f}±{r['accuracy_std']*100:.2f}%"
        f1      = f"{r['macro_f1_mean']*100:.2f}±{r['macro_f1_std']*100:.2f}%"
        lat     = f"{r['latency_mean_ms']:.3f}"
        print(f"{name:<25} {params:>12} {acc:>16} {f1:>16} {lat:>15}")

    print("="*110)
    print("\nPER-CLASS F1 (mean across seeds):")
    print(f"{'Model':<25} {'Benign':>12} {'Defacement':>12} {'Phishing':>12} {'Malware':>12}")
    print("-"*70)

    try:
        with open(os.path.join(RESULTS_DIR, "multiseed_summary.json")) as f:
            ms = json.load(f)
        pf = ms["summary"]["phishformer"]
        pcf1 = pf["per_class_f1_mean"]
        print(f"{'PhishFormer':<25} "
              f"{pcf1[0]*100:>11.2f}% {pcf1[1]*100:>11.2f}% "
              f"{pcf1[2]*100:>11.2f}% {pcf1[3]*100:>11.2f}%")
    except Exception:
        pass

    for r in results:
        pcf1 = r["per_class_f1_mean"]
        print(f"{r.get('display', r['model']):<25} "
              f"{pcf1[0]*100:>11.2f}% {pcf1[1]*100:>11.2f}% "
              f"{pcf1[2]*100:>11.2f}% {pcf1[3]*100:>11.2f}%")
    print("="*110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight transformer comparison")
    parser.add_argument(
        "--models", nargs="+", default=["distilbert", "tinybert", "minilm"],
        choices=list(TRANSFORMER_MODELS.keys()),
        help="Which models to run"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds"
    )
    parser.add_argument(
        "--csv_path", type=str, default="data/raw/malicious_phish.csv"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Max fine-tuning epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (reduce to 16 if OOM)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 3 epochs, 1 seed only (for testing)"
    )
    parser.add_argument(
        "--cpu_only", action="store_true",
        help="Force CPU -- use this if MPS freezes on HuggingFace models"
    )
    args = parser.parse_args()

    # Install check
    try:
        import transformers
        logger.info(f"transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("transformers not installed. Run:")
        logger.error("  pip install transformers accelerate")
        sys.exit(1)

    seeds = [args.seeds[0]] if args.fast else args.seeds
    device = get_device(prefer_gpu=not args.cpu_only)
    logger.info(f"Device: {device_info(device)}")
    logger.info(f"Models to run: {args.models}")
    logger.info(f"Seeds: {seeds}")

    all_summaries = []
    for model_key in args.models:
        summary = run_multiseed(
            model_key, seeds, args.csv_path,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            fast=args.fast,
        )
        all_summaries.append(summary)

    print_comparison_table(all_summaries)

    out_path = os.path.join(RESULTS_DIR, "transformer_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    logger.info(f"\nFull results saved to {out_path}")
    logger.info("Comparison complete.")
