"""
fast_distilbert.py -- PhishFormer
Fast DistilBERT fine-tuning with disk-cached tokenization.

Tokenizes once, saves to disk as tensors, loads fast for all subsequent
epochs and seeds. Reduces per-epoch time from 7 hours to ~20-40 minutes.

Strategy:
  - Tokenize all 651,191 URLs once (~15 min), save as .pt cache
  - All subsequent runs load from cache instantly
  - Train DistilBERT and TinyBERT for 5 epochs, 3 seeds each
  - Report test accuracy, macro-F1, per-class F1, latency

Usage:
  python3 src/fast_distilbert.py                    # full run
  python3 src/fast_distilbert.py --models distilbert --seeds 42  # single
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger
from data import load_and_split, IDX2LABEL, NUM_CLASSES

logger = get_logger()
RESULTS_DIR = "results"
CKPT_DIR    = "checkpoints"
CACHE_DIR   = "data/tokenized_cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)

MODELS = {
    "distilbert": {
        "hf_name":  "distilbert-base-uncased",
        "display":  "DistilBERT-base",
        "params":   66_956_548,
        "max_len":  128,
    },
    "tinybert": {
        "hf_name":  "huawei-noah/TinyBERT_General_4L_312D",
        "display":  "TinyBERT-4L",
        "params":   None,   # measured at runtime
        "max_len":  128,
    },
}


# ── Cached tokenized dataset ──────────────────────────────────────────────────

class CachedURLDataset(Dataset):
    """
    Loads pre-tokenized tensors from disk.
    After the first run, __init__ is near-instant.
    """
    def __init__(self, urls, labels, tokenizer, model_key, split_name, max_len=128):
        cache_path = os.path.join(
            CACHE_DIR, f"{model_key}_{split_name}_{len(urls)}.pt"
        )

        if os.path.exists(cache_path):
            logger.info(f"Loading tokenized cache: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.input_ids      = cached["input_ids"]
            self.attention_mask = cached["attention_mask"]
        else:
            logger.info(
                f"Tokenizing {len(urls):,} URLs for {split_name} "
                f"(first run only, will be cached)..."
            )
            # Tokenize in batches of 1000 for progress visibility
            all_ids, all_masks = [], []
            batch_size = 1000
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i+batch_size]
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt",
                )
                all_ids.append(enc["input_ids"])
                all_masks.append(enc["attention_mask"])
                if (i // batch_size) % 50 == 0:
                    logger.info(
                        f"  Tokenized {min(i+batch_size, len(urls)):,}/{len(urls):,}"
                    )

            self.input_ids      = torch.cat(all_ids,   dim=0)
            self.attention_mask = torch.cat(all_masks, dim=0)
            torch.save(
                {"input_ids": self.input_ids, "attention_mask": self.attention_mask},
                cache_path,
            )
            logger.info(f"Cache saved: {cache_path}")

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ── Train / evaluate ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        optimizer.zero_grad()
        out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item() * len(lbls)
        all_preds.extend(out.logits.argmax(1).cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
        total_loss += out.loss.item() * len(lbls)
        all_preds.extend(out.logits.argmax(1).cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
    n = len(all_labels)
    return {
        "loss":         total_loss / n,
        "accuracy":     accuracy_score(all_labels, all_preds),
        "macro_f1":     f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class_f1": f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        "preds":        all_preds,
        "labels":       all_labels,
    }


@torch.no_grad()
def latency_benchmark(model, tokenizer, device, n=200, warmup=20, max_len=128):
    model.eval()
    urls = ["https://paypal.com-secure-login.phishing.xyz/verify",
            "https://google.com", "http://malware.ru/payload.exe"] * 100
    lats = []
    for i in range(warmup + n):
        enc = tokenizer(urls[i % len(urls)], truncation=True,
                        padding="max_length", max_length=max_len,
                        return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        t0 = time.perf_counter()
        _ = model(input_ids=ids, attention_mask=mask)
        t1 = time.perf_counter()
        if i >= warmup:
            lats.append((t1 - t0) * 1000)
    return {
        "mean_ms": float(np.mean(lats)),
        "std_ms":  float(np.std(lats)),
        "p95_ms":  float(np.percentile(lats, 95)),
    }


# ── Main training function ────────────────────────────────────────────────────

def train_one(model_key, seed, csv_path, max_epochs=5, batch_size=64, lr=2e-5,
              patience=3, device=None):
    try:
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   get_linear_schedule_with_warmup)
    except ImportError:
        logger.error("Run: pip install transformers accelerate")
        sys.exit(1)

    cfg = MODELS[model_key]
    cache_path = os.path.join(RESULTS_DIR, f"{model_key}_seed{seed}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            r = json.load(f)
        logger.info(f"[CACHED] {cfg['display']} seed={seed} "
                    f"acc={r['test_accuracy']:.4f} f1={r['test_macro_f1']:.4f}")
        return r

    set_seed(seed)
    if device is None:
        device = torch.device("cpu")

    logger.info(f"\n{'='*60}")
    logger.info(f"{cfg['display']} | seed={seed} | device={device}")
    logger.info(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_name"], num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {param_count:,}")

    # Load data
    train_ds, val_ds, test_ds, _ = load_and_split(csv_path, seed=seed)

    # Build cached datasets
    train_hf = CachedURLDataset(
        train_ds.urls, train_ds.labels, tokenizer, model_key, f"train_s{seed}", cfg["max_len"]
    )
    val_hf = CachedURLDataset(
        val_ds.urls, val_ds.labels, tokenizer, model_key, f"val_s{seed}", cfg["max_len"]
    )
    test_hf = CachedURLDataset(
        test_ds.urls, test_ds.labels, tokenizer, model_key, f"test_s{seed}", cfg["max_len"]
    )

    train_loader = DataLoader(train_hf, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_hf,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_hf,  batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    best_f1, patience_ctr, best_epoch = -1.0, 0, 0
    ckpt = os.path.join(CKPT_DIR, f"{model_key}_seed{seed}_best.pt")
    history = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tm = train_epoch(model, train_loader, optimizer, scheduler, device)
        vm = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:>2}/{max_epochs} | "
            f"train F1={tm['macro_f1']:.4f} loss={tm['loss']:.4f} | "
            f"val F1={vm['macro_f1']:.4f} loss={vm['loss']:.4f} | "
            f"time={elapsed/60:.1f}m"
        )
        history.append({
            "epoch": epoch, "train_f1": tm["macro_f1"],
            "val_f1": vm["macro_f1"], "elapsed_min": elapsed / 60,
        })

        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, patience_ctr = vm["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt)
            logger.info(f"  Best val macro-F1: {best_f1:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best and test
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_m = evaluate(model, test_loader, device)

    report = classification_report(
        test_m["labels"], test_m["preds"],
        target_names=[IDX2LABEL[i] for i in range(NUM_CLASSES)], digits=4
    )
    logger.info(f"\nTEST: acc={test_m['accuracy']:.4f} macro-F1={test_m['macro_f1']:.4f}")
    logger.info(report)

    logger.info("Running latency benchmark (CPU)...")
    lat = latency_benchmark(model, tokenizer, device)
    logger.info(f"Latency: {lat['mean_ms']:.2f}+/-{lat['std_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms")

    result = {
        "model": model_key, "display": cfg["display"], "seed": seed,
        "param_count": param_count, "best_epoch": best_epoch,
        "test_accuracy": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"],
        "per_class_f1": test_m["per_class_f1"],
        "latency_mean_ms": lat["mean_ms"], "latency_std_ms": lat["std_ms"],
        "history": history, "report": report,
    }
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def run_multiseed(model_key, seeds, csv_path, **kw):
    results = [train_one(model_key, s, csv_path, **kw) for s in seeds]
    accs = [r["test_accuracy"]  for r in results]
    f1s  = [r["test_macro_f1"]  for r in results]
    pcf1 = np.array([r["per_class_f1"] for r in results])
    summary = {
        "model": model_key,
        "display": MODELS[model_key]["display"],
        "seeds": seeds,
        "param_count": results[0]["param_count"],
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),  "macro_f1_std": float(np.std(f1s)),
        "per_class_f1_mean": pcf1.mean(0).tolist(),
        "per_class_f1_std":  pcf1.std(0).tolist(),
        "latency_mean_ms": float(np.mean([r["latency_mean_ms"] for r in results])),
        "per_seed": results,
    }
    path = os.path.join(RESULTS_DIR, f"{model_key}_multiseed.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\n{MODELS[model_key]['display']} SUMMARY: "
                f"acc={np.mean(accs)*100:.2f}+/-{np.std(accs)*100:.2f}% "
                f"F1={np.mean(f1s)*100:.2f}+/-{np.std(f1s)*100:.2f}%")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["distilbert", "tinybert"],
                        choices=list(MODELS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--csv_path", default="data/raw/malicious_phish.csv")
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    try:
        import transformers
    except ImportError:
        print("Run: pip install transformers accelerate")
        sys.exit(1)

    device = torch.device("cpu")
    logger.info(f"Device: CPU | Models: {args.models} | Seeds: {args.seeds}")
    logger.info("NOTE: Tokenization cached to disk/data/tokenized_cache/")
    logger.info("      First run per model: ~15min tokenization, then fast epochs")

    all_summaries = []
    for model_key in args.models:
        s = run_multiseed(model_key, args.seeds, args.csv_path,
                          max_epochs=args.epochs, batch_size=args.batch_size,
                          device=device)
        all_summaries.append(s)

    # Print comparison table
    print("\n" + "="*90)
    print("LIGHTWEIGHT TRANSFORMER COMPARISON (mean+/-std, 3 seeds)")
    print("="*90)
    print(f"{'Model':<25} {'Params':>12} {'Accuracy':>18} {'Macro-F1':>18} {'CPU lat':>12}")
    print("-"*90)
    # PhishFormer reference
    print(f"{'PhishFormer (proposed)':<25} {'1,791,108':>12} "
          f"{'97.75+/-0.37%':>18} {'96.94+/-0.39%':>18} {'1.57ms':>12}")
    for s in all_summaries:
        print(f"{s['display']:<25} {s['param_count']:>12,} "
              f"{s['accuracy_mean']*100:.2f}+/-{s['accuracy_std']*100:.2f}%{' ':>8} "
              f"{s['macro_f1_mean']*100:.2f}+/-{s['macro_f1_std']*100:.2f}%{' ':>8} "
              f"{s['latency_mean_ms']:.2f}ms{' ':>5}")
    print("="*90)

    with open(os.path.join(RESULTS_DIR, "transformer_comparison.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    logger.info("Complete.")
