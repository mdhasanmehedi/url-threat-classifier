"""
mps_distilbert.py -- PhishFormer
DistilBERT and TinyBERT fine-tuning on Apple Silicon GPU (MPS).
Uses pre-cached tokenized tensors from data/tokenized_cache/
so no tokenizer is called during training -- pure tensor ops on MPS.

Tokenization cache must exist first (run fast_distilbert.py once on CPU
to generate it, or it will be generated here automatically on CPU then
training switches to MPS).

Usage:
  python3 src/mps_distilbert.py                         # all models, 3 seeds
  python3 src/mps_distilbert.py --models distilbert --seeds 42
  python3 src/mps_distilbert.py --models distilbert --seeds 42 123 456
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_logger
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
        "hf_name": "distilbert-base-uncased",
        "display": "DistilBERT-base",
        "max_len": 128,
    },
    "tinybert": {
        "hf_name": "huawei-noah/TinyBERT_General_4L_312D",
        "display": "TinyBERT-4L",
        "max_len": 128,
    },
}


# ── Dataset: loads from disk cache, moves to GPU in batches ──────────────────

class CachedDataset(Dataset):
    """
    Loads pre-tokenized tensors from disk.
    Keeps everything on CPU; DataLoader moves batches to GPU.
    This avoids MPS freezes caused by tokenizer calls inside workers.
    """

    def __init__(self, urls, labels, tokenizer, model_key, split_name, max_len=128):
        cache_path = os.path.join(
            CACHE_DIR, f"{model_key}_{split_name}_{len(urls)}.pt"
        )

        if os.path.exists(cache_path):
            logger.info(f"Loading cache: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.input_ids      = cached["input_ids"]
            self.attention_mask = cached["attention_mask"]
            logger.info(f"  Loaded {len(self.input_ids):,} samples")
        else:
            logger.info(f"Cache not found. Tokenizing {len(urls):,} URLs on CPU...")
            logger.info(f"(This runs once only, then cached to {cache_path})")
            all_ids, all_masks = [], []
            bs = 1000
            for i in range(0, len(urls), bs):
                enc = tokenizer(
                    urls[i:i+bs],
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt",
                )
                all_ids.append(enc["input_ids"])
                all_masks.append(enc["attention_mask"])
                if (i // bs) % 50 == 0:
                    logger.info(f"  Tokenized {min(i+bs, len(urls)):,}/{len(urls):,}")
            self.input_ids      = torch.cat(all_ids,   dim=0)
            self.attention_mask = torch.cat(all_masks, dim=0)
            torch.save(
                {"input_ids": self.input_ids, "attention_mask": self.attention_mask},
                cache_path,
            )
            logger.info(f"Saved cache: {cache_path}")

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
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += out.loss.item() * len(lbls)
        preds = out.logits.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.cpu().numpy())

    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        out = model(input_ids=ids, attention_mask=mask, labels=lbls)
        total_loss += out.loss.item() * len(lbls)
        preds = out.logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
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
def benchmark_latency(model, tokenizer, device, n=300, warmup=50, max_len=128):
    """Measure per-URL CPU latency (deployment metric, always on CPU)."""
    model_cpu = model.to("cpu").eval()
    urls = [
        "https://paypal.com-secure-login.phishing.xyz/verify",
        "https://google.com",
        "http://malware-cdn.ru/payload.exe",
    ]
    lats = []
    for i in range(warmup + n):
        enc = tokenizer(
            urls[i % 3], truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        t0 = time.perf_counter()
        _ = model_cpu(**enc)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = lats[warmup:]
    # Move model back
    model.to(device)
    return {
        "mean_ms": float(np.mean(lats)),
        "std_ms":  float(np.std(lats)),
        "p95_ms":  float(np.percentile(lats, 95)),
    }


# ── Main training function ────────────────────────────────────────────────────

def train_one(model_key, seed, csv_path, max_epochs=5, batch_size=32,
              lr=2e-5, patience=3, device=None):
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        logger.error("Run: pip install transformers accelerate")
        sys.exit(1)

    cfg        = MODELS[model_key]
    cache_path = os.path.join(RESULTS_DIR, f"{model_key}_seed{seed}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            r = json.load(f)
        if r.get("epochs_trained", 0) >= max_epochs:
            logger.info(
                f"[CACHED] {cfg['display']} seed={seed} "
                f"acc={r['test_accuracy']:.4f} f1={r['test_macro_f1']:.4f}"
            )
            return r
        else:
            logger.info(
                f"Cache found but only {r.get('epochs_trained',0)} epochs. Retraining..."
            )
            os.remove(cache_path)

    set_seed(seed)
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    logger.info(f"\n{'='*60}")
    logger.info(f"{cfg['display']} | seed={seed} | device={device}")
    logger.info(f"{'='*60}")

    # Load tokenizer (only used for tokenization cache + latency benchmark)
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])

    # Load model onto MPS
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_name"], num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mb    = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    logger.info(f"Parameters: {param_count:,} ({param_mb:.1f} MB) on {device}")

    # Load data splits
    train_ds, val_ds, test_ds, _ = load_and_split(csv_path, seed=seed)

    # Build cached datasets (tokenization on CPU, training on MPS)
    train_hf = CachedDataset(
        train_ds.urls, train_ds.labels, tokenizer,
        model_key, f"train_s{seed}", cfg["max_len"]
    )
    val_hf = CachedDataset(
        val_ds.urls, val_ds.labels, tokenizer,
        model_key, f"val_s{seed}", cfg["max_len"]
    )
    test_hf = CachedDataset(
        test_ds.urls, test_ds.labels, tokenizer,
        model_key, f"test_s{seed}", cfg["max_len"]
    )

    # num_workers=0 required for MPS stability
    train_loader = DataLoader(train_hf, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_hf,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_hf,  batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(
        f"Data ready: train={len(train_hf):,} val={len(val_hf):,} test={len(test_hf):,}"
    )

    # Optimiser + linear warmup scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1, patience_ctr, best_epoch = -1.0, 0, 0
    ckpt_path = os.path.join(CKPT_DIR, f"{model_key}_seed{seed}_best.pt")
    history   = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tm = train_epoch(model, train_loader, optimizer, scheduler, device)
        vm = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:>2}/{max_epochs} | "
            f"train loss={tm['loss']:.4f} F1={tm['macro_f1']:.4f} | "
            f"val loss={vm['loss']:.4f} F1={vm['macro_f1']:.4f} acc={vm['accuracy']:.4f} | "
            f"time={elapsed/60:.1f}m"
        )
        history.append({
            "epoch": epoch,
            "train_loss": tm["loss"], "train_f1": tm["macro_f1"],
            "val_loss":   vm["loss"], "val_f1":   vm["macro_f1"],
            "elapsed_min": elapsed / 60,
        })

        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, patience_ctr = vm["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  ✓ Best val macro-F1: {best_f1:.4f} — saved")
        else:
            patience_ctr += 1
            logger.info(f"  No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best checkpoint and evaluate on test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_m = evaluate(model, test_loader, device)

    report = classification_report(
        test_m["labels"], test_m["preds"],
        target_names=[IDX2LABEL[i] for i in range(NUM_CLASSES)],
        digits=4,
    )
    logger.info(f"\nTEST RESULTS -- {cfg['display']} (seed {seed})")
    logger.info(f"  Accuracy : {test_m['accuracy']:.4f}")
    logger.info(f"  Macro-F1 : {test_m['macro_f1']:.4f}")
    logger.info(f"\n{report}")

    # Latency on CPU (deployment metric)
    logger.info("Benchmarking CPU latency...")
    lat = benchmark_latency(model, tokenizer, device)
    logger.info(
        f"CPU latency: {lat['mean_ms']:.2f}+/-{lat['std_ms']:.2f}ms "
        f"p95={lat['p95_ms']:.2f}ms"
    )

    result = {
        "model":            model_key,
        "display":          cfg["display"],
        "seed":             seed,
        "param_count":      param_count,
        "param_mb":         float(param_mb),
        "epochs_trained":   best_epoch,
        "best_val_f1":      float(best_f1),
        "test_accuracy":    float(test_m["accuracy"]),
        "test_macro_f1":    float(test_m["macro_f1"]),
        "per_class_f1":     test_m["per_class_f1"],
        "latency_mean_ms":  lat["mean_ms"],
        "latency_std_ms":   lat["std_ms"],
        "latency_p95_ms":   lat["p95_ms"],
        "history":          history,
        "report":           report,
    }
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {cache_path}")
    return result


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_multiseed(model_key, seeds, csv_path, device, **kw):
    results = []
    for seed in seeds:
        r = train_one(model_key, seed, csv_path, device=device, **kw)
        results.append(r)

    accs = [r["test_accuracy"]  for r in results]
    f1s  = [r["test_macro_f1"]  for r in results]
    pcf1 = np.array([r["per_class_f1"] for r in results])

    summary = {
        "model":            model_key,
        "display":          MODELS[model_key]["display"],
        "seeds":            seeds,
        "param_count":      results[0]["param_count"],
        "param_mb":         results[0]["param_mb"],
        "accuracy_mean":    float(np.mean(accs)),
        "accuracy_std":     float(np.std(accs)),
        "macro_f1_mean":    float(np.mean(f1s)),
        "macro_f1_std":     float(np.std(f1s)),
        "per_class_f1_mean": pcf1.mean(0).tolist(),
        "per_class_f1_std":  pcf1.std(0).tolist(),
        "latency_mean_ms":  float(np.mean([r["latency_mean_ms"] for r in results])),
        "per_seed":         results,
    }

    path = os.path.join(RESULTS_DIR, f"{model_key}_multiseed.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"{MODELS[model_key]['display']} MULTI-SEED SUMMARY")
    logger.info(f"  Accuracy : {np.mean(accs)*100:.2f}+/-{np.std(accs)*100:.2f}%")
    logger.info(f"  Macro-F1 : {np.mean(f1s)*100:.2f}+/-{np.std(f1s)*100:.2f}%")
    logger.info(f"{'='*60}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT/TinyBERT on MPS")
    parser.add_argument("--models", nargs="+",
                        default=["distilbert", "tinybert"],
                        choices=list(MODELS.keys()))
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--csv_path", type=str, default="data/raw/malicious_phish.csv")
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Reduce to 16 if MPS runs out of memory")
    args = parser.parse_args()

    try:
        import transformers
        logger.info(f"transformers {transformers.__version__}")
    except ImportError:
        print("Run: pip install transformers accelerate")
        sys.exit(1)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available -- using CPU")

    all_summaries = []
    for model_key in args.models:
        summary = run_multiseed(
            model_key, args.seeds, args.csv_path, device,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_summaries.append(summary)

    # Final comparison table
    print("\n" + "="*95)
    print("FINAL COMPARISON TABLE (mean+/-std across seeds)")
    print("="*95)
    print(f"{'Model':<25} {'Params':>12} {'Accuracy':>18} {'Macro-F1':>18} {'CPU lat (ms)':>14}")
    print("-"*95)
    print(f"{'PhishFormer (proposed)':<25} {'1,791,108':>12} "
          f"{'97.75+/-0.37%':>18} {'96.94+/-0.39%':>18} {'1.57+/-0.02':>14}")
    for s in all_summaries:
        print(
            f"{s['display']:<25} {s['param_count']:>12,} "
            f"{s['accuracy_mean']*100:.2f}+/-{s['accuracy_std']*100:.2f}%{' ':>8} "
            f"{s['macro_f1_mean']*100:.2f}+/-{s['macro_f1_std']*100:.2f}%{' ':>8} "
            f"{s['latency_mean_ms']:.2f}+/-{s.get('latency_std_ms',0):.2f}{' ':>4}"
        )
    print("="*95)

    with open(os.path.join(RESULTS_DIR, "transformer_comparison.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    logger.info("All done.")
