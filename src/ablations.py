"""
ablations.py — PhishFormer
Three targeted ablation studies validating PhishFormer's core contributions.

Ablation A — Obfuscation-subset fusion analysis (Contribution 1)
Ablation B — Attention faithfulness evaluation (Contribution 2)
Ablation C — Binary-collapse comparison (Contribution 3)

Usage:
  # Run all three ablations
  python3 src/ablations.py --all

  # Run individually
  python3 src/ablations.py --ablation A
  python3 src/ablations.py --ablation B
  python3 src/ablations.py --ablation C
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, device_info, load_checkpoint
from data import (
    load_and_split, get_dataloaders,
    IDX2LABEL, LABEL2IDX, NUM_CLASSES, MAX_LEN, tokenize_url, PAD_IDX
)
from models import PhishFormer, CNNOnly, TransformerOnly, get_model
from train import train_model, DEFAULTS

logger = get_logger()

RESULTS_DIR = "results"
CKPT_DIR    = "checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation A — Obfuscation-Subset Fusion Analysis
# ─────────────────────────────────────────────────────────────────────────────

def detect_obfuscation(url: str) -> bool:
    """
    Heuristically identify URLs containing character-level obfuscation.
    Patterns:
      - Digit-letter homoglyphs: 0→o, 1→l/i, 3→e, 4→a, 5→s
      - Consecutive repeated characters (e.g. 'gooogle')
      - Mixed case within a single token (e.g. 'PayPaL')
      - Excessive hyphenation (3+ hyphens in domain)
      - IP-address-like patterns in domain
    """
    url_lower = url.lower()

    # Homoglyph digit patterns
    homoglyph_patterns = [
        r'[a-z]0[a-z]',    # letter-zero-letter (o→0)
        r'[a-z]1[a-z]',    # letter-one-letter  (l/i→1)
        r'[a-z]3[a-z]',    # letter-three-letter (e→3)
        r'pay[0p]a[1l]',   # paypal variants
        r'g[0o]{2}g[1l]e', # google variants
        r'arnazon|arnaz0n', # amazon variants
    ]
    for pat in homoglyph_patterns:
        if re.search(pat, url_lower):
            return True

    # Consecutive character repetition (3+ same chars)
    if re.search(r'(.)\1{2,}', url_lower):
        return True

    # Excessive hyphens in domain portion
    domain = url.split('/')[2] if len(url.split('/')) > 2 else url
    if domain.count('-') >= 3:
        return True

    # IP address in domain
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
        return True

    return False


def run_ablation_a(seed: int = 42, cfg: dict = None) -> dict:
    """
    Ablation A: Compare PhishFormer, CNN-only, and Transformer-only
    on (a) the full test set and (b) an obfuscated URL subset.

    Validates Contribution 1: full-resolution fusion matters most
    on structurally ambiguous, obfuscated URLs.
    """
    if cfg is None:
        cfg = DEFAULTS.copy()

    logger.info("=" * 60)
    logger.info("ABLATION A — Obfuscation-Subset Fusion Analysis")
    logger.info("=" * 60)

    set_seed(seed)
    device = get_device()

    # Load data
    _, _, test_ds, class_weights = load_and_split(cfg["csv_path"], seed=seed)
    _, _, test_loader = get_dataloaders(
        *load_and_split(cfg["csv_path"], seed=seed)[:3]
    )

    # Identify obfuscated subset from test set
    test_urls   = test_ds.urls
    test_labels = test_ds.labels

    obf_indices     = [i for i, u in enumerate(test_urls) if detect_obfuscation(u)]
    nonobf_indices  = [i for i, u in enumerate(test_urls) if not detect_obfuscation(u)]

    logger.info(f"Test set size       : {len(test_urls):,}")
    logger.info(f"Obfuscated subset   : {len(obf_indices):,} ({100*len(obf_indices)/len(test_urls):.1f}%)")
    logger.info(f"Non-obfuscated      : {len(nonobf_indices):,}")

    # Tokenize subsets
    def subset_tensors(indices):
        xs = torch.tensor(
            [tokenize_url(test_urls[i]) for i in indices], dtype=torch.long
        )
        ys = [test_labels[i] for i in indices]
        return xs, ys

    obf_x,    obf_y    = subset_tensors(obf_indices)
    nonobf_x, nonobf_y = subset_tensors(nonobf_indices)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    models_to_eval = {
        "PhishFormer":      (PhishFormer,      f"{CKPT_DIR}/phishformer_seed{seed}_best.pt"),
        "CNN Only":         (CNNOnly,           f"{CKPT_DIR}/cnn_seed{seed}_best.pt"),
        "Transformer Only": (TransformerOnly,   f"{CKPT_DIR}/transformer_seed{seed}_best.pt"),
    }

    results = {}

    for name, (ModelClass, ckpt_path) in models_to_eval.items():
        model = ModelClass().to(device)
        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path, device=device)
            logger.info(f"Loaded {name} from {ckpt_path}")
        else:
            logger.warning(f"Checkpoint not found for {name} at {ckpt_path} — using untrained weights")

        model.eval()

        def predict_batch(x_tensor, batch_size=512):
            all_preds = []
            with torch.no_grad():
                for i in range(0, len(x_tensor), batch_size):
                    xb = x_tensor[i:i+batch_size].to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
            return all_preds

        # Full test set
        _, _, test_loader = get_dataloaders(
            *load_and_split(cfg["csv_path"], seed=seed)[:3]
        )
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.to(device))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(yb.numpy())
        full_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Obfuscated subset
        obf_preds = predict_batch(obf_x)
        obf_f1    = f1_score(obf_y, obf_preds, average="macro", zero_division=0)

        # Non-obfuscated subset
        nonobf_preds = predict_batch(nonobf_x)
        nonobf_f1    = f1_score(nonobf_y, nonobf_preds, average="macro", zero_division=0)

        logger.info(
            f"{name:<20s} | Full F1: {full_f1:.4f} | "
            f"Obfuscated F1: {obf_f1:.4f} | Non-obfuscated F1: {nonobf_f1:.4f} | "
            f"Obf margin: {obf_f1 - nonobf_f1:+.4f}"
        )

        results[name] = {
            "full_macro_f1":       full_f1,
            "obfuscated_macro_f1": obf_f1,
            "nonobf_macro_f1":     nonobf_f1,
            "obf_margin":          obf_f1 - nonobf_f1,
        }

    # Summary
    pf_obf  = results["PhishFormer"]["obfuscated_macro_f1"]
    cnn_obf = results["CNN Only"]["obfuscated_macro_f1"]
    tf_obf  = results["Transformer Only"]["obfuscated_macro_f1"]

    logger.info("\n── Ablation A Summary ──")
    logger.info(f"PhishFormer vs CNN-only on obfuscated subset: {pf_obf - cnn_obf:+.4f}")
    logger.info(f"PhishFormer vs Transformer-only on obfuscated subset: {pf_obf - tf_obf:+.4f}")
    logger.info(f"Obfuscated subset size: {len(obf_indices):,} URLs")

    out = {
        "ablation": "A",
        "obfuscated_subset_size": len(obf_indices),
        "nonobf_subset_size":     len(nonobf_indices),
        "models": results,
    }
    path = os.path.join(RESULTS_DIR, "ablation_A.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Ablation A results saved to {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ablation B — Attention Faithfulness Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_b(seed: int = 42, n_samples: int = 500, cfg: dict = None) -> dict:
    """
    Ablation B: Perturbation-based faithfulness test for PhishFormer attention.

    For each sampled URL:
      1. Extract per-character attention importance scores
      2. Create three masked variants (top-20%, random-20%, bottom-20%)
      3. Measure confidence drop under each masking strategy
      4. Wilcoxon signed-rank test: top-attention vs random masking

    Validates Contribution 2: attention maps are causally faithful,
    not decorative.
    """
    if cfg is None:
        cfg = DEFAULTS.copy()

    logger.info("=" * 60)
    logger.info("ABLATION B — Attention Faithfulness Evaluation")
    logger.info("=" * 60)

    set_seed(seed)
    device = get_device()

    _, _, test_ds, _ = load_and_split(cfg["csv_path"], seed=seed)

    # Sample correctly-classified malicious URLs
    ckpt_path = f"{CKPT_DIR}/phishformer_seed{seed}_best.pt"
    model = PhishFormer().to(device)
    if os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path, device=device)
        logger.info(f"Loaded PhishFormer from {ckpt_path}")
    else:
        logger.warning("No checkpoint found — using untrained model (results will be random)")

    model.eval()

    # Find correctly classified malicious URLs
    logger.info("Finding correctly classified malicious URLs...")
    malicious_indices = []
    batch_size = 512

    for start in range(0, len(test_ds), batch_size):
        end   = min(start + batch_size, len(test_ds))
        xs    = torch.tensor(
            [tokenize_url(test_ds.urls[i]) for i in range(start, end)],
            dtype=torch.long
        ).to(device)
        ys    = [test_ds.labels[i] for i in range(start, end)]

        with torch.no_grad():
            logits = model(xs)
            preds  = logits.argmax(dim=1).cpu().numpy()

        for local_i, (pred, true) in enumerate(zip(preds, ys)):
            global_i = start + local_i
            if true != LABEL2IDX["benign"] and pred == true:
                malicious_indices.append(global_i)

        if len(malicious_indices) >= n_samples * 3:
            break

    logger.info(f"Found {len(malicious_indices):,} correctly classified malicious URLs")

    # Stratified sample across malicious classes
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        malicious_indices,
        size=min(n_samples, len(malicious_indices)),
        replace=False
    ).tolist()
    logger.info(f"Sampled {len(sampled)} URLs for faithfulness test")

    # Faithfulness evaluation
    top_drops    = []  # confidence drop: top-attention masking
    random_drops = []  # confidence drop: random masking
    bottom_drops = []  # confidence drop: bottom-attention masking

    MASK_RATIO = 0.20  # mask 20% of characters

    for idx in sampled:
        url   = test_ds.urls[idx]
        label = test_ds.labels[idx]
        tokens = torch.tensor([tokenize_url(url)], dtype=torch.long).to(device)

        with torch.no_grad():
            # Get original confidence and attention weights
            logits, importance = model.get_attention_weights(tokens)
            probs    = torch.softmax(logits, dim=1)
            orig_conf = probs[0, label].item()

            # importance shape: (1, MAX_LEN)
            imp = importance[0].cpu().numpy()  # (MAX_LEN,)

        seq_len    = (tokens[0] != PAD_IDX).sum().item()
        n_mask     = max(1, int(seq_len * MASK_RATIO))

        # Sort by importance
        sorted_idx = np.argsort(imp[:seq_len])  # ascending
        top_chars    = sorted_idx[-n_mask:]      # highest attention
        bottom_chars = sorted_idx[:n_mask]       # lowest attention
        random_chars = rng.choice(seq_len, size=n_mask, replace=False)

        def masked_confidence(char_positions):
            masked = tokens.clone()
            for pos in char_positions:
                masked[0, pos] = PAD_IDX  # replace with PAD
            with torch.no_grad():
                logits_m = model(masked)
                probs_m  = torch.softmax(logits_m, dim=1)
            return probs_m[0, label].item()

        top_conf    = masked_confidence(top_chars)
        random_conf = masked_confidence(random_chars)
        bottom_conf = masked_confidence(bottom_chars)

        top_drops.append(orig_conf - top_conf)
        random_drops.append(orig_conf - random_conf)
        bottom_drops.append(orig_conf - bottom_conf)

    top_drops    = np.array(top_drops)
    random_drops = np.array(random_drops)
    bottom_drops = np.array(bottom_drops)

    # Wilcoxon signed-rank test: top vs random
    stat, p_value = wilcoxon(top_drops, random_drops, alternative="greater")

    logger.info("\n── Ablation B Results ──")
    logger.info(f"Samples tested          : {len(sampled)}")
    logger.info(f"Mask ratio              : {MASK_RATIO*100:.0f}% of URL characters")
    logger.info(f"Mean confidence drop:")
    logger.info(f"  Top-attention masking : {top_drops.mean():.4f} ± {top_drops.std():.4f}")
    logger.info(f"  Random masking        : {random_drops.mean():.4f} ± {random_drops.std():.4f}")
    logger.info(f"  Bottom-attention mask : {bottom_drops.mean():.4f} ± {bottom_drops.std():.4f}")
    logger.info(f"Wilcoxon statistic      : {stat:.2f}")
    logger.info(f"p-value (top > random)  : {p_value:.6f}")
    logger.info(f"Faithful? {'YES ✓' if p_value < 0.05 else 'NO ✗'} (α=0.05)")

    out = {
        "ablation": "B",
        "n_samples": len(sampled),
        "mask_ratio": MASK_RATIO,
        "top_attention_drop_mean":    float(top_drops.mean()),
        "top_attention_drop_std":     float(top_drops.std()),
        "random_drop_mean":           float(random_drops.mean()),
        "random_drop_std":            float(random_drops.std()),
        "bottom_attention_drop_mean": float(bottom_drops.mean()),
        "bottom_attention_drop_std":  float(bottom_drops.std()),
        "wilcoxon_statistic":         float(stat),
        "p_value":                    float(p_value),
        "faithful":                   bool(p_value < 0.05),
    }

    path = os.path.join(RESULTS_DIR, "ablation_B.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Ablation B results saved to {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ablation C — Binary-Collapse Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_c(seed: int = 42, cfg: dict = None) -> dict:
    """
    Ablation C: Train a binary PhishFormer (malicious vs benign)
    on collapsed labels from the same data, then quantify
    what operational information the four-class model preserves
    that the binary model cannot.

    Validates Contribution 3: four-class classification captures
    operationally meaningful distinctions a binary classifier discards.
    """
    if cfg is None:
        cfg = DEFAULTS.copy()

    logger.info("=" * 60)
    logger.info("ABLATION C — Binary-Collapse Comparison")
    logger.info("=" * 60)

    set_seed(seed)
    device = get_device()

    # ── Step 1: Train binary PhishFormer ──────────────────────────────────────
    # Collapse phishing=2, malware=3, defacement=1 → malicious=1; benign=0 stays
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import Dataset, DataLoader

    logger.info("Building binary-collapsed dataset from dedup-safe splits...")
    # Reuse the SAME dedup-safe, group-stratified splits as the four-class
    # model (via load_and_split), so the binary and four-class models are
    # trained and evaluated on identical URL populations. Collapsing to
    # binary labels here (rather than re-splitting from the raw CSV with a
    # fresh train_test_split) avoids reintroducing the duplicate-URL
    # leakage that was fixed for every other model in this study, and
    # ensures a fair, same-test-population binary-vs-four-class comparison.
    train_ds_4c, val_ds_4c, test_ds_4c, _ = load_and_split(cfg["csv_path"], seed=seed)

    def to_binary(ds):
        urls = ds.urls
        binary_labels = [0 if IDX2LABEL[l] == "benign" else 1 for l in ds.labels]
        return urls, binary_labels

    X_train, y_train = to_binary(train_ds_4c)
    X_val,   y_val   = to_binary(val_ds_4c)
    X_test,  y_test  = to_binary(test_ds_4c)

    logger.info(f"Binary split (from dedup-safe splits) — train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")

    # Simple Dataset for binary task
    class BinaryURLDataset(Dataset):
        def __init__(self, urls, labels):
            self.urls   = urls
            self.labels = labels
        def __len__(self):
            return len(self.urls)
        def __getitem__(self, i):
            x = torch.tensor(tokenize_url(self.urls[i]), dtype=torch.long)
            y = torch.tensor(self.labels[i], dtype=torch.long)
            return x, y

    train_ds_b = BinaryURLDataset(X_train, y_train)
    val_ds_b   = BinaryURLDataset(X_val,   y_val)
    test_ds_b  = BinaryURLDataset(X_test,  y_test)

    train_loader_b = DataLoader(train_ds_b, batch_size=512, shuffle=True,  num_workers=0)
    val_loader_b   = DataLoader(val_ds_b,   batch_size=512, shuffle=False, num_workers=0)
    test_loader_b  = DataLoader(test_ds_b,  batch_size=512, shuffle=False, num_workers=0)

    # Binary class weights
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(y_train))
    binary_weights = torch.tensor(cw, dtype=torch.float32).to(device)

    # Binary PhishFormer: identical architecture, 2-unit output
    from models import (
        VOCAB_SIZE, EMBED_DIM, CNN_FILTERS, CNN_KERNELS,
        TRANSFORMER_HEADS, TRANSFORMER_LAYERS, TRANSFORMER_FF_DIM, DROPOUT
    )
    binary_model = PhishFormer(num_classes=2).to(device)
    criterion_b  = nn.CrossEntropyLoss(weight=binary_weights)
    optimizer_b  = torch.optim.AdamW(binary_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_b  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=30)

    best_val_f1_b = -1.0
    patience_ctr  = 0
    ckpt_b        = os.path.join(CKPT_DIR, f"phishformer_binary_seed{seed}_best.pt")

    logger.info("Training binary PhishFormer...")
    for epoch in range(1, 31):
        binary_model.train()
        for xb, yb in train_loader_b:
            optimizer_b.zero_grad()
            logits = binary_model(xb.to(device))
            loss   = criterion_b(logits, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(binary_model.parameters(), 1.0)
            optimizer_b.step()
        scheduler_b.step()

        # Validation
        binary_model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader_b:
                preds = binary_model(xb.to(device)).argmax(1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(yb.numpy())
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        logger.info(f"Binary epoch {epoch:>2} | Val macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1_b:
            best_val_f1_b = val_f1
            patience_ctr  = 0
            torch.save(binary_model.state_dict(), ckpt_b)
        else:
            patience_ctr += 1
            if patience_ctr >= 5:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best binary model
    binary_model.load_state_dict(torch.load(ckpt_b, map_location=device))
    binary_model.eval()

    # Binary test results
    test_preds_b, test_labels_b = [], []
    with torch.no_grad():
        for xb, yb in test_loader_b:
            preds = binary_model(xb.to(device)).argmax(1).cpu().numpy()
            test_preds_b.extend(preds)
            test_labels_b.extend(yb.numpy())

    binary_acc = accuracy_score(test_labels_b, test_preds_b)
    binary_f1  = f1_score(test_labels_b, test_preds_b, average="macro", zero_division=0)
    logger.info(f"Binary PhishFormer — Accuracy: {binary_acc:.4f} | Macro-F1: {binary_f1:.4f}")

    # ── Step 2: Operational information analysis ──────────────────────────────
    # Load four-class PhishFormer
    four_class_model = PhishFormer(num_classes=4).to(device)
    ckpt_4class = f"{CKPT_DIR}/phishformer_seed{seed}_best.pt"
    if os.path.exists(ckpt_4class):
        load_checkpoint(four_class_model, ckpt_4class, device=device)
    four_class_model.eval()

    # Reuse the same test split already loaded in Step 1 (test_ds_4c),
    # guaranteeing the binary and four-class models are evaluated on
    # identical test URLs -- required for the fair, same-test-population
    # comparison this ablation is meant to establish.
    _, _, test_loader_4 = get_dataloaders(train_ds_4c, val_ds_4c, test_ds_4c)

    fc_preds, fc_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader_4:
            preds = four_class_model(xb.to(device)).argmax(1).cpu().numpy()
            fc_preds.extend(preds)
            fc_labels.extend(yb.numpy())

    fc_preds  = np.array(fc_preds)
    fc_labels = np.array(fc_labels)

    # Non-phishing malicious URLs (malware + defacement)
    # that four-class model correctly distinguishes from phishing
    nonphish_malicious_mask = np.isin(fc_labels, [LABEL2IDX["malware"], LABEL2IDX["defacement"]])
    nonphish_correct_mask   = (fc_preds == fc_labels) & nonphish_malicious_mask

    total_nonphish_malicious   = nonphish_malicious_mask.sum()
    correctly_distinguished    = nonphish_correct_mask.sum()
    pct_distinguished          = 100 * correctly_distinguished / max(total_nonphish_malicious, 1)

    # Under binary classification these would all get label "malicious"
    # losing the malware vs defacement distinction entirely
    binary_blind_spot = total_nonphish_malicious

    logger.info("\n── Ablation C Results ──")
    logger.info(f"Four-class PhishFormer accuracy : {accuracy_score(fc_labels, fc_preds):.4f}")
    logger.info(f"Binary PhishFormer accuracy     : {binary_acc:.4f}")
    logger.info(f"Binary PhishFormer macro-F1     : {binary_f1:.4f}")
    logger.info(f"Non-phishing malicious URLs in test set  : {total_nonphish_malicious:,}")
    logger.info(f"Correctly distinguished by 4-class model : {correctly_distinguished:,} ({pct_distinguished:.1f}%)")
    logger.info(f"These {total_nonphish_malicious:,} URLs would receive identical 'malicious'")
    logger.info(f"label under binary classification, losing threat-type information.")

    report_binary = classification_report(
        test_labels_b, test_preds_b,
        target_names=["benign", "malicious"], digits=4
    )
    logger.info(f"\nBinary Classification Report:\n{report_binary}")

    out = {
        "ablation": "C",
        "four_class_accuracy":        float(accuracy_score(fc_labels, fc_preds)),
        "binary_accuracy":            float(binary_acc),
        "binary_macro_f1":            float(binary_f1),
        "total_nonphish_malicious":   int(total_nonphish_malicious),
        "correctly_distinguished":    int(correctly_distinguished),
        "pct_distinguished":          float(pct_distinguished),
        "binary_report":              report_binary,
    }

    path = os.path.join(RESULTS_DIR, "ablation_C.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Ablation C results saved to {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PhishFormer ablation studies")
    parser.add_argument("--ablation", type=str, choices=["A", "B", "C"],
                        help="Which ablation to run")
    parser.add_argument("--all", action="store_true",
                        help="Run all three ablations sequentially")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_path", type=str, default=DEFAULTS["csv_path"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = DEFAULTS.copy()
    cfg["csv_path"] = args.csv_path

    if args.all:
        run_ablation_a(seed=args.seed, cfg=cfg)
        run_ablation_b(seed=args.seed, cfg=cfg)
        run_ablation_c(seed=args.seed, cfg=cfg)
    elif args.ablation == "A":
        run_ablation_a(seed=args.seed, cfg=cfg)
    elif args.ablation == "B":
        run_ablation_b(seed=args.seed, cfg=cfg)
    elif args.ablation == "C":
        run_ablation_c(seed=args.seed, cfg=cfg)
    else:
        print("Specify --ablation A/B/C or --all")
