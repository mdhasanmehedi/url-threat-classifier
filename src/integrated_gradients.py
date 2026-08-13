"""
integrated_gradients.py -- PhishFormer

Faithfulness comparison of three explanation methods on the SAME sample,
using the SAME perturbation-masking protocol as Ablation B, so results
are directly comparable:

  1. Raw attention weights        (Ablation B -- known to FAIL, p=0.239)
  2. Input x Gradient             (gradient-based attribution)
  3. Integrated Gradients         (Sundararajan et al., 2017)

For each method we mask the top-20% highest-attributed characters and
measure the drop in predicted-class confidence, versus masking a random
20%. A faithful method should cause a significantly larger drop under
top-masking than random-masking (Wilcoxon signed-rank, one-sided).

This converts the negative attention result into a constructive finding
IF a gradient-based method passes where attention fails.

Reuses the existing trained checkpoint -- NO retraining required.

Usage:
  python3 src/integrated_gradients.py
  python3 src/integrated_gradients.py --n_samples 500 --seed 42
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint, device_info
from data import (
    load_and_split, tokenize_url,
    IDX2LABEL, LABEL2IDX, NUM_CLASSES, MAX_LEN, PAD_IDX
)
from models import PhishFormer

logger = get_logger()
RESULTS_DIR = "results"
CKPT_DIR    = "checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)

MASK_RATIO   = 0.20   # identical to Ablation B
IG_STEPS     = 32     # integration steps for integrated gradients


# ── Attribution methods ───────────────────────────────────────────────────────

def get_embedding_layer(model: PhishFormer):
    """Return the character embedding layer for gradient hooks."""
    return model.embedding


def _forward_from_embeddings(model: PhishFormer, embeddings: torch.Tensor,
                             pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Run PhishFormer forward starting from pre-computed embeddings
    (needed for integrated gradients, which interpolates in embedding space).
    Mirrors PhishFormer.forward but takes embeddings instead of token ids.
    """
    emb = embeddings.transpose(1, 2)          # (B, embed_dim, seq_len)
    seq_len = emb.size(2)
    conv_outs = []
    for conv in model.conv_layers:
        out = F.relu(conv(emb))
        out = out[:, :, :seq_len]
        conv_outs.append(out)
    cnn_out = torch.cat(conv_outs, dim=1)     # (B, 384, seq_len)
    cnn_out = cnn_out.transpose(1, 2)         # (B, seq_len, 384)
    transformer_out = model.transformer(cnn_out, src_key_padding_mask=pad_mask)
    mask = (~pad_mask).float().unsqueeze(-1)
    pooled = (transformer_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    logits = model.classifier(model.dropout(pooled))
    return logits


def attribution_input_x_gradient(model: PhishFormer, tokens: torch.Tensor,
                                 target: int, device) -> np.ndarray:
    """
    Input x Gradient attribution.
    Returns per-character importance (seq_len,).
    """
    model.eval()
    pad_mask = (tokens == PAD_IDX)
    embed_layer = get_embedding_layer(model)

    with torch.enable_grad():
        # Build embeddings as a fresh differentiable leaf (avoids inference-mode taint)
        base = embed_layer.weight[tokens].detach().clone()   # (1, seq_len, embed_dim)
        embeddings = base.requires_grad_(True)
        logits = _forward_from_embeddings(model, embeddings, pad_mask)
        score  = logits[0, target]
        model.zero_grad()
        grad = torch.autograd.grad(score, embeddings)[0]     # explicit, robust

    inp  = embeddings[0].detach()
    attr = (grad[0] * inp).sum(dim=1).abs()
    return attr.detach().cpu().numpy()


def attribution_integrated_gradients(model: PhishFormer, tokens: torch.Tensor,
                                     target: int, device,
                                     steps: int = IG_STEPS) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al., 2017).
    Baseline = all-PAD embedding. Integrate gradients along the path
    from baseline to the actual input embedding.
    Returns per-character importance (seq_len,).
    """
    model.eval()
    pad_mask = (tokens == PAD_IDX)
    embed_layer = get_embedding_layer(model)

    input_emb = embed_layer.weight[tokens].detach().clone()      # (1, seq_len, embed_dim)

    # Baseline: embedding of all-PAD sequence
    baseline_tokens = torch.full_like(tokens, PAD_IDX)
    baseline_emb = embed_layer.weight[baseline_tokens].detach().clone()

    # Accumulate gradients along interpolation path
    total_grad = torch.zeros_like(input_emb)
    for alpha in np.linspace(0.0, 1.0, steps):
        with torch.enable_grad():
            interp = (baseline_emb + alpha * (input_emb - baseline_emb)).clone().requires_grad_(True)
            logits = _forward_from_embeddings(model, interp, pad_mask)
            score  = logits[0, target]
            model.zero_grad()
            g = torch.autograd.grad(score, interp)[0]
            total_grad += g.detach()

    avg_grad = total_grad / steps
    ig = (input_emb - baseline_emb) * avg_grad     # (1, seq_len, embed_dim)
    attr = ig[0].sum(dim=1).abs()                  # (seq_len,)
    return attr.detach().cpu().numpy()


@torch.no_grad()
def attribution_attention(model: PhishFormer, tokens: torch.Tensor,
                          target: int, device) -> np.ndarray:
    """Raw attention importance (same as Ablation B)."""
    _, importance = model.get_attention_weights(tokens)
    return importance[0].cpu().numpy()


# ── Faithfulness test (shared protocol) ───────────────────────────────────────

@torch.no_grad()
def masked_confidence(model, tokens, target, positions, device):
    """Confidence of target class after masking given positions with PAD."""
    masked = tokens.clone()
    for pos in positions:
        masked[0, int(pos)] = PAD_IDX
    logits = model(masked)
    probs  = torch.softmax(logits, dim=1)
    return probs[0, target].item()


def run_faithfulness(method_name, attr_fn, model, sampled, test_ds,
                     device, rng):
    """
    Run the perturbation-masking faithfulness test for one attribution method.
    Identical protocol to Ablation B.
    Returns (top_drops, random_drops, bottom_drops).
    """
    top_drops, random_drops, bottom_drops = [], [], []

    for count, idx in enumerate(sampled):
        url    = test_ds.urls[idx]
        label  = test_ds.labels[idx]
        tokens = torch.tensor([tokenize_url(url)], dtype=torch.long).to(device)

        # Original confidence
        with torch.no_grad():
            probs = torch.softmax(model(tokens), dim=1)
            orig_conf = probs[0, label].item()

        # Attribution scores for the true/predicted class
        attr = attr_fn(model, tokens, label, device)

        seq_len = int((tokens[0] != PAD_IDX).sum().item())
        n_mask  = max(1, int(seq_len * MASK_RATIO))

        order        = np.argsort(attr[:seq_len])   # ascending
        top_chars    = order[-n_mask:]
        bottom_chars = order[:n_mask]
        random_chars = rng.choice(seq_len, size=n_mask, replace=False)

        top_drops.append(orig_conf - masked_confidence(model, tokens, label, top_chars, device))
        random_drops.append(orig_conf - masked_confidence(model, tokens, label, random_chars, device))
        bottom_drops.append(orig_conf - masked_confidence(model, tokens, label, bottom_chars, device))

        if (count + 1) % 100 == 0:
            logger.info(f"  [{method_name}] processed {count+1}/{len(sampled)}")

    return (np.array(top_drops), np.array(random_drops), np.array(bottom_drops))


def summarize(method_name, top, random_, bottom):
    """Compute stats and Wilcoxon test; return result dict."""
    stat, p = wilcoxon(top, random_, alternative="greater")
    faithful = bool(p < 0.05)
    logger.info(f"\n── {method_name} ──")
    logger.info(f"  Top-attr masking    : {top.mean():.4f} ± {top.std():.4f}")
    logger.info(f"  Random masking      : {random_.mean():.4f} ± {random_.std():.4f}")
    logger.info(f"  Bottom-attr masking : {bottom.mean():.4f} ± {bottom.std():.4f}")
    logger.info(f"  Wilcoxon W          : {stat:.1f}")
    logger.info(f"  p-value (top>random): {p:.6f}")
    logger.info(f"  FAITHFUL? {'YES ✓' if faithful else 'NO ✗'} (α=0.05)")
    return {
        "method":              method_name,
        "top_drop_mean":       float(top.mean()),
        "top_drop_std":        float(top.std()),
        "random_drop_mean":    float(random_.mean()),
        "random_drop_std":     float(random_.std()),
        "bottom_drop_mean":    float(bottom.mean()),
        "bottom_drop_std":     float(bottom.std()),
        "wilcoxon_statistic":  float(stat),
        "p_value":             float(p),
        "faithful":            faithful,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(seed: int = 42, n_samples: int = 500, csv_path: str = "data/raw/malicious_phish.csv"):
    set_seed(seed)
    device = get_device()
    # IG needs gradients; MPS gradient support for this is fine but CPU is safest
    # for reproducibility of the gradient path. Use CPU to avoid MPS edge cases.
    device = torch.device("cpu")
    logger.info(f"Device: {device_info(device)} (CPU forced for gradient stability)")

    # Load model
    model = PhishFormer().to(device)
    ckpt  = os.path.join(CKPT_DIR, f"phishformer_seed{seed}_best.pt")
    if not os.path.exists(ckpt):
        logger.error(f"Checkpoint not found: {ckpt}")
        sys.exit(1)
    load_checkpoint(model, ckpt, device=device)
    model.eval()
    # Ensure parameters can participate in the autograd graph for attribution
    for p in model.parameters():
        p.requires_grad_(True)
    logger.info(f"Loaded {ckpt}")

    # Build the SAME sample as Ablation B: correctly classified malicious URLs
    _, _, test_ds, _ = load_and_split(csv_path, seed=seed)
    logger.info("Finding correctly classified malicious URLs...")
    malicious_indices = []
    bs = 512
    for start in range(0, len(test_ds), bs):
        end = min(start + bs, len(test_ds))
        xs = torch.tensor(
            [tokenize_url(test_ds.urls[i]) for i in range(start, end)],
            dtype=torch.long
        ).to(device)
        ys = [test_ds.labels[i] for i in range(start, end)]
        with torch.no_grad():
            preds = model(xs).argmax(1).cpu().numpy()
        for li, (p, t) in enumerate(zip(preds, ys)):
            if t != LABEL2IDX["benign"] and p == t:
                malicious_indices.append(start + li)
        if len(malicious_indices) >= n_samples * 3:
            break

    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        malicious_indices, size=min(n_samples, len(malicious_indices)), replace=False
    ).tolist()
    logger.info(f"Sampled {len(sampled)} URLs (same protocol as Ablation B)\n")

    # Run all three methods on the SAME sample
    results = {}

    logger.info("="*55)
    logger.info("METHOD 1/3: Raw Attention (expected: FAIL)")
    logger.info("="*55)
    t, r, b = run_faithfulness("Attention", attribution_attention,
                               model, sampled, test_ds, device, np.random.default_rng(seed))
    results["attention"] = summarize("Raw Attention", t, r, b)

    logger.info("\n" + "="*55)
    logger.info("METHOD 2/3: Input x Gradient")
    logger.info("="*55)
    t, r, b = run_faithfulness("InputxGrad", attribution_input_x_gradient,
                               model, sampled, test_ds, device, np.random.default_rng(seed))
    results["input_x_gradient"] = summarize("Input x Gradient", t, r, b)

    logger.info("\n" + "="*55)
    logger.info("METHOD 3/3: Integrated Gradients")
    logger.info("="*55)
    t, r, b = run_faithfulness("IntGrad",
                               lambda m, tok, tgt, dev: attribution_integrated_gradients(m, tok, tgt, dev, IG_STEPS),
                               model, sampled, test_ds, device, np.random.default_rng(seed))
    results["integrated_gradients"] = summarize("Integrated Gradients", t, r, b)

    # ── Final comparison ──────────────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("FAITHFULNESS COMPARISON SUMMARY")
    logger.info("="*70)
    logger.info(f"{'Method':<25} {'Top drop':>12} {'Random drop':>14} {'p-value':>10} {'Faithful':>10}")
    logger.info("-"*70)
    for key in ["attention", "input_x_gradient", "integrated_gradients"]:
        r = results[key]
        logger.info(
            f"{r['method']:<25} {r['top_drop_mean']:>12.4f} "
            f"{r['random_drop_mean']:>14.4f} {r['p_value']:>10.4f} "
            f"{'YES' if r['faithful'] else 'NO':>10}"
        )
    logger.info("="*70)

    out = {
        "seed": seed,
        "n_samples": len(sampled),
        "mask_ratio": MASK_RATIO,
        "ig_steps": IG_STEPS,
        "methods": results,
    }
    path = os.path.join(RESULTS_DIR, "attribution_faithfulness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nResults saved to {path}")

    # Interpretation hint
    att = results["attention"]["faithful"]
    ig  = results["integrated_gradients"]["faithful"]
    ixg = results["input_x_gradient"]["faithful"]
    logger.info("\n── INTERPRETATION ──")
    if not att and (ig or ixg):
        logger.info("POSITIVE FINDING: attention fails but a gradient method passes.")
        logger.info("This converts the negative result into a constructive contribution.")
    elif not att and not ig and not ixg:
        logger.info("All methods fail: a stronger cautionary finding about URL explainability.")
    elif att:
        logger.info("Attention passed here -- re-examine vs Ablation B protocol differences.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--csv_path", type=str, default="data/raw/malicious_phish.csv")
    args = parser.parse_args()
    main(seed=args.seed, n_samples=args.n_samples, csv_path=args.csv_path)
