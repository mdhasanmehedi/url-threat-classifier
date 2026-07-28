"""
figures.py — PhishFormer
Generates all figures needed for the paper:

  Figure 1: PhishFormer architecture diagram (SVG/PNG)
  Figure 2: Confusion matrix heatmap (PhishFormer, seed=42)
  Figure 3: ROC curves per class (PhishFormer, seed=42)
  Figure 4: Training curves — loss and macro-F1 (all models)
  Figure 5: Attention heatmap examples on real URLs
  Figure 6: Per-class F1 comparison bar chart (all models)

Usage:
  python3 src/figures.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint
from data import (
    load_and_split, get_dataloaders, tokenize_url,
    IDX2LABEL, LABEL2IDX, NUM_CLASSES, MAX_LEN, PAD_IDX
)
from models import PhishFormer

logger = get_logger()
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
CKPT_DIR    = "checkpoints"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Consistent colour palette across all figures
CLASS_COLORS = {
    "benign":     "#2196F3",   # blue
    "defacement": "#FF9800",   # orange
    "phishing":   "#F44336",   # red
    "malware":    "#9C27B0",   # purple
}
CLASS_ORDER = ["benign", "defacement", "phishing", "malware"]

MODEL_DISPLAY = {
    "phishformer": "PhishFormer",
    "cnn":         "CNN Only",
    "transformer": "Transformer Only",
    "lstm":        "LSTM",
    "bilstm":      "BiLSTM",
    "random_forest": "Random Forest",
}
MODEL_COLORS = {
    "phishformer":   "#E53935",
    "cnn":           "#1E88E5",
    "transformer":   "#43A047",
    "lstm":          "#FB8C00",
    "bilstm":        "#8E24AA",
    "random_forest": "#6D4C41",
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(seed: int = 42) -> str:
    """Generate confusion matrix heatmap for PhishFormer (seed=42)."""
    logger.info("Generating Figure 2: Confusion Matrix...")

    result_path = os.path.join(RESULTS_DIR, f"phishformer_seed{seed}.json")
    if not os.path.exists(result_path):
        logger.error(f"Result file not found: {result_path}")
        return None

    with open(result_path) as f:
        result = json.load(f)

    cm = np.array(result["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    cmap = LinearSegmentedColormap.from_list(
        "phish", ["#ffffff", "#1565C0"], N=256
    )

    for ax, normalize, title_suffix in zip(
        axes, [False, True], ["(Counts)", "(Row-Normalised)"]
    ):
        if normalize:
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            vmax = 1.0
        else:
            cm_plot = cm
            fmt = "d"
            vmax = cm.max()

        im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap,
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        tick_labels = [IDX2LABEL[i].capitalize() for i in range(NUM_CLASSES)]
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=11)
        ax.set_yticklabels(tick_labels, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix {title_suffix}", fontsize=13, fontweight="bold")

        thresh = cm_plot.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = cm_plot[i, j]
                text = f"{val:{fmt}}" if not normalize else f"{val:.2f}"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text, ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    plt.suptitle(
        "PhishFormer — Confusion Matrix on Malicious-Phish Test Set (n=97,680)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig2_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_roc_curves(seed: int = 42) -> str:
    """Generate per-class ROC curves for PhishFormer."""
    logger.info("Generating Figure 3: ROC Curves...")

    device = get_device()
    model  = PhishFormer().to(device)
    ckpt   = os.path.join(CKPT_DIR, f"phishformer_seed{seed}_best.pt")

    if not os.path.exists(ckpt):
        logger.error(f"Checkpoint not found: {ckpt}")
        return None

    load_checkpoint(model, ckpt, device=device)
    model.eval()

    set_seed(seed)
    _, _, test_ds, _ = load_and_split(
        "data/raw/malicious_phish.csv", seed=seed
    )
    _, _, test_loader = get_dataloaders(
        *load_and_split("data/raw/malicious_phish.csv", seed=seed)[:3]
    )

    all_probs  = []
    all_labels = []

    for xb, yb in test_loader:
        logits = model(xb.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(yb.numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.array(all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-class ROC
    ax = axes[0]
    for i, cls in enumerate(CLASS_ORDER):
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=2,
                label=f"{cls.capitalize()} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Per Class", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    # Right: zoomed in on high-performance region
    ax2 = axes[1]
    for i, cls in enumerate(CLASS_ORDER):
        y_true  = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=2,
                 label=f"{cls.capitalize()} (AUC = {roc_auc:.4f})")

    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax2.set_xlim([0.0, 0.05])
    ax2.set_ylim([0.90, 1.01])
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title("ROC Curves — Zoomed (FPR ≤ 0.05)", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        "PhishFormer — ROC Curves on Malicious-Phish Test Set (n=97,680)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig3_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(seed: int = 42) -> str:
    """Generate training loss and macro-F1 curves for all models."""
    logger.info("Generating Figure 4: Training Curves...")

    models_to_plot = ["phishformer", "cnn", "transformer", "lstm", "bilstm"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name in models_to_plot:
        result_path = os.path.join(RESULTS_DIR, f"{model_name}_seed{seed}.json")
        if not os.path.exists(result_path):
            logger.warning(f"No result file for {model_name} seed={seed}, skipping")
            continue

        with open(result_path) as f:
            result = json.load(f)

        if "history" not in result or not result["history"]:
            logger.warning(f"No training history for {model_name}, skipping")
            continue

        history = result["history"]
        epochs     = [h["epoch"]     for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss   = [h["val_loss"]   for h in history]
        train_f1   = [h["train_f1"]   for h in history]
        val_f1     = [h["val_f1"]     for h in history]

        color = MODEL_COLORS[model_name]
        label = MODEL_DISPLAY[model_name]

        # Loss plot
        axes[0].plot(epochs, val_loss, color=color, lw=2, label=label)
        axes[0].plot(epochs, train_loss, color=color, lw=1,
                     linestyle="--", alpha=0.4)

        # F1 plot
        axes[1].plot(epochs, val_f1, color=color, lw=2, label=label)
        axes[1].plot(epochs, train_f1, color=color, lw=1,
                     linestyle="--", alpha=0.4)

    for ax, ylabel, title in zip(
        axes,
        ["Cross-Entropy Loss", "Macro-F1 Score"],
        ["Validation Loss (solid) / Train Loss (dashed)",
         "Validation Macro-F1 (solid) / Train Macro-F1 (dashed)"]
    ):
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Training Curves — All Models on Malicious-Phish Dataset",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig4_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Attention Heatmap Examples
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_attention_heatmaps(seed: int = 42) -> str:
    """Generate attention heatmap visualisations for example URLs."""
    logger.info("Generating Figure 5: Attention Heatmaps...")

    device = get_device()
    model  = PhishFormer().to(device)
    ckpt   = os.path.join(CKPT_DIR, f"phishformer_seed{seed}_best.pt")

    if not os.path.exists(ckpt):
        logger.error(f"Checkpoint not found: {ckpt}")
        return None

    load_checkpoint(model, ckpt, device=device)
    model.eval()

    example_urls = [
        ("http://paypa1.com-secure-1ogin.xyz/verify",     "phishing",   "Phishing — homoglyph substitution"),
        ("http://update-flash.malware-cdn.ru/install.exe", "malware",    "Malware — suspicious path/domain"),
        ("http://www.garage-defaced-site.be/hacked.html",  "defacement", "Defacement — hacked site pattern"),
        ("https://www.google.com/search?q=python",         "benign",     "Benign — legitimate URL"),
    ]

    fig, axes = plt.subplots(len(example_urls), 1, figsize=(16, 3 * len(example_urls)))

    for ax, (url, true_class, description) in zip(axes, example_urls):
        tokens = torch.tensor([tokenize_url(url, MAX_LEN)], dtype=torch.long).to(device)
        logits, importance = model.get_attention_weights(tokens)
        pred_class = IDX2LABEL[logits.argmax(dim=1).item()]

        imp = importance[0].cpu().numpy()
        url_chars = list(url)
        n_chars   = min(len(url_chars), MAX_LEN)
        imp_chars = imp[:n_chars]

        # Normalise for display
        if imp_chars.max() > imp_chars.min():
            imp_norm = (imp_chars - imp_chars.min()) / (imp_chars.max() - imp_chars.min())
        else:
            imp_norm = imp_chars

        # Display only first 80 characters for readability
        disp_n   = min(n_chars, 80)
        imp_disp = imp_norm[:disp_n]
        chars    = url_chars[:disp_n]

        cmap = plt.cm.YlOrRd
        for i, (ch, score) in enumerate(zip(chars, imp_disp)):
            color = cmap(score)
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax.text(i + 0.5, 0.5, ch, ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white" if score > 0.6 else "black")

        ax.set_xlim(0, disp_n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(
            f"{description} | True: {true_class} | Predicted: {pred_class}",
            fontsize=10, fontweight="bold", pad=4,
            color=CLASS_COLORS.get(true_class, "black")
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.02)
    cbar.set_label("Attention Importance", fontsize=10)

    plt.suptitle(
        "PhishFormer — Character-Level Attention Heatmaps (darker = higher attention)",
        fontsize=12, fontweight="bold", y=1.01
    )

    path = os.path.join(FIGURES_DIR, "fig5_attention_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Per-Class F1 Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_perclass_f1_comparison(seed: int = 42) -> str:
    """Generate grouped bar chart comparing per-class F1 across all models."""
    logger.info("Generating Figure 6: Per-Class F1 Comparison...")

    models = ["random_forest", "transformer", "cnn", "lstm", "bilstm", "phishformer"]

    # Load per-class F1 from result files
    f1_data = {}
    for m in models:
        path = os.path.join(RESULTS_DIR, f"{m}_seed{seed}.json")
        if not os.path.exists(path):
            logger.warning(f"Missing results for {m}, skipping")
            continue
        with open(path) as f:
            result = json.load(f)
        f1_data[m] = result.get("per_class_f1", [0, 0, 0, 0])

    if not f1_data:
        logger.error("No result files found")
        return None

    n_models  = len(f1_data)
    n_classes = NUM_CLASSES
    x         = np.arange(n_classes)
    width     = 0.13
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (m, f1s) in enumerate(f1_data.items()):
        label  = MODEL_DISPLAY.get(m, m)
        color  = MODEL_COLORS.get(m, "grey")
        bars   = ax.bar(x + offsets[i], [f * 100 for f in f1s],
                        width, label=label, color=color, alpha=0.85,
                        edgecolor="white", linewidth=0.5)

        # Highlight PhishFormer bars with border
        if m == "phishformer":
            for bar in bars:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)

    ax.set_xlabel("Threat Category", fontsize=12)
    ax.set_ylabel("F1-Score (%)", fontsize=12)
    ax.set_title(
        "Per-Class F1-Score Comparison — All Models (seed=42)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.capitalize() for c in CLASS_ORDER],
        fontsize=12
    )
    ax.set_ylim(70, 101)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="lower right", ncol=2)

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig6_perclass_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Multi-seed performance distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_multiseed_comparison() -> str:
    """Box/bar plot showing mean±std across seeds for all models."""
    logger.info("Generating Figure 7: Multi-seed Performance Distribution...")

    summary_path = os.path.join(RESULTS_DIR, "multiseed_summary.json")
    if not os.path.exists(summary_path):
        logger.warning("No multiseed_summary.json found, skipping Figure 7")
        return None

    with open(summary_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    models  = ["random_forest", "transformer", "cnn", "lstm", "bilstm", "phishformer"]
    models  = [m for m in models if m in summary]

    means = [summary[m]["macro_f1_mean"] * 100 for m in models]
    stds  = [summary[m]["macro_f1_std"]  * 100 for m in models]
    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "grey") for m in models]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=0.5,
                  error_kw={"elinewidth": 2, "ecolor": "black", "capthick": 2})

    # Highlight PhishFormer
    pf_idx = models.index("phishformer") if "phishformer" in models else -1
    if pf_idx >= 0:
        bars[pf_idx].set_edgecolor("black")
        bars[pf_idx].set_linewidth(2)

    # Annotate with mean±std
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.3, f"{mean:.2f}\n±{std:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Macro-F1 (%)", fontsize=12)
    ax.set_title(
        "Multi-Seed Macro-F1 Comparison (mean±std, seeds 42/123/456)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(85, 102)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig7_multiseed_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(42)

    logger.info("Generating all PhishFormer paper figures...")
    logger.info(f"Output directory: {FIGURES_DIR}")

    generated = []

    try:
        p = plot_confusion_matrix()
        if p: generated.append(p)
    except Exception as e:
        logger.error(f"Figure 2 failed: {e}")

    try:
        p = plot_roc_curves()
        if p: generated.append(p)
    except Exception as e:
        logger.error(f"Figure 3 failed: {e}")

    try:
        p = plot_training_curves()
        if p: generated.append(p)
    except Exception as e:
        logger.error(f"Figure 4 failed: {e}")

    try:
        p = plot_attention_heatmaps()
        if p: generated.append(p)
    except Exception as e:
        logger.error(f"Figure 5 failed: {e}")

    try:
        p = plot_perclass_f1_comparison()
        if p: generated.append(p)
    except Exception as e:
        logger.error(f"Figure 6 failed: {e}")

    try:
        p = plot_multiseed_comparison()
        if p: generated.append(p)
    except Exception as e:
        logger.error(f"Figure 7 failed: {e}")

    logger.info(f"\nGenerated {len(generated)} figures:")
    for p in generated:
        logger.info(f"  {p}")
    logger.info("\nAll figures complete.")
