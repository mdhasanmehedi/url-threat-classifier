"""
multiseed_eval.py — PhishFormer
Multi-seed evaluation for statistical significance (Q1 requirement).

Runs all six models with seeds [42, 123, 456], reports mean ± std
for accuracy and macro-F1, and runs Wilcoxon signed-rank tests
comparing PhishFormer against each baseline.

Usage:
  python3 src/multiseed_eval.py

Results saved to:
  results/multiseed_summary.json
  results/multiseed_summary.txt  (paper-ready table)
"""

import os
import sys
import json
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger
from train import train_model, train_random_forest, DEFAULTS

logger = get_logger()

SEEDS   = [42, 123, 456]
MODELS  = ["phishformer", "cnn", "transformer", "lstm", "bilstm", "random_forest"]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_all_seeds(cfg: dict) -> dict:
    """
    Train every model with every seed.
    Skips already-completed runs by checking for saved result files.
    Safe to resume if interrupted.
    """
    device = get_device()
    all_results = {}

    for model_name in MODELS:
        all_results[model_name] = {}

        for seed in SEEDS:
            cache_path = os.path.join(
                RESULTS_DIR, f"{model_name}_seed{seed}.json"
            )

            # Load from cache if already run
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    result = json.load(f)
                logger.info(
                    f"[CACHED] {model_name} seed={seed} — "
                    f"acc={result['test_accuracy']:.4f} "
                    f"f1={result['test_macro_f1']:.4f}"
                )
            else:
                logger.info(f"Training {model_name} seed={seed}...")
                if model_name == "random_forest":
                    result = train_random_forest(seed=seed, cfg=cfg)
                else:
                    result = train_model(
                        model_name, seed=seed, cfg=cfg, device=device
                    )
                with open(cache_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)

            all_results[model_name][seed] = result

    return all_results


def compute_summary(all_results: dict) -> dict:
    """
    Compute mean ± std across seeds for each model.
    Returns summary dict ready for paper table.
    """
    summary = {}

    for model_name in MODELS:
        accs  = [all_results[model_name][s]["test_accuracy"]  for s in SEEDS]
        f1s   = [all_results[model_name][s]["test_macro_f1"]  for s in SEEDS]

        # Per-class F1 mean ± std
        pcf1_per_seed = [
            all_results[model_name][s]["per_class_f1"] for s in SEEDS
        ]
        pcf1_mean = np.mean(pcf1_per_seed, axis=0)
        pcf1_std  = np.std(pcf1_per_seed,  axis=0)

        summary[model_name] = {
            "accuracy_mean":  float(np.mean(accs)),
            "accuracy_std":   float(np.std(accs)),
            "macro_f1_mean":  float(np.mean(f1s)),
            "macro_f1_std":   float(np.std(f1s)),
            "per_class_f1_mean": pcf1_mean.tolist(),
            "per_class_f1_std":  pcf1_std.tolist(),
            "per_seed_acc":   accs,
            "per_seed_f1":    f1s,
        }

    return summary


def statistical_tests(all_results: dict, summary: dict) -> dict:
    """
    Wilcoxon signed-rank tests: PhishFormer vs each baseline.
    Uses per-seed macro-F1 scores (3 paired observations).

    Note: with only 3 seeds the Wilcoxon test has low power.
    We report the statistic and p-value with this caveat noted.
    """
    pf_f1s = [all_results["phishformer"][s]["test_macro_f1"] for s in SEEDS]
    tests  = {}

    for baseline in ["cnn", "transformer", "lstm", "bilstm", "random_forest"]:
        bl_f1s = [all_results[baseline][s]["test_macro_f1"] for s in SEEDS]

        # With only 3 observations Wilcoxon may not converge —
        # fall back to reporting differences directly
        diffs = [p - b for p, b in zip(pf_f1s, bl_f1s)]
        mean_diff = np.mean(diffs)

        try:
            if len(set(diffs)) == 1:
                # All differences identical — test undefined
                stat, p = float("nan"), float("nan")
            else:
                stat, p = wilcoxon(pf_f1s, bl_f1s)
                stat, p = float(stat), float(p)
        except Exception:
            stat, p = float("nan"), float("nan")

        tests[baseline] = {
            "phishformer_f1s": pf_f1s,
            "baseline_f1s":    bl_f1s,
            "mean_diff":       float(mean_diff),
            "wilcoxon_stat":   stat,
            "p_value":         p,
        }
        logger.info(
            f"PhishFormer vs {baseline:>14s}: "
            f"mean diff={mean_diff:+.4f} | "
            f"W={stat} | p={p:.4f}"
        )

    return tests


def format_paper_table(summary: dict, tests: dict) -> str:
    """
    Format results as a paper-ready plain-text table.
    Copy this directly into Section 6.1 of the paper.
    """
    from data import IDX2LABEL

    label_order = ["benign", "defacement", "phishing", "malware"]
    model_display = {
        "random_forest": "Random Forest",
        "transformer":   "Transformer Only",
        "cnn":           "CNN Only",
        "lstm":          "LSTM",
        "bilstm":        "BiLSTM",
        "phishformer":   "PhishFormer (proposed)",
    }
    display_order = ["random_forest", "transformer", "cnn", "lstm", "bilstm", "phishformer"]

    lines = []
    lines.append("=" * 100)
    lines.append("MULTI-SEED RESULTS TABLE (3 seeds: 42, 123, 456)")
    lines.append("=" * 100)
    lines.append(
        f"{'Model':<25} {'Accuracy':>16} {'Macro-F1':>16} "
        f"{'Benign F1':>16} {'Defacement F1':>16} "
        f"{'Phishing F1':>16} {'Malware F1':>16}"
    )
    lines.append("-" * 100)

    for m in display_order:
        s = summary[m]
        acc_str = f"{s['accuracy_mean']*100:.2f}±{s['accuracy_std']*100:.2f}%"
        f1_str  = f"{s['macro_f1_mean']*100:.2f}±{s['macro_f1_std']*100:.2f}%"
        pcf1_strs = [
            f"{s['per_class_f1_mean'][i]*100:.2f}±{s['per_class_f1_std'][i]*100:.2f}%"
            for i in range(4)
        ]
        lines.append(
            f"{model_display[m]:<25} {acc_str:>16} {f1_str:>16} "
            f"{pcf1_strs[0]:>16} {pcf1_strs[1]:>16} "
            f"{pcf1_strs[2]:>16} {pcf1_strs[3]:>16}"
        )

    lines.append("=" * 100)
    lines.append("")
    lines.append("STATISTICAL TESTS (PhishFormer vs each baseline, Wilcoxon signed-rank)")
    lines.append("Note: 3-seed Wilcoxon has low statistical power; mean differences are the primary metric.")
    lines.append("-" * 60)
    for baseline, t in tests.items():
        lines.append(
            f"PhishFormer vs {model_display[baseline]:<22}: "
            f"mean diff={t['mean_diff']:+.4f} | "
            f"W={t['wilcoxon_stat']} | p={t['p_value']:.4f}"
        )
    lines.append("=" * 100)

    return "\n".join(lines)


if __name__ == "__main__":
    cfg = DEFAULTS.copy()

    logger.info("=" * 60)
    logger.info("MULTI-SEED EVALUATION — PhishFormer")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Models: {MODELS}")
    logger.info("=" * 60)

    # Run all models across all seeds (cached runs are skipped)
    all_results = run_all_seeds(cfg)

    # Compute summary statistics
    summary = compute_summary(all_results)

    # Statistical tests
    logger.info("\nRunning statistical tests...")
    tests = statistical_tests(all_results, summary)

    # Format paper table
    table = format_paper_table(summary, tests)
    logger.info(f"\n{table}")

    # Save everything
    output = {
        "seeds":      SEEDS,
        "summary":    summary,
        "tests":      tests,
        "per_model":  {
            m: {str(s): all_results[m][s] for s in SEEDS}
            for m in MODELS
        }
    }

    json_path = os.path.join(RESULTS_DIR, "multiseed_summary.json")
    txt_path  = os.path.join(RESULTS_DIR, "multiseed_summary.txt")

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    with open(txt_path, "w") as f:
        f.write(table)

    logger.info(f"\nResults saved to {json_path}")
    logger.info(f"Paper table saved to {txt_path}")
    logger.info("Multi-seed evaluation complete.")
