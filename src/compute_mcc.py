"""
compute_mcc.py -- PhishFormer
Computes Matthews Correlation Coefficient (MCC) for all trained models
across the 3 seeds, reusing the saved prediction results. MCC is a
balanced metric robust to class imbalance (reviewer request).

Usage:
  python3 src/compute_mcc.py
"""
import os, sys, json
import numpy as np
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint
from data import load_and_split, get_dataloaders, IDX2LABEL
from models import get_model
import torch

logger = get_logger()
SEEDS  = [42, 123, 456]
MODELS = ["phishformer", "cnn", "transformer", "lstm", "bilstm"]
CKPT   = "checkpoints"

def mcc_for(model_name, seed, device):
    set_seed(seed)
    _, _, test_ds, _ = load_and_split("data/raw/malicious_phish.csv", seed=seed)
    _, _, test_loader = get_dataloaders(
        *load_and_split("data/raw/malicious_phish.csv", seed=seed)[:3]
    )
    model = get_model(model_name).to(device)
    ckpt = f"{CKPT}/{model_name}_seed{seed}_best.pt"
    if not os.path.exists(ckpt):
        return None
    load_checkpoint(model, ckpt, device=device)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(p); labels.extend(yb.numpy())
    return matthews_corrcoef(labels, preds)

if __name__ == "__main__":
    device = get_device()
    results = {}
    for m in MODELS:
        vals = []
        for s in SEEDS:
            v = mcc_for(m, s, device)
            if v is not None:
                vals.append(v)
                logger.info(f"{m} seed={s}: MCC={v:.4f}")
        if vals:
            results[m] = {"mcc_mean": float(np.mean(vals)), "mcc_std": float(np.std(vals))}
            logger.info(f"{m}: MCC = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Random Forest from its saved predictions if available
    print("\n" + "="*50)
    print("MCC SUMMARY (mean +/- std across 3 seeds)")
    print("="*50)
    for m, r in results.items():
        print(f"{m:<15} MCC = {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f}")
    json.dump(results, open("results/mcc_summary.json", "w"), indent=2)
    print("\nSaved to results/mcc_summary.json")
