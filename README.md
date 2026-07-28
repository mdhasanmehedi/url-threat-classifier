# Attention Lies, Gradients Don't

**Faithful Explanations and Multi-Class Threat Categorization for Malicious URL Detection**

Code repository for the paper by **Md Mehedi Hasan** (2026).

> **Note on scope:** This repository contains **defensive, detection-oriented research code** for classifying malicious URLs (phishing, malware, defacement). It does **not** contain, generate, or host any phishing content, attack tooling, or malicious payloads. All code here is for academic research into detection methods.

## Overview

This paper uses **PhishFormer** — a lightweight (1.79M-parameter) character-level CNN-Transformer hybrid — as a controlled testbed to interrogate two assumptions common in malicious URL detection research: that architectural novelty drives performance, and that attention weights are trustworthy explanations. All experiments are run on a 651,191-URL, four-class dataset (benign, phishing, malware, defacement), across three random seeds.

Three findings:

1. **Architecture is (mostly) not the lever.** PhishFormer, CNN-only, LSTM, and BiLSTM cluster within 0.7 macro-F1 points of each other. Only the PhishFormer-vs-CNN-only ranking is genuinely unresolved across seeds; a pure-Transformer variant and a Random Forest baseline trail the cluster by a clear, consistent margin.
2. **Four-class labeling earns its place.** A binary-collapse ablation shows that four-class categorization preserves malware/defacement/phishing distinctions — each mapping to a different incident-response action — that a binary detector discards, at negligible accuracy cost.
3. **Attention is not a faithful explanation; gradients are.** A perturbation-based faithfulness audit shows raw attention weights **fail** a causal faithfulness test (Wilcoxon p=0.281), while gradient-based attribution (Integrated Gradients, Input×Gradient) **passes decisively** (p<0.001). Analyst-facing explanations should be derived from gradient attribution, not attention heatmaps.

The repository also includes cross-dataset domain adaptation (PhiUSIIL, 235,795 URLs), deployment latency/memory benchmarking, and a comparison against fine-tuned DistilBERT.

## Repository Structure

```
.
├── src/
│   ├── models.py                  # PhishFormer + all baseline architectures
│   ├── data.py                    # Tokenization, dataset class, stratified splitting
│   ├── train.py                   # Training loop (all deep learning models)
│   ├── multiseed_eval.py          # 3-seed evaluation + Wilcoxon significance testing
│   ├── ablations.py               # Ablation A/B/C (fusion, faithfulness, binary-collapse)
│   ├── integrated_gradients.py    # Gradient-based attribution methods
│   ├── compute_mcc.py             # Matthews Correlation Coefficient computation
│   ├── benchmark.py               # Inference latency / throughput / memory benchmarks
│   ├── cross_dataset_eval.py      # PhiUSIIL domain-adaptation experiments
│   ├── finetune_phiusiil.py       # Few-shot / full fine-tuning on PhiUSIIL
│   ├── distilbert_comparison.py   # DistilBERT baseline comparison
│   ├── fast_distilbert.py         # Optimized DistilBERT inference
│   ├── mps_distilbert.py          # Apple Silicon (MPS) DistilBERT variant
│   ├── figures.py                 # Paper figure generation
│   └── utils.py                   # Logging, seed-setting, shared utilities
├── data/                           # See data/README.md — raw data not committed here
├── results/                        # Per-model, per-seed JSON results (metrics, confusion matrices)
└── requirements.txt
```

Hyperparameters (sequence length, vocabulary size, embedding dimensions, learning rate, batch size, etc.) are defined as constants and function arguments directly within the relevant `src/` files (e.g., `data.py`, `models.py`, `train.py`) rather than in separate config files. Exact values match Table 2 of the paper.

## Setup

```bash
git clone https://github.com/mdhasanmehedi/url-threat-classifier.git
cd url-threat-classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

This project uses the **Malicious-Phish** dataset (651,191 URLs; sid321axn, 2021), publicly available on Kaggle. See [`data/README.md`](data/README.md) for download instructions — the raw CSV is not committed to this repository.

Cross-dataset domain adaptation experiments use **PhiUSIIL** (235,795 URLs; 100,945 legitimate, 134,850 phishing).

## Reproducibility

Trained model checkpoints are not distributed with this repository. All reported results are exactly reproducible from the provided code using the documented seeds (42, 123, 456) — see `src/train.py` and `src/multiseed_eval.py`.

## Reproducing Results

```bash
# Train PhishFormer (single seed)
python src/train.py --model phishformer --seed 42

# Multi-seed evaluation with significance testing
python src/multiseed_eval.py

# Run all three ablation studies
python src/ablations.py

# Faithfulness audit (attention vs. gradient attribution)
python src/integrated_gradients.py

# Cross-dataset domain adaptation
python src/cross_dataset_eval.py
```

Exact hyperparameters and the stratified 70/15/15 train/validation/test split (seed=42) are defined directly in `src/data.py`, `src/models.py`, and `src/train.py`.

## Citation

If you use this code, please cite:

```bibtex
@article{hasan2026attentionlies,
  title   = {Attention Lies, Gradients Don't: Faithful Explanations and Multi-Class Threat Categorization for Malicious URL Detection},
  author  = {Hasan, Md Mehedi},
  year    = {2026}
}
```

*(Full citation details — journal, volume, DOI — will be added upon publication.)*

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

Md Mehedi Hasan — Independent Researcher, alumnus of the Moscow Institute of Physics and Technology (MIPT). Corresponding author details are listed in the published paper.
