# PhishFormer: Faithful Explanations and Multi-Class Threat Categorization for Malicious URL Detection

This repository accompanies the paper **"When Attention Lies and Gradients Don't: Faithful Explanations and Multi-Class Threat Categorization for Malicious URL Detection."**

PhishFormer is a 1.79M-parameter character-level CNN-Transformer hybrid for four-class malicious URL detection (benign, phishing, malware, defacement), evaluated on a 651,191-URL dataset. Rather than claiming architectural superiority, this study uses PhishFormer as a controlled testbed to examine two questions of practical consequence:

1. Are attention weights faithful explanations of model behavior, or should analysts rely on gradient-based attribution instead?
2. Does four-class threat categorization preserve operationally meaningful information that binary detection discards?

A perturbation-based faithfulness audit (the first of its kind in this domain) shows raw attention weights fail the faithfulness test (Wilcoxon p=0.941), while gradient-based attribution — Integrated Gradients and Input×Gradient — passes decisively (p<0.001). As context for this finding and the four-class analysis, we also show that architectural choice matters less than commonly assumed: across three random seeds, PhishFormer, CNN-only, LSTM, and BiLSTM cluster within 0.9 macro-F1 points of one another, with no architecture claiming formal statistical superiority.

## Repository structure

```
.
├── src/                    # All source code
│   ├── train.py                    # Main training script (6 architectures)
│   ├── data.py                     # Data loading, dedup-safe splitting
│   ├── models.py                   # Model architectures
│   ├── utils.py                    # Shared utilities (checkpointing, device, logging)
│   ├── dedup_group_split.py        # Generates leakage-safe train/val/test splits
│   ├── finetune_phiusiil.py        # Cross-dataset domain adaptation (PhiUSIIL)
│   ├── cross_dataset_eval.py       # Cross-dataset evaluation harness
│   ├── ablations.py                # Ablation studies (A/B/C)
│   ├── integrated_gradients.py     # Gradient-based attribution methods
│   ├── compute_mcc.py              # Matthews Correlation Coefficient computation
│   ├── paired_bootstrap.py         # Bootstrap significance testing
│   ├── benchmark.py                # Inference latency/throughput/memory benchmarking
│   ├── distilbert_comparison.py    # DistilBERT baseline comparison
│   ├── fast_distilbert.py          # DistilBERT training utilities
│   ├── mps_distilbert.py           # Apple Silicon (MPS) DistilBERT variant
│   └── figures.py                  # Figure generation
├── results/                 # Verified result artifacts (JSON)
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Dataset

This work uses two publicly available datasets, **not included in this repository** due to size and licensing:

- **Malicious-Phish** (651,191 URLs; benign/phishing/malware/defacement) — sourced from Kaggle.
- **PhiUSIIL** (235,795 URLs; used for cross-dataset domain adaptation) — sourced from the PhiUSIIL dataset release.

Download both and place them at:
```
data/raw/malicious_phish.csv
data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

**1. Generate deduplication-safe splits** (guarantees no URL leaks across train/val/test). Run once per seed, matching the three seeds used in Step 2:
```bash
cd src
for seed in 42 123 456; do
    python3 dedup_group_split.py --csv_path ../data/raw/malicious_phish.csv --seed $seed
done
```

**2. Train all six architectures** (PhishFormer, CNN-only, Transformer-only, LSTM, BiLSTM, Random Forest) across three seeds:
```bash
python3 train.py --model all --seeds 42 123 456 --csv_path ../data/raw/malicious_phish.csv
```
(`--model` accepts `phishformer`, `cnn`, `transformer`, `lstm`, `bilstm`, `random_forest`, or `all`. Add `--benchmark` to run the inference latency benchmark automatically after training, or `--cpu_only` to force CPU execution.)

**3. Cross-dataset domain adaptation** (zero-shot / few-shot / full fine-tune on PhiUSIIL):
```bash
python3 finetune_phiusiil.py --csv_path ../data/raw/PhiUSIIL_Phishing_URL_Dataset.csv --seed 42
```

**4. DistilBERT comparison baseline:**
```bash
python3 distilbert_comparison.py --models distilbert --seeds 42
```

**5. Inference latency/throughput/memory benchmark.** Reported results were produced with each model benchmarked in its own isolated process, CPU thread count explicitly pinned, and batched-throughput warmup increased to 50 iterations (see Section 5.4 of the paper); run one model per process rather than all five in a single invocation:
```bash
for m in phishformer cnn transformer lstm bilstm; do
    python3 benchmark.py --model $m
done
```

**6. Faithfulness audit** (gradient-based attribution vs. attention):
```bash
python3 integrated_gradients.py
```

**7. Ablation studies:**
```bash
python3 ablations.py --ablation A
python3 ablations.py --ablation B
python3 ablations.py --ablation C
```

Model checkpoints are not included in this repository (regenerable via the scripts above, or contact the author for pretrained weights — DistilBERT's checkpoint exceeds GitHub's file-size limits).

## Results summary

The faithfulness audit is the paper's central finding: raw attention fails a causal faithfulness test (p=0.941), while gradient-based attribution passes decisively (p<0.001). Architecturally, the six models compare as follows:

| Model | Params | Accuracy (3-seed) | Macro-F1 (3-seed) |
|---|---|---|---|
| PhishFormer (proposed) | 1,791,108 | 97.82±0.30% | 97.01±0.27% |
| DistilBERT-base | 66,956,548 | 98.89% | 98.30% (single seed) |

See `results/` for full per-model, per-class, and per-seed breakdowns, and the paper for complete discussion.

## Citation

If you use this code or refer to this work, please cite:

```bibtex
@misc{hasan_phishformer,
  title  = {When Attention Lies and Gradients Don't: Faithful Explanations and Multi-Class Threat Categorization for Malicious URL Detection},
  author = {Hasan, Md Mehedi},
  year   = {2026},
  note   = {Preprint}
}
```

## Author

**Md Mehedi Hasan**
Independent Researcher; alumnus, Moscow Institute of Physics and Technology (MIPT)
Email: khasan.m@phystech.edu

## License

MIT License — see [LICENSE](./LICENSE).
