# Data

## Malicious-Phish (primary training dataset)

This project uses the **Malicious-Phish** dataset (651,191 URLs across four classes: benign, phishing, malware, defacement), created by sid321axn and published on Kaggle:

**https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset**

The raw CSV is not committed to this repository (to keep the repo lightweight and to always point users to the canonical, up-to-date source). To reproduce:

1. Download `malicious_phish.csv` from the Kaggle link above (requires a free Kaggle account).
2. Place it at `data/raw/malicious_phish.csv`.
3. Run `src/data.py` to verify preprocessing and see the class-distribution sanity check.

Expected columns: `url`, `type` (one of `benign`, `phishing`, `malware`, `defacement`).

## PhiUSIIL (cross-dataset domain adaptation)

Used in `src/cross_dataset_eval.py` and `src/finetune_phiusiil.py` for the domain-adaptation experiments (Section 6.7 of the paper). 235,795 URLs (100,945 legitimate, 134,850 phishing).

Dataset source: PhiUSIIL (see paper references for full citation). Download and place under `data/phiusiil/`.

## Splits

All train/validation/test splits (70/15/15, stratified, `seed=42`) are generated deterministically by `src/data.py` — no split files are stored separately. Running `load_and_split()` with the same seed will reproduce the exact same split used in all reported experiments.
