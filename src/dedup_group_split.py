"""
Dedup-aware, group-stratified train/val/test split for the Malicious-Phish
dataset (651,191 URLs). Fixes the leakage flagged by both reviewers: the
10,072 exact-duplicate URLs are now guaranteed to stay entirely within a
single split (never split across train/val/test).

Usage:
    python dedup_group_split.py --seed 42
    python dedup_group_split.py --seed 123
    python dedup_group_split.py --seed 456

Adjust CSV_PATH, URL_COL, LABEL_COL below to match your dataset file.
"""

import argparse
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ---- CONFIG: pre-filled for Mehedi's setup ---------------------------------
CSV_PATH = "/Users/mehedi/Projects/phishformer/data/raw/malicious_phish.csv"
URL_COL = "url"
LABEL_COL = "type"
TEST_FRAC = 0.15                   # matches your original 15% test split
VAL_FRAC = 0.15                    # matches your original 15% val split
OUT_DIR_TEMPLATE = "splits_seed{seed}"
# -----------------------------------------------------------------------------


def group_stratified_split(df: pd.DataFrame, label_col: str, group_col: str,
                            frac: float, seed: int):
    """
    Split df into (held_out, remainder) such that:
      - held_out is approximately `frac` of df
      - class balance in held_out approximately matches df
      - no group (duplicate-URL cluster) appears in both held_out and remainder
    """
    n_splits = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df[label_col].values
    groups = df[group_col].values

    # Pick the fold whose held-out size is closest to the target fraction
    best_fold_idx, best_diff = None, float("inf")
    fold_indices = list(sgkf.split(df, y, groups))
    for i, (_, held_idx) in enumerate(fold_indices):
        diff = abs(len(held_idx) / len(df) - frac)
        if diff < best_diff:
            best_diff, best_fold_idx = diff, i

    train_idx, held_idx = fold_indices[best_fold_idx]
    return df.iloc[train_idx].copy(), df.iloc[held_idx].copy()


def main(seed: int):
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df):,}")

    # Group ID: all rows with an identical URL string share a group.
    df["_dup_group"] = df.groupby(URL_COL).ngroup()
    n_dup_groups = (df["_dup_group"].value_counts() > 1).sum()
    n_dup_rows = df["_dup_group"].map(df["_dup_group"].value_counts()) .gt(1).sum()
    print(f"Duplicate URL groups: {n_dup_groups:,} (covering {n_dup_rows:,} rows)")

    # Stage 1: carve off test set
    trainval_df, test_df = group_stratified_split(
        df, LABEL_COL, "_dup_group", TEST_FRAC, seed
    )

    # Stage 2: carve off val set from remaining train+val pool
    # (val fraction is relative to the ORIGINAL dataset size, so rescale)
    val_frac_of_trainval = VAL_FRAC / (1 - TEST_FRAC)
    train_df, val_df = group_stratified_split(
        trainval_df, LABEL_COL, "_dup_group", val_frac_of_trainval, seed
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.drop(columns=["_dup_group"], inplace=True)

    # ---- Verification: assert zero cross-split leakage ----
    train_urls = set(train_df[URL_COL])
    val_urls = set(val_df[URL_COL])
    test_urls = set(test_df[URL_COL])

    leak_test = test_urls & train_urls
    leak_val = val_urls & train_urls
    leak_val_test = val_urls & test_urls

    print(f"\nSplit sizes -> train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
    print(f"Class distribution (test):\n{test_df[LABEL_COL].value_counts(normalize=True).round(4)}")
    print(f"\nLeakage check:")
    print(f"  train ∩ test URL overlap: {len(leak_test)} rows  (expect 0)")
    print(f"  train ∩ val  URL overlap: {len(leak_val)} rows  (expect 0)")
    print(f"  val   ∩ test URL overlap: {len(leak_val_test)} rows  (expect 0)")
    assert len(leak_test) == 0 and len(leak_val) == 0 and len(leak_val_test) == 0, \
        "Leakage still present — check group assignment logic."
    print("  PASS: zero exact-duplicate leakage across splits.")

    out_dir = OUT_DIR_TEMPLATE.format(seed=seed)
    import os
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    val_df.to_csv(f"{out_dir}/val.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)
    print(f"\nSaved to {out_dir}/{{train,val,test}}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[42, 123, 456])
    args = parser.parse_args()
    main(args.seed)
