"""
data.py — PhishFormer
Character-level URL tokenization, dataset class, stratified splits,
and class-weight computation for weighted cross-entropy loss.
"""

import os
import string
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils import get_logger, set_seed

logger = get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

# Label mapping — order is fixed; do not change after training begins
LABEL2IDX: Dict[str, int] = {
    "benign": 0,
    "defacement": 1,
    "phishing": 2,
    "malware": 3,
}
IDX2LABEL: Dict[int, str] = {v: k for k, v in LABEL2IDX.items()}
NUM_CLASSES: int = len(LABEL2IDX)

# Character vocabulary:
# printable ASCII minus whitespace, plus a dedicated PAD and UNK token.
# This covers every character realistically appearing in a URL.
_CHARS = string.printable.strip()           # 94 printable non-whitespace chars
PAD_TOKEN = "<PAD>"                         # index 0  — padding
UNK_TOKEN = "<UNK>"                         # index 1  — unknown character
VOCAB: List[str] = [PAD_TOKEN, UNK_TOKEN] + list(_CHARS)
CHAR2IDX: Dict[str, int] = {ch: idx for idx, ch in enumerate(VOCAB)}
VOCAB_SIZE: int = len(VOCAB)               # 96

PAD_IDX: int = CHAR2IDX[PAD_TOKEN]        # 0
UNK_IDX: int = CHAR2IDX[UNK_TOKEN]        # 1

# Maximum URL length — covers >99% of URLs in this dataset.
# URLs longer than this are truncated from the right;
# shorter ones are right-padded with PAD_IDX.
MAX_LEN: int = 200


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_url(url: str, max_len: int = MAX_LEN) -> List[int]:
    """
    Convert a URL string to a fixed-length list of character indices.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Map each character to its index (UNK_IDX for unseen chars).
      3. Truncate to max_len or right-pad with PAD_IDX.

    Returns:
        List[int] of length max_len.
    """
    url = url.strip()
    indices = [CHAR2IDX.get(ch, UNK_IDX) for ch in url[:max_len]]
    # Pad if shorter than max_len
    indices += [PAD_IDX] * (max_len - len(indices))
    return indices


# ── Dataset class ─────────────────────────────────────────────────────────────

class URLDataset(Dataset):
    """
    PyTorch Dataset for character-level URL classification.

    Args:
        urls:   List of raw URL strings.
        labels: List of integer class indices (0–3).
        max_len: Sequence length after tokenization.
    """

    def __init__(
        self,
        urls: List[str],
        labels: List[int],
        max_len: int = MAX_LEN,
    ) -> None:
        assert len(urls) == len(labels), "urls and labels must have equal length"
        self.urls = urls
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = tokenize_url(self.urls[idx], self.max_len)
        x = torch.tensor(token_ids, dtype=torch.long)       # shape: (max_len,)
        y = torch.tensor(self.labels[idx], dtype=torch.long) # scalar
        return x, y


# ── Data loading and splitting ────────────────────────────────────────────────

def load_and_split(
    csv_path: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    # test_ratio is implicitly 1 - train_ratio - val_ratio = 0.15
) -> Tuple[URLDataset, URLDataset, URLDataset, torch.Tensor]:
    """
    Load the Malicious-Phish CSV, encode labels, perform stratified
    train/val/test split, and compute class weights.

    Args:
        csv_path:    Path to malicious_phish.csv.
        seed:        Random seed for reproducibility.
        train_ratio: Fraction of data for training.
        val_ratio:   Fraction of data for validation.

    Returns:
        (train_dataset, val_dataset, test_dataset, class_weights)
        class_weights is a float32 Tensor of shape (NUM_CLASSES,),
        intended for use as the `weight` argument to nn.CrossEntropyLoss.
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate expected columns
    assert "url" in df.columns and "type" in df.columns, (
        f"Expected columns 'url' and 'type', got {list(df.columns)}"
    )

    # Drop rows with null URLs or unknown labels
    df = df.dropna(subset=["url", "type"])
    unknown = ~df["type"].isin(LABEL2IDX.keys())
    if unknown.sum() > 0:
        logger.warning(f"Dropping {unknown.sum()} rows with unrecognised labels")
        df = df[~unknown]

    urls = df["url"].tolist()
    labels = [LABEL2IDX[t] for t in df["type"].tolist()]

    logger.info(f"Total samples after cleaning: {len(urls):,}")
    for name, idx in LABEL2IDX.items():
        count = labels.count(idx)
        logger.info(f"  {name:>12s}: {count:>7,}  ({100*count/len(labels):.1f}%)")

    # ── Stratified split: train / (val + test) ────────────────────────────────
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    X = np.array(urls)
    y = np.array(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=seed,
    )
    # Split the temp portion into val and test, maintaining stratification
    relative_val = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - relative_val),
        stratify=y_temp,
        random_state=seed,
    )

    logger.info(
        f"Split sizes — train: {len(X_train):,} | "
        f"val: {len(X_val):,} | test: {len(X_test):,}"
    )

    # ── Class weights for weighted cross-entropy ──────────────────────────────
    # Using sklearn's 'balanced' mode: weight_i = n_samples / (n_classes * n_i)
    # This up-weights the minority malware class automatically.
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)
    logger.info(f"Class weights (balanced): {dict(zip(IDX2LABEL.values(), cw.round(4)))}")

    train_ds = URLDataset(X_train.tolist(), y_train.tolist())
    val_ds   = URLDataset(X_val.tolist(),   y_val.tolist())
    test_ds  = URLDataset(X_test.tolist(),  y_test.tolist())

    return train_ds, val_ds, test_ds, class_weights


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    train_ds: URLDataset,
    val_ds: URLDataset,
    test_ds: URLDataset,
    batch_size: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap datasets in DataLoaders.

    num_workers=0 is required on macOS with MPS to avoid multiprocessing
    issues with PyTorch's MPS backend.  Increase to 2–4 on Linux/CUDA.
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,   # pin_memory is CUDA-only; not used with MPS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    set_seed(42)

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/malicious_phish.csv"

    train_ds, val_ds, test_ds, class_weights = load_and_split(csv_path)
    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds)

    # ── Tokenization sanity checks ────────────────────────────────────────────
    logger.info("── Tokenization sanity checks ──")
    sample_urls = [
        "https://paypal.com-secure-login.phishing.xyz/account",   # phishing
        "https://google.com",                                       # benign
        "http://malware-c2.ru/payload.exe",                        # malware
    ]
    for url in sample_urls:
        tokens = tokenize_url(url)
        logger.info(f"URL   : {url}")
        logger.info(f"Tokens: {tokens[:20]} ... (len={len(tokens)})")
        assert len(tokens) == MAX_LEN, "Token length mismatch"

    # ── Batch shape check ─────────────────────────────────────────────────────
    logger.info("── Batch shape check ──")
    x_batch, y_batch = next(iter(train_loader))
    logger.info(f"x_batch shape : {x_batch.shape}")   # expect (512, 200)
    logger.info(f"y_batch shape : {y_batch.shape}")   # expect (512,)
    logger.info(f"x dtype       : {x_batch.dtype}")   # expect torch.int64
    logger.info(f"y dtype       : {y_batch.dtype}")   # expect torch.int64
    logger.info(f"y unique vals : {y_batch.unique()}")
    logger.info(f"Class weights : {class_weights}")
    logger.info(f"Vocab size    : {VOCAB_SIZE}")
    logger.info("data.py OK")
