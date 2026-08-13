"""
utils.py — PhishFormer
Shared utilities: reproducibility seeding, device selection, logging.
"""

import os
import random
import logging
import numpy as np
import torch


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Pin all random sources to the same seed for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS does not expose a direct seed API but torch.manual_seed covers it
        torch.mps.manual_seed(seed)
    # Make cuDNN deterministic if CUDA is present
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Device selection ─────────────────────────────────────────────────────────

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Return the best available device in priority order:
      1. MPS  (Apple Silicon GPU)
      2. CUDA (NVIDIA GPU — for cloud/Colab runs)
      3. CPU  (fallback)

    Setting prefer_gpu=False forces CPU, which is useful for
    the per-URL inference latency benchmarks (Ablation A / Section 5.3).
    """
    if prefer_gpu:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Return a human-readable string describing the active device."""
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    if device.type == "cuda":
        return f"NVIDIA GPU — {torch.cuda.get_device_name(0)}"
    return "CPU"


# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str = "phishformer", level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes timestamped messages to stdout.
    Call once at the top of each script; re-calling returns the same logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    path: str,
) -> None:
    """Save model + optimizer state with epoch and best metric recorded."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a saved checkpoint into model (and optionally optimizer)."""
    map_location = device if device is not None else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    logger = get_logger()
    logger.info(f"Seed set to 42")
    logger.info(f"Active device: {device_info(device)}")
    # Smoke-test: create a small tensor on the selected device
    t = torch.tensor([1.0, 2.0, 3.0]).to(device)
    logger.info(f"Tensor on {device}: {t}")
    logger.info("utils.py OK")
