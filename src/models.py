"""
models.py — PhishFormer
All model architectures in one place:
  1. PhishFormer      — hybrid CNN + Transformer (full-resolution fusion)
  2. CNNOnly          — Baseline 2: multi-filter CNN with global max pooling
  3. TransformerOnly  — Baseline 3: positional embedding + Transformer encoder
  4. LSTMModel        — Baseline 4: single-layer LSTM
  5. BiLSTMModel      — Baseline 5: bidirectional LSTM
  6. RandomForest     — Baseline 1: classical ML (sklearn, not nn.Module)

Architecture constants match Section 4 of the paper exactly.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB_SIZE, NUM_CLASSES, MAX_LEN, PAD_IDX

# ── Shared hyperparameters (Section 4.7) ─────────────────────────────────────
EMBED_DIM        = 128   # character embedding dimension
CNN_FILTERS      = 128   # filters per kernel size in CNN component
CNN_KERNELS      = (3, 4, 5)  # n-gram window sizes
TRANSFORMER_HEADS    = 4
TRANSFORMER_LAYERS   = 2
TRANSFORMER_FF_DIM   = 256   # feed-forward dimension inside transformer
DROPOUT          = 0.3
LSTM_HIDDEN      = 256


# ─────────────────────────────────────────────────────────────────────────────
# 1. PhishFormer  (proposed model)
# ─────────────────────────────────────────────────────────────────────────────

class PhishFormer(nn.Module):
    """
    Hybrid CNN-Transformer for character-level URL classification.

    Key design choice (Contribution 1):
      The CNN extracts local n-gram features while PRESERVING the full
      sequence length (200 positions).  The resulting position-aware
      feature map is passed directly to the Transformer encoder, so
      self-attention can reason over spatial relationships between
      local patterns — unlike prior hybrids (Hu & Xu 2023; Asiri 2023)
      that pool the CNN output to a single vector before the Transformer,
      destroying positional information.

    Architecture:
      Embedding(96, 128)
        → [Conv1d(k=3), Conv1d(k=4), Conv1d(k=5)]  each: 128 filters
        → ReLU + concatenate along filter dim  →  (B, 384, 200)
        → transpose  →  (B, 200, 384)           [full-resolution sequence]
        → TransformerEncoder(d_model=384, nhead=4, layers=2)
        → mean pooling over sequence  →  (B, 384)
        → Dropout → Linear(384, 4)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        num_filters: int = CNN_FILTERS,
        kernel_sizes: Tuple[int, ...] = CNN_KERNELS,
        num_heads: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        ff_dim: int = TRANSFORMER_FF_DIM,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        max_len: int = MAX_LEN,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        # ── Character embedding ───────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # ── Parallel CNN towers (one per kernel size) ─────────────────────────
        # Input to Conv1d: (B, embed_dim, seq_len)
        # Output per tower: (B, num_filters, seq_len)  [same-length via padding]
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,   # 'same' padding to preserve sequence length
            )
            for k in kernel_sizes
        ])

        # Total CNN output channels after concatenation along filter dim
        cnn_out_dim = num_filters * len(kernel_sizes)  # 128*3 = 384

        # ── Transformer encoder ───────────────────────────────────────────────
        # d_model must equal cnn_out_dim so we feed CNN output directly
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,    # input/output: (B, seq_len, d_model)
            norm_first=True,     # pre-norm (more stable than post-norm)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # required for stability on macOS MPS/CPU
        )

        # ── Classification head ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(cnn_out_dim, num_classes)

        # Store for attention extraction (Ablation B)
        self._attention_weights = None
        self._register_attention_hook()

    def _register_attention_hook(self) -> None:
        """
        Register a forward hook on the last TransformerEncoderLayer's
        self-attention module to cache attention weights for
        the faithfulness evaluation (Ablation B / Section 4.6).
        """
        last_layer = self.transformer.layers[-1]

        def hook(module, input, output):
            # output is (attn_output, attn_weights) when need_weights=True
            # but TransformerEncoderLayer doesn't expose weights by default.
            # We handle attention extraction explicitly in get_attention_weights().
            pass

        last_layer.register_forward_hook(hook)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: LongTensor of shape (B, seq_len) — character indices
            return_attention: if True, also return attention weights
                              from the last transformer layer

        Returns:
            logits of shape (B, num_classes)
            [optionally: (logits, attention_weights)]
        """
        # Padding mask: True where x == PAD_IDX (Transformer ignores these)
        pad_mask = (x == self.pad_idx)  # (B, seq_len)

        # ── Embedding ─────────────────────────────────────────────────────────
        emb = self.embedding(x)          # (B, seq_len, embed_dim)
        emb = emb.transpose(1, 2)        # (B, embed_dim, seq_len) for Conv1d

        # ── Parallel CNN towers ───────────────────────────────────────────────
        seq_len = emb.size(2)
        conv_outs = []
        for conv in self.conv_layers:
            out = F.relu(conv(emb))          # (B, num_filters, ~seq_len)
            out = out[:, :, :seq_len]        # trim to exact seq_len
            conv_outs.append(out)

        # Concatenate along filter dimension, keep sequence length intact
        cnn_out = torch.cat(conv_outs, dim=1)   # (B, 384, seq_len)
        cnn_out = cnn_out.transpose(1, 2)        # (B, seq_len, 384)

        # ── Transformer encoder ───────────────────────────────────────────────
        if return_attention:
            # Manual pass through layers to extract last-layer attention
            hidden = cnn_out
            attn_weights = None
            for i, layer in enumerate(self.transformer.layers):
                if i == len(self.transformer.layers) - 1:
                    normed = layer.norm1(hidden)
                    attn_out, attn_weights = layer.self_attn(
                        normed, normed, normed,
                        key_padding_mask=pad_mask,
                        need_weights=True,
                        average_attn_weights=True,
                    )
                    hidden = hidden + layer.dropout1(attn_out)
                    hidden = hidden + layer._ff_block(layer.norm2(hidden))
                else:
                    hidden = layer(hidden, src_key_padding_mask=pad_mask)
            transformer_out = hidden
        else:
            transformer_out = self.transformer(
                cnn_out,
                src_key_padding_mask=pad_mask,
            )  # (B, seq_len, 384)

        # ── Mean pooling (ignore padding positions) ───────────────────────────
        # Build a mask to zero out padded positions before averaging
        mask = (~pad_mask).float().unsqueeze(-1)     # (B, seq_len, 1)
        pooled = (transformer_out * mask).sum(dim=1) # (B, 384)
        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)

        # ── Classification ────────────────────────────────────────────────────
        logits = self.classifier(self.dropout(pooled))   # (B, num_classes)

        if return_attention:
            return logits, attn_weights   # attn_weights: (B, seq_len, seq_len)
        return logits

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method used in ablations.py (Ablation B).
        Returns (logits, per-character importance scores).

        Importance score for position i = mean attention weight received
        by position i across all query positions in the last layer.
        Shape: (B, seq_len).
        """
        logits, attn_weights = self.forward(x, return_attention=True)
        # attn_weights: (B, seq_len, seq_len) — [query, key]
        # Sum over query dimension to get how much each key position
        # was attended to in total
        importance = attn_weights.sum(dim=1)    # (B, seq_len)
        # Normalise per sample to [0, 1]
        importance = importance / importance.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return logits, importance

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN Only  (Baseline 2)
# ─────────────────────────────────────────────────────────────────────────────

class CNNOnly(nn.Module):
    """
    Character-level CNN with parallel multi-size filters and
    global max pooling — the standard TextCNN baseline.

    Architecture:
      Embedding → [Conv1d(k=3), Conv1d(k=4), Conv1d(k=5)]
      → ReLU → GlobalMaxPool → concatenate → Dropout → Linear(4)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        num_filters: int = CNN_FILTERS,
        kernel_sizes: Tuple[int, ...] = CNN_KERNELS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)   # (B, embed_dim, seq_len)
        pooled = []
        for conv in self.conv_layers:
            out = F.relu(conv(emb))                # (B, num_filters, seq_len)
            out = out.max(dim=2).values            # (B, num_filters) global max
            pooled.append(out)
        cat = torch.cat(pooled, dim=1)             # (B, 384)
        return self.classifier(self.dropout(cat))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transformer Only  (Baseline 3)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = MAX_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerOnly(nn.Module):
    """
    Character-level Transformer encoder with sinusoidal positional encoding
    and mean pooling for classification.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        num_heads: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        ff_dim: int = TRANSFORMER_FF_DIM,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        max_len: int = MAX_LEN,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_mask = (x == self.pad_idx)
        emb = self.pos_encoding(self.embedding(x))    # (B, seq_len, embed_dim)
        out = self.transformer(emb, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM  (Baseline 4)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Single-layer unidirectional LSTM over character embeddings."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = LSTM_HIDDEN,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                   # (B, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(emb)              # h_n: (1, B, hidden_dim)
        return self.classifier(self.dropout(h_n.squeeze(0)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BiLSTM  (Baseline 5)
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMModel(nn.Module):
    """Single-layer bidirectional LSTM over character embeddings."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = LSTM_HIDDEN,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional: final hidden = [forward; backward] → 2 * hidden_dim
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.bilstm(emb)    # h_n: (2, B, hidden_dim)
        # Concatenate forward and backward final hidden states
        h = torch.cat([h_n[0], h_n[1]], dim=1)   # (B, hidden_dim*2)
        return self.classifier(self.dropout(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Random Forest  (Baseline 1 — classical ML, sklearn)
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestBaseline:
    """
    Random Forest over hand-crafted URL features.
    Not a nn.Module — uses sklearn directly.

    Features (18 total, matching Section 5.1 of the paper):
      - URL length
      - Number of dots, hyphens, underscores, slashes, question marks,
        equals signs, at signs, percent signs, ampersands
      - Number of digits, uppercase letters
      - Presence of 'https' (binary)
      - Subdomain depth (number of dots in hostname)
      - Path depth (number of slashes)
      - Presence of IP address pattern (binary)
      - Length of longest numeric run
      - Entropy of URL characters
    """

    def __init__(self, n_estimators: int = 100, seed: int = 42) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )

    @staticmethod
    def extract_features(urls: list) -> "np.ndarray":
        import re
        import math
        import numpy as np
        from collections import Counter

        def entropy(s: str) -> float:
            if not s:
                return 0.0
            counts = Counter(s)
            probs = [c / len(s) for c in counts.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        def longest_numeric_run(s: str) -> int:
            runs = re.findall(r"\d+", s)
            return max((len(r) for r in runs), default=0)

        def ip_present(s: str) -> int:
            return int(bool(re.search(r"\d{1,3}(\.\d{1,3}){3}", s)))

        feats = []
        for url in urls:
            url = str(url)
            feats.append([
                len(url),
                url.count("."),
                url.count("-"),
                url.count("_"),
                url.count("/"),
                url.count("?"),
                url.count("="),
                url.count("@"),
                url.count("%"),
                url.count("&"),
                sum(c.isdigit() for c in url),
                sum(c.isupper() for c in url),
                int("https" in url.lower()),
                url.split("/")[2].count(".") if len(url.split("/")) > 2 else url.count("."),
                url.count("/") - 2 if url.count("/") > 2 else 0,
                ip_present(url),
                longest_numeric_run(url),
                entropy(url),
            ])
        return np.array(feats, dtype=np.float32)

    def fit(self, urls: list, labels: list) -> None:
        X = self.extract_features(urls)
        self.model.fit(X, labels)

    def predict(self, urls: list) -> "np.ndarray":
        X = self.extract_features(urls)
        return self.model.predict(X)

    def predict_proba(self, urls: list) -> "np.ndarray":
        X = self.extract_features(urls)
        return self.model.predict_proba(X)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(name: str) -> nn.Module:
    """
    Return an untrained model by name.
    Valid names: 'phishformer', 'cnn', 'transformer', 'lstm', 'bilstm'
    Use RandomForestBaseline() directly for the RF baseline.
    """
    name = name.lower().strip()
    registry = {
        "phishformer": PhishFormer,
        "cnn":         CNNOnly,
        "transformer": TransformerOnly,
        "lstm":        LSTMModel,
        "bilstm":      BiLSTMModel,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name]()


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import get_logger, get_device, set_seed, device_info

    set_seed(42)
    device = get_device()
    logger = get_logger()
    logger.info(f"Device: {device_info(device)}")

    B, L = 8, MAX_LEN
    dummy = torch.randint(2, VOCAB_SIZE, (B, L)).to(device)

    models_to_test = [
        ("PhishFormer",      PhishFormer()),
        ("CNN Only",         CNNOnly()),
        ("Transformer Only", TransformerOnly()),
        ("LSTM",             LSTMModel()),
        ("BiLSTM",           BiLSTMModel()),
    ]

    for name, model in models_to_test:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        params = model.count_parameters()
        logger.info(
            f"{name:<20s} | output: {tuple(out.shape)} "
            f"| params: {params:,}"
        )
        assert out.shape == (B, NUM_CLASSES), f"Shape mismatch for {name}"

    # Test attention extraction for PhishFormer
    pf = PhishFormer().to(device)
    pf.eval()
    with torch.no_grad():
        logits, importance = pf.get_attention_weights(dummy)
    logger.info(f"PhishFormer attention importance shape: {tuple(importance.shape)}")
    assert importance.shape == (B, L), "Attention importance shape mismatch"

    # Test Random Forest feature extraction (no training, just feature shape)
    rf = RandomForestBaseline()
    sample_urls = ["https://google.com", "http://phish.xyz/login", "malware.ru/pay.exe"]
    feats = rf.extract_features(sample_urls)
    logger.info(f"RF feature matrix shape: {feats.shape}")
    assert feats.shape == (3, 18), f"RF feature shape mismatch: {feats.shape}"

    logger.info("models.py OK")
