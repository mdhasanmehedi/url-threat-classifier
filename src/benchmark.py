"""
benchmark.py — PhishFormer
Measures inference latency, throughput, memory usage, and model size
for all trained models on both CPU and MPS (Apple Silicon GPU).

Results support the deployment claim in Section 5.3 and provide
the latency/throughput table for Section 6 (or a new Section 5.4).

Usage:
  python3 src/benchmark.py
  python3 src/benchmark.py --n_urls 1000 --n_warmup 100
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, get_logger, load_checkpoint, device_info
from data import tokenize_url, MAX_LEN, PAD_IDX
from models import PhishFormer, CNNOnly, TransformerOnly, LSTMModel, BiLSTMModel

logger = get_logger()
RESULTS_DIR  = "results"
CKPT_DIR     = "checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "phishformer": (PhishFormer,      "phishformer_seed42_best.pt"),
    "cnn":         (CNNOnly,          "cnn_seed42_best.pt"),
    "transformer": (TransformerOnly,  "transformer_seed42_best.pt"),
    "lstm":        (LSTMModel,        "lstm_seed42_best.pt"),
    "bilstm":      (BiLSTMModel,      "bilstm_seed42_best.pt"),
}

# Representative URL samples for benchmarking
SAMPLE_URLS = [
    "https://paypal.com-secure-login.phishing.xyz/account?token=abc123",
    "https://google.com",
    "http://malware-distribution-c2-server.ru/payload.exe?id=99",
    "https://legitimate-bank.com/login/secure",
    "http://192.168.1.1/admin/panel",
    "https://amazon.com-deals.phishing-site.tk/verify",
    "http://update-adobe-flash.malware.ru/install.exe",
    "https://www.facebook.com",
    "http://hack3d-s1te.defacement.xyz/index.html",
    "https://microsoft.com",
]


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Return model size in MB (parameter memory only)."""
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def get_model_disk_size_mb(ckpt_path: str) -> float:
    """Return checkpoint file size on disk in MB."""
    if os.path.exists(ckpt_path):
        return os.path.getsize(ckpt_path) / (1024 ** 2)
    return float("nan")


def benchmark_model(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    n_urls: int = 1000,
    n_warmup: int = 100,
    batch_sizes: list = None,
) -> dict:
    """
    Benchmark a model on a given device.

    Measures:
    - Single-URL latency (mean ± std, p50, p95, p99) in ms
    - Batch throughput (URLs/second) for batch sizes [1, 32, 128, 512]
    - Peak memory usage during inference (MB)
    - Model parameter memory (MB)
    - Checkpoint disk size (MB)
    """
    if batch_sizes is None:
        batch_sizes = [1, 32, 128, 512]

    model = model.to(device)
    model.eval()

    # Prepare token sequences
    tokens_list = []
    for i in range(max(n_urls, max(batch_sizes))):
        url = SAMPLE_URLS[i % len(SAMPLE_URLS)]
        tokens_list.append(tokenize_url(url, MAX_LEN))
    all_tokens = torch.tensor(tokens_list, dtype=torch.long)

    results = {}

    # ── 1. Single-URL latency ──────────────────────────────────────────────────
    logger.info(f"  [{device.type.upper()}] Single-URL latency ({n_urls} URLs)...")
    single_token = all_tokens[0].unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(single_token)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for i in range(n_urls):
            x = all_tokens[i % len(all_tokens)].unsqueeze(0).to(device)
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)
    results["latency_mean_ms"]   = float(np.mean(latencies))
    results["latency_std_ms"]    = float(np.std(latencies))
    results["latency_p50_ms"]    = float(np.percentile(latencies, 50))
    results["latency_p95_ms"]    = float(np.percentile(latencies, 95))
    results["latency_p99_ms"]    = float(np.percentile(latencies, 99))

    logger.info(
        f"    Latency: {results['latency_mean_ms']:.3f} ± "
        f"{results['latency_std_ms']:.3f} ms | "
        f"p50={results['latency_p50_ms']:.3f} | "
        f"p95={results['latency_p95_ms']:.3f} | "
        f"p99={results['latency_p99_ms']:.3f}"
    )

    # ── 2. Throughput by batch size ────────────────────────────────────────────
    throughputs = {}
    for bs in batch_sizes:
        batch = all_tokens[:bs].to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = model(batch)
        if device.type == "mps":
            torch.mps.synchronize()

        # Timed
        n_batches = max(50, 1000 // bs)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_batches):
                _ = model(batch)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed   = t1 - t0
        total_urls = n_batches * bs
        urls_per_sec = total_urls / elapsed
        throughputs[bs] = float(urls_per_sec)
        logger.info(f"    Throughput (batch={bs}): {urls_per_sec:,.0f} URLs/sec")

    results["throughput_urls_per_sec"] = throughputs

    # ── 3. Peak memory usage ───────────────────────────────────────────────────
    peak_memory_mb = float("nan")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            batch = all_tokens[:512].to(device)
            _ = model(batch)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    elif device.type == "mps":
        # MPS doesn't expose peak memory directly
        # Estimate from model size + activation memory
        param_mb = get_model_size_mb(model)
        # Rough activation estimate: batch_size * seq_len * d_model * 4 bytes
        activation_mb = (512 * MAX_LEN * 384 * 4) / (1024 ** 2)
        peak_memory_mb = param_mb + activation_mb

    elif device.type == "cpu":
        import tracemalloc
        tracemalloc.start()
        with torch.no_grad():
            batch = all_tokens[:512].to(device)
            _ = model(batch)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / (1024 ** 2)

    results["peak_memory_mb"] = float(peak_memory_mb)
    logger.info(f"    Peak memory: {peak_memory_mb:.1f} MB")

    return results


def run_all_benchmarks(
    n_urls: int = 1000,
    n_warmup: int = 100,
    seed: int = 42,
) -> dict:
    """Run benchmarks for all models on both CPU and MPS."""
    set_seed(seed)
    all_results = {}

    for model_name, (ModelClass, ckpt_filename) in MODEL_REGISTRY.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_filename)
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_name.upper()}")
        logger.info(f"{'='*60}")

        model = ModelClass()

        # Load checkpoint
        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path, device=torch.device("cpu"))
            logger.info(f"Loaded checkpoint from {ckpt_path}")
        else:
            logger.warning(f"No checkpoint at {ckpt_path} — using untrained weights")

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_memory_mb = get_model_size_mb(model)
        disk_size_mb = get_model_disk_size_mb(ckpt_path)

        logger.info(f"Parameters     : {param_count:,}")
        logger.info(f"Param memory   : {param_memory_mb:.2f} MB")
        logger.info(f"Disk size      : {disk_size_mb:.2f} MB")

        model_results = {
            "param_count":      param_count,
            "param_memory_mb":  float(param_memory_mb),
            "disk_size_mb":     float(disk_size_mb),
            "devices":          {},
        }

        # Benchmark on CPU
        logger.info(f"\n  Running CPU benchmark...")
        model_cpu = ModelClass()
        if os.path.exists(ckpt_path):
            load_checkpoint(model_cpu, ckpt_path, device=torch.device("cpu"))
        cpu_results = benchmark_model(
            model_name, model_cpu, torch.device("cpu"),
            n_urls=n_urls, n_warmup=n_warmup
        )
        model_results["devices"]["cpu"] = cpu_results

        # Benchmark on MPS if available
        if torch.backends.mps.is_available():
            logger.info(f"\n  Running MPS (Apple Silicon GPU) benchmark...")
            model_mps = ModelClass()
            if os.path.exists(ckpt_path):
                load_checkpoint(model_mps, ckpt_path, device=torch.device("mps"))
            mps_results = benchmark_model(
                model_name, model_mps, torch.device("mps"),
                n_urls=n_urls, n_warmup=n_warmup
            )
            model_results["devices"]["mps"] = mps_results

        all_results[model_name] = model_results

    return all_results


def format_benchmark_table(all_results: dict) -> str:
    """Format results as paper-ready table."""
    model_display = {
        "phishformer": "PhishFormer (proposed)",
        "cnn":         "CNN Only",
        "transformer": "Transformer Only",
        "lstm":        "LSTM",
        "bilstm":      "BiLSTM",
    }

    lines = []
    lines.append("=" * 110)
    lines.append("INFERENCE BENCHMARK RESULTS")
    lines.append("=" * 110)

    # CPU latency table
    lines.append("\nSINGLE-URL LATENCY (ms) — mean±std | p95")
    lines.append("-" * 90)
    lines.append(f"{'Model':<25} {'Params':>10} {'Param MB':>10} {'Disk MB':>10} "
                 f"{'CPU lat (ms)':>18} {'CPU p95 (ms)':>15} "
                 f"{'MPS lat (ms)':>18} {'MPS p95 (ms)':>15}")
    lines.append("-" * 90)

    for m, r in all_results.items():
        name = model_display.get(m, m)
        cpu = r["devices"].get("cpu", {})
        mps = r["devices"].get("mps", {})

        cpu_lat = f"{cpu.get('latency_mean_ms', float('nan')):.3f}±{cpu.get('latency_std_ms', float('nan')):.3f}"
        cpu_p95 = f"{cpu.get('latency_p95_ms', float('nan')):.3f}"
        mps_lat = f"{mps.get('latency_mean_ms', float('nan')):.3f}±{mps.get('latency_std_ms', float('nan')):.3f}" if mps else "N/A"
        mps_p95 = f"{mps.get('latency_p95_ms', float('nan')):.3f}" if mps else "N/A"

        lines.append(
            f"{name:<25} {r['param_count']:>10,} {r['param_memory_mb']:>10.2f} "
            f"{r['disk_size_mb']:>10.2f} {cpu_lat:>18} {cpu_p95:>15} "
            f"{mps_lat:>18} {mps_p95:>15}"
        )

    lines.append("\nTHROUGHPUT (URLs/second) by batch size")
    lines.append("-" * 90)
    lines.append(f"{'Model':<25} {'CPU bs=1':>12} {'CPU bs=32':>12} "
                 f"{'CPU bs=128':>12} {'CPU bs=512':>12} "
                 f"{'MPS bs=512':>12}")
    lines.append("-" * 90)

    for m, r in all_results.items():
        name = model_display.get(m, m)
        cpu = r["devices"].get("cpu", {}).get("throughput_urls_per_sec", {})
        mps = r["devices"].get("mps", {}).get("throughput_urls_per_sec", {})

        lines.append(
            f"{name:<25} "
            f"{cpu.get(1, float('nan')):>12,.0f} "
            f"{cpu.get(32, float('nan')):>12,.0f} "
            f"{cpu.get(128, float('nan')):>12,.0f} "
            f"{cpu.get(512, float('nan')):>12,.0f} "
            f"{mps.get(512, float('nan')):>12,.0f}"
        )

    lines.append("=" * 110)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_urls",   type=int, default=1000)
    parser.add_argument("--n_warmup", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    logger.info("PhishFormer Inference Benchmark")
    logger.info(f"n_urls={args.n_urls}, n_warmup={args.n_warmup}")

    results = run_all_benchmarks(
        n_urls=args.n_urls,
        n_warmup=args.n_warmup,
        seed=args.seed,
    )

    table = format_benchmark_table(results)
    logger.info(f"\n{table}")

    # Save
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    txt_path  = os.path.join(RESULTS_DIR, "benchmark_results.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(txt_path, "w") as f:
        f.write(table)

    logger.info(f"\nResults saved to {json_path}")
    logger.info(f"Table saved to {txt_path}")
    logger.info("Benchmark complete.")
