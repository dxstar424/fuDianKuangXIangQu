#!/usr/bin/env python3
"""
Benchmark HIP FlashAttention kernel vs PyTorch SDPA.

Run on SCNet to validate correctness and measure speedup BEFORE
integrating into vLLM and submitting to platform.

Usage:
    python scripts/bench_flash_attn.py [--seq-len 8192] [--warmup 5] [--iters 20]
"""

import argparse
import os
import sys
import time

import torch


def bench_torch_sdpa(q, k, v, scale, iters, causal=True):
    """Benchmark PyTorch scaled_dot_product_attention."""
    # Warmup
    for _ in range(5):
        _ = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            scale=scale, is_causal=causal,
        ).transpose(0, 1)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            scale=scale, is_causal=causal,
        ).transpose(0, 1)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out, dt / iters * 1000  # ms


def bench_hip_kernel(q, k, v, scale, iters):
    """Benchmark our HIP FlashAttention kernel."""
    # Import the loader — it will try to load the .so
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from fdu_vllm.flash_attn_loader import flash_attn_prefill, is_available

    if not is_available():
        print("[BENCH] HIP kernel not available (compile first: bash scripts/compile_kernels.sh)")
        return None, float("inf")

    # Warmup
    for _ in range(5):
        _ = flash_attn_prefill(q, k, v, scale)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = flash_attn_prefill(q, k, v, scale)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out, dt / iters * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-2,
                        help="Max relative error for correctness check")
    args = parser.parse_args()

    device = "cuda"
    if not torch.cuda.is_available():
        print("[BENCH] No CUDA/ROCm device — exiting")
        sys.exit(1)

    # Qwen3.5-27B params
    n_heads = 64
    n_kv_heads = 32
    head_dim = 128
    seq_len = args.seq_len
    dtype = torch.bfloat16
    scale = head_dim ** -0.5

    print(f"[BENCH] seq_len={seq_len}, n_heads={n_heads}, "
          f"n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"[BENCH] tensor size: Q={seq_len*64*128*2/1024/1024:.1f}MB "
          f"KV={2*seq_len*32*128*2/1024/1024:.1f}MB")

    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(seq_len, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)

    # ── Bench torch SDPA ──
    print("\n--- PyTorch SDPA ---")
    out_torch, dt_torch = bench_torch_sdpa(q, k, v, scale, args.iters)
    print(f"  Time: {dt_torch:.2f} ms")

    # ── Bench HIP kernel ──
    print("\n--- HIP FlashAttn ---")
    out_hip, dt_hip = bench_hip_kernel(q, k, v, scale, args.iters)
    if out_hip is None:
        print("[BENCH] HIP kernel not available. Compile first:")
        print("  bash scripts/compile_kernels.sh")
        print("Then copy .so to src/attention/hip_kernels/build/")
        sys.exit(1)
    print(f"  Time: {dt_hip:.2f} ms")

    # ── Correctness check ──
    max_err = (out_torch.float() - out_hip.float()).abs().max().item()
    rel_err = max_err / out_torch.float().abs().max().item()
    print(f"\n--- Correctness ---")
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Max relative error: {rel_err:.6f}")

    if rel_err < args.tolerance:
        print(f"  ✓ PASS (rel_err < {args.tolerance})")
    else:
        print(f"  ✗ FAIL (rel_err >= {args.tolerance}) — kernel has bugs!")

    # ── Speedup ──
    speedup = dt_torch / dt_hip
    print(f"\n--- Speedup ---")
    print(f"  {speedup:.2f}× vs PyTorch SDPA")
    if speedup > 1.5:
        print(f"  ✓ Worth integrating into vLLM")
    elif speedup > 1.0:
        print(f"  ⚠ Marginal gain — test at longer seq_len")
    else:
        print(f"  ✗ Slower than PyTorch — needs kernel optimization")


if __name__ == "__main__":
    main()
