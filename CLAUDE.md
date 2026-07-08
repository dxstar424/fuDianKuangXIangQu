# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026 先导杯 (Pioneer Cup) competition entry — **Qwen3.5-27B (bf16) inference optimization on a single DCU BW1000 accelerator** (CDNA3, gfx942) using vLLM 0.18.1. The goal is maximizing output throughput while staying within TTFT/TPOT SLA constraints (≤ baseline × 1.5).

## Commands

```bash
# Local dev benchmarking (pure stdlib, no pip install needed)
python scripts/benchmark.py --tier short                          # single tier only

# Full benchmark suite
bash launch.sh &                                                  # start optimized server
python scripts/benchmark.py --host localhost --port 8000 --output results/

# Baseline comparison
bash scripts/run_baseline.sh                                      # one-shot baseline pipeline
python scripts/compare.py results/baseline.json results/opt.json

# HIP kernel compilation (requires hipcc from DTK)
bash scripts/compile_kernels.sh
# or manually:
hipcc -O3 --offload-arch=gfx942 -std=c++17 -fPIC -shared \
  -o build/kernels/dcu_flash_attn.so src/attention/hip_kernels/dcu_flash_attn.cpp

# Environment check
bash scripts/check_env.sh
```

**Platform commands** (run on competition container, server on port 8001):
```bash
./run_throughput.sh all 10           # all tiers, 10 samples
./run_throughput.sh 4-8K 10          # single tier
./run_accuracy.sh hotpotqa 10        # F1 metric
./run_accuracy.sh gov_report 10      # ROUGE metric
```

## Container Environment

- **Platform**: SCNet → 容器服务 → 核心节点分区一 / hx1hdexclu06 队列
- **Image**: `qwen3.5-dtk26.04:0509` (clone to personal registry first)
- **Hardware**: 1× DCU BW1000 (CDNA3, gfx942), x86 host, ≥96 GB RAM
- **Stack**: Python 3.10.12, PyTorch 2.10.0 (ROCm), vLLM 0.18.1, transformers 5.5.0, DTK 26.04
- **Persistent storage**: `/public/home/xdzs2026_c415/` — 模型权重、源码、testdata 都在这里，重启不丢
- **Container home**: `/root/` — 非持久化，每次重启清空
- **Model**: Qwen3.5-27B (bf16), SHA256-locked
  - 持久路径: `/public/home/xdzs2026_c415/Qwen3.5-27B`
  - 容器内快速加载路径: `/data/Qwen3.5-27B` 或 `/root/Qwen3.5-27B`
- **vLLM source**: `http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git` branch `v0.18.1`
- **Platform scripts**: `/public/home/xdzs2026_c415/testdata/` (run_throughput.sh, run_accuracy.sh)
- **References**: HIP Programming Guide & DTK Software Guide at https://pra.xtnl.org.cn/

### Container Restart Procedure

每次容器重启后执行：

```bash
# 1. 复制模型到快速加载路径
cp -r /public/home/xdzs2026_c415/Qwen3.5-27B /root/Qwen3.5-27B

# 2. 重新编译安装 vLLM（仅改过源码时需要）
cd ~/vllm_cscc
python setup.py bdist_wheel
pip install dist/vllm-*.whl --no-deps --force-reinstall

# 3. 启动 baseline 服务（端口 8001，配合平台脚本）
cd /public/home/xdzs2026_c415/testdata
./start_vllm.sh &

# 4. Smoke test
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

## Architecture

The entire optimization suite is injected into vLLM through a single entry point: **`src/plugin.py → apply()`**. It monkey-patches vLLM internals at import time — no vLLM source modification needed. The plugin is activated by `export FDU_OPTIMIZE=1` (set in `launch.sh`).

### Module Dependency Flow

```
launch.sh (sets FDU_* env vars, starts vLLM server)
  └─ src/plugin.py:apply()
       ├─ _register_attention_backend()  → src/attention/dcu_attention.py
       │     └─ hip_kernels/dcu_flash_attn.cpp  (ctypes-loaded .so)
       ├─ _replace_scheduler()           → src/scheduler/custom_scheduler.py
       ├─ _replace_block_allocator()     → src/kv_cache/block_allocator.py
       └─ _inject_kv_quantization()      → src/quantization/kv_quant.py
                                            src/kv_cache/cache_manager.py
                                            src/executor/exec_path.py
```

### Key Design Decisions

- **Dual-path attention**: `DCUAttentionBackend` tries the compiled HIP `.so` first (ctypes), falls back to `F.scaled_dot_product_attention` for local dev without DCU hardware.
- **PyTorch ROCm compatibility**: On DCU, `torch.cuda.*` APIs transparently map to HIP — no separate `torch.hip` module exists. `torch.cuda.CUDAGraph` maps to `hipGraphCreate/Launch` underneath.
- **Config layering**: `src/config.py` merges Defaults → config.yaml → env vars (`FDU_*`), with validation.
- **All scripts use pure stdlib** (`scripts/benchmark.py`, `scripts/compare.py`) — no pip install needed for the evaluation toolchain.
- **KV FP8 quantization is online/non-persistent** per competition rules — quantize on write, dequantize on read, no cached quantized weights.

### vLLM Integration

The plugin patches vLLM at multiple levels (with multi-path fallback for v0.18.x compatibility):
1. Attention backend registration → `vllm.attention.selector` or `vllm.attention.backends.registry`
2. Scheduler replacement → `vllm.core.scheduler.Scheduler._schedule` or `.schedule`
3. Block allocator → `vllm.core.block_manager.BlockSpaceManager.allocate`
4. KV quantization → standalone injector (not a monkey-patch)

### Evaluation Model

Three-tier weighted scoring: short (20%, ~512 tokens, concurrency 8), medium (50%, ~4096 tokens, concurrency 32), long (30%, ~16384 tokens, concurrency 8). SLA violation on any tier zeroes that tier's throughput score. Final score = weighted throughput × accuracy coefficient (Δ ≤ 1% → 1.00).

## Competition Rules

- **Allowed**: KV Cache online FP8 quant, custom HIP kernels (hipcc/DTK), vLLM plugin/scheduler customization, operator-level low-precision compute, custom env vars (must document in `docs/env_vars.md`)
- **Prohibited**: Model weight modification, persistent quantization, structural changes (pruning/skipping), auxiliary models, speculative decoding, pre-cached answers, input truncation, network downloads during evaluation

## Coding Conventions

- **Python**: 4-space indent, `snake_case`, `PascalCase` classes. Scripts use pure stdlib.
- **HIP C++**: 2-space indent, `snake_case` globals, `UPPER_SNAKE` macros. `extern "C"` on host wrapper only (NOT on `__global__` kernel). Architecture: `--offload-arch=gfx942`.
- **Shell**: `set -e`, lowercase `snake_case`, `${VAR:-default}` for overrides.
- **Config naming**: Use `hip_graph` / `dcu_*` — never `cuda_*`. No nvidia-ml-py references.
- **Git commits**: `feat:` / `fix:` / `refactor:` / `docs:` / `perf:` prefixes. Push to GitLab only (not GitHub).

## Pre-Submission Checklist

- vLLM compiles: `python setup.py bdist_wheel`
- Server starts + health check passes
- `run_throughput.sh` and `run_accuracy.sh` complete
- `config.yaml` declares all tunable params
- `changelog.md` updated with change → expected → actual
- `docs/env_vars.md` lists all custom env vars with justification
- `checksum.txt` present (platform-generated)
