# Repository Guidelines

> 2026 先导杯 · FDU SCCSCC26 · Qwen3.5-27B (bf16) × vLLM 0.18.1 推理优化 @ DCU BW1000

## Path Warning

**On SCNet containers, `~` = `/root` (VOLATILE, lost on restart).**
Persistent home: `/public/home/xdzs2026_c415`. All paths below use this absolute path.
Do NOT store work under `/root` or `~`.

## Project Structure

```
fdu-sccscc26/
├── src/                   # All optimization modules (plugin entry: src/plugin.py → apply())
│   ├── kv_cache/          #   Tiered allocator (16/64/256), defrag, prefix cache, watermark
│   ├── attention/         #   DCU attention backend + HIP FlashAttention kernel (hip_kernels/)
│   ├── scheduler/         #   Length-aware prefill/decode decoupling
│   ├── quantization/      #   KV FP8 online quant (E4M3, non-persistent)
│   ├── executor/          #   HIP Graph capture, warmup, batch scheduling
│   └── utils/             #   RequestProfiler + DCUHardwareProfiler
├── baseline/              # Stock vLLM reference (launch.sh + config.yaml)
├── scripts/               # benchmark.py, compare.py, check_env.sh, compile_kernels.sh, run_baseline.sh
├── docs/env_vars.md       # Custom env-var documentation (required for submission)
├── launch.sh              # Optimized server entrypoint (accepts --model --port --tensor-parallel-size)
├── config.yaml            # All tunable parameters declared here
├── Dockerfile             # Based on competition/vllm-0.18.1-base:v1.0
├── changelog.md           # Per-submission delta log (required)
└── report.md              # Optimization analysis (required for final submission)
```

## Platform Environment

- **Platform**: SCNet → 容器服务 → 核心节点分区一 / hx1hdexclu06 队列
- **Image**: `qwen3.5-dtk26.04:0509` (clone to personal registry first)
- **Hardware**: 1× DCU BW1000 (CDNA3, gfx942), x86 host, ≥96 GB RAM
- **Stack**: Python 3.10.12, PyTorch 2.10.0 (ROCm), vLLM 0.18.1, transformers 5.5.0, DTK 26.04
- **Model**: Qwen3.5-27B (bf16), SHA256-locked, at `/public/home/xdzs2026_c415/Qwen3.5-27B` (persistent) → copy to `/root` for fast load
- **vLLM source**: `http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git` branch `v0.18.1`
- **References**: HIP Programming Guide & DTK Software Guide at https://pra.xtnl.org.cn/

## Container Lifecycle (every restart)

```bash
# 1. Copy model to local disk (fast load)
cp -r /public/home/xdzs2026_c415/Qwen3.5-27B /root/Qwen3.5-27B

# 2. Recompile vLLM — build on /tmp (local disk), NOT network storage
cp -r /public/home/xdzs2026_c415/vllm_cscc /tmp/vllm_cscc && cd /tmp/vllm_cscc
python setup.py bdist_wheel
cp dist/vllm-*.whl /public/home/xdzs2026_c415/vllm_cscc/dist/
pip install dist/vllm-*.whl --no-deps --force-reinstall

# 3. Launch server (listens on 127.0.0.1:8001)
cd /public/home/xdzs2026_c415/testdata && ./start_vllm.sh &

# 4. Smoke test (after model loads, ~10 min)
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

**Tip**: Run vLLM compile in one terminal and testdata download in another to save time.

## Build, Test, and Development Commands

```bash
# Platform throughput tests (run in /public/home/xdzs2026_c415/testdata, server on 8001)
./run_throughput.sh all 10          # All tiers, 10 samples each
./run_throughput.sh 4-8K 10         # Single tier: 4-8K / 8-16K / 16-32K

# Platform accuracy tests
./run_accuracy.sh hotpotqa 10       # F1 metric (QA)
./run_accuracy.sh gov_report 10     # ROUGE metric (summarization)

# Local dev (pure stdlib, no pip install needed)
python scripts/benchmark.py --tier short
python scripts/compare.py results/baseline.json results/opt.json
bash scripts/check_env.sh
bash scripts/compile_kernels.sh     # HIP kernel → .so via hipcc
```

## Evaluation & Scoring

Three-tier throughput with SLA hard constraints:

| Tier | Context | Weight | TTFT SLA (P99) | TPOT SLA (P99) |
|------|---------|--------|----------------|-----------------|
| 4-8K | 4K–8K | 20% | ≤ Baseline × 1.5 | ≤ Baseline × 1.5 |
| 8-16K | 8K–16K | 50% | ≤ Baseline × 1.5 | ≤ Baseline × 1.5 |
| 16-32K | 16K–32K | 30% | ≤ Baseline × 1.5 | ≤ Baseline × 1.5 |

Score = throughput_score × accuracy_coefficient. SLA violation zeroes that tier's score.
Accuracy: 4 tasks (hotpotqa F1, gov_report ROUGE, retrieval_multi_point, aggregation_keyword_aggregation), Δ ≤ 1% → coefficient 1.00.

## Competition Rules

**Allowed**: KV Cache online FP8 quant, custom HIP kernels (hipcc/DTK), vLLM plugin customization, operator-level low-precision compute, custom env vars (must document).

**Prohibited**: Model weight modification, persistent quantization, structural changes (pruning/skipping), auxiliary models, speculative decoding, pre-cached answers, input truncation, network downloads during evaluation.

## Coding Style & Naming Conventions

- **Python**: 4-space indent, `snake_case`, `PascalCase` classes. Scripts use pure stdlib.
- **HIP C++**: 2-space indent, `snake_case` globals, `UPPER_SNAKE` macros. Arch: `--offload-arch=gfx942`. `extern "C"` on host wrapper only, NOT on `__global__`. Follow official HIP Programming Guide.
- **Shell**: `set -e`, lowercase `snake_case`, `${VAR:-default}` for overrides.
- **Config naming**: `use_hip_graph` not `use_cuda_graph`. No nvidia-ml-py in requirements.

## Pre-Submission Checklist

- [ ] vLLM compiles: `python setup.py bdist_wheel`
- [ ] Wheel installs: `pip install dist/vllm-*.whl --no-deps --force-reinstall`
- [ ] Server starts + health check passes
- [ ] `curl` single inference returns valid response
- [ ] `run_throughput.sh` completes, `run_accuracy.sh` completes
- [ ] `config.yaml` declares all tunable parameters
- [ ] `changelog.md` updated (change → expected → actual)
- [ ] `docs/env_vars.md` lists all custom env vars with justification
- [ ] `checksum.txt` present (generated on platform)
- [ ] `README.md` credits third-party code/libraries

## Commit Conventions

```
feat: <what was added>
fix: <what was fixed>
refactor: <what was restructured>
docs: <documentation change>
perf: <performance improvement>
```

Push only to GitLab: `https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git`. Never push to GitHub.
