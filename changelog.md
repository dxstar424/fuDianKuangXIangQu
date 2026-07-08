# 变更日志

## [v0.2.1] - 2026-07-08

### 最容易拿分（跑分指南对齐）
- `launch.sh`：显存 0.94、prefix cache、关闭 log、bf16、KV FP8 默认关
- `warmup_server.py`：按 4-8K / 8-16K / 16-32K 分档 prefill warmup（稳 TTFT P99）
- `vllm_env.py`：import 前 ROCm/日志优化
- `scripts/scnet_start_optimized.sh`：SCNet testdata 端口 8001 一键启动
- `docs/easy_scoring.md`：提分优先级说明

## [v0.2.0] - 2026-07-08

### 多阶段合规提分方案实施

**Phase 0 — SCNet / Baseline**
- 新增 `scripts/scnet_setup.sh`：SCNet 一键初始化（vLLM 编译、模型、testdata）
- 新增 `scripts/record_baseline.sh`：三档吞吐 baseline 记录
- 新增 `scripts/gate_check.sh`：quick/full 精度性能门禁

**Phase 1 — ROCm / launch**
- 重写 `launch.sh`：合规参数、warmup、`fdu_vllm.server` 入口
- 新增 `scripts/rocm_env.sh`、`scripts/compile_vllm.sh`、`scripts/warmup_server.py`
- 移除 `FDU_SCHEDULER_POLICY`（违规/无收益）

**Phase 2 — vLLM 内置路径**
- 新增 `src/fdu_vllm/gqa_decode.py`：GQA einsum 路径（64Q/32KV）
- Prefix caching 通过 `--enable-prefix-caching` 启用
- KV block/prefix hooks：`fdu_vllm/kv_cache.py`

**Phase 3 — KV FP8**
- `fdu_vllm/kv_fp8.py`：在线非持久化 FP8

**Phase 4 — HIP attention**
- `dcu_attention.py`：HIP 失败自动 fallback PyTorch
- 新增 `scripts/verify_token_consistency.py`

**Phase 5 — HIP Graph**
- 默认 `FDU_ENABLE_HIP_GRAPH=0`；`fdu_vllm/hip_graph.py` opt-in

**Phase 6 — 工程化**
- `patches/vllm_cscc/` + `scripts/apply_vllm_patches.sh`
- 更新 `Dockerfile`、`config.yaml`、`docs/env_vars.md`、`report.md`

---

## [v0.1.1] - 2026-07-07

### 改动
- HIP/ROCm 语义与 profiling 工具链

---

## [v0.1.0] - 2026-07-06

### 改动
- 项目初始化
