# 变更日志

## [v0.2.7] - 2026-07-10

### 平台提交修复（missing setup.py）
- 新增根目录 `setup.py`：克隆/使用 `vllm_cscc/`、应用 FDU 补丁、`bdist_wheel`
- 新增 `scripts/prepare_submit.sh`：SCNet 上 vendor vllm 源码进仓库（离线评测必需）
- `launch.sh`：SCNet 家目录模型路径；warmup 失败非致命；健康检查 900s
- `warmup_server.py`：16-32K warmup 降至 16k tokens，降低 OOM 风险
- `gate_check.sh` / `scnet_start_optimized.sh` / `scnet_resume.sh`：SCNet 路径解析

## [v0.2.6] - 2026-07-09

### 最有把握提分项加固（Phase 1）
- 重写 `docs/easy_scoring.md`：相对 stock 的 1.1–1.7 清单 + 代码落点
- `docs/deep_optimization_guide.md` §三增加「最有把握提分步骤」表
- `launch.sh`：模型路径 `/root`→`/data` 自动解析；健康检查支持 `/v1/models`；超时 600s
- `warmup_server.py`：`tier=all` 时 **先 8–16K**；chat 失败回退 completions；缩短 decode
- `vllm_env.py`：与 `rocm_env.sh` 对齐（SDMA / expandable_segments 等）
- `config.yaml`：Phase 1 默认 `strategy=none`、`backend=vllm_default`
- `report.md` / `parameter_tuning.md` / `env_vars.md`：同步最有把握项说明

## [v0.2.5] - 2026-07-09

### Phase 1 代码闭环
- 新增 `scripts/phase1_env.sh`：1.1–1.7 专用环境，默认关闭 Phase 2+ 钩子
- 新增 `src/fdu_vllm/phase1.py`：Phase 1 配置校验与日志
- 新增 `scripts/verify_phase1_config.sh`、`scripts/run_phase1_gate.sh`
- `hooks.py`：`FDU_PHASE=1` 时仅 launch CLI 优化，不安装 GQA/KV/attention 钩子
- `config.py`：Phase 1 默认 `kv_quant=false`、`gpu=0.94`、`gqa=false`
- `launch.sh`：启动时打印 Phase 1 配置摘要
- `Dockerfile`：写入 Phase 1 默认 ENV
- 修复 `scnet_setup.sh` testdata 存在性判断

## [v0.2.4] - 2026-07-09

### DCU decode 访存实测解读
- 新增 `docs/dcu_decode_benchmark_interpretation.md`（gfx936 微基准：权重 IO 95%、双缓冲证伪）
- 调整 roadmap：HIP Graph↑、GEMV 双缓冲↓、FlashAttn 主攻 prefill/TTFT、KV FP8 预期修正

## [v0.2.3] - 2026-07-09

### 四人分工与协作
- 新增 `docs/team_division.md`：I 整合/Git、P1 KV/Prefill、P2 Decode、G 门禁/定期评测
- 新增 `docs/parameter_tuning.md`：参数作用、A/B 表、合并冲突区（整合负责人维护）
- 新增 `docs/deep_optimization_guide.md`：必须(M0–M3) vs 冲刺(S1–S4) 深度提分指南
- 更新 `optimization_roadmap.md` §4 分工、赛程负责人、7/13 main 大合并里程碑
- `easy_scoring.md` / roadmap 增加深度指南入口

## [v0.2.2] - 2026-07-09

### 官方指导对齐
- 新增 `docs/official_guidance_interpretation.md`（zhaorq 2026-07-09 技术指导解读）
- 调整 `optimization_roadmap.md`：KV 块/defrag 上调 P1；HIP Graph 提前 P2；KV FP8 融合门禁
- 更新 `easy_scoring.md`：Prefill/Decode 分阶段优先级 + 提交核对清单
- 更新 `report.md`：优化贡献量化表 + 合规声明补充
- `launch.sh`：修复 `FDU_ENABLE_PREFIX_CACHE` 默认值

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
