# 变更日志

## [v0.4.0] - 2026-07-12

### ★ 去 FP8 + 最大化系统优化（60 → 90 冲刺）

**根因诊断**：
- 四轮实验 A-D 全部 ~60 分，都启用了 `--quantization fp8`
- DCU 讲义实测：FP8 量化在 prefill 是负优化（反量化开销 > 权重 IO 节省），"正是我们 down 量化拖累 16–32K 长档的原因"
- 16-32K 5.77 < baseline 7.75（-26%）直接由 FP8 造成

**v0.4.0 改动**：
- `launch.sh`：`ENABLE_FP8_WEIGHT_QUANT=0`（关 FP8），移除 `--quantization fp8`
- `launch.sh`：`GPU_MEMORY_UTILIZATION=0.97`（激进显存），`--block-size 32`（减半 32K 页表遍历）
- `launch.sh`：`cudagraph_capture_sizes=[1,2,4,8]`（扩展 HIP Graph 覆盖）
- `scripts/rocm_env.sh`：`VLLM_ROCM_USE_AITER=0`（关 FP8 GEMM 后端），`VLLM_ROCM_USE_SKINNY_GEMM=1`（确认 decode GEMV 走手写 HIP kernel）
- `scripts/rocm_env.sh`：新增 `ROCBLAS_LAYER=4`、`MIOPEN_FIND_MODE=1`（ROCm 自调优）
- `scripts/rocm_env.sh`：新增 `VLLM_ROCM_USE_AITER_RMSNORM=1`（纯 bf16 RMSNorm 加速）
- `scripts/rocm_env.sh`：去掉 `VLLM_USE_TRITON_FLASH_ATTN`（cargo cult，非 vLLM 原生 env var）
- `Dockerfile`：同步所有 env var 变更
- `config.yaml`：`fp8_weight_quant.enabled=false`，`block_size=32`，`gpu=0.97`
- `docs/env_vars.md`：v0.4.0 全量更新

**预期效果**：
- 16-32K：去掉 FP8 负优化 → 从 5.77 恢复到 >7.75 baseline，目标 10-12 tok/s
- 8-16K：GPU 0.97 + block_size 32 + rocBLAS 自调优 → 目标 14-16 tok/s
- 4-8K：无回归
- 总分目标：75-85（保守）/ 85-90（激进）

## [v0.3.0] - 2026-07-11

### FP8 在线权重量化（W8A8）— 60 → 88 分冲刺

**物理依据**（`doc/大模型decode访存瓶颈与双缓冲_DCU实测(1).html`）：
- DCU decode = 95% 权重 HBM IO（54GB bf16 ÷ 1.2TB/s = 45ms/token），计算仅用 0.3% 算力
- 双缓冲实测 ±1.5%（已撞带宽墙），唯一出路：减少搬运字节（量化）
- 原文结论："这就是我们把 TPOT 从 ~49ms 压到 ~40ms 的主武器"

**改动**：
- `launch.sh`：新增 `ENABLE_FP8_WEIGHT_QUANT=1`，添加 `--quantization fp8` 到 vLLM CLI
- `scripts/rocm_env.sh`：`VLLM_ROCM_USE_AITER=1`（FP8 W8A8 GEMM 走 AITER Triton BMM）
- `Dockerfile`：同步 `VLLM_ROCM_USE_AITER=1` + `ENABLE_FP8_WEIGHT_QUANT=1`
- `config.yaml`：新增 `fp8_weight_quant` 配置节
- `docs/env_vars.md`：文档化 `ENABLE_FP8_WEIGHT_QUANT`、`VLLM_ROCM_USE_AITER`、`VLLM_ROCM_USE_SKINNY_GEMM`

**技术路径**：vLLM 内置 `Fp8OnlineLinearMethod`
— 加载 bf16 权重 → `ops.scaled_fp8_quant()` 在线量化到 FP8 FNUZ → 存储 `weight`(FP8 27GB) + `weight_scale`
— Forward: per-token 激活量化 → W8A8 FP8 GEMM (AITER Triton BMM / `torch._scaled_mm`)
— 预期权重 HBM IO: 45ms → 22.5ms (-50%)，8-16K 吞吐 ~18-20 tok/s → 平台分 ~85-90

**预期效果**：8-16K 吞吐 2×，总分 85-90。
**实测结果**：待 SCNet 验证 + 平台提交。

## [v0.2.19-ExpB] - 2026-07-11

### 实验 B 结果 — GPU 0.95 + eager=0 + warmup=1

**预期效果**：HF Graph 关 eager + warmup 压制 TTFT 尖刺 → 吞吐略升。
**实测结果**：
```
最终得分=59.8594; 4K-8K=12.85; 8K-16K=10.02; 16K-32K=5.77; SLA=0; 精度=0
```
对比 Exp A (59.98)：4-8K 微降 (−0.12), 8-16K 持平 (−0.01), 16-32K 不变。总分略低于 A。
**结论**：eager=0 (HF Graph) + warmup 未带来任何增益。16-32K 死卡 5.77，与 A 完全相同 → decode 带宽墙无法用 launch flags 突破。实验 C/D 预期同样 ~60。

## [v0.2.15] - 2026-07-11

### Phase2 三板全部证伪 → 回归 S1 recover

**结论：Decode 端在 DCU 上已触碰物理极限。GQA / HIP Graph / KV FP8 三项全部放弃。**

#### GQA wrap — 源码审查确认无效

- **改动**：深度审查 `gqa_backend_wrap.py` 实际拦截点
- **发现**：`wrap_attn_backend()` 只重写 `_forward_encoder_attention`，该方法在 vLLM V1 的 rocm_attn.py、flash_attn.py、triton_attn.py 中**仅在 `AttentionType.ENCODER_ONLY` / `ENCODER` / `ENCODER_DECODER` 时调用**。Qwen3.5-27B 是 decoder-only，全部层用 `AttentionType.DECODER`，不经过此路径。
- **预期效果**：无（已经不影响 Decode）
- **实测结果**：未上机（源码分析已确认为死代码路径）。真正 Decode 走 `RocmAttentionImpl.forward()` → `chunked_prefill_paged_decode()` → PagedAttention kernel，已原生接收 `num_kv_heads` 参数。
- **→ 冻结。代码保留，不再投入。**

#### HIP Graph — 物理分析 + S2b 实测证伪

- **改动**：审查 `hip_graph.py` + `exec_path.py` 实际工作状态 + DCU decode 物理瓶颈分析
- **发现**：
  1. 代码层面：`hip_graph.py` 从未调用 `capture_graph()`，`_graphs` 字典始终为空。当前 wrapper 只加开销无任何收益（骨架代码）。
  2. 物理层面：DCU decode 权重 IO 占 95%（54GB bf16 ÷ 1.2TB/s HBM = 45ms/token），kernel launch overhead 仅 ~2ms（<3%）。HIP Graph 最佳收益 = 消除 2ms → +2.8% 吞吐 ≈ +0.34 tok/s，不到 go/no-go 门槛（+0.5）。
  3. 历史实测：S2b `ENFORCE_EAGER=0`（原生 vLLM CUDA/HIP Graph）vs `ENFORCE_EAGER=1` → 12.17 vs 12.19 tok/s（噪声级）。
- **预期效果**：+0.34 tok/s（理论上限）
- **实测结果**：原生 Graph 已证伪（+0.02 tok/s 噪声）；自研捕获实现无意义。
- **→ 冻结。不实现真实 capture。代码保留，ENFORCE_EAGER=1 保持。**

#### KV FP8 — 历史实测全档倒退

- **改动**：重新评估 KV FP8 对 16-32K HBM IO 缩减的理论收益 vs 历史实测数据
- **理论**：32K 时 KV 读占 28.6ms → FP8 减至 14.3ms（-19% TPOT）
- **实测**（2026-07-09，`--kv-cache-dtype fp8`，原生 vLLM，非 FDU hooks）：
  - 4-8K：12.21 → **10.65 tok/s（-12.8%）**，TTFT P99 4546 → **26627ms（崩溃）**
  - 8-16K：7.24 → **7.10 tok/s（-1.9%）**
  - 16-32K：3.22 → **2.90 tok/s（-9.9%）**
- **预期效果**：16-32K TPOT 降 ~19%（理论）
- **实测结果**：全档倒退。ROCm fp8 attention 路径未优化，量化/反量化开销超过 KV 带宽节省。
- **→ 冻结。不开 KV FP8。FDU_ENABLE_KV_QUANT=0 保持。**

#### 物理根源总结

DCU decode 瓶颈 = 54GB bf16 模型权重 HBM 搬移（~45ms/token，占 95%）。
任何不减少模型权重 HBM IO 的优化（Graph/GQA/scheduler/KV quant）都无法显著改善 TPOT。
唯一合规的权重 IO 缩减手段（INT4 权重量化）属于"持久化量化"→ 违规。

#### S1 recover 回归

- `launch.sh` 保持 S1 默认：stock api_server, GPU 0.94, ENFORCE_EAGER=1, DO_WARMUP=0
- `CLAUDE.md` 更新：Phase2 证伪表 + S1 recover 行动清单
- 计划从 Phase2 冲刺切换为 S1 recover：确认配置 → 可选单档 warmup A/B → 平台提交 → 7/15 截止

## [v0.2.14] - 2026-07-11

### 少次多阶段冲刺落地（S1–S4 + Phase2 三板）
- 扩充 `docs/sprint_strategy_0711.md`：GQA/defrag/Graph SCNet 协议、一人一块、合 main 规则
- 新增 `scripts/stage3_defrag_launch.sh`、`scripts/run_phase2_bench.sh`
- S1：`scripts/verify_recover_config.sh`（锁 0.94/eager/stock/warmup=0）
- S2：`scripts/ab_stage2.sh`（eager-off / warmup-816 单变量）
- S3：GQA **真实接线** — `gqa_backend_wrap.py` 包装 stock AttentionBackend；修 `attention.py` API（`load_kernel`）；`stage3_gqa_launch.sh`
- S4：`hip_graph.py` model-runner 补丁 + `stage4_graph_launch.sh`（默认关）
- Phase2 默认：仅 GQA；`FDU_KV_CACHE_STRATEGY=none`（defrag/FP8 不开）
- overlay：`patches/vllm_cscc/overlay/vllm/v1/attention/backends/fdu_gqa_attn.py`
- 同步 easy_scoring / parameter_tuning / env_vars / roadmap

## [v0.2.13] - 2026-07-10

### 紧急恢复（本队 59.97 ≈ Baseline；纠正「84=本队」误标）
- **澄清**：排行榜「富贵花开」84.74 **不是本队账号**；本队正式分以 lutinayi 59.97 为准
- `launch.sh` 默认 stock `api_server`；gpu **0.94**；`DO_WARMUP=0`；`ENFORCE_EAGER=1`
- 更新 scoring / easy_scoring / report / roadmap / 提交清单中的账号归属

### 评分与实测结果深度解读
- `docs/scoring_and_results_interpretation.md`：官方公式、Baseline 反推、7/6 vs 7/10 复盘
- `optimization_roadmap.md` §5.3 追加 7/10 平台结果（59.97）

## [v0.2.12] - 2026-07-10

### 高性价比加固 + 合并冲突清理
- **修复** `launch.sh` / `config.yaml` / `report.md` / `Dockerfile` / `README.md` / `requirements.txt` / `dcu_attention.py` 中误提交的 `<<<<<<<` 冲突标记（否则评测无法启动）
- `GPU_MEMORY_UTILIZATION` 默认 **0.94 → 0.95**（相对 stock 0.92）；OOM 回退链 0.94→0.93→0.92
- 明确不传 `--enforce-eager`，保留 vLLM 原生 CUDAGraph/HIP Graph（减 decode launch）
- 同步 `easy_scoring.md` / `env_vars.md` / `parameter_tuning.md` / `report.md` / `phase1.py`

## [v0.2.11] - 2026-07-10

### SCNet GitLab 权限排查工具链
- `scripts/scnet_gitlab_diagnose.sh`：DNS/proxy/curl/git 全量诊断
- `scripts/scnet_gitlab_fix_hosts.sh`：修复容器 DNS（111.6.188.181）
- `scripts/scnet_gitlab_clone.sh`：GITLAB_TOKEN 克隆私有仓库
- `docs/scnet_gitlab_access.md`：10 类原因 + 决策树

## [v0.2.10] - 2026-07-10

### 双通道评测工作流（官方无 log + SCNet 无 git）
- 新增 `scripts/platform_build.sh`：镜像评测机编译，log 写入 `results/platform_build_*.log`
- 新增 `scripts/scnet_import_repo.sh`：zip 导入仓库，绕过 GitLab 403
- 新增 `docs/dual_eval_workflow.md`：平台提交 vs SCNet 调试分工说明
- `scnet_resume.sh` 增加 `import` / `platform-build` 子命令

## [v0.2.9] - 2026-07-10

### 平台编译修复（vLLM build failed）
- 补全 vendor 遗漏的 `requirements/`（`setup.py` 的 `get_requirements()` 依赖）与 `LICENSE`
- `setup.py`：ROCm 版本文件缺失时降级警告，避免 `get_version_add` 崩溃

## [v0.2.8] - 2026-07-10

### 平台提交修复（vLLM 源码 vendor 到根目录）
- 根目录合入官方 `vllm_cscc` v0.18.1 的 `setup.py`、`vllm/`、`cmake/`、`csrc/` 等编译必需文件
- `fdu_vllm` 插件注册到 `vllm/__init__.py`；`prepare_submit.sh` 改为铺平到仓库根
- **注意**：竞赛平台默认拉取 **`main`** 分支，须同步 push 到 main 后再提交

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
