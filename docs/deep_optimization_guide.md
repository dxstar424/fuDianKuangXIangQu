# 分阶段深度优化提分指南

> **分支**：`lutinayi_branch`（勿改 `main`）  
> **模型 / 框架**：Qwen3.5-27B bf16 · vLLM 0.18.1 · DCU 单卡 · 并发=1  
> **评分**：吞吐（4–8K 20% / **8–16K 50%** / 16–32K 30%）× 精度系数；SLA 熔断  
> **配套**：[`optimization_roadmap.md`](./optimization_roadmap.md) · [`official_guidance_interpretation.md`](./official_guidance_interpretation.md) · [`easy_scoring.md`](./easy_scoring.md) · [`parameter_tuning.md`](./parameter_tuning.md) · [`env_vars.md`](./env_vars.md)

---

## 一、总览

**在 SLA（TTFT/TPOT P99 ≤ Baseline×1.5）与精度（Δ≤1%）硬门槛内，把 Decode 阶段的 HBM/KV 读路径和 Prefill 首字路径抠到极致，主攻 8–16K（50% 权重）；一切优化必须可 A/B、可写进 report，且绝不碰 batch scheduler / 投机解码 / 持久化量化。**

物理图景（决策用）：

| 阶段 | 瓶颈 | 主指标 | 主攻手段 |
|------|------|--------|----------|
| Prefill | DCU 浮点算力（GEMM）+ 首路径开销 | TTFT P99（分档） | warmup、prefix cache、显存预算、减 Python/launch |
| Decode | HBM 带宽 + KV 反复读写 | TPOT P99（全局）+ 输出吞吐 | GQA、PagedAttention 块策略、Graph capture、**融合** KV FP8、HIP Attention |

并发=1 ⇒ 吞吐 ≈ 单请求变快；**改 `max-num-seqs` / `max-num-batched-tokens` / batch scheduler = 违规且无收益。**

---

## 二、Phase 0：赛前准备与 Baseline 锚定（必须做）

> 没有 stock baseline，后面所有「提分」都无法证明，也写不出官方要求的量化贡献表。

### 0.1 SCNet 环境跑通【必做】

- **操作**：登录 scnet.cn → 容器服务 → 克隆镜像 `qwen3.5-dtk26.04:0509` → 队列 `hx1hdexclu06` → SSH；执行 `bash scripts/scnet_setup.sh`（或按选手 PDF：编译 `vllm_cscc` v0.18.1、下载 Qwen3.5-27B、下载 testdata）。
- **为什么必须**：评测机与调试机同栈；本机 Windows 无 DCU，无法替代实机数据。

### 0.2 Stock Baseline 三档全量指标【必做】

- **操作**：
  ```bash
  cd ~/testdata && ./start_vllm.sh          # stock，勿开 FDU 优化
  # 另开终端
  ./run_throughput.sh 4-8K 10
  ./run_throughput.sh 8-16K 20              # 主攻档多采几条
  ./run_throughput.sh 16-32K 15
  ./run_accuracy.sh hotpotqa 10
  ./run_accuracy.sh gov_report 10
  ```
  将 **吞吐 / TTFT P99 / TPOT P99** 填入 `report.md` 与 `docs/optimization_roadmap.md` §5。
- **为什么必须**：SLA 门限 = Baseline×1.5；相对提升率也相对 Baseline；无表则无法判断熔断与提分。

### 0.3 Decode 步 Profiling（KV 读路径）【必做】

- **操作**：对 8–16K、16–32K 各跑若干请求，用 `hy-smi` + `rocprof`（或 `src/utils/profiling.py`）记录：decode 单步时间、Attention/KV load 占比、显存占用峰值。
- **为什么必须**：官方强调「长输入 TPOT 恶化要看 KV 读取路径，不能只看吞吐」；决定 Phase 2/3 该抠块分配还是写 HIP kernel。

### 0.4 平台流水线冒烟【必做】

- **操作**：GitLab 推送 **`lutinayi_branch`** → 竞赛平台触发评测 → 确认编译通过、`launch.sh` 能起服务。
- **为什么必须**：自测脚本口径 ≠ 官方 `vllm bench serve`；流水线不通等于零分。

### 0.5 文档骨架就位【必做】

- **操作**：准备并保持同步：`docs/env_vars.md`（变量名/取值/原因）、`report.md`（技术路线 + 贡献表空表）、`changelog.md`。
- **为什么必须**：官方：缺 env 说明或无可量化贡献 → 提交可能无效。

### Phase 0 完成门禁

- [ ] 三档 stock 吞吐 + TTFT/TPOT P99 有数  
- [ ] 至少一次长档 Decode profiling 结论（KV-bound？算力-bound？）  
- [ ] 平台至少一次编译成功  
- [ ] `env_vars.md` / `report.md` 模板就绪  

---

## 三、Phase 1：低风险高回报优化（必须做，保分项）

> 目标：在**不碰红线、精度系数≈1.0** 前提下，稳住 SLA，主攻档拿到「保底分 + 小幅提升」。对应仓库 v0.2.1+ `launch.sh` 默认策略。

### 1.1 提高 GPU 显存利用率（KV 池更大）

| 项 | 内容 |
|----|------|
| **优化点** | `gpu_memory_utilization`：0.90–0.92 → **0.94**（可 A/B 到 0.93/0.95） |
| **具体操作** | 在 `launch.sh` 设 `GPU_MEMORY_UTILIZATION=0.94`；对应 vLLM `--gpu-memory-utilization 0.94`。**禁止**改 `max-num-seqs` / `max-num-batched-tokens`。一次只改 0.01，记入 `parameter_tuning.md` A/B 表。 |
| **收益侧重** | **8–16K、16–32K**（KV 线性膨胀，池子不够会 OOM/抖动拖垮吞吐） |
| **风险** | >0.95 易 OOM → 完成率下降或 TTFT 尖刺。**应对**：OOM 立即回退 0.93→0.92；每次改完跑 `run_throughput.sh 8-16K 20`。 |
| **精度** | 无直接影响。 |

### 1.2 分档 Warmup（稳 TTFT P99）

| 项 | 内容 |
|----|------|
| **优化点** | 服务就绪后、正式压测前，对 4–8K / **8–16K** / 16–32K 量级做 dummy prefill+decode |
| **具体操作** | `DO_WARMUP=1`；`python scripts/warmup_server.py --port 8000 --rounds 1 --tier all`（或 `--tier 8-16K` 先保主档）。触发 ROCm/HIP kernel JIT 与显存池预热。 |
| **收益侧重** | **全档 TTFT P99**（尤其首条请求）；间接保护各档 SLA |
| **风险** | 启动多耗数分钟。**应对**：评测机可接受；勿关 warmup 赌启动速度。 |
| **精度** | 无。 |

### 1.3 开启 Prefix Caching

| 项 | 内容 |
|----|------|
| **优化点** | vLLM `--enable-prefix-caching` |
| **具体操作** | `ENABLE_PREFIX_CACHING=1` 且 `FDU_ENABLE_PREFIX_CACHE=1`；启动日志确认 flag 生效。 |
| **收益侧重** | Prefill / **TTFT**（有共享前缀时）；盲测收益不确定但成本极低 |
| **风险** | 略增元数据开销。**应对**：若显存紧张优先保 1.1，prefix 仍建议开。 |
| **精度** | 无（不改计算语义）。 |

### 1.4 关闭请求/统计日志

| 项 | 内容 |
|----|------|
| **优化点** | 减少 Python 侧 I/O 与格式化开销 |
| **具体操作** | `launch.sh` 增加 `--disable-log-requests --disable-log-stats`；`vllm_env.py` 设 `VLLM_LOGGING_LEVEL=WARNING`。 |
| **收益侧重** | 全档微降 TPOT / 稳吞吐（并发=1 仍有 Python 路径） |
| **风险** | 排障变难。**应对**：调试时临时打开，提交版保持关。 |
| **精度** | 无。 |

### 1.5 ROCm/DCU 环境变量（带宽与队列）

| 项 | 内容 |
|----|------|
| **优化点** | DCU 侧 SDMA / 显存分配 / 设备可见性 |
| **具体操作** | `scripts/rocm_env.sh`：`HIP_VISIBLE_DEVICES=0`、`HSA_ENABLE_SDMA=1`、`GPU_MAX_HW_QUEUES=2`、`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`、`HIP_FORCE_DEV_KERNARG=1`；全部写入 `docs/env_vars.md`。 |
| **收益侧重** | Decode HBM 搬运与分配稳定性（**8–16K / 16–32K**） |
| **风险** | 个别 DTK 版本对某变量敏感。**应对**：单变量 A/B；异常则回退该变量。 |
| **精度** | 无。 |

### 1.6 默认关闭 KV 在线量化（保精度系数）

| 项 | 内容 |
|----|------|
| **优化点** | `FDU_ENABLE_KV_QUANT=0`（Phase 1 强制） |
| **具体操作** | `launch.sh` / `rocm_env.sh` 默认 0；Phase 2 融合方案验证前禁止打开。 |
| **收益侧重** | 保证精度系数 **1.0**（乘数不被打折） |
| **风险** | 长档显存更紧。**应对**：用 1.1 显存利用率与 Phase 2 块策略补；勿用持久化量化「偷」显存。 |
| **精度** | 本项是**保精度**策略。 |

### 1.7 dtype 明确为 bf16 + 服务接口合规

| 项 | 内容 |
|----|------|
| **优化点** | 与官方权重一致，避免隐式转换 |
| **具体操作** | `--dtype bfloat16 --served-model-name Qwen3.5-27B --max-model-len 32768`；**不要**改 temperature/max_tokens（评测脚本锁定）。 |
| **收益侧重** | 全档稳定性 |
| **风险** | 误加非法 flag 导致评测机拒识。**应对**：只保留赛题允许项。 |

### Phase 1 完成门禁

```bash
bash scripts/scnet_start_optimized.sh
cd ~/testdata
./run_throughput.sh 8-16K 20
./run_accuracy.sh hotpotqa 10
bash scripts/gate_check.sh quick
```

- [ ] 8–16K 吞吐 ≥ stock  
- [ ] 各档 TTFT P99 ≤ Baseline×1.5；全局 TPOT P99 ≤ Baseline×1.5  
- [ ] 精度抽样 Δ≤1%  
- [ ] 平台提交一次，SLA/精度扣分=0  

---

## 四、Phase 2：深度优化（冲刺项，拉分项）

> 目标：主攻 **8–16K**，对齐官方「KV Cache + Decode 访存 + Graph」。每项必须 deep hook 进 `vllm_cscc` 运行时，禁止只改 PYTHONPATH 空转。

### 2.1 PagedAttention 块分配 / 碎片整理（官方最强调）

| 项 | 内容 |
|----|------|
| **优化点** | 分级块大小 + HBM burst 对齐 + 主动 defrag |
| **具体操作** | ① 将 `src/kv_cache/block_allocator.py` 的 tiered blocks（如 16/64/256 tokens）与 128B 对齐逻辑，经 `scripts/apply_vllm_patches.sh` **合入 vLLM `CacheEngine`/block manager**，不是独立未调用模块。② `defrag_threshold`（如 0.7）在空闲碎片过高时整理。③ `FDU_KV_CACHE_STRATEGY=defrag`。④ 用 16–32K 压测对比碎片率与 OOM 次数。 |
| **收益侧重** | **16–32K（30%）+ 8–16K（50%）** 显存效率与 Decode 读连续性 |
| **SLA 风险** | Defrag 若在请求中同步停顿 → **TTFT/TPOT 尖刺**。**应对**：仅在空闲/请求间隙整理；监控 P99；尖刺则提高阈值或关 defrag。 |
| **精度风险** | 无（布局优化）。**验证**：token 一致性抽样。 |

### 2.2 GQA Decode 路径（减 KV 读放大）

| 项 | 内容 |
|----|------|
| **优化点** | Qwen 64Q/32KV：避免 `repeat_interleave` 物化 2× KV |
| **具体操作** | 使用 `fdu_vllm/gqa_decode.py` 的分组 einsum / 专用 GQA kernel 分支；在 ROCm attention 后端注册；`FDU_ENABLE_GQA_OPT=1`。 |
| **收益侧重** | **全档 TPOT**，尤其 **8–16K**（Decode 步数×带宽） |
| **SLA 风险** | 错误实现导致数值不稳或变慢 → TPOT P99 熔断。**应对**：`python scripts/verify_token_consistency.py`；TPOT 回归则关 GQA。 |
| **精度风险** | 中。**验证**：temperature=0 逐 token 对比 stock；`run_accuracy.sh` 四类抽样。 |

### 2.3 HIP Graph Capture（减 Decode launch 开销）

| 项 | 内容 |
|----|------|
| **优化点** | 对并发=1 的固定 decode shape 捕获 HIP Graph（PyTorch `CUDAGraph`→ROCm） |
| **具体操作** | `fdu_vllm/hip_graph.py`；`export FDU_ENABLE_HIP_GRAPH=1`；warmup 后再 capture；**禁止**改采样/自回归语义。SCNet 连续 `run_throughput.sh` ≥30min。 |
| **收益侧重** | 全档 **TPOT**（逐步 kernel launch） |
| **SLA 风险** | DCU 上 graph 可能 segfault 或首步异常 → TTFT/完成率。**应对**：失败立即 `FDU_ENABLE_HIP_GRAPH=0`；勿强行上评测机。 |
| **精度风险** | 低（同图回放）。**验证**：一致性脚本 + accuracy 抽样。 |

### 2.4 Prefill 路径减负（非改 scheduler）

| 项 | 内容 |
|----|------|
| **优化点** | 降低首 token 前 Python/拷贝开销（配合已有 warmup/prefix） |
| **具体操作** | ① 模型放到 `/root/Qwen3.5-27B`（PDF：加载更快）。② 避免请求路径多余 CPU↔HBM 同步。③ 编译 `vllm_cscc` 时 `HIPCC_COMPILE_FLAGS_APPEND=-O3 --offload-arch=<rocminfo 的 gfx*>`。 |
| **收益侧重** | **TTFT**（各档，长档更敏感） |
| **SLA 风险** | 过度异步若同步点错误可能偶发超时。**应对**：保持与 stock 相同 API 语义；只减开销不改算法。 |
| **精度风险** | 无。 |

### 2.5 长档 TPOT 专项分析驱动迭代

| 项 | 内容 |
|----|------|
| **优化点** | 用数据决定下一刀砍 KV 还是 Attention |
| **具体操作** | 每完成 2.1–2.3 一项：跑 `8-16K`+`16-32K`，填 report「±X% 吞吐 / ±Y ms TPOT」；若 KV read >50% 步时 → 优先 2.1/Phase3 FP8；若 Attention compute 为主 → Phase 3 HIP Attn。 |
| **收益侧重** | 避免无效冲刺 |
| **风险** | 无技术风险；不做则 Phase 3 易走弯路。 |

### Phase 2 完成门禁

- [ ] KV 块逻辑确认进入 vLLM 运行时（日志/计数器可证）  
- [ ] GQA：token 一致性通过 + 8–16K TPOT↓  
- [ ] Graph：长测稳定才开；否则保持关  
- [ ] `gate_check.sh full` 四类 Δ≤1%  
- [ ] report 每项有量化贡献  

---

## 五、Phase 3：极限压榨（高危冲刺项，慎入）

> **非必需。** 仅当 Phase 1–2 完成、平台分已 ≥87 或 8–16K 仍落后榜首 >2 tok/s，且距截止 ≥3 天时考虑。

### 3.1 KV Cache 运行时 FP8（必须与 Attention 融合）【非必需】

| 项 | 内容 |
|----|------|
| **优化点** | 写入 KV 时 FP8(E4M3)+动态 scale；Attention load 时 **kernel 内**反量化；**保留全部历史 token KV** |
| **具体操作** | ① 禁止独立「dequant kernel → 再 Attention」。② 在 PagedAttention/FlashAttention HIP 路径内融合 dequant（改 `vllm_cscc` attention + `kv_fp8` 逻辑）。③ `FDU_ENABLE_KV_QUANT=1` 仅双门禁通过后。④ **禁止**写量化权重/KV 缓存文件到盘。 |
| **收益侧重** | **16–32K** 显存与带宽；间接 **8–16K** |
| **SLA 风险** | 融合不佳 → 额外访存 → **TPOT 升**甚至熔断。**应对**：A/B「融合 vs 独立 vs 关闭」；无净收益立即关。 |
| **精度风险** | **高**。**验证**：`gate_check full`；四类 Δ≤1% 才保留；否则关。 |
| **退出机制** | ① 任一任务 Δ>1%；② TPOT P99 相对关 FP8 变差；③ 2 天内做不出融合路径 → **放弃，保持 0**。 |

### 3.2 HIP FlashAttention / Decode Attention Kernel【非必需】

| 项 | 内容 |
|----|------|
| **优化点** | 针对 DCU CDNA：LDS double-buffer、MFMA 16×16×16、64-wide wavefront、tile 对齐 HBM 128B（见 `hip_kernels/dcu_flash_attn.cpp`） |
| **具体操作** | ① **仅当** profiling 证明 Attention 是 TPOT 主因。② 先做 decode batch=1 热路径，再考虑 prefill。③ `hipcc -O3 --offload-arch=gfx*`；JIT/`load_inline`；**必须** PyTorch fallback。④ 每版 `verify_token_consistency.py`。 |
| **收益侧重** | 全档 TPOT（理论上限高） |
| **SLA 风险** | 编译失败/数值错/偶发慢 → 评测 0 分。**应对**：fallback 默认安全路径；评测机只上验证过的 so/wheel。 |
| **精度风险** | **高**。**验证**：token 一致 + full accuracy。 |
| **退出机制** | ① profiling 不支持「Attn 主因」；② 编译/调试 >2 天无稳定收益；③ 截止前 <3 天 → **停，回退 Phase 2**。 |

### 3.3 Linear / GEMM 融合与 DCU 算力压榨（Prefill）【非必需】

| 项 | 内容 |
|----|------|
| **优化点** | Prefill 大 GEMM：QKV 融合、更好的 hipBLAS/rocBLAS tile；算子内临时低精度（非权重量化落盘） |
| **具体操作** | 在 `vllm_cscc` 的 linear/attention prep 路径做 fusion；用 rocprof 看 MFMA 利用率；仅临时 cast，不写 INT8 权重文件。 |
| **收益侧重** | **TTFT**（Prefill）；对 8–16K/16–32K 首字 |
| **SLA 风险** | 融合错误拖慢或数值漂移。**应对**：小步 A/B；TTFT 回归则回退。 |
| **精度风险** | 中高（若用激进低精度）。**验证**：accuracy full。 |
| **退出机制** | Prefill 已非主矛盾（Decode 仍占 TPOT 大头）→ 优先 3.1/3.2 或停止极限项。 |

### Phase 3 总退出原则

```
分数已够冲榜且 SLA/精度危险 → 冻结代码，只修编译
距截止 <48h → 禁止新开 HIP/FP8 大改
任何极限项导致平台分下降 → 立即回滚到 Phase 2 门禁通过版本（打 tag）
```

---

## 六、禁忌清单速查

| # | 禁止行为 | 后果 |
|---|----------|------|
| 1 | 改模型结构 / 换权重 / 微调·蒸馏·后训练 | 成绩无效 |
| 2 | 持久化 INT8/INT4 等权重量化文件或可复用量化缓存 | 违规 |
| 3 | 投机解码（draft / MTP / 多头预测 / early-exit draft） | 违规 |
| 4 | 截断输入、跳层、token pruning、early-exit | 违规 |
| 5 | 改 `max_model_len` / `max_num_seqs` / `max_num_batched_tokens` / batch scheduler 代码 | 违规；并发=1 也无收益 |
| 6 | 预缓存答案、评测 if-else 过拟合 | 取消成绩 |
| 7 | KV 量化时删除/跳过历史 token | 违规 |
| 8 | 反量化独立算子、未与 Attention 融合却宣称 KV 量化加速 | 可能负优化 + 难过审 |
| 9 | 自定义 env 不写 `env_vars.md` | 提交可能无效 |
| 10 | report 无分项量化贡献 | 官方可复现性要求，可能扣分/无效 |
| 11 | 未跑 gate 就平台提交 | 历史掉分（如 84→77） |
| 12 | 启用 `custom_scheduler.py` 当正式方案 | 红线风险 + 无收益 |

**SLA 速记**：TTFT P99（分档）与 TPOT P99（全局）任一超 Baseline×1.5 → **该档吞吐分清零**。  
**精度速记**：Δ≤1% → k=1.0；Δ>10% 单类 → 该类系数 0，总分可崩。

---

## 七、提分优先级总结表

| 优化项 | 难度 | 收益档位侧重 | 风险等级 | 建议阶段 |
|--------|------|--------------|----------|----------|
| SCNet + stock 三档 baseline + profiling | 低 | 全档（方法论） | 低 | **Phase 0 必做** |
| 平台编译/`launch.sh` 冒烟 | 低 | — | 低 | **Phase 0 必做** |
| `gpu_memory_utilization=0.94` | 低 | 8–16K、16–32K | 中（OOM） | **Phase 1 必做** |
| 分档 warmup | 低 | 全档 TTFT | 低 | **Phase 1 必做** |
| `--enable-prefix-caching` | 低 | TTFT | 低 | **Phase 1 必做** |
| 关 log + ROCm env | 低 | 全档微收益 | 低 | **Phase 1 必做** |
| 默认关 KV FP8（保精度） | 低 | 精度乘数 | 低 | **Phase 1 必做** |
| PagedAttention 块/defrag deep hook | 中高 | **8–16K、16–32K** | 中（P99 尖刺） | **Phase 2 冲刺主线** |
| GQA decode（64Q/32KV） | 中 | **全档 TPOT，主 8–16K** | 中（精度/TPOT） | **Phase 2 冲刺主线** |
| HIP Graph capture | 中 | 全档 TPOT | 中高（DCU 稳定性） | Phase 2 冲刺 |
| Prefill 减负 / hipcc -O3 | 低中 | TTFT | 低 | Phase 2 |
| **融合** KV FP8 | 高 | 16–32K → 8–16K | **高**（精度+TPOT） | Phase 3 慎入 |
| HIP FlashAttention | 很高 | 全档 TPOT | **很高** | Phase 3 慎入 |
| Prefill GEMM/Linear 融合 | 高 | TTFT | 高 | Phase 3 慎入 |
| 改 batch scheduler / 投机解码 / 持久化量化 | — | — | **禁止** | 不做 |

---

## 附：推荐执行顺序（到 7/15）

```
Phase 0  →  Phase 1（平台保分）  →  Phase 2（KV块+GQA±Graph，死磕 8-16K）
                ↓
         分≥87 且有余力？──否──→ 冻结，写 report，反复 gate + 提交
                │是
                ↓
         Phase 3：仅融合 FP8 或（profiling 支持时）HIP Attn；随时准备退出回滚
```

**主攻口诀**：先保 SLA 与 k=1.0 → 再抠 Decode（KV+GQA）抬 8–16K → Graph 锦上添花 → FP8/HIP 能进则进、不能则撤。
