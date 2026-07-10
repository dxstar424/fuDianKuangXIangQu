# 最有把握提分清单（Phase 1 · 必做）

> **深度指南**：[deep_optimization_guide.md](./deep_optimization_guide.md) §三  
> **总规划**：[optimization_roadmap.md](./optimization_roadmap.md)  
> **调参**：[parameter_tuning.md](./parameter_tuning.md)  
> **分支**：`lutinayi_branch` · 入口：`launch.sh`

本文只列 **一定有把握、低风险、已合入默认提交路径** 的步骤。  
不做 HIP kernel / KV FP8 / GQA deep hook（那些是 Phase 2+，需门禁后再开）。

---

## 相对 stock 改了什么

| # | 优化项 | stock baseline | 我们的默认 | 主攻指标 | 把握 |
|---|--------|----------------|------------|----------|------|
| 1.1 | 显存利用率 | `0.92` | **`0.95`** | 8–16K / 16–32K 吞吐 | ★★★★★ |
| 1.2 | 分档 warmup | 无 | **开**（先 8–16K） | 全档 TTFT P99 | ★★★★★ |
| 1.3 | Prefix caching | 关 | **开** | TTFT | ★★★★☆ |
| 1.4 | 关请求/统计日志 | 开日志 | **关** | 微降 TPOT | ★★★★☆ |
| 1.5 | ROCm/DCU env | 默认 | **SDMA + expandable_segments 等** | Decode 稳定性 | ★★★★☆ |
| 1.6 | KV 在线量化 | — | **强制关** | 精度系数 = 1.0 | ★★★★★（保分） |
| 1.7 | dtype / 接口 | bf16 隐式 | **显式 bf16 + 合规 served-name** | 稳定性 | ★★★★★ |

**禁止动**：`max-num-seqs` / `max-num-batched-tokens` / batch scheduler（与 stock 同为 256）。

---

## 代码落点（评测机实际跑这些）

```
launch.sh
 ├─ source scripts/rocm_env.sh → phase1_env.sh   # 1.5 + Phase 门控
 ├─ MODEL_PATH: /root → /data → $HOME            # Prefill 加载加速
 ├─ --gpu-memory-utilization 0.95                # 1.1（OOM → 0.94）
 ├─ --enable-prefix-caching                      # 1.3
 ├─ --disable-log-requests --disable-log-stats   # 1.4
 ├─ --dtype bfloat16 --served-model-name ...     # 1.7
 ├─ python -m fdu_vllm.server                    # vllm_env.py 再设 ROCm/日志
 └─ scripts/warmup_server.py --tier all          # 1.2（8-16K 优先）
```

`FDU_PHASE=1`（默认）：**不**安装 GQA / KV defrag / HIP Graph 钩子，避免未验证路径拖垮 SLA。

---

## 为什么这些「一定有把握」

1. **不改计算语义** → 精度系数不易掉。  
2. **不碰红线参数** → 不会因违规零分。  
3. **相对 stock 增量清晰** → 可写进 `report.md` 量化贡献。  
4. **平台已有先例**（富贵花开 84.74，SLA/精度扣分=0）说明同类 launch 路径可过评测。  
5. **OOM 有回退**：默认 0.95；OOM → `0.94` → `0.93` → `0.92`。

---

## SCNet 验证顺序（主攻 8–16K）

```bash
# 0) 静态检查
bash scripts/verify_phase1_config.sh

# 1) stock 对照（另开终端记 TTFT/TPOT/吞吐）
cd ~/testdata && ./start_vllm.sh
./run_throughput.sh 8-16K 20

# 2) Phase 1 优化版
bash scripts/scnet_start_optimized.sh   # PORT=8001
cd ~/testdata
# 对 8001 跑同等吞吐/精度
bash ~/…/scripts/gate_check.sh quick
```

门禁：8–16K ≥ stock；TTFT/TPOT P99 ≤ baseline×1.5；精度 Δ≤1%；再平台提交。

---

## 下一步（有把握项跑通之后）

| 优先级 | 项 | 条件 |
|--------|----|------|
| Phase 2 | KV 块/defrag deep hook、GQA | Phase 1 门禁通过 |
| Phase 2 | HIP Graph | SCNet 长测稳定 |
| Phase 3 | 融合 KV FP8 / HIP Attn | 分≥87 或 8–16K 仍差 >2 tok/s |

详见 [deep_optimization_guide.md](./deep_optimization_guide.md) §四–§五。
