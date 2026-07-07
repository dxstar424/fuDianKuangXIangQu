# 环境变量说明

| 变量名 | 取值 | 作用 | 配置原因 |
|--------|------|------|----------|
| `FDU_KV_CACHE_STRATEGY` | `defrag` / `prealloc` / `dynamic` | KV Cache 管理策略 | 按场景切换分配策略 |
| `FDU_ATTENTION_BACKEND` | `dcu_optimized` | Attention 后端选择 | 启用自研 DCU kernel |
| `FDU_ENABLE_KV_QUANT` | `0` / `1` | KV Cache 在线量化开关 | 控制显存用量 |
| `FDU_SCHEDULER_POLICY` | `length_aware` / `fcfs` | 调度策略 | 按负载特征选择 |
