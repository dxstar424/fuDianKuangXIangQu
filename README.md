# FDU SCCSCC26 - 基于国产加速卡的 Qwen 大模型推理服务优化

> 2026 全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）

## 赛题概要

| 项目 | 说明 |
|------|------|
| 模型 | Qwen3.5-27B (bf16) |
| 框架 | vLLM 0.18.1 |
| 硬件 | DCU 加速卡（初赛单卡） |
| 目标 | 在 SLA 约束下最大化推理吞吐量 |

## 优化策略总览

```
┌─────────────────────────────────────────────┐
│              最终得分 = 吞吐量 × 精度系数      │
├─────────────┬─────────────┬─────────────────┤
│  KV Cache   │  Decode算子  │   执行路径       │
│  优化        │  优化        │   优化           │
├─────────────┼─────────────┼─────────────────┤
│ ·分级块分配  │ ·HIP Attention│ ·CUDA Graph     │
│ ·碎片整理    │ ·算子融合     │ ·调度批量化      │
│ ·KV FP8量化  │ ·GQA优化      │ ·异步传输       │
│ ·Prefix缓存  │ ·LDS利用      │ ·预热           │
└─────────────┴─────────────┴─────────────────┘
```

## 项目结构

```
fdu-sccscc26/
├── Dockerfile              # 容器构建
├── launch.sh               # 服务启动脚本
├── config.yaml             # 可调参数
├── requirements.txt        # 额外依赖
├── src/
│   ├── kv_cache/           # KV Cache 管理
│   ├── attention/          # DCU Attention 后端
│   │   └── hip_kernels/    # HIP 内核源码
│   ├── scheduler/          # 自定义调度器
│   ├── quantization/       # KV Cache 在线量化
│   ├── executor/           # 执行路径优化
│   └── utils/              # 性能分析工具
├── scripts/                # 构建 & 评测脚本
├── docs/                   # 文档
├── changelog.md            # 提交变更日志
└── report.md               # 优化方案报告
```

## 快速开始

```bash
# 构建镜像
docker build -t fdu-sccscc26 .

# 启动服务
docker run --gpus all -p 8000:8000 fdu-sccscc26

# 健康检查
curl http://localhost:8000/health
```

## 提交清单

- [ ] `launch.sh` - 服务启动脚本
- [ ] `config.yaml` - 参数声明
- [ ] `changelog.md` - 变更日志
- [ ] `report.md` - 优化报告
- [ ] `docs/env_vars.md` - 环境变量说明
- [ ] `checksum.txt` - 权重 SHA256
- [ ] 完整源码（`src/`）
