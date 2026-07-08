# 变更日志

## [v0.3.0] - 2026-07-09

### 改动
- **策略变更**: 放弃 monkey-patch 运行时注入，改为直接修改 vLLM 源码
- `launch.sh`: 新增 `--enable-prefix-caching`、`--compilation-config`（HIP Graph）、`--max-num-seqs 256`、`--max-num-batched-tokens 8192`
- `config.yaml`: 精简为两层（vLLM 原生参数 + FDU patch 参数）
- `scripts/explore_vllm.sh`: 新增，DCU fork 结构探索脚本
- `plugin.py`: 保留为参考文档，不再运行时注入

### 预期效果
- Prefix Caching: prefill 减少 30-50%
- HIP Graph: kernel launch 开销消除
- 更大 batch: 吞吐量提升

### 实际结果
- 待 DCU 平台验证

---

## [v0.2.0] - 2026-07-08
（同上，略）
