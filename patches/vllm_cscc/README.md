# vLLM CSCC 补丁说明

将 `overlay/` 合入官方 `vllm_cscc` v0.18.1 源码树：

```bash
bash scripts/apply_vllm_patches.sh ~/vllm_cscc
bash scripts/compile_vllm.sh
```

`apply_vllm_patches.sh` 会：
1. 复制 `overlay/` 文件
2. 在 `vllm/__init__.py` 注册 `fdu_vllm.activate()`
3. 同步 `src/fdu_vllm/` 到 `vllm_cscc/fdu_vllm/`

## 合规边界

- 不修改 batch scheduler / max-num-batched-tokens
- KV FP8 仅在线、非持久化
- 无投机解码、无权重量化文件
