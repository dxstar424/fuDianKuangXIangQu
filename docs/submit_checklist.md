# 当前平台提交检查清单

> 适用路线：2026-07-14 原生 gfx936 BF16 保底 + gated online W8/W4。当前平台分支为 `dx_branch`；平台必须显式绑定该分支，或把同一提交同步到平台实际拉取的 `main`。

## 代码与 Git

- [ ] 工作树干净：`git status --short` 无输出。
- [ ] `git rev-parse HEAD` 与 GitLab `origin/dx_branch` 一致。
- [ ] GitHub `dx_branch` 和 `main` 指向同一提交，作为 GitLab 502 时的备份。
- [ ] 仓库根目录包含 `setup.py`、`vllm/`、`csrc/`、`launch.sh` 和 `Dockerfile`。
- [ ] 不运行历史 `scripts/prepare_submit.sh` 覆盖当前 vendored 源码；它只用于从已知匹配的上游重新 vendor，当前路线不需要。

## 静态验证

```bash
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v
python3 -m py_compile \
  scripts/build_gfx936_quant_jit.py \
  scripts/preflight_gfx936_quant.py \
  scripts/bench_gfx936_quant.py \
  vllm/model_executor/layers/gfx936_online_quant.py
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_ab_gfx936.sh
git diff --check
```

- [ ] 全部测试通过。
- [ ] 活跃启动文件没有 AWQ、bitsandbytes、`--quantization` 或架构伪装。
- [ ] `FDU_ENABLE=0`、`VLLM_ROCM_USE_AITER=0`。

## SCNet 门禁

- [ ] 按 [SCNET_RUN.md](SCNET_RUN.md) 使用隔离 wrapper，不直接修改模型或 testdata。
- [ ] JIT 首次编译 `<=50s`，第二次命中同一 `/tmp` `.so`。
- [ ] 六 shape JSON 含精确 commit、HIP source hash、有限数值和逐 shape admission。
- [ ] 候选服务 probe 成功，日志模式与请求的 `FDU_GFX936_QUANT_MODE` 完全一致。
- [ ] 三档无吞吐回退，TTFT/TPOT P99 留有 `< baseline×1.45` 余量。
- [ ] HotpotQA 与 Retrieval MultiPoint 抽样精度下降 `<=1%`；正式提交后检查四项完整精度。
- [ ] 无 OOM、Traceback、失败请求、非有限指标或静默 BF16 fail-open。

## 默认模式选择

- [ ] hybrid 只有在两个 MLP W4 数值通过、各自比 W8 microbenchmark `>=1.05x`，并且端到端 8–16K 比 W8 `>=1.03x` 时才可选。
- [ ] 否则 W8 只有在 8–16K `>=12.60 tok/s`、三档/SLA/精度通过且加权投影高于 66.8175 时才可选。
- [ ] 否则 `scripts/rocm_env.sh` 和 Dockerfile 继续保持 `FDU_GFX936_QUANT_MODE=off`。
- [ ] 如果修改默认模式，同一提交同步更新 `docs/env_vars.md`、`docs/GFX936_HANDOFF.md`、`report.md` 和 `changelog.md`。

## 平台结果

- [ ] 分支绑定和提交号在消耗评测次数前再次确认。
- [ ] 记录三档 throughput、TTFT P99、TPOT P99、SLA 扣分、四项精度与最终得分。
- [ ] 只有实测数据写入 `report.md`；三样本快速门禁必须标为方向性结果。
- [ ] 平台 build failed 时，在同一提交上复现 Docker/wheel 构建，不临时引入另一条优化路线。
