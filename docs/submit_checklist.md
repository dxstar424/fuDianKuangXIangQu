# 官方平台提交检查清单（Phase 6）

## 代码
- [ ] **仓库根目录存在真实 vLLM 源码**：`setup.py` + `vllm/`（评测机检查 `/coursegrader/submit/setup.py`）
- [ ] 在 SCNet 执行 `bash scripts/prepare_submit.sh` 后 commit + push（首次 vendor 或更新补丁后）
- [ ] **平台默认拉 `main` 分支**；若提交 `lutinayi_branch` 须在平台/GitLab 绑定对应分支
- [ ] `bash scripts/verify_recover_config.sh`（S1）或对应阶段脚本自检通过
- [ ] `bash scripts/compile_vllm.sh` 在 SCNet/评测容器编译通过（S3 overlay 后）
- [ ] `bash launch.sh` 启动成功，curl 推理正常
- [ ] `bash scripts/gate_check.sh full` 通过（Δ≤1%）— S3/S4 必做；S1 至少 quick

## 冲刺配额（见 docs/sprint_strategy_0711.md）
- [ ] 本阶段仅 1 次平台提交；全赛程约 ≤4 次
- [ ] 8–16K 相对上一 tag +≥0.5 tok/s 才提交
- [ ] 未开未接线项（defrag / KV FP8 / FlashQLA 整库）

## 文档
- [ ] `report.md` 已填 baseline / 优化后三档数据
- [ ] `changelog.md` 记录本版本改动
- [ ] `docs/env_vars.md` 与 `launch.sh` 实际 env 一致

## GitLab
- [ ] push 到 `gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu`
- [ ] 竞赛平台触发评测（**本队账号**，分支 **`lutinayi_branch` / `main`**；「富贵花开」非本队）
- [ ] 平台仅显示 build failed 时 → SCNet 跑 `bash scripts/platform_build.sh` 拿 log（见 [dual_eval_workflow.md](./dual_eval_workflow.md)）

## 分支建议
- `baseline-test`：stock launch 验证流水线
- `main` / `optimize`：含 `fdu_vllm` 优化
