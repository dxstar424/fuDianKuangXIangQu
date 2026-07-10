# 官方平台提交检查清单（Phase 6）

## 代码
- [ ] **仓库根目录存在 `setup.py`**（评测机检查 `/coursegrader/submit/setup.py`）
- [ ] 离线编译：在 SCNet 执行 `bash scripts/prepare_submit.sh` 后 `git add vllm_cscc/` 并 push
- [ ] `bash scripts/compile_vllm.sh` 在 SCNet/评测容器编译通过
- [ ] `bash launch.sh` 启动成功，curl 推理正常
- [ ] `bash scripts/gate_check.sh full` 通过（Δ≤1%）

## 文档
- [ ] `report.md` 已填 baseline / 优化后三档数据
- [ ] `changelog.md` 记录本版本改动
- [ ] `docs/env_vars.md` 与 `launch.sh` 实际 env 一致

## GitLab
- [ ] push 到 `gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu`
- [ ] 竞赛平台触发评测，查看编译 log

## 分支建议
- `baseline-test`：stock launch 验证流水线
- `main` / `optimize`：含 `fdu_vllm` 优化
