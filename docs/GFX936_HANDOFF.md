# gfx936 BF16 原生 Kernel 路线交接文档

> 更新日期：2026-07-14  
> 目标：Qwen3.5-27B BF16 / vLLM 0.18.1 / 单 DCU（原生 `gfx936`），在不牺牲精度系数和 SLA 的前提下提升 decode 吞吐。

## 1. 一页结论

- 实现已推送到 GitLab 和 GitHub 的 `dx_branch`。
- 当前实现锚点是 `91ea5e9`，交接文档提交可用 `git log -1` 查看。
- 第一个平台可跑分锚点是 `6af6666`。
- 当前白名单故意为空，所以代码可安全运行，但自定义 skinny GEMM **尚未真正进入模型推理**。
- 必须先在 SCNet `gfx936` 上完成直接 kernel benchmark，由脚本生成实测 shape 白名单。
- 未得到 SCNet 数据前，不能声称已提速，也不能声称达到 90 分。

## 2. 分支与远端

| 项目 | 值 |
|---|---|
| GitLab | `https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git` |
| GitHub | `https://github.com/dxstar424/fuDianKuangXIangQu.git` |
| 使用分支 | `dx_branch` |
| 实现锚点 | `91ea5e9` |
| 安全可跑分锚点 | `6af6666` |

GitLab 在 SCNet 命令行中会询问账号/令牌，所以当前 SCNet 优先从 GitHub 的 `dx_branch` 拉取。不要把账号、密码或 token 写入日志、文档或 Git remote URL。

## 3. 已完成的实现

### 3.1 原生 gfx936 构建

- `setup.py` 在 HIP 构建中强制生成 `vllm._rocm_C`。
- `csrc/rocm/skinny_gemms.cu` 只将 `gfx936` 加入 BF16/FP16 GFX9 skinny kernel 编译范围。
- `gfx936` 没有被加入全局 `_ON_GFX9` 或 `_ON_MI3XX` 判定，避免误启用其他架构专属功能。
- 不设置 `HSA_OVERRIDE_GFX_VERSION`，不伪装成 `gfx942`。

### 3.2 精确 shape 调度

- `vllm/model_executor/layers/utils.py` 中的 `rocm_unquantized_gemm_impl` 增加了独立 `gfx936` 分支。
- 仅对 BF16/FP16、无 bias、连续 weight、`K % 8 == 0`、`N in {1,2,4}` 的实测 shape 使用 `wvSplitK`。
- 精确白名单位于 `vllm/model_executor/layers/rocm_skinny_shapes.py`。
- 当前值为：

```python
VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset()
```

因此当前提交会自动走 stock BF16 linear，不会调用未验证 kernel。

### 3.3 模型精度与回滚

- 活跃路径使用 BF16 原始权重。
- 不使用 AWQ、INT4、FP8 权重转换、持久量化权重或投机解码。
- `FDU_ENABLE=0`，关闭历史 FDU 插件钩子。
- `VLLM_ROCM_USE_AITER=0`，直接 A/B 时保持单变量。
- `FDU_FORCE_STOCK_GEMM=1` 可在不重编 wheel 的情况下立即恢复 stock linear。

### 3.4 失败前置门禁

`scripts/preflight_rocm.py` 在加载 27B 模型前检查：

- Python、`vllm`、`vllm._C` 和 `vllm._rocm_C` 来自预期 venv；
- 真实架构为 `gfx936`；
- `torch.ops._rocm_C.wvSplitK` 和 `LLMM1` 存在。

任一项不符合都应立即停止，不进入模型加载。

## 4. 已验证的 SCNet 环境

2026-07-14 在当前容器中已观察到：

| 项目 | 实测值 |
|---|---|
| 原生架构 | `gfx936:sramecc+:xnack-` |
| PyTorch | `2.10.0` |
| HIP | `6.3.26093` |
| Compute Units | `80` |
| 持久目录 | `/public/home/xdzs2026_c415` |
| 模型 | `/public/home/xdzs2026_c415/Qwen3.5-27B` 已存在 |
| 评测数据 | `/public/home/xdzs2026_c415/testdata` 已存在 |

### 当前已知阻塞

SCNet 系统 Python 最初缺少 `ensurepip/python3.10-venv`，导致：

1. `vllm_baseline` 虚拟环境创建失败；
2. 后续 `setup.py` 使用残缺 venv，报 `ModuleNotFoundError: torch`；
3. preflight 继而报 `vllm._C` / `_rocm_C` / architecture 全部缺失。

这些是同一个 venv 初始化失败引起的连锁错误，**不是 kernel benchmark 失败**。

## 5. 从当前状态继续：可复制命令

### 5.1 拉取 GitHub `dx_branch`

```bash
export ROOT=/public/home/xdzs2026_c415
export EXP=$ROOT/experiments/gfx936_skinny

mkdir -p "$EXP" "$ROOT/results/gfx936_skinny"
cd "$EXP"

# 仅在 source 是前一次未完成 clone 时执行。改名保留，不直接删除。
if [ -e source ] && [ ! -d source/.git ]; then
  mv source "source.incomplete.$(date +%s)"
fi

if [ -d source/.git ]; then
  cd source
  git fetch origin dx_branch
  git checkout dx_branch
  git merge --ff-only origin/dx_branch
else
  git clone --depth 1 --branch dx_branch \
    https://github.com/dxstar424/fuDianKuangXIangQu.git source
  cd source
fi

git rev-parse --short HEAD
```

实现锚点应至少为 `91ea5e9`；交接文档合入后可能是更新的提交。

### 5.2 修复 venv 支持

当前容器为 root：

```bash
apt-get update
apt-get install -y python3.10-venv
```

如果包名不存在，再使用：

```bash
apt-get install -y python3-venv
```

清理的只是刚才创建失败的两个 venv：

```bash
rm -rf \
  /public/home/xdzs2026_c415/venvs/vllm_baseline \
  /public/home/xdzs2026_c415/venvs/vllm_gfx936
```

### 5.3 初始化、检查 torch、构建 control

用 `&&` 确保前一步失败时不会继续：

```bash
cd /public/home/xdzs2026_c415/experiments/gfx936_skinny/source

bash scripts/scnet_ab_gfx936.sh init && \
/public/home/xdzs2026_c415/venvs/vllm_baseline/bin/python -c \
'import torch; p=torch.cuda.get_device_properties(0); print(torch.__version__, torch.version.hip, p.gcnArchName, p.multi_processor_count)' && \
bash scripts/scnet_ab_gfx936.sh build-control
```

预期中间检查包含：

```text
2.10.0 6.3.26093 gfx936:sramecc+:xnack- 80
```

control wheel 构建成功后查看 hash：

```bash
cat /public/home/xdzs2026_c415/results/gfx936_skinny/wheels/control/SHA256SUMS
```

### 5.4 直接 kernel 门禁

```bash
cd /public/home/xdzs2026_c415/experiments/gfx936_skinny/source
bash scripts/scnet_ab_gfx936.sh bench
echo "bench_exit=$?"
```

结果文件：

```text
/public/home/xdzs2026_c415/results/gfx936_skinny/microbench.json
```

摘要命令：

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("/public/home/xdzs2026_c415/results/gfx936_skinny/microbench.json")
d = json.loads(p.read_text())
print("arch:", d.get("arch"))
print("passed:", d.get("passed"))
print("projected:", d.get("projected_linear_speedup"))
print("fatal_error:", d.get("fatal_error"))
for row in d.get("rows", []):
    print(
        row.get("n"), row.get("family"),
        "speedup=", row.get("speedup"),
        "stock_p99=", row.get("stock_p99_ms"),
        "candidate_p99=", row.get("candidate_p99_ms"),
        "admitted=", row.get("admitted"),
        "error=", row.get("error"),
    )
PY
```

## 6. 直接 kernel 决策门禁

benchmark 会覆盖 Qwen3.5-27B 的 6 类主要 linear，每类测 `N=1/2/4`，共 18 行：

1. `gdn_qkvz`
2. `gdn_ba`
3. `full_attention_qkv_gate`
4. `attention_output`
5. `mlp_gate_up`
6. `mlp_down`

每行必须同时满足：

| 门禁 | 阈值 |
|---|---:|
| 输出有限 | `finite=True` |
| PyTorch close | `rtol=0.03, atol=0.5` |
| 余弦相似度 | `>= 0.999` |
| 相对 L2 | `<= 0.01` |
| 中位加速 | `>= 1.15` |
| P99 | candidate `<=` stock |

全局还必须满足：

- 18 行全部通过；
- N=1、2、4 的主导 linear 总延迟投影加速均 `>= 1.6`。

### 通过：`bench_exit=0`

脚本会原子写入白名单。继续：

```bash
cd /public/home/xdzs2026_c415/experiments/gfx936_skinny/source
git diff -- vllm/model_executor/layers/rocm_skinny_shapes.py
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v

git add vllm/model_executor/layers/rocm_skinny_shapes.py
git commit -m "perf: admit measured gfx936 skinny shapes"

# 当前 origin 为 GitHub。
git push origin HEAD:dx_branch

bash scripts/scnet_ab_gfx936.sh build-candidate
cat /public/home/xdzs2026_c415/results/gfx936_skinny/wheels/candidate/SHA256SUMS
```

### 失败：`bench_exit=2`

- 不构建性能 candidate；
- 不加载 27B 模型；
- 保留 `microbench.json` 和日志；
- 后续诊断服务强制设置：

```bash
export FDU_FORCE_STOCK_GEMM=1
```

## 7. candidate 通过后的模型 A/B

### 7.1 同 wheel stock/candidate 探针

```bash
cd /public/home/xdzs2026_c415/experiments/gfx936_skinny/source

bash scripts/scnet_ab_gfx936.sh start-candidate-stock
bash scripts/scnet_ab_gfx936.sh probe candidate-stock
bash scripts/scnet_ab_gfx936.sh stop

bash scripts/scnet_ab_gfx936.sh start-candidate
bash scripts/scnet_ab_gfx936.sh probe candidate
bash scripts/scnet_ab_gfx936.sh stop
```

检查回答一致：

```bash
/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("/public/home/xdzs2026_c415/results/gfx936_skinny/probes")
stock = json.loads((root / "candidate-stock.json").read_text())
candidate = json.loads((root / "candidate.json").read_text())
assert stock["prompts"] == candidate["prompts"]
assert stock["responses"] == candidate["responses"]
print("token-consistency smoke passed")
PY
```

出现 import error、illegal instruction、GPU fault、OOM、超时或回答不一致，立即拒绝 candidate。

### 7.2 两轮三档吞吐

按得分权重优先测 8–16K，再测 16–32K 和 4–8K：

```bash
for round in 1 2; do
  for mode in control candidate; do
    bash scripts/scnet_ab_gfx936.sh "start-$mode"
    for tier in 8-16K 16-32K 4-8K; do
      bash scripts/scnet_ab_gfx936.sh throughput "$tier" 10 "$mode-r$round"
    done
    bash scripts/scnet_ab_gfx936.sh stop
  done
done
```

继续门禁：

- candidate 8–16K 吞吐中位数相对 control 至少提升 50%；
- 其他两档不回退；
- TTFT P99 不超过 control 的 1.5x；
- TPOT P99 不超过 control 的 1.5x；
- 无失败请求。

### 7.3 四项精度

```bash
for mode in control candidate; do
  bash scripts/scnet_ab_gfx936.sh "start-$mode"
  for task in hotpotqa gov_report retrieval_multi_point aggregation_keyword_aggregation; do
    bash scripts/scnet_ab_gfx936.sh accuracy "$task" 10 "$mode-accuracy"
  done
  bash scripts/scnet_ab_gfx936.sh stop
done
```

任一任务降低超过 1% 都拒绝 candidate。本 BF16 路线不接受精度系数小于 1.0 作为正常结果。

### 7.4 复现投影分数

```bash
/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python \
  scripts/score_gfx936.py \
  --results-root /public/home/xdzs2026_c415/results/gfx936_skinny \
  --control-run control-r1 --control-run control-r2 \
  --candidate-run candidate-r1 --candidate-run candidate-r2 \
  --accuracy-coefficient 1.0 \
  --output /public/home/xdzs2026_c415/results/gfx936_skinny/score.json
```

## 8. 跑分与提交决策

### 现在直接提交 `91ea5e9`

- 可以启动；
- preflight 会检查真实 `gfx936` 和安装 wheel；
- 由于白名单为空，linear 实际使用 stock BF16；
- 适合做安全基准，不应预期明显性能提升。

### 目标是冲 90

必须至少完成：

```text
control wheel
  -> 18-row direct kernel gate
  -> generated whitelist
  -> candidate wheel
  -> stock/candidate response consistency
  -> two-run three-tier throughput
  -> SLA
  -> four accuracy tasks
  -> reproduced score
```

任一门禁失败，不能因为目标分数高就强行提交。

## 9. 紧急回滚

运行时回滚：

```bash
export FDU_FORCE_STOCK_GEMM=1
bash scripts/scnet_start_optimized.sh
```

如果平台环境不是实测的原生 `gfx936`，应让 preflight 失败或使用 stock 回滚，不得恢复架构伪装。

## 10. 重要文件地图

| 文件 | 作用 |
|---|---|
| `setup.py` | 请求构建 `vllm._rocm_C` |
| `csrc/rocm/skinny_gemms.cu` | gfx936 HIP skinny kernels |
| `vllm/platforms/rocm_capabilities.py` | 狭义架构能力判定 |
| `vllm/model_executor/layers/rocm_skinny_policy.py` | 纯 Python shape 门禁 |
| `vllm/model_executor/layers/rocm_skinny_shapes.py` | SCNet 生成的精确白名单 |
| `vllm/model_executor/layers/utils.py` | stock/custom GEMM 实际调度 |
| `scripts/preflight_rocm.py` | 模型加载前环境/扩展门禁 |
| `scripts/bench_gfx936_skinny.py` | 18 行直接 kernel benchmark 和白名单生成 |
| `scripts/scnet_ab_gfx936.sh` | 隔离 venv、wheel、服务和官方评测 A/B |
| `scripts/probe_gfx936.py` | 固定三问顺序 smoke probe |
| `scripts/score_gfx936.py` | 按官方曲线复现投影分数 |
| `launch.sh` | 平台 BF16 启动入口 |
| `scripts/rocm_env.sh` | 保守 ROCm 环境与缓存目录 |
| `docs/SCNET_RUN.md` | 精简 SCNet 操作手册 |

## 11. 本地验证状态

`91ea5e9` 前已完成：

- `tests/fdu` 共 74 个测试通过；
- `launch.sh`、`rocm_env.sh`、`scnet_start_optimized.sh`、`scnet_ab_gfx936.sh` 通过 `bash -n`；
- Python 脚本通过 `compileall`；
- 活跃路径无 AWQ/INT4/FP8 权重转换命令；
- 无 `HSA_OVERRIDE_GFX_VERSION=` 或 `ROCBLAS_LAYER=` 导出。

本地 macOS 没有 DCU，所以尚未完成的唯一关键事实是：**SCNet 上的实际 kernel 正确性和加速数据**。

## 12. 接手人的第一个动作

不要直接启动 27B 模型。先执行：

```bash
apt-get install -y python3.10-venv
rm -rf /public/home/xdzs2026_c415/venvs/vllm_baseline \
       /public/home/xdzs2026_c415/venvs/vllm_gfx936
cd /public/home/xdzs2026_c415/experiments/gfx936_skinny/source
bash scripts/scnet_ab_gfx936.sh init
bash scripts/scnet_ab_gfx936.sh build-control
bash scripts/scnet_ab_gfx936.sh bench
```

然后只根据 `microbench.json` 决定是否继续。
