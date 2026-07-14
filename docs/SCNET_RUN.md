# SCNet gfx936 最快测试流程

目标是用最少的模型加载次数判断在线 W8/W4 是否值得提交。最新平台结果为 15.00 / 11.97 / 6.11 tok/s、最终 66.7878，与上一轮 66.8175 统计等价；两次评测都没有可用启动日志/精确 commit，因此快速阶段只把约 66.8 分当方向基线。正式候选模式仍由 `FDU_GFX936_QUANT_MODE` 选择。

所有脚本只写 `/tmp` 和以下隔离目录，不修改模型与原始 testdata：

```text
/public/home/xdzs2026_c415/experiments/gfx936_skinny
/public/home/xdzs2026_c415/venvs/vllm_gfx936
/public/home/xdzs2026_c415/results/gfx936_skinny
```

## 0. 拉最新 `dx_branch`

```bash
set -o pipefail
cd /public/home/xdzs2026_c415/vllm_cscc
git status --short
git fetch origin dx_branch
git checkout dx_branch
git merge --ff-only origin/dx_branch
git rev-parse HEAD
```

如果 `git status --short` 有本地改动，先停下确认，不要用强制 reset 覆盖。

## 1. 初始化并重建当前 gfx936 wheel

```bash
set -o pipefail
cd /public/home/xdzs2026_c415/vllm_cscc
bash scripts/scnet_ab_gfx936.sh init
bash scripts/scnet_ab_gfx936.sh build-candidate
```

正式 W8 内核现在编入 `vllm._rocm_C`，因此切到本提交后必须重建 candidate wheel；`sync-candidate-python` 不会更新 native ABI，不能用于验证该候选。

## 2. 可选：复现独立 JIT kernel 微基准

```bash
set -o pipefail
SOURCE=/public/home/xdzs2026_c415/experiments/gfx936_skinny/source
PY=/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python
COLD_CACHE="$(mktemp -d /tmp/fdu_gfx936_quant_cold.XXXXXX)"
trap 'rm -rf "$COLD_CACHE"' EXIT

if /usr/bin/time -f 'compile_wall_s=%e' \
  "$PY" "$SOURCE/scripts/build_gfx936_quant_jit.py" \
    --source "$SOURCE/csrc/fdu/gfx936_quant_gemv.hip" \
    --arch gfx936 --timeout 45 --cache-root "$COLD_CACHE" \
  2>&1 | tee /tmp/fdu_gfx936_quant_compile.log; then
  COMPILE_STATUS=0
else
  COMPILE_STATUS=$?
fi
if [ "$COMPILE_STATUS" -ne 0 ]; then
  echo "cold_compile_failed=$COMPILE_STATUS" >&2
  exit "$COMPILE_STATUS"
fi

SO1="$("$PY" "$SOURCE/scripts/build_gfx936_quant_jit.py" \
  --source "$SOURCE/csrc/fdu/gfx936_quant_gemv.hip" --arch gfx936 --timeout 45 \
  --cache-root "$COLD_CACHE")"
SO2="$("$PY" "$SOURCE/scripts/build_gfx936_quant_jit.py" \
  --source "$SOURCE/csrc/fdu/gfx936_quant_gemv.hip" --arch gfx936 --timeout 45 \
  --cache-root "$COLD_CACHE")"
test "$SO1" = "$SO2" && test -f "$SO1" && echo "jit_cache_ok=$SO1"
rm -rf "$COLD_CACHE"
trap - EXIT
```

该步骤只复现历史单 kernel 数据；正式服务使用 wheel 内 `_rocm_C`，不依赖这里的 `/tmp` `.so`。首次退出 0、`compile_wall_s <=50` 和重复缓存路径仍可用于确认 HIP 源本身可编译。

## 3. 六 shape W8 门禁

不要再给下面的命令套外层 `tee`；脚本已经把输出写到 `/tmp/fdu_gfx936_quant_w8.log`，且会保留真实退出码。

```bash
set -o pipefail
cd /public/home/xdzs2026_c415/vllm_cscc
if bash scripts/scnet_ab_gfx936.sh quant-bench-w8; then
  W8_STATUS=0
else
  W8_STATUS=$?
fi
echo "w8_status=$W8_STATUS"

python3 - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/fdu_gfx936_quant_w8.json')
d = json.loads(p.read_text())
print('commit=', d.get('git_commit'), 'arch=', d.get('arch'), 'passed=', d.get('passed'))
admitted = {(r['M'], r['K']) for r in d['rows'] if r['admitted']}
for r in d['rows']:
    print((r['M'], r['K']), r['selected_kind'], r['admitted'],
          'nrmse=', r['nrmse'], 'cos=', r['cosine'], 'speedup=', r['speedup'],
          'reason=', r['reason'])
mlp = {(34816, 5120), (5120, 17408)}
print('fast_partial_allowed=', mlp <= admitted and len(admitted) >= 4)
PY
```

完整通过要求六行均满足：有限值、W8 NRMSE `<= 0.015`、cosine `>= 0.999`、相对当前 BF16/LLMM1 至少 `1.10x`。

为了快速筛选，退出码为 2 时只允许一种“部分继续”：两个 MLP shape `(34816, 5120)`、`(5120, 17408)` 和至少两个其他 shape 已通过。未通过的 shape 必须在模型加载时保留 BF16。少于这个条件就停止，不加载 27B 模型。

## 4. 只加载一次 W8，先打最高权重档

命令顺序不能颠倒：启动脚本会拒绝“请求 W8、实际 fail-open 到 BF16”的假健康状态；生成探针成功后才跑吞吐。

```bash
set -o pipefail
cd /public/home/xdzs2026_c415/vllm_cscc
bash scripts/scnet_ab_gfx936.sh start-candidate-w8
bash scripts/scnet_ab_gfx936.sh probe-candidate-w8
bash scripts/scnet_ab_gfx936.sh throughput 8-16K 3 w8-fast
```

W8 继续条件：

- 8–16K 吞吐至少 `12.60 tok/s`（相对最新 11.97 约 +5.3%）；
- TTFT P99 与 TPOT P99 均不超过官方 baseline 的 `1.45x`，为 1.5x 硬门槛留余量；
- `/tmp/fdu_gfx936_w8.log` 没有 OOM、Traceback、非有限指标或 `keeping BF16 path`；
- 日志确认高字节量 shape 的量化层已被接纳；被拒 shape 明确回退 BF16。

三条样本只用于方向判断，不是最终分数。如果 W8 不满足以上条件，执行 `bash scripts/scnet_ab_gfx936.sh stop` 并显式回滚为 `off`。

## 5. 仅在 W8 有收益时尝试 hybrid W4

W8 已超过 12.60、但距离目标仍明显不足时才做这一轮：

```bash
set -o pipefail
cd /public/home/xdzs2026_c415/vllm_cscc
bash scripts/scnet_ab_gfx936.sh stop
if bash scripts/scnet_ab_gfx936.sh quant-bench-hybrid; then
  HYBRID_STATUS=0
else
  HYBRID_STATUS=$?
fi
echo "hybrid_status=$HYBRID_STATUS"

python3 - <<'PY'
import json
from pathlib import Path
w8 = json.loads(Path('/tmp/fdu_gfx936_quant_w8.json').read_text())
hybrid = json.loads(Path('/tmp/fdu_gfx936_quant_hybrid_w4.json').read_text())
w8_rows = {(r['M'], r['K']): r for r in w8['rows']}
for row in hybrid['rows']:
    shape = (row['M'], row['K'])
    if shape in {(34816, 5120), (5120, 17408)}:
        old = w8_rows[shape]['candidate_ms']
        new = row['candidate_ms']
        ratio = old / new if old and new else None
        print(shape, 'kind=', row['selected_kind'], 'admitted=', row['admitted'],
              'nrmse=', row['nrmse'], 'cos=', row['cosine'],
              'w4_vs_w8=', ratio)
PY
```

只有两个 MLP 行都选择 `w4`、满足 NRMSE `<= 0.08` 与 cosine `>= 0.995`，且 `w4_vs_w8 >= 1.05` 时才启动 hybrid：

```bash
set -o pipefail
cd /public/home/xdzs2026_c415/vllm_cscc
bash scripts/scnet_ab_gfx936.sh start-candidate-hybrid
bash scripts/scnet_ab_gfx936.sh probe-candidate-hybrid
bash scripts/scnet_ab_gfx936.sh throughput 8-16K 3 hybrid-fast
```

端到端 hybrid 还必须比 W8 的 8–16K 吞吐高 `1.03x`，并满足同一 1.45x SLA 余量；否则 W8 胜出。

## 6. 胜者短验：三档 + 两项精度

设置 `WINNER=w8` 或 `WINNER=hybrid_w4`，会再启动一次最终候选：

```bash
set -o pipefail
cd /public/home/xdzs2026_c415/vllm_cscc
WINNER=w8

case "$WINNER" in
  w8) bash scripts/scnet_ab_gfx936.sh start-candidate-w8 ;;
  hybrid_w4) bash scripts/scnet_ab_gfx936.sh start-candidate-hybrid ;;
  *) echo "invalid WINNER=$WINNER" >&2; exit 2 ;;
esac

for tier in 8-16K 16-32K 4-8K; do
  bash scripts/scnet_ab_gfx936.sh throughput "$tier" 3 "$WINNER-fast"
done
bash scripts/scnet_ab_gfx936.sh accuracy hotpotqa 3 "$WINNER-fast"
bash scripts/scnet_ab_gfx936.sh accuracy retrieval_multi_point 3 "$WINNER-fast"
bash scripts/scnet_ab_gfx936.sh stop
```

不要直接在 `~/testdata` 运行；wrapper 会先复制只读 testdata，避免污染官方输入。

胜者必须满足：三档均无吞吐回退、TTFT/TPOT 都在 1.45x 余量内、两项抽样精度相对保底路径下降不超过 1%、无失败请求/OOM/非有限值，并且加权投影高于 66.7878。完整平台评测仍是最终统计与四项精度验证。

## 7. 把这些结果发回

最小结果集：

```text
/tmp/fdu_gfx936_quant_compile.log
/tmp/fdu_gfx936_quant_w8.json
/tmp/fdu_gfx936_quant_w8.log
/tmp/fdu_gfx936_w8.log
/public/home/xdzs2026_c415/results/gfx936_skinny/throughput/w8-fast/8-16K.json
```

若测 hybrid，再附：

```text
/tmp/fdu_gfx936_quant_hybrid_w4.json
/tmp/fdu_gfx936_hybrid_w4.log
/public/home/xdzs2026_c415/results/gfx936_skinny/throughput/hybrid-fast/8-16K.json
```

通常的最终选择规则只有三种：hybrid 明确胜出则选 `hybrid_w4`；否则 W8 通过则选 `w8`；否则保持 `off`。本轮已停止继续 SCNet；正式盲测候选使用 wheel 内置选择性 `w8`。
