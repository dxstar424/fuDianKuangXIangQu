# SCNet gfx936 BF16 流水线

本流程只操作以下持久目录：

```text
/public/home/xdzs2026_c415/experiments/gfx936_skinny
/public/home/xdzs2026_c415/venvs/vllm_baseline
/public/home/xdzs2026_c415/venvs/vllm_gfx936
/public/home/xdzs2026_c415/results/gfx936_skinny
```

`Qwen3.5-27B` 和 `testdata` 是只读输入。所有官方评测都在 `results/gfx936_skinny/eval_work` 的独立副本中运行。

## 1. 拉取 dx_branch 并确认硬件

```bash
export ROOT=/public/home/xdzs2026_c415
export EXP=$ROOT/experiments/gfx936_skinny
mkdir -p "$EXP" "$ROOT/results/gfx936_skinny"
git clone --branch dx_branch --single-branch \
  https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git \
  "$EXP/source"
cd "$EXP/source"
git rev-parse HEAD | tee "$ROOT/results/gfx936_skinny/source_commit.txt"

unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER PYTHONPATH
rocminfo | sed -n '/Name:.*gfx/,+3p' | head
python3 - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(torch.__version__, torch.version.hip, p.name, p.gcnArchName, p.multi_processor_count)
assert p.gcnArchName.split(":", 1)[0] == "gfx936"
PY
```

## 2. 构建空白名单 control，直测 kernel

```bash
bash scripts/scnet_ab_gfx936.sh init
bash scripts/scnet_ab_gfx936.sh build-control
bash scripts/scnet_ab_gfx936.sh bench
```

`bench` 只在 18 行全部满足以下门禁时写入白名单：余弦相似度 `>=0.999`、相对 L2 `<=0.01`、`assert_close(rtol=0.03, atol=0.5)`、单 shape 中位加速 `>=1.15`、P99 不回退，且 N=1/2/4 的主导 linear 投影加速均 `>=1.6`。

如果命令退出码为 2，保留 `microbench.json` 并停止；不加载大模型。通过后：

```bash
git diff -- vllm/model_executor/layers/rocm_skinny_shapes.py
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v
git add vllm/model_executor/layers/rocm_skinny_shapes.py
git commit -m "perf: admit measured gfx936 skinny shapes"
git push origin HEAD:dx_branch
bash scripts/scnet_ab_gfx936.sh build-candidate
```

## 3. 同 wheel 探针与吞吐 A/B

```bash
bash scripts/scnet_ab_gfx936.sh start-candidate-stock
bash scripts/scnet_ab_gfx936.sh probe candidate-stock
bash scripts/scnet_ab_gfx936.sh stop
bash scripts/scnet_ab_gfx936.sh start-candidate
bash scripts/scnet_ab_gfx936.sh probe candidate
bash scripts/scnet_ab_gfx936.sh stop
```

对比 `probes/candidate-stock.json` 与 `probes/candidate.json` 的 `prompts` 和 `responses`，必须完全一致。

每个模式都按 8–16K、16–32K、4–8K 顺序跑两轮：

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

继续条件：8–16K 至少提升 50%，其他档不回退，TTFT/TPOT P99 都在 control 的 1.5x 内，且无失败请求。

## 4. 精度与投影得分

```bash
for mode in control candidate; do
  bash scripts/scnet_ab_gfx936.sh "start-$mode"
  for task in hotpotqa gov_report retrieval_multi_point aggregation_keyword_aggregation; do
    bash scripts/scnet_ab_gfx936.sh accuracy "$task" 10 "$mode-accuracy"
  done
  bash scripts/scnet_ab_gfx936.sh stop
done

/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python scripts/score_gfx936.py \
  --results-root /public/home/xdzs2026_c415/results/gfx936_skinny \
  --control-run control-r1 --control-run control-r2 \
  --candidate-run candidate-r1 --candidate-run candidate-r2 \
  --accuracy-coefficient 1.0 \
  --output /public/home/xdzs2026_c415/results/gfx936_skinny/score.json
```

任一精度任务降低超过 1% 就拒绝 candidate。未实测前不声称达到 90 分。

## 5. 紧急回退

```bash
export FDU_FORCE_STOCK_GEMM=1
bash scripts/scnet_start_optimized.sh
```

回退只跳过自定义 linear 调度，不需重编 wheel。禁止使用 `pkill`/`killall`；流水线只终止它自己记录的 PID。
