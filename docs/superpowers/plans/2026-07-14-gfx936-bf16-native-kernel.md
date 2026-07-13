# gfx936 BF16 Native Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and validate a competition-compliant BF16 vLLM wheel that activates native `gfx936` skinny GEMM only for SCNet-measured safe shapes, preserves the BF16 accuracy coefficient, and has an immediate stock-GEMM rollback.

**Architecture:** Build `vllm._rocm_C` for native `gfx936`, keep `gfx936` outside the global MI300/GFX9 feature predicates, and add a narrow architecture plus exact-shape policy at `rocm_unquantized_gemm_impl`. Start with an empty runtime whitelist, directly benchmark the compiled op on SCNet, generate the whitelist deterministically from passing rows, rebuild the candidate wheel, then compare an empty-whitelist BF16 control with the candidate under identical launch conditions.

**Tech Stack:** Python 3.10+, pure-stdlib `unittest`, PyTorch 2.10 ROCm/HIP 6.3, CMake, HIP C++, vLLM 0.18.1, Bash, SCNet WebShell, JSON benchmark artifacts.

---

## Source specification and fixed gates

Implement against [the approved design](../specs/2026-07-14-gfx936-bf16-native-kernel-design.md). Do not reintroduce AWQ, FP8 weight conversion, persistent quantized weights, speculative decoding, scheduler overrides, `HSA_OVERRIDE_GFX_VERSION`, or source-tree `PYTHONPATH` shadowing.

Use these persistent SCNet paths exactly:

```text
/public/home/xdzs2026_c415/experiments/gfx936_skinny
/public/home/xdzs2026_c415/venvs/vllm_baseline
/public/home/xdzs2026_c415/venvs/vllm_gfx936
/public/home/xdzs2026_c415/results/gfx936_skinny
```

Treat `/public/home/xdzs2026_c415/Qwen3.5-27B` and `/public/home/xdzs2026_c415/testdata` as read-only inputs.

The candidate may proceed from one gate to the next only when all requirements in the approved design pass. In particular, the direct-kernel gate is cosine similarity `>= 0.999`, relative L2 `<= 0.01`, `assert_close(rtol=0.03, atol=0.5)`, median speedup `>= 1.15`, no P99 regression, and projected dominant linear speedup `>= 1.6` for every measured decode batch size.

## Task 1: Create the isolated implementation worktree

**Files:** none

- [ ] **Step 1: Confirm the source worktree is clean and record the design commit**

Run:

```bash
cd /Users/dxstarpomdx/studyu/self_ss/FDU_SCCSCC26/fdu-sccscc26
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
```

Expected: no modified files other than this plan before it is committed; the approved design commit `f0cf41a` is in history.

- [ ] **Step 2: Create a dedicated branch and worktree**

Run after this plan commit exists:

```bash
cd /Users/dxstarpomdx/studyu/self_ss/FDU_SCCSCC26/fdu-sccscc26
git worktree add \
  -b codex/gfx936-bf16 \
  /Users/dxstarpomdx/studyu/self_ss/FDU_SCCSCC26/fdu-sccscc26-gfx936 \
  dx_branch
cd /Users/dxstarpomdx/studyu/self_ss/FDU_SCCSCC26/fdu-sccscc26-gfx936
git status --short --branch
```

Expected: branch `codex/gfx936-bf16`, clean worktree, based on `dx_branch`.

- [ ] **Step 3: Run the pre-change local contract checks**

Run:

```bash
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_start_optimized.sh
```

Expected: the existing `tests/fdu` suite is empty or passes; all three shell files parse.

## Task 2: Make the ROCm extension and gfx936 BF16 kernel buildable

**Files:**

- Create: `tests/fdu/test_gfx936_build_contract.py`
- Modify: `setup.py`
- Modify: `csrc/rocm/skinny_gemms.cu`

- [ ] **Step 1: Write the failing build-contract test**

Create `tests/fdu/test_gfx936_build_contract.py` with this contract:

```python
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Gfx936BuildContractTest(unittest.TestCase):
    def test_hip_build_requests_rocm_extension(self) -> None:
        text = (ROOT / "setup.py").read_text()
        self.assertRegex(
            text,
            re.compile(
                r"if _is_hip\(\):\s+"
                r"ext_modules\.append\(CMakeExtension\(name=\"vllm\._rocm_C\"\)\)"
            ),
        )

    def test_gfx936_compiles_only_the_gfx9_skinny_family(self) -> None:
        text = (ROOT / "csrc/rocm/skinny_gemms.cu").read_text()
        gfx9_block, mi3xx_tail = text.split("#if defined(__HIPCC__)", 2)[1:]
        self.assertIn("defined(__gfx936__)", gfx9_block)
        mi3xx_block = mi3xx_tail.split("#endif", 1)[0]
        self.assertNotIn("defined(__gfx936__)", mi3xx_block)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm it fails for both missing contracts**

Run:

```bash
python3 -m unittest tests.fdu.test_gfx936_build_contract -v
```

Expected: failure because the `_rocm_C` line is commented and `__gfx936__` is absent from the GFX9 compiler macro.

- [ ] **Step 3: Request `_rocm_C` in HIP wheels**

Replace the commented block in `setup.py` with:

```python
if _is_hip():
    ext_modules.append(CMakeExtension(name="vllm._rocm_C"))
```

Do not mark the extension optional: a candidate wheel without it must fail to build or fail preflight.

- [ ] **Step 4: Compile gfx936 only in the BF16/FP16 GFX9 branch**

Change the first architecture macro in `csrc/rocm/skinny_gemms.cu` to:

```cpp
#if defined(__HIPCC__) &&                                             \
    (defined(__gfx90a__) || defined(__gfx936__) ||                    \
     defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif
```

Leave the `__HIP__MI3XX__` macro exactly limited to `gfx942` and `gfx950`.

- [ ] **Step 5: Run the test and static formatting checks**

Run:

```bash
python3 -m unittest tests.fdu.test_gfx936_build_contract -v
git diff --check
```

Expected: two tests pass; no whitespace errors.

- [ ] **Step 6: Commit the build contract**

```bash
git add setup.py csrc/rocm/skinny_gemms.cu tests/fdu/test_gfx936_build_contract.py
git commit -m "fix: build gfx936 ROCm skinny extension"
```

## Task 3: Add narrow architecture and exact-shape policy modules

**Files:**

- Create: `vllm/platforms/rocm_capabilities.py`
- Create: `vllm/model_executor/layers/rocm_skinny_shapes.py`
- Create: `vllm/model_executor/layers/rocm_skinny_policy.py`
- Create: `tests/fdu/test_rocm_skinny_policy.py`
- Modify: `vllm/platforms/rocm.py`

- [ ] **Step 1: Write failing pure-Python tests that do not import torch or vLLM**

Use `importlib.util.spec_from_file_location` in `tests/fdu/test_rocm_skinny_policy.py` so the local macOS check does not execute `vllm/__init__.py`. For the policy test, register empty `ModuleType` parents named `vllm`, `vllm.model_executor`, and `vllm.model_executor.layers` with `__path__` pointing at the corresponding source directories; load `rocm_skinny_shapes.py` under its fully-qualified name, then load the policy under its fully-qualified name. Cover these cases:

```python
self.assertEqual(canonical_rocm_arch("gfx936:sramecc+:xnack-"), "gfx936")
self.assertTrue(is_gfx936_arch("gfx936:sramecc+:xnack-"))
self.assertTrue(supports_rocm_skinny_gemm_arch("gfx936"))
self.assertFalse(supports_rocm_skinny_gemm_arch("gfx1100"))

passing = frozenset({(1, 4096, 4096, "bfloat16", False)})
self.assertTrue(
    is_gfx936_skinny_eligible(
        n=1,
        m=4096,
        k=4096,
        dtype_name="bfloat16",
        bias_present=False,
        weight_contiguous=True,
        activation_reshapeable=True,
        validated_shapes=passing,
    )
)
```

Also assert rejection for `n=5`, float32, bias, non-contiguous weight, non-reshapeable activation, `k % 8 != 0`, and a shape absent from `validated_shapes`. Parse `vllm/platforms/rocm.py` and assert `gfx936` is absent from the `_ON_MI3XX` and `_ON_GFX9` assignment lines.

- [ ] **Step 2: Run the test and confirm the modules are missing**

Run:

```bash
python3 -m unittest tests.fdu.test_rocm_skinny_policy -v
```

Expected: import/file-not-found failure for the new capability or policy module.

- [ ] **Step 3: Implement the architecture helper**

Create `vllm/platforms/rocm_capabilities.py`:

```python
from __future__ import annotations


ROCM_SKINNY_GEMM_ARCHES = frozenset({"gfx90a", "gfx936", "gfx942", "gfx950"})


def canonical_rocm_arch(gcn_arch: str | None) -> str:
    return (gcn_arch or "").split(":", 1)[0].strip().lower()


def is_gfx936_arch(gcn_arch: str | None) -> bool:
    return canonical_rocm_arch(gcn_arch) == "gfx936"


def supports_rocm_skinny_gemm_arch(gcn_arch: str | None) -> bool:
    return canonical_rocm_arch(gcn_arch) in ROCM_SKINNY_GEMM_ARCHES
```

- [ ] **Step 4: Create an intentionally empty generated whitelist**

Create `vllm/model_executor/layers/rocm_skinny_shapes.py`:

```python
from __future__ import annotations


SkinnyShape = tuple[int, int, int, str, bool]

# This remains empty until scripts/bench_gfx936_skinny.py admits measured rows.
VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset()
```

An empty set is the safe local implementation state: direct op benchmarking does not depend on runtime dispatch.

- [ ] **Step 5: Implement the pure shape policy**

Create `vllm/model_executor/layers/rocm_skinny_policy.py`:

```python
from __future__ import annotations

from collections.abc import AbstractSet

from .rocm_skinny_shapes import (
    VALIDATED_GFX936_SHAPES,
    SkinnyShape,
)


SUPPORTED_BATCH_SIZES = frozenset({1, 2, 4})
SUPPORTED_DTYPES = frozenset({"bfloat16", "float16"})


def is_gfx936_skinny_eligible(
    *,
    n: int,
    m: int,
    k: int,
    dtype_name: str,
    bias_present: bool,
    weight_contiguous: bool,
    activation_reshapeable: bool,
    validated_shapes: AbstractSet[SkinnyShape] = VALIDATED_GFX936_SHAPES,
) -> bool:
    shape = (n, m, k, dtype_name, bias_present)
    return (
        n in SUPPORTED_BATCH_SIZES
        and dtype_name in SUPPORTED_DTYPES
        and not bias_present
        and weight_contiguous
        and activation_reshapeable
        and k % 8 == 0
        and shape in validated_shapes
    )
```

- [ ] **Step 6: Expose only targeted runtime predicates**

In `vllm/platforms/rocm.py`, import the two pure helpers and add:

```python
_ON_GFX936 = is_gfx936_arch(_GCN_ARCH)
_SUPPORTS_ROCM_SKINNY_GEMM = supports_rocm_skinny_gemm_arch(_GCN_ARCH)


def on_gfx936() -> bool:
    return _ON_GFX936


def supports_rocm_skinny_gemm() -> bool:
    return _SUPPORTS_ROCM_SKINNY_GEMM
```

Do not add `gfx936` to `_ON_GFX9`, `_ON_MI3XX`, `on_gfx9()`, `on_mi3xx()`, attention selection, FP8 selection, or AITER selection.

- [ ] **Step 7: Run tests and commit**

```bash
python3 -m unittest tests.fdu.test_rocm_skinny_policy -v
python3 -m compileall -q \
  vllm/platforms/rocm_capabilities.py \
  vllm/model_executor/layers/rocm_skinny_shapes.py \
  vllm/model_executor/layers/rocm_skinny_policy.py
git diff --check
git add vllm/platforms/rocm.py vllm/platforms/rocm_capabilities.py \
  vllm/model_executor/layers/rocm_skinny_shapes.py \
  vllm/model_executor/layers/rocm_skinny_policy.py \
  tests/fdu/test_rocm_skinny_policy.py
git commit -m "perf: add guarded gfx936 skinny GEMM policy"
```

Expected: all policy tests pass and the committed whitelist is empty.

## Task 4: Wire the rollback flag and guarded dispatch

**Files:**

- Create: `tests/fdu/test_gfx936_dispatch_contract.py`
- Modify: `vllm/envs.py`
- Modify: `vllm/model_executor/layers/utils.py`

- [ ] **Step 1: Write the failing dispatch-contract tests**

The test must verify textually, without importing torch, that:

1. `FDU_FORCE_STOCK_GEMM: bool = False` is declared in `envs.py`;
2. the environment registry maps it from `FDU_FORCE_STOCK_GEMM`, default false;
3. `rocm_unquantized_gemm_impl` returns `torch.nn.functional.linear` when the flag is true before any AITER or skinny call;
4. gfx936 calls `is_gfx936_skinny_eligible` before `ops.wvSplitK`;
5. existing `gfx90a`, `gfx942`, and `gfx950` behavior remains behind the pre-existing `on_gfx9()` condition.

Run:

```bash
python3 -m unittest tests.fdu.test_gfx936_dispatch_contract -v
```

Expected: failure because the flag and policy call do not exist.

- [ ] **Step 2: Add the emergency environment flag**

Add the typed declaration beside `VLLM_ROCM_USE_SKINNY_GEMM`:

```python
FDU_FORCE_STOCK_GEMM: bool = False
```

Add the registry entry:

```python
"FDU_FORCE_STOCK_GEMM": lambda: (
    os.getenv("FDU_FORCE_STOCK_GEMM", "False").lower() in ("true", "1")
),
```

- [ ] **Step 3: Guard the dispatch**

In `rocm_unquantized_gemm_impl`, calculate `n`, `m`, and `k`, then immediately add:

```python
if envs.FDU_FORCE_STOCK_GEMM:
    return torch.nn.functional.linear(x, weight, bias)
```

Import `on_gfx936`, `on_gfx9`, `on_gfx950`, and `supports_rocm_skinny_gemm`. Preserve the existing `wvSplitKrc` and AITER branches for their original architectures. Replace only the small-batch skinny predicate with:

```python
use_skinny = (
    envs.VLLM_ROCM_USE_SKINNY_GEMM
    and supports_rocm_skinny_gemm()
    and x.dtype in [torch.float16, torch.bfloat16]
    and weight.dtype == x.dtype
    and k % 8 == 0
)

if use_skinny and on_gfx936():
    from vllm.model_executor.layers.rocm_skinny_policy import (
        is_gfx936_skinny_eligible,
    )

    use_skinny = is_gfx936_skinny_eligible(
        n=n,
        m=m,
        k=k,
        dtype_name=str(x.dtype).removeprefix("torch."),
        bias_present=bias is not None,
        weight_contiguous=weight.is_contiguous(),
        activation_reshapeable=x.size(-1) == k,
    )
elif use_skinny:
    use_skinny = on_gfx9()
```

Keep the existing `ops.wvSplitK(weight, x_view, cu_count, bias)` call and stock fallback. Never catch a GPU exception and continue.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m unittest \
  tests.fdu.test_rocm_skinny_policy \
  tests.fdu.test_gfx936_dispatch_contract -v
python3 -m compileall -q vllm/envs.py vllm/model_executor/layers/utils.py
git diff --check
git add vllm/envs.py vllm/model_executor/layers/utils.py \
  tests/fdu/test_gfx936_dispatch_contract.py
git commit -m "perf: dispatch measured gfx936 skinny GEMMs"
```

Expected: both suites pass; runtime gfx936 dispatch still falls back because the measured whitelist is empty.

## Task 5: Replace the AWQ startup path with a fail-fast BF16 wheel contract

**Files:**

- Create: `scripts/preflight_rocm.py`
- Create: `tests/fdu/test_preflight_rocm.py`
- Create: `tests/fdu/test_runtime_contract.py`
- Modify: `fdu_vllm/hooks.py`
- Modify: `launch.sh`
- Modify: `scripts/rocm_env.sh`
- Modify: `scripts/scnet_start_optimized.sh`
- Modify: `Dockerfile`

- [ ] **Step 1: Write failing runtime tests**

`tests/fdu/test_runtime_contract.py` must assert:

- `launch.sh`, `scripts/rocm_env.sh`, and `Dockerfile` contain none of `awq`, `bitsandbytes`, `pre_quantize`, or `/tmp/awq_model` (case-insensitive);
- neither `HSA_OVERRIDE_GFX_VERSION` nor `ROCBLAS_LAYER` is assigned/exported; both names may occur only in the explicit `unset` safety line in `scripts/rocm_env.sh`;
- `launch.sh` does not export `PYTHONPATH`, invokes `scripts/preflight_rocm.py`, uses `--dtype bfloat16`, and does not set evaluator-owned `--max-num-seqs` or `--max-num-batched-tokens`;
- `scripts/rocm_env.sh` defaults `FDU_ENABLE=0`, `VLLM_ROCM_USE_AITER=0`, `VLLM_ROCM_USE_SKINNY_GEMM=1`, and `FDU_FORCE_STOCK_GEMM=0`;
- `hooks.py` reads config and returns when disabled before any runtime patch import;
- `Dockerfile` builds and installs the local wheel rather than appending code to a preinstalled `vllm/__init__.py`.

`tests/fdu/test_preflight_rocm.py` imports `scripts/preflight_rocm.py` by file path and tests `validate_report` with synthetic dictionaries. Required failures are wrong architecture, vLLM outside the expected prefix, missing `_C`, missing `_rocm_C`, missing `wvSplitK`, and missing `LLMM1`.

- [ ] **Step 2: Run the tests and confirm AWQ-era failures**

```bash
python3 -m unittest \
  tests.fdu.test_runtime_contract \
  tests.fdu.test_preflight_rocm -v
```

Expected: runtime-contract failures for AWQ, architecture override, `PYTHONPATH`, and missing preflight module.

- [ ] **Step 3: Implement a JSON preflight with a pure validator**

`scripts/preflight_rocm.py` must expose:

```python
def collect_report() -> dict[str, object]: ...

def validate_report(
    report: dict[str, object],
    *,
    expected_prefix: str,
    required_arch: str | None,
    require_skinny: bool,
) -> list[str]: ...
```

`collect_report` records `sys.executable`, `sys.prefix`, `vllm.__file__`, module paths for `vllm._C` and `vllm._rocm_C`, `torch.version.hip`, native `gcnArchName`, and booleans for `torch.ops._rocm_C.wvSplitK` and `LLMM1`. It catches import failures into report fields but does not suppress validation errors.

The CLI prints one JSON object with `errors`, exits `0` only for an empty error list, and exits `2` otherwise. Supported arguments are:

```text
--expected-prefix PATH
--require-arch gfx936
--require-skinny
```

Canonicalize `gfx936:sramecc+:xnack-` to `gfx936` before comparing. When `require_skinny` is false, `_rocm_C` and its ops are reported but not required; `_C` and the prefix check are always required.

- [ ] **Step 4: Disable all plugin behavior before patch imports**

At the start of `fdu_vllm/hooks.py::activate`, call `get_config()`, return immediately when `cfg.enable` is false, and only then load optional plugin code. Remove the `quant_force` import/call entirely from `activate`; leave the historical module unreferenced.

- [ ] **Step 5: Replace `scripts/rocm_env.sh` with native conservative defaults**

The active defaults must be equivalent to:

```bash
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-1}"
export FDU_FORCE_STOCK_GEMM="${FDU_FORCE_STOCK_GEMM:-0}"
export FDU_ENABLE="${FDU_ENABLE:-0}"
```

Keep cache directories under `${FDU_CACHE_ROOT:-/public/home/xdzs2026_c415/cache}`. Explicitly `unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER` so inherited values cannot impersonate `gfx942` or enable profiling.

- [ ] **Step 6: Replace `launch.sh` with the canonical BF16 wheel launch**

The launch sequence is:

1. resolve `PYTHON_BIN`, `MODEL_PATH`, and `PORT`;
2. source `scripts/rocm_env.sh`;
3. `unset PYTHONPATH` and `cd /tmp`;
4. run preflight using the selected interpreter and `sys.prefix`;
5. require `gfx936` plus skinny ops only when skinny is enabled and stock fallback is off;
6. exec `"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server` with BF16 model args and append `"$@"` last.

Use this fixed argument set; do not add evaluator-owned batch or sampling flags:

```bash
VLLM_ARGS=(
  --model "$MODEL_PATH"
  --port "$PORT"
  --tensor-parallel-size 1
  --max-model-len 32768
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.94}"
  --dtype bfloat16
  --trust-remote-code
  --served-model-name Qwen3.5-27B
  --load-format auto
  --no-enable-log-requests
)
```

The preflight must complete before the first model weight is read.

- [ ] **Step 7: Make the SCNet helper select an installed environment**

Remove `export PYTHONPATH="$PROJ/src"` from `scripts/scnet_start_optimized.sh`. Set `PYTHON_BIN` from `VLLM_ENV`, defaulting to `/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python`, verify it is executable, and delegate to `launch.sh` from `/tmp`.

- [ ] **Step 8: Replace the Dockerfile patching path with a real local wheel build**

Use `COPY . /workspace`, set `PYTORCH_ROCM_ARCH=gfx936`, build `python setup.py bdist_wheel`, install the resulting wheel with `--no-deps --force-reinstall`, and run `scripts/preflight_rocm.py` at container runtime rather than Docker build time. Remove the inline Python patch, bitsandbytes install, AWQ comments, AITER defaults, architecture override, and rocBLAS profiling variable. Keep `FDU_ENABLE=0`, skinny enabled, and the emergency flag off by default.

- [ ] **Step 9: Run local tests and commit**

```bash
python3 -m unittest \
  tests.fdu.test_preflight_rocm \
  tests.fdu.test_runtime_contract -v
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_start_optimized.sh
python3 -m compileall -q scripts/preflight_rocm.py fdu_vllm/hooks.py
git diff --check
git add Dockerfile launch.sh scripts/rocm_env.sh scripts/scnet_start_optimized.sh \
  scripts/preflight_rocm.py fdu_vllm/hooks.py \
  tests/fdu/test_preflight_rocm.py tests/fdu/test_runtime_contract.py
git commit -m "fix: restore compliant BF16 runtime contract"
```

## Task 6: Build the deterministic gfx936 microbenchmark and whitelist generator

**Files:**

- Create: `scripts/bench_gfx936_skinny.py`
- Create: `tests/fdu/test_skinny_benchmark_math.py`
- Modify when SCNet gate passes: `vllm/model_executor/layers/rocm_skinny_shapes.py`

- [ ] **Step 1: Write failing pure tests for shape derivation, metrics, and source generation**

Test these public pure functions without importing torch:

```python
def derive_qwen35_shapes(config: dict[str, object]) -> list[dict[str, int | str]]: ...
def cosine_similarity(reference: list[float], actual: list[float]) -> float: ...
def relative_l2(reference: list[float], actual: list[float]) -> float: ...
def percentile(values: list[float], q: float) -> float: ...
def project_linear_speedup(rows: list[dict[str, object]]) -> dict[int, float]: ...
def render_whitelist_module(rows: list[dict[str, object]]) -> str: ...
```

For the default Qwen3.5 text config, assert the six `(M, K)` families are exactly:

```python
{
    (12288, 4096),
    (64, 4096),
    (10240, 4096),
    (4096, 4096),
    (24576, 4096),
    (4096, 12288),
}
```

Derive them from config fields, not constants:

- GDN QKVZ: `2 * linear_key_head_dim * linear_num_key_heads + 2 * linear_value_head_dim * linear_num_value_heads`;
- GDN BA: `2 * linear_num_value_heads`;
- full attention: `num_attention_heads * head_dim * (1 + attn_output_gate) + 2 * num_key_value_heads * head_dim`;
- output: `hidden_size` by `hidden_size`;
- MLP gate/up: `2 * intermediate_size` by `hidden_size`;
- MLP down: `hidden_size` by `intermediate_size`.

Test that `render_whitelist_module` includes only rows with all correctness gates true, median speedup `>= 1.15`, and candidate P99 `<=` stock P99, sorted by `(dtype, n, m, k, bias)`. A no-pass result must render an empty `frozenset()`.

- [ ] **Step 2: Run the test and confirm the script is missing**

```bash
python3 -m unittest tests.fdu.test_skinny_benchmark_math -v
```

Expected: file/import failure for `scripts/bench_gfx936_skinny.py`.

- [ ] **Step 3: Implement the pure half of the benchmark first**

Keep torch imports inside `main()` or a `run_gpu_benchmark()` function so local tests can load the module. `percentile` uses sorted linear interpolation. `project_linear_speedup` weights GDN rows by the number of `linear_attention` layers and full-attention rows by the number of `full_attention` layers from the checkpoint config; output and MLP rows are weighted by all layers. Report one projected speedup for each `n` in `{1, 2, 4}`.

- [ ] **Step 4: Implement direct GPU measurement**

The CLI must support:

```text
--model-config /public/home/xdzs2026_c415/Qwen3.5-27B/config.json
--output /public/home/xdzs2026_c415/results/gfx936_skinny/microbench.json
--dtype bfloat16
--warmup 100
--iterations 500
--repeats 5
--write-whitelist PATH
```

For every derived `(M, K)` and `N` in `1,2,4`:

1. seed with `torch.manual_seed(20260714)`;
2. allocate contiguous BF16 `x[N,K]` and `weight[M,K]` on `cuda`;
3. compare `torch.nn.functional.linear(x, weight)` with `ops.wvSplitK(weight, x, num_compute_units(), None)`;
4. reject non-finite output;
5. compute cosine and relative L2 in float32;
6. call `torch.testing.assert_close(..., rtol=0.03, atol=0.5)` and record pass/failure text;
7. warm each callable 100 times;
8. use paired `torch.cuda.Event(enable_timing=True)` objects around each of 500 calls, synchronize once after recording, and retain every elapsed time;
9. repeat five times, alternating stock-first and candidate-first;
10. store median and P99 latency, effective weight bandwidth, and speedup in JSON.

A row that fails correctness or performance is recorded with its rejection reason and excluded from the generated whitelist. The process exits `2` when the remaining admitted rows cannot cover every dominant layer family, when any projected per-`N` linear speedup is below `1.6`, or when a GPU/runtime fault prevents reliable measurement. `--write-whitelist` writes the deterministic module only when those global gates pass. It never modifies the model directory.

- [ ] **Step 5: Run tests and commit the benchmark with an empty whitelist**

```bash
python3 -m unittest tests.fdu.test_skinny_benchmark_math -v
python3 -m compileall -q scripts/bench_gfx936_skinny.py
git diff --check
git add scripts/bench_gfx936_skinny.py \
  tests/fdu/test_skinny_benchmark_math.py
git commit -m "perf: add gfx936 skinny GEMM microbenchmark"
```

Do not populate `VALIDATED_GFX936_SHAPES` on macOS.

## Task 7: Add the isolated SCNet build and A/B harness

**Files:**

- Create: `scripts/scnet_ab_gfx936.sh`
- Create: `scripts/probe_gfx936.py`
- Create: `scripts/score_gfx936.py`
- Create: `tests/fdu/test_scnet_ab_contract.py`
- Create: `tests/fdu/test_probe_gfx936.py`
- Create: `tests/fdu/test_score_gfx936.py`

- [ ] **Step 1: Write failing harness contract tests**

The test must require the four persistent paths, `"$SYSTEM_PYTHON" -m venv --system-site-packages`, native `PYTORCH_ROCM_ARCH=gfx936`, PID-specific stopping, SHA-256 wheel recording, no `killall`/`pkill`, no writes under model/testdata, and modes `init`, `build-control`, `bench`, `build-candidate`, `start-control`, `start-candidate-stock`, `start-candidate`, `stop`, `probe`, `throughput`, and `accuracy`.

Run:

```bash
python3 -m unittest tests.fdu.test_scnet_ab_contract -v
```

Expected: failure because the harness does not exist.

- [ ] **Step 2: Implement deterministic environment and wheel actions**

`init` creates both venvs and result directories. `build-control` builds the current source while the committed whitelist is empty, copies the wheel to `results/gfx936_skinny/wheels/control/`, records `sha256sum`, and installs it only in `vllm_baseline`. `bench` uses the control environment to run the direct op benchmark and writes a candidate whitelist into the experiment source. `build-candidate` runs local tests again, rebuilds from the source containing the generated whitelist, copies the wheel to `wheels/candidate/`, records its hash, and installs it only in `vllm_gfx936`.

All builds use:

```bash
PYTORCH_ROCM_ARCH=gfx936 MAX_JOBS="${MAX_JOBS:-16}" \
  "$PYTHON" setup.py bdist_wheel
```

Before each build, remove only that source tree's `build/`, `dist/`, and `*.egg-info`; never remove persistent result or model data.

- [ ] **Step 3: Implement PID-safe service actions**

All three start modes use the same BF16 CLI arguments and save PID plus logs under the result directory:

- `start-control`: control venv, `FDU_FORCE_STOCK_GEMM=1`;
- `start-candidate-stock`: candidate venv, `FDU_FORCE_STOCK_GEMM=1`;
- `start-candidate`: candidate venv, `FDU_FORCE_STOCK_GEMM=0`.

Set `FDU_ENABLE=0`, `VLLM_ROCM_USE_AITER=0`, and `VLLM_ROCM_USE_SKINNY_GEMM=1` in all modes. Run preflight before backgrounding the server. After backgrounding, poll `http://127.0.0.1:${PORT}/health` every two seconds for at most 20 minutes; fail and print the last 200 log lines if health never succeeds. `stop` sends `TERM` only to the recorded PID, waits up to 30 seconds, then sends `KILL` only to that same PID if necessary.

- [ ] **Step 4: Implement a sequential deterministic smoke probe**

`scripts/probe_gfx936.py` uses only `argparse`, `json`, `urllib.request`, and `pathlib`. It sends these exact prompts to `/v1/chat/completions` with `temperature=0.0`, `seed=20260714`, `max_tokens=64`, and `stream=false`:

```python
PROMPTS = (
    "用一句话介绍复旦大学。",
    "计算 37*19，只输出整数。",
    "Return the word BLUE exactly.",
)
```

The CLI accepts `--host`, `--port`, `--model`, `--label`, and `--output`, then writes one stable JSON document containing the label, prompts, response strings, finish reasons, and usage. `probe LABEL` in the shell harness calls this helper against the currently recorded server and writes under the result directory. `tests/fdu/test_probe_gfx936.py` mocks `urllib.request.urlopen` and verifies request payloads plus stable output ordering.

- [ ] **Step 5: Implement official-eval capture and score calculation**

`throughput TIER COUNT LABEL` creates a fresh scratch copy at `results/gfx936_skinny/eval_work/LABEL/throughput-TIER`, using `rsync -a` from testdata while excluding generated `test/`, `accuracy_debug/`, and `outputs/` directories. It runs the official script only inside the scratch copy, tees the log, then copies `test/TIER_throughput/result.json` to `results/gfx936_skinny/throughput/LABEL/TIER.json`. `accuracy TASK COUNT LABEL` uses its own scratch copy and preserves all OpenCompass outputs beneath `results/gfx936_skinny/accuracy/LABEL/TASK/`. The original testdata directory is never a process working directory and is never modified.

`scripts/score_gfx936.py` accepts `--results-root`, repeatable `--control-run`, repeatable `--candidate-run`, and `--accuracy-coefficient`. It reads the three saved vLLM benchmark JSON files per run, uses the median throughput across repeats and the worst P99 across repeats, applies weights `0.2/0.5/0.3` and the official curve

```python
max_score * (0.6 + 0.4 * (1.0 - math.exp(-1.3 * relative_gain)))
```

and prints JSON containing per-tier medians, gains, worst TTFT/TPOT P99, SLA decisions, weighted raw score, coefficient, and final score. Default `max_score` is `100`. `tests/fdu/test_score_gfx936.py` uses temporary result fixtures to verify medians, tier weights, the formula, and SLA failure handling.

- [ ] **Step 6: Run local tests and commit**

```bash
python3 -m unittest \
  tests.fdu.test_scnet_ab_contract \
  tests.fdu.test_probe_gfx936 \
  tests.fdu.test_score_gfx936 -v
bash -n scripts/scnet_ab_gfx936.sh
python3 -m compileall -q scripts/probe_gfx936.py scripts/score_gfx936.py
git diff --check
git add scripts/scnet_ab_gfx936.sh scripts/probe_gfx936.py \
  scripts/score_gfx936.py tests/fdu/test_scnet_ab_contract.py \
  tests/fdu/test_probe_gfx936.py tests/fdu/test_score_gfx936.py
git commit -m "feat: add isolated SCNet gfx936 A/B workflow"
```

## Task 8: Remove stale active AWQ configuration and document the BF16 path

**Files:**

- Modify: `config.yaml`
- Modify: `docs/env_vars.md`
- Modify: `docs/SCNET_RUN.md`
- Modify: `report.md`
- Modify: `changelog.md`
- Create: `tests/fdu/test_no_active_quantization.py`

- [ ] **Step 1: Write the failing stale-configuration test**

Check active runtime/config/documentation files for AWQ/FP8 claims. Historical files such as `fdu_vllm/quant_force.py` may remain, but `launch.sh`, `Dockerfile`, `config.yaml`, `scripts/rocm_env.sh`, `docs/env_vars.md`, `docs/SCNET_RUN.md`, and the current sections of `report.md`/`changelog.md` must describe BF16 gfx936. Assert `config.yaml` defaults all FDU hooks off.

- [ ] **Step 2: Run the test and confirm stale claims fail**

```bash
python3 -m unittest tests.fdu.test_no_active_quantization -v
```

Expected: failures for current AWQ/FP8 configuration and documentation.

- [ ] **Step 3: Update configuration and runbook**

Document only active variables, including `VLLM_ROCM_USE_SKINNY_GEMM`, `FDU_FORCE_STOCK_GEMM`, `FDU_ENABLE=0`, `VLLM_ROCM_USE_AITER=0`, native architecture detection, cache paths, and preflight behavior. Add the exact SCNet sequence from Task 10 below. Mark quantization modules as inactive historical experiments rather than supported runtime paths.

- [ ] **Step 4: Update report and changelog without claiming unmeasured speedup**

Before SCNet results exist, state only the measured hardware/root-cause facts and the hard gates. After Task 11, append actual wheel hashes, admitted shapes, throughput, SLA, accuracy, and score. Do not state that 90 was achieved unless the official or reproduced calculation proves it.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest tests.fdu.test_no_active_quantization -v
git diff --check
git add config.yaml docs/env_vars.md docs/SCNET_RUN.md report.md changelog.md \
  tests/fdu/test_no_active_quantization.py
git commit -m "docs: document native gfx936 BF16 path"
```

## Task 9: Run complete local verification before SCNet

**Files:** all changed files

- [ ] **Step 1: Run all focused tests from a clean command**

```bash
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v
```

Expected: all tests pass; no torch/vLLM import is required on macOS.

- [ ] **Step 2: Run syntax and static safety checks**

```bash
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_start_optimized.sh \
  scripts/scnet_ab_gfx936.sh
python3 -m compileall -q fdu_vllm scripts \
  vllm/platforms/rocm_capabilities.py \
  vllm/model_executor/layers/rocm_skinny_policy.py \
  vllm/model_executor/layers/rocm_skinny_shapes.py
git diff --check
rg -n "/tmp/awq_model|--quantization[ =]|bitsandbytes|pre_quantize" \
  launch.sh Dockerfile scripts/rocm_env.sh config.yaml docs/env_vars.md
rg -n "export (HSA_OVERRIDE_GFX_VERSION|ROCBLAS_LAYER)=" \
  launch.sh Dockerfile scripts/rocm_env.sh
```

Expected: syntax checks pass and the final `rg` returns no matches.

- [ ] **Step 3: Inspect the exact branch delta**

```bash
git status --short --branch
git log --oneline dx_branch..HEAD
git diff --stat dx_branch...HEAD
git diff dx_branch...HEAD -- setup.py csrc/rocm/skinny_gemms.cu \
  vllm/platforms/rocm.py vllm/model_executor/layers/utils.py \
  launch.sh scripts/rocm_env.sh
```

Expected: only the approved build, dispatch, runtime, benchmark, harness, test, and documentation changes.

- [ ] **Step 4: Push the isolated branch**

```bash
git push origin codex/gfx936-bf16
```

If GitLab returns 502, retry without rewriting history. Do not substitute the submission branch until SCNet gates pass.

## Task 10: Build the empty-whitelist control and run the SCNet microbenchmark

**Files:**

- Generated on SCNet: `vllm/model_executor/layers/rocm_skinny_shapes.py`
- Generated artifacts: `/public/home/xdzs2026_c415/results/gfx936_skinny/**`

- [ ] **Step 1: Materialize the experiment source under persistent storage**

From the SCNet WebShell:

```bash
export ROOT=/public/home/xdzs2026_c415
export EXP=$ROOT/experiments/gfx936_skinny
mkdir -p "$EXP" "$ROOT/results/gfx936_skinny"
cd "$EXP"
git clone --branch codex/gfx936-bf16 --single-branch \
  https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git \
  source
cd source
git rev-parse HEAD | tee "$ROOT/results/gfx936_skinny/source_commit.txt"
```

Use the repository's configured authenticated origin URL; do not print credentials in logs.

- [ ] **Step 2: Confirm native hardware and inherited environment**

```bash
unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER PYTHONPATH
rocminfo | sed -n '/Name:.*gfx/,+3p' | head
python3 - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(torch.__version__, torch.version.hip, p.name, p.gcnArchName, p.multi_processor_count)
PY
```

Expected: `gfx936:sramecc+:xnack-`, HIP 6.3, 80 compute units. Stop if the native architecture differs.

- [ ] **Step 3: Build and install the empty-whitelist control wheel**

```bash
bash scripts/scnet_ab_gfx936.sh init
bash scripts/scnet_ab_gfx936.sh build-control
```

Expected: wheel build succeeds; archive contains `vllm/_C*.so`, `vllm/_rocm_C*.so`, and `fdu_vllm/`; preflight reports the venv path, native gfx936, `wvSplitK=true`, and `LLMM1=true`.

- [ ] **Step 4: Run direct correctness/performance gate and generate the whitelist**

```bash
bash scripts/scnet_ab_gfx936.sh bench
```

Expected: JSON records 18 BF16 rows; every admitted row passes all correctness gates, has speedup `>=1.15`, and has no P99 regression; rejected rows retain an explicit reason; every dominant layer family remains covered; and projected speedup for `N=1,2,4` is each `>=1.6`. The command exits `2` and does not write a whitelist if the global gate fails.

- [ ] **Step 5: Apply the hard decision**

If the command exits `2`, preserve the JSON/logs and stop before loading Qwen3.5-27B. Set `FDU_FORCE_STOCK_GEMM=1` for any subsequent diagnostic run.

If it exits `0`, inspect and commit the generated whitelist:

```bash
git diff -- vllm/model_executor/layers/rocm_skinny_shapes.py
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v
git add vllm/model_executor/layers/rocm_skinny_shapes.py
git commit -m "perf: admit measured gfx936 skinny shapes"
git push origin codex/gfx936-bf16
bash scripts/scnet_ab_gfx936.sh build-candidate
```

Expected: candidate preflight passes and the candidate wheel hash is recorded separately from the empty-whitelist control.

## Task 11: Run model smoke, ordered throughput A/B, accuracy, and score gates

**Files:**

- Generated artifacts: `/public/home/xdzs2026_c415/results/gfx936_skinny/**`
- Modify after measured results: `report.md`
- Modify after measured results: `changelog.md`

- [ ] **Step 1: Run same-wheel stock control smoke**

Run the two modes sequentially on the single DCU:

```bash
bash scripts/scnet_ab_gfx936.sh start-candidate-stock
bash scripts/scnet_ab_gfx936.sh probe candidate-stock
bash scripts/scnet_ab_gfx936.sh stop

bash scripts/scnet_ab_gfx936.sh start-candidate
bash scripts/scnet_ab_gfx936.sh probe candidate
bash scripts/scnet_ab_gfx936.sh stop

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

Reject on import error, missing symbol, illegal instruction, GPU fault, OOM, graph failure, timeout, or response mismatch.

- [ ] **Step 2: Run first-pass throughput in score-priority order**

For each server mode, use fresh service starts and identical test inputs. Run:

```bash
bash scripts/scnet_ab_gfx936.sh start-control
bash scripts/scnet_ab_gfx936.sh throughput 8-16K 10 control-r1
bash scripts/scnet_ab_gfx936.sh throughput 16-32K 10 control-r1
bash scripts/scnet_ab_gfx936.sh throughput 4-8K 10 control-r1
bash scripts/scnet_ab_gfx936.sh stop

bash scripts/scnet_ab_gfx936.sh start-candidate
bash scripts/scnet_ab_gfx936.sh throughput 8-16K 10 candidate-r1
bash scripts/scnet_ab_gfx936.sh throughput 16-32K 10 candidate-r1
bash scripts/scnet_ab_gfx936.sh throughput 4-8K 10 candidate-r1
bash scripts/scnet_ab_gfx936.sh stop
```

- [ ] **Step 3: Enforce the throughput/SLA continuation gate**

Continue only when the candidate improves reproduced 8–16K throughput by at least 50%, no tier regresses, all TTFT P99 values remain within `1.5x` baseline, global TPOT P99 remains within `1.5x` baseline, and no request fails. Otherwise set `FDU_FORCE_STOCK_GEMM=1`, preserve artifacts, and reject the candidate.

- [ ] **Step 4: Repeat all three tiers**

Repeat Step 2 exactly, changing only run labels to `control-r2` and `candidate-r2`. Use the median throughput across the two runs for score projection; use the worse P99 for SLA decisions.

- [ ] **Step 5: Run all four accuracy tasks against control and candidate**

Run exactly:

```bash
for mode in control candidate; do
  bash scripts/scnet_ab_gfx936.sh "start-${mode}"
  for task in hotpotqa gov_report retrieval_multi_point aggregation_keyword_aggregation; do
    bash scripts/scnet_ab_gfx936.sh accuracy "$task" 10 "${mode}-accuracy"
  done
  bash scripts/scnet_ab_gfx936.sh stop
done
```

Any candidate task delta above 1% rejects the affected whitelist shape set or the entire candidate; do not accept an accuracy coefficient below 1.00 on this BF16 line as normal behavior.

- [ ] **Step 6: Calculate the projected score**

Run:

```bash
/public/home/xdzs2026_c415/venvs/vllm_gfx936/bin/python \
  scripts/score_gfx936.py \
  --results-root /public/home/xdzs2026_c415/results/gfx936_skinny \
  --control-run control-r1 \
  --control-run control-r2 \
  --candidate-run candidate-r1 \
  --candidate-run candidate-r2 \
  --accuracy-coefficient 1.0 \
  > /public/home/xdzs2026_c415/results/gfx936_skinny/score.json
```

Record both raw and final score. A projected 90 is a stretch success condition, not a reason to ignore an SLA or accuracy failure.

- [ ] **Step 7: Record measured evidence and commit it**

Update `report.md` and `changelog.md` with:

- native architecture and runtime versions;
- control/candidate commit and wheel hashes;
- exact admitted whitelist;
- direct-kernel correctness, median, P99, and projected linear speedup;
- two-run throughput medians by tier;
- TTFT/TPOT P99 and request failures;
- four accuracy deltas;
- raw and final projected score;
- accept/reject decision and rollback command.

Run:

```bash
git add report.md changelog.md
git commit -m "docs: record gfx936 BF16 benchmark results"
git push origin codex/gfx936-bf16
```

## Task 12: Submission integration only after every gate passes

**Files:** branch integration only

- [ ] **Step 1: Re-run the complete candidate gate after the final result commit**

Rebuild the wheel from the exact final commit, verify its hash, rerun preflight, the 8–16K throughput smoke, and all four 10-sample accuracy tasks. Do not reuse an older wheel after source changes.

- [ ] **Step 2: Review the final branch**

Use the required `superpowers:requesting-code-review` and `superpowers:verification-before-completion` skills. Resolve only evidence-backed findings, then rerun affected tests.

- [ ] **Step 3: Integrate without rewriting submission history**

Merge `codex/gfx936-bf16` into the platform-bound `lutinayi_branch`, run `scripts/prepare_submit.sh`, and push `lutinayi_branch` to `origin`. Confirm the platform branch binding before spending a submission attempt.

- [ ] **Step 4: Keep rollback deployable**

Record that `FDU_FORCE_STOCK_GEMM=1` restores stock linear dispatch without rebuilding. If the platform environment differs from the measured SCNet contract, fail preflight or use the stock flag; never fall back to architecture impersonation or quantized-model generation.
