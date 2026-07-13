# gfx936 BF16 Native Decode Kernel Design

- **Status:** Approved in chat on 2026-07-14
- **Target:** Qwen3.5-27B, vLLM 0.18.1, one BW DCU, ROCm/HIP 6.3
- **Primary objective:** maximize final competition score, with 90 as a stretch target
- **Accuracy budget:** keep every task within 1% of the BF16 baseline on this main line

## 1. Decision summary

Replace the current AWQ/FP8 experimentation path with a compliant BF16 path that makes the existing ROCm skinny-GEMM kernels buildable, importable, and selectively usable on the actual `gfx936` device.

The implementation will:

1. build and package `vllm._rocm_C` in the submitted wheel;
2. add narrowly scoped `gfx936` support for BF16/FP16 skinny GEMM without globally pretending that the device is `gfx942` or MI300;
3. benchmark and whitelist only matrix shapes that are both numerically correct and measurably faster;
4. run from one installed wheel with fail-fast startup checks, avoiding source-tree shadowing;
5. gate every optimization with SCNet throughput, TTFT, TPOT, stability, and accuracy measurements.

Persistent weight quantization, model conversion, speculative decoding, scheduler changes, and output-semantic changes are excluded.

## 2. Evidence and root cause

### 2.1 SCNet facts measured on 2026-07-13

The SCNet WebShell reported:

- device name: `BW`;
- native architecture: `gfx936:sramecc+:xnack-`;
- compute units: 80;
- PyTorch: 2.10.0;
- HIP runtime: 6.3.26093;
- `HSA_OVERRIDE_GFX_VERSION` was unset during inspection;
- vLLM was not installed in the fresh container;
- `vllm._rocm_C`, `wvSplitK`, `LLMM1`, and ROCm paged-attention ops were therefore unavailable;
- `/public/home/xdzs2026_c415/vllm_cscc/setup.py` had the `_rocm_C` extension declaration commented out.

All persistent SCNet work must stay under `/public/home/xdzs2026_c415`. The model and official test data must not be edited.

### 2.2 Build gap

`CMakeLists.txt` defines `_rocm_C`, but `setup.py` does not add `vllm._rocm_C` to `ext_modules`. A normal `python setup.py bdist_wheel` therefore does not request or install that CMake component. Setting `VLLM_ROCM_USE_SKINNY_GEMM=1` cannot compensate for a missing shared library.

### 2.3 Dispatch and compilation gaps

There are two independent `gfx936` exclusions:

- `csrc/rocm/skinny_gemms.cu` defines its GFX9 implementation only for `gfx90a`, `gfx942`, and `gfx950`. Other targets compile unreachable stubs for the `wvSplitK` kernels.
- `vllm/platforms/rocm.py` and `vllm/model_executor/layers/utils.py` route skinny GEMM through the existing `on_gfx9()` predicate, which does not include `gfx936`.

Changing only the build or only the Python dispatch cannot activate the fast path. Both layers must agree.

### 2.4 Invalid architecture override

The repository currently exports `HSA_OVERRIDE_GFX_VERSION=9.4.2`, which makes a `gfx936` device advertise itself as `gfx942`. This can select incompatible code objects and unrelated MI300/AITER paths. The design removes this override and compiles for the native architecture.

### 2.5 Why accuracy sacrifice is not the primary lever

With the competition score curve, a uniform final throughput score of 90 requires about 2.07 times baseline throughput when the accuracy coefficient is 1.00. With coefficients 0.97 and 0.94, the required throughput rises to about 2.32 and 2.72 times baseline. Accuracy loss therefore makes a 90-point final score harder unless the low-precision speedup is exceptionally large.

This BF16 design targets coefficient 1.00. A later low-precision design would require a separate specification and rule review.

## 3. Goals and non-goals

### 3.1 Goals

- Produce one self-contained vLLM wheel that includes `_C`, `_rocm_C`, and `fdu_vllm` as required.
- Use the native `gfx936` target with no architecture impersonation.
- Reduce decode GEMV/GEMM time for Qwen3.5 batch sizes 1 through 4.
- Preserve BF16 model weights, inference semantics, sampling behavior, and scheduler behavior.
- Provide deterministic preflight checks and an immediate stock-GEMM rollback switch.
- Obtain reproducible SCNet A/B evidence before modifying the submission launch path.
- Keep final per-task accuracy deltas at or below 1%.

### 3.2 Non-goals

- Persistent AWQ, GPTQ, FP8, INT4, or other converted weight files.
- In-memory service-initialization weight conversion or reusable quantized weight caches.
- Speculative decoding, layer skipping, pruning, token pruning, or answer caching.
- Scheduler, `max-num-seqs`, `max-num-batched-tokens`, temperature, or evaluator-owned parameter changes.
- Broadly enabling every `on_gfx9()` or MI300 optimization on `gfx936`.
- Replacing the Gated Delta Net, attention backend, or KV-cache format in the first implementation cycle.

## 4. Architecture

### 4.1 Canonical build and import contract

`setup.py` will explicitly add `CMakeExtension(name="vllm._rocm_C")` for HIP builds. The wheel is the only runtime source of the `vllm` package.

The launch path must:

- start from a directory such as `/tmp`, not the repository root;
- avoid prepending the repository root to `PYTHONPATH`;
- invoke the Python interpreter belonging to the selected baseline or candidate environment;
- verify that `vllm.__file__` resolves inside that environment;
- import `vllm._C` and `vllm._rocm_C` before loading the model;
- verify the presence of `torch.ops._rocm_C.wvSplitK` and `LLMM1`;
- fail before model loading if the candidate configuration requests skinny GEMM but the extension or ops are missing.

Silent plugin or extension failures are not acceptable for the candidate. The baseline environment remains independently runnable.

### 4.2 Narrow gfx936 capability boundary

The implementation will not add `gfx936` to the global MI300 predicate and will not use `HSA_OVERRIDE_GFX_VERSION`.

Instead it will introduce a narrowly named capability such as `supports_rocm_skinny_gemm()` whose supported set is limited to architectures validated for these BF16/FP16 kernels, including `gfx936` after the direct microbenchmark passes. `rocm_unquantized_gemm_impl` will use this capability instead of treating `gfx936` as a generic MI300.

In `skinny_gemms.cu`, `__gfx936__` will be added only to the macro branch that compiles BF16/FP16 GFX9 skinny kernels. It will not be added to the MI3XX/FP8 branch unless a separate experiment proves instruction and numerical compatibility.

The ROCm attention source remains unchanged in this cycle. This isolates the experiment to linear layers.

### 4.3 Shape-aware dispatch

The candidate dispatch accepts a call only when all of these conditions hold:

- the device is a validated skinny-GEMM target;
- the feature flag is enabled and the emergency stock flag is disabled;
- activation and weight dtypes are BF16 or FP16 and match;
- the weight is contiguous;
- the activation is reshapeable to two dimensions;
- decode batch size is in the kernel-supported range 1 through 4;
- `K` is divisible by 8;
- the exact `(N, M, K, dtype, bias)` family passed the SCNet correctness and performance gates.

Every other call falls back to `torch.nn.functional.linear`. There is no attempt to catch asynchronous GPU faults and continue; unsupported cases are excluded before launch.

The candidate logs its selected architecture, extension path, and enabled shape families once at startup. Per-token logging is prohibited.

### 4.4 Representative Qwen3.5 shapes

The microbenchmark covers at least these BF16 matrix families for `N` in `{1, 2, 4}`:

| Layer family | M | K |
|---|---:|---:|
| GDN input QKVZ | 12,288 | 4,096 |
| GDN input BA | 64 | 4,096 |
| Full-attention QKV/gate | 10,240 | 4,096 |
| GDN/attention output | 4,096 | 4,096 |
| MLP gate/up | 24,576 | 4,096 |
| MLP down | 4,096 | 12,288 |

If the actual model configuration differs, the benchmark reads its dimensions from `/public/home/xdzs2026_c415/Qwen3.5-27B/config.json` and adds the resulting shapes before any model-level test. Hard-coded defaults do not override the checkpoint configuration.

### 4.5 Runtime configuration

The first candidate uses:

- BF16 weights and activations;
- native architecture detection;
- skinny GEMM enabled only through the validated dispatch;
- `ROCBLAS_LAYER` unset, because it is a profiling/logging control rather than an autotuner;
- AITER, custom attention, KV FP8, custom HIP Graph, and all quantization hooks disabled;
- no AWQ preprocessing and no `/tmp/awq_model`;
- platform-owned CLI parameters left untouched.

After the BF16 candidate passes, AITER FlashAttention and GDN prefill tuning may be tested one at a time as a second phase. They are not bundled into the first A/B result.

## 5. SCNet isolation and data flow

### 5.1 Persistent layout

Use these isolated locations:

- experiment source/build: `/public/home/xdzs2026_c415/experiments/gfx936_skinny`;
- baseline environment: `/public/home/xdzs2026_c415/venvs/vllm_baseline`;
- candidate environment: `/public/home/xdzs2026_c415/venvs/vllm_gfx936`;
- benchmark artifacts: `/public/home/xdzs2026_c415/results/gfx936_skinny`.

Both environments use `--system-site-packages` so they share the competition PyTorch/ROCm stack but install separate vLLM wheels. The system Python installation is not overwritten during A/B testing.

The following directories are read-only inputs to this work:

- `/public/home/xdzs2026_c415/Qwen3.5-27B`;
- `/public/home/xdzs2026_c415/testdata`.

### 5.2 Data flow

1. Build the baseline and candidate wheels.
2. Record wheel hashes and inspect their contents.
3. Install each wheel into its dedicated environment.
4. Run preflight import and op-registration checks.
5. Run direct linear microbenchmarks without loading the 27B model.
6. Generate a shape whitelist from passing benchmark rows.
7. Run candidate model smoke tests.
8. Run baseline and candidate throughput tests with identical test inputs and environment controls.
9. Run the four accuracy tasks.
10. Calculate predicted competition score from measured per-tier throughput and accuracy coefficients.

Benchmark output is JSON plus a concise text summary so results can be compared mechanically and audited later.

## 6. Correctness, performance, and score gates

### 6.1 Direct kernel correctness

For every candidate shape and each `N` value:

- generate deterministic BF16 inputs with a fixed seed;
- compare `wvSplitK` against the stock PyTorch linear result;
- reject NaN or infinity;
- require cosine similarity of at least 0.999;
- require relative L2 error no greater than 0.01;
- require `torch.testing.assert_close` with `rtol=0.03` and `atol=0.5`.

These tolerances only admit a shape to model testing. The official accuracy tasks remain the final correctness authority.

### 6.2 Direct kernel performance

Each result uses:

- 100 warm-up iterations;
- 500 measured iterations;
- HIP events with explicit synchronization;
- five independent repeats;
- median latency and effective weight bandwidth;
- P99 latency from the collected iterations.

A shape is whitelisted only if its median speedup is at least 1.15 times stock and its P99 is not slower than stock. The weighted set of dominant decode shapes must project to at least 1.6 times linear-layer speedup before spending time on a full model load.

### 6.3 Model smoke gate

Before throughput testing, the candidate must:

- pass preflight;
- load Qwen3.5-27B without illegal-instruction, missing-symbol, OOM, or graph-capture failures;
- become healthy within the normal startup budget;
- answer fixed deterministic prompts;
- match baseline output tokens on the token-consistency smoke set.

### 6.4 Throughput and SLA gate

Run baseline and candidate under the same fresh-container conditions.

The progression is:

1. `8-16K`, 10 samples;
2. `16-32K`, 10 samples;
3. `4-8K`, 10 samples;
4. repeat all three tiers when the first pass succeeds.

Continue to final accuracy testing only when:

- the 8-16K candidate improves by at least 50% over the locally reproduced baseline;
- no tier regresses in throughput;
- every tier remains inside the official TTFT P99 limit;
- global TPOT P99 remains inside the official limit;
- no request fails.

The final report calculates raw throughput score. A 90-point final target requires raw throughput score at least 90 with coefficient 1.00, or at least 92.79 with coefficient 0.97.

### 6.5 Accuracy gate

Run `hotpotqa`, `gov_report`, `retrieval_multi_point`, and `aggregation_keyword_aggregation` against the reproduced BF16 baseline. Because this line does not intentionally reduce precision, any task delta above 1% is treated as a kernel correctness defect and causes rollback of the affected shape or the entire kernel path.

## 7. Failure handling and rollback

- `FDU_FORCE_STOCK_GEMM=1` disables the custom dispatch without rebuilding.
- A missing `_rocm_C`, missing op, unexpected architecture, or wheel/source mismatch fails preflight before model loading.
- An unsupported shape always falls back to stock linear.
- A failing benchmark row is excluded from the whitelist and recorded with its reason.
- An illegal instruction, GPU fault, model-output mismatch, accuracy regression, or SLA failure rejects the candidate.
- Baseline and candidate environments stay separate, allowing immediate restart with the baseline interpreter.
- The existing model and official datasets are never modified, moved, or converted.

## 8. Planned repository changes

The implementation plan may modify only files serving this design, principally:

- `setup.py` — request and package `_rocm_C` for HIP builds;
- `csrc/rocm/skinny_gemms.cu` — compile the BF16/FP16 GFX9 path for `gfx936`;
- `vllm/platforms/rocm.py` — expose a targeted skinny-GEMM capability;
- `vllm/model_executor/layers/utils.py` — validated shape-aware dispatch and fallback;
- `launch.sh` — compliant BF16 launch using the installed wheel;
- `Dockerfile` — syntactically valid, matching runtime configuration;
- `scripts/preflight_rocm.py` — build/import/device/op contract checks;
- `scripts/bench_gfx936_skinny.py` — correctness and latency microbenchmark;
- `scripts/scnet_ab_gfx936.sh` — reproducible isolated A/B workflow;
- `docs/env_vars.md`, `report.md`, and `changelog.md` — configuration and measured contribution records;
- focused tests for architecture detection, dispatch eligibility, fallback, and preflight failures.

The AWQ modules may remain in history, but the launch path, activation chain, Docker image, and submission report must not activate or claim them.

## 9. Test strategy

### 9.1 Local static tests

- Python compilation and import-isolation tests with mocks;
- shell syntax checks;
- Dockerfile parser/build-context checks where available;
- wheel-content assertions;
- tests that `gfx936` enables only skinny GEMM, not MI300/FP8/AITER capabilities;
- tests that all unsupported shapes and dtypes fall back to stock linear;
- tests that preflight rejects a missing or shadowed extension.

### 9.2 SCNet tests

- native architecture detection with `HSA_OVERRIDE_GFX_VERSION` unset;
- `_C` and `_rocm_C` build/import checks;
- direct op-registration checks;
- representative-shape correctness and performance microbenchmarks;
- deterministic model smoke tests;
- per-tier throughput, TTFT, and TPOT A/B;
- four-task accuracy A/B;
- repeated candidate runs to detect variance and delayed GPU faults.

No success claim is made from macOS-only static analysis.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `gfx936` is not instruction-compatible with the existing skinny kernel | Compile natively, run direct correctness tests first, and retain stock fallback. Do not use the gfx942 override. |
| The kernel is correct but slower for some shapes | Per-shape benchmark whitelist with a 1.15x admission floor. |
| Global architecture classification enables unrelated kernels | Add a targeted skinny capability instead of changing MI300/AITER predicates. |
| Wheel builds but omits `_rocm_C` | Inspect wheel contents and import from an isolated environment before model loading. |
| Source-tree shadowing loads Python without matching shared libraries | Launch outside the repo and verify `vllm.__file__` plus extension paths. |
| Numerical reduction order changes accuracy | Direct numerical gates, token consistency, and four-task accuracy rollback at delta greater than 1%. |
| Full-model testing consumes too much urgent time | Require a projected 1.6x linear speedup at the microbenchmark gate first. |
| The 90-point target remains physically unreachable in BF16 | Preserve all benchmark evidence, keep the stable BF16 improvement, and scope a separate compliant operator-level precision design only after measuring the remaining bottleneck. |

## 11. Rejected alternatives

### 11.1 AITER/Graph-only tuning

This remains a secondary A/B phase. It cannot repair the missing extension, native-architecture mismatch, or weight-bandwidth bottleneck and is unlikely by itself to supply the approximately two-times throughput needed for 90.

### 11.2 Persistent AWQ or service-start quantization

This violates the current competition rule against persistent model conversion or reusable quantized weights. The present implementation also contains packing, completion-sentinel, dtype, and layer-coverage defects. It is removed from the active design.

### 11.3 Global gfx942 impersonation

This may force selectors to choose code compiled for a different device and can enable unsupported MI300/FP8 paths. Native `gfx936` compilation plus narrow capability checks is the selected approach.

### 11.4 Immediate dynamic FP8/KV FP8 work

This has higher correctness and compatibility risk, while any accuracy-coefficient loss raises the throughput required for a 90-point final score. It requires separate evidence and a separate approved design.

## 12. Expected outcome

The design is intended to turn an inactive ROCm optimization into a measurable native `gfx936` decode path. The working target is a stable 85–92 score range, with 90 treated as a stretch objective rather than a guarantee. The first decisive result is the representative-shape microbenchmark; model-level work proceeds only if that result justifies it.
