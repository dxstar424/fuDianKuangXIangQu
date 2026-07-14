# gfx936 Online Weight-Only Quantization JIT Design

**Date:** 2026-07-14
**Status:** Approved by the user; SCNet gate shortened by user direction
**Target:** Qwen3.5-27B, vLLM 0.18.1, one native `gfx936` DCU

## 1. Objective

Raise the platform score above the current `dx_branch` result while preserving
the accuracy coefficient and SLA gates. The selected approach is runtime-only
weight quantization plus a small, shape-specialized HIP decode kernel compiled
on the final evaluation machine.

The latest recorded platform result is:

| Metric | Value |
|---|---:|
| 4K–8K throughput | 15.03 tok/s |
| 8K–16K throughput | 12.00 tok/s |
| 16K–32K throughput | 6.09 tok/s |
| SLA deduction | 0.0 |
| Accuracy deduction | 0.0 |
| Final score | 66.8175 |

With 1,024 output tokens, these throughputs correspond to approximately
68.1, 85.3, and 168.1 seconds per request. The absence of TTFT/TPOT detail
means the design must assume that long-context requests contain both a large
prefill component and a bandwidth-bound decode component.

The goal is a platform submission with a credible path toward 90 points. This
is a stretch target, not a guaranteed outcome. W8 is the conservative
candidate; selective W4 is the higher-risk candidate with enough memory-
traffic reduction to make 90 physically plausible.

## 2. Hard constraints

1. Runtime HIP compilation must complete within a 50-second platform budget.
   The implementation uses a 45-second subprocess timeout and reserves the
   final five seconds for loading and validation.
2. Compilation happens during final platform startup, not during an offline
   SCNet build and not as part of the vendored vLLM wheel.
3. Quantization is online and non-persistent. No quantized weights, scales, or
   transformed model directory may be written outside ephemeral `/tmp` or
   included in the submission.
4. The checkpoint, layer count, model topology, scheduler-locked parameters,
   requested token counts, and sampling behavior remain unchanged.
5. Accuracy degradation must remain at or below the competition's 1% full-
   coefficient boundary. TTFT and TPOT must remain within their 1.5x SLA
   limits.
6. The implementation targets the real `gfx936` device. It must not use
   `HSA_OVERRIDE_GFX_VERSION` or enable MI300/gfx942 capability predicates.
7. Compile, load, validation, or speed-gate failure must leave a working
   inference path. The current five-shape BF16 `LLMM1` route remains the
   pre-quantization fallback.

## 3. Scope and non-goals

### In scope

- Runtime `hipcc` compilation of a small `.so` without PyTorch C++ headers.
- Online W8A16 quantization for six dominant Qwen3.5 linear shape families.
- Optional group-32 W4A16 for the two largest MLP shape families.
- A custom N=1 HIP GEMV path and a correctness-preserving BF16 prefill path.
- Per-shape numerical and performance gates, structured logging, and explicit
  fallback behavior.
- SCNet microbenchmark, short throughput/SLA probes, and two representative
  accuracy probes before a platform submission is selected; the platform run
  supplies the exhaustive final validation.

### Out of scope for this iteration

- `wvSplitK`; its existing gfx936 benchmark failed and it remains disabled.
- Repairing the historical custom FlashAttention source. Its `Br=64` kernel
  computes only `q_start` for each chunk, leaving the other query rows
  unwritten, and it was authored for gfx942 rather than gfx936.
- Persistent AWQ/GPTQ/bitsandbytes artifacts or downloading another model.
- FP8 or `torch._scaled_mm`; prior platform attempts crashed on this device
  class and no supported gfx936 kernel has been demonstrated.
- AITER, custom HIP Graph, KV FP8, scheduler changes, or unrelated environment
  tuning in the first candidate. They would confound the quantized-kernel A/B.
- A custom prefill attention or GDN implementation. That is a separate project
  if profiling later proves it necessary.

## 4. Alternatives considered

### 4.1 Extend the BF16 `LLMM1` path

This is the lowest-risk option and can fix the current `mlp_down` launch-size
problem, but measured BF16 GEMV already approaches the device's HBM bandwidth.
Changing tiling cannot remove enough bytes to support a 90-point target. It is
retained only as fallback and as the baseline for speed gates.

### 4.2 JIT-compile a prefill FlashAttention kernel

This attacks the long-tier TTFT component, but the existing source is
numerically incomplete and does not integrate correctly with vLLM's packed,
chunked, paged metadata. Rewriting and validating it under the remaining time
would put the zero-deduction accuracy/SLA result at unacceptable risk.

### 4.3 Online weight-only quantization plus HIP GEMV

This is the selected design. Decode is dominated by reading model weights for
every token. W8 approximately halves those bytes; selective W4 reduces the
largest MLP matrices to slightly above one quarter of their BF16 footprint.
A raw HIP library avoids unsupported vendor quantization paths and keeps the
compile unit small enough for the startup budget.

## 5. Runtime profiles

One environment variable selects the candidate:

| `FDU_GFX936_QUANT_MODE` | Behavior |
|---|---|
| `off` | Current BF16 behavior, including the five validated `LLMM1` shapes |
| `w8` | Per-output-channel symmetric W8A16 for all admitted shapes |
| `hybrid_w4` | Group-32 W4A16 for the two MLP shapes; W8A16 for the other four |

Unknown values are treated as `off` with a warning. The implementation will
support both candidates, but the submitted default is selected only after the
SCNet gates in section 11.

The six candidate `(M, K)` shapes are:

| Family | M | K | `w8` | `hybrid_w4` |
|---|---:|---:|---|---|
| GDN QKVZ | 16,384 | 5,120 | W8 | W8 |
| GDN BA | 96 | 5,120 | W8 | W8 |
| Full-attention QKV gate | 14,336 | 5,120 | W8 | W8 |
| Attention/GDN output | 5,120 | 6,144 | W8 | W8 |
| MLP gate/up | 34,816 | 5,120 | W8 | W4 group 32 |
| MLP down | 5,120 | 17,408 | W8 | W4 group 32 |

Eligibility also requires native gfx936, BF16 activations, no bias, contiguous
weights, `N=1` for the custom decode call, and exact shape equality. Other
linears remain BF16.

## 6. Components

### 6.1 Runtime builder

`scripts/build_gfx936_quant_jit.py` owns compilation and no inference logic.
It will:

1. verify that the requested architecture is exactly `gfx936`;
2. locate `hipcc` without downloading or installing anything;
3. hash the kernel source, compiler identity, flags, and architecture;
4. compile to a temporary path using:

   ```text
   hipcc -O3 -std=c++17 -shared -fPIC --offload-arch=gfx936
   ```

5. enforce a 45-second timeout, terminate the compiler on timeout, and rename
   the finished library atomically into `/tmp/fdu_gfx936_quant/<hash>.so`;
6. print only the successful library path to stdout and diagnostics to stderr.

The HIP source includes only ROCm/HIP runtime headers. It must not include
Torch, ATen, pybind11, or vLLM headers. This keeps compile latency predictable.

`launch.sh` runs the existing preflight first, then invokes the builder. On
success it exports `FDU_GFX936_QUANT_SO`; on any failure it sets
`FDU_GFX936_QUANT_MODE=off` and continues with the current server path.

### 6.2 HIP library

`csrc/fdu/gfx936_quant_gemv.hip` exports four C-linkage host functions:

- `fdu_gfx936_w8a16_gemv`
- `fdu_gfx936_w4a16_gemv`
- `fdu_gfx936_w8_dequant`
- `fdu_gfx936_w4_dequant`

Each wrapper receives raw device pointers, `M`, `K`, scale metadata, output,
and the current `hipStream_t`. It returns a status code and never synchronizes
the whole device.

Both GEMV kernels use:

- a fixed 256-thread block rather than making thread count proportional to K;
- dynamic LDS for the BF16 activation row, at most 34,816 bytes for K=17,408;
- four output rows per block initially, with compile-time specializations for
  the six admitted K values;
- vectorized, coalesced weight loads;
- FP32 accumulation and wave64 reduction;
- bounds checks for the final output-row tile;
- no bias path.

This removes the current `LLMM1` failure mode in which K=17,408 implies 2,176
threads in one block, above the device limit.

W8 stores signed int8 weights and one FP32 scale per output row. W4 stores two
signed 4-bit values per byte and one FP16 scale for every group of 32 K values.
W4 quantized integers are clamped to `[-7, 7]` and encoded in two's-complement
nibbles.

The two dequantization launchers fill one caller-allocated BF16 matrix on the
current stream. Prefill therefore needs one reconstructed matrix, rather than
the multiple full-size temporaries produced by cast/unpack/multiply Torch
expressions. The dequantization kernels are simple one-element-per-thread
grids and do not share the decode GEMV reduction code.

### 6.3 Python loader and custom op

`vllm/model_executor/layers/gfx936_online_quant.py` has three responsibilities:

1. load `FDU_GFX936_QUANT_SO` through `ctypes` and declare exact signatures;
2. provide pure shape/mode policy helpers and online packing functions;
3. implement the runtime body for a registered opaque Torch custom op.

The opaque custom op receives activation, packed weight, scale, quantization
kind, original M/K, and optional bias. Its fake implementation returns the
correct output shape for vLLM tracing.

For N=1, it allocates the BF16 output and calls the HIP library on
`torch.cuda.current_stream().cuda_stream`. For N>1, it allocates one temporary
BF16 weight, fills it through the matching HIP dequantization launcher, and
calls `torch.nn.functional.linear` on the same stream. If a dequantization
launcher fails after conversion, a Torch reference unpack/dequant path is the
last-resort correctness fallback. A missing library before conversion, a bias,
or a rejected shape stays on the original BF16 dispatch.

The op boundary prevents Torch tracing from entering `ctypes` code. The first
candidate does not enable CUDA/HIP graph capture around this path.

### 6.4 Weight conversion hook

The active integration point is
`UnquantizedLinearMethod.process_weights_after_loading`. For each eligible
layer it will:

1. retain the original BF16 tensor while the shape gate runs;
2. allocate the final packed tensor and scales;
3. fill the W8 candidate, or the W4 candidate in `hybrid_w4` mode, in
   output-row chunks whose original BF16 slice is at most 64 MiB, keeping
   packing temporaries bounded to three such chunks in addition to the final
   packed tensor;
4. validate and benchmark the first encountered layer for that exact shape;
5. if W4 fails, try W8 before falling back to BF16;
6. cache the decision per `(mode, M, K)` and apply it to later layers;
7. replace the BF16 parameter with the packed non-trainable parameter and
   registered scale buffer only after admission.

After model-wide post-load processing, the loader releases cached BF16
allocator blocks once so KV-cache sizing sees the reduced resident weight
footprint. No packed data is copied to the CPU or filesystem.

`UnquantizedLinearMethod.apply` dispatches layers carrying the quantization
metadata to the new custom op. Unmodified layers continue through the current
ROCm unquantized dispatch and therefore retain the five-shape `LLMM1` route.

## 7. Data flow

### Startup

```text
launch.sh
  -> native gfx936/vLLM preflight
  -> hipcc builder (45 s timeout)
  -> export ephemeral .so path or force mode=off
  -> start vLLM
  -> load all BF16 checkpoint tensors
  -> per-shape correctness/performance admission
  -> replace admitted weights with W8/W4 buffers
  -> release freed BF16 allocator cache
```

### Prefill (`N > 1`)

```text
packed weight + scale
  -> one caller-allocated BF16 matrix
  -> HIP dequantization on the current stream
  -> existing ROCm/rocBLAS linear
  -> discard temporary BF16 tensor
```

### Decode (`N = 1`)

```text
BF16 activation + packed weight + scale
  -> opaque Torch op
  -> ctypes C launcher on current stream
  -> shape-specialized HIP GEMV
  -> BF16 output
```

## 8. Admission gates

### 8.1 Compile gate

- exact native architecture: `gfx936`;
- `hipcc` exists and identifies a ROCm/HIP compiler;
- compile and link finish within 45 seconds;
- resulting `.so` loads and exposes all four required symbols.

Failure disables online quantization before model conversion begins.

### 8.2 Synthetic kernel smoke gate

Before model loading, a small GPU check uses all three real K values (5,120,
6,144, and 17,408) with a small M. It validates launch success, finite output,
and agreement with a CPU/PyTorch dequantized reference. Failure disables the
corresponding quantization kind.

### 8.3 Per-shape model-weight gate

The first real layer of every candidate shape is checked with deterministic
BF16 activation vectors:

- W8: normalized RMSE at most 0.015 and cosine similarity at least 0.999;
- W4: normalized RMSE at most 0.080 and cosine similarity at least 0.995;
- all outputs must be finite;
- candidate median latency must be at least 1.10x faster than the current
  baseline over synchronized warmups and 30 timed iterations.

For the existing five admitted shapes, the speed baseline is `LLMM1`; for
`mlp_down`, it is stock BF16 linear. These numerical thresholds are kernel
admission checks, not substitutes for the full task-level accuracy gate.

### 8.4 End-to-end gate

Candidates proceed in this speed-first order:

1. model load and a deterministic server probe;
2. server readiness within the existing platform startup watchdog;
3. three 8K–16K cases with TTFT P99 and TPOT P99, compared with the recorded
   dx_branch result rather than spending another server load on `off`;
4. three cases in each throughput tier for the selected candidate;
5. three HotpotQA and three Retrieval MultiPoint accuracy cases;
6. a directional score projection, explicitly labeled as a fast SCNet gate.

Any failed request, non-finite value, sampled accuracy delta above 1%, tier
throughput regression, or SLA ratio above 1.45 rejects that profile. These
short samples deliberately trade confidence for turnaround time; the platform
evaluation remains the final statistical and four-task accuracy test.

## 9. Error handling and rollback

| Failure | Behavior |
|---|---|
| Wrong architecture or missing `hipcc` | Log once, force `mode=off`, start BF16 server |
| Compile timeout/error | Remove partial file, force `mode=off`, start BF16 server |
| Missing symbol or load error | Force `mode=off` before weight conversion |
| W4 shape fails admission | Retry that shape as W8 |
| W8 shape fails admission | Keep that shape's BF16 parameter and current dispatch |
| Runtime HIP launcher returns error | Reconstruct BF16 weight and execute linear for that call; log once |
| End-to-end gate fails | Do not bind that mode to the platform branch |

The environment switch `FDU_GFX936_QUANT_MODE=off` is the immediate rollback.
No source rebuild is required to return to the 66.8175 candidate behavior.

## 10. Testing strategy

### Local macOS tests

Local tests must remain importable without Torch, vLLM, ROCm, or `hipcc`. Pure
Python tests cover:

- mode parsing and exact shape policy;
- compile command construction, hash keys, timeout, atomic output, and failure
  cleanup using a fake compiler;
- signed W4 nibble packing/unpacking reference logic;
- W8/W4 scale and quantization reference calculations on small Python arrays;
- source invariants: gfx936 target, fixed block size, four required C symbols, no
  Torch/ATen includes;
- dispatch metadata and fallback decisions.

Existing `tests/fdu` must remain green. GPU-only tests are skipped explicitly
when Torch/HIP is unavailable rather than silently passing.

### SCNet tests

SCNet supplies rapid directional evidence:

- actual compile wall time and `.so` load;
- synthetic and six-shape kernel JSON output;
- W8 versus `LLMM1`/stock latency and numerical error;
- hybrid W4 versus W8 latency and numerical error;
- model-load peak/resident memory;
- three-case throughput probes with TTFT/TPOT;
- three-case HotpotQA and Retrieval MultiPoint accuracy probes.

Every result record includes commit hash, mode, compiler version, source hash,
and whether each shape selected W4, W8, or BF16.

## 11. Submission selection

Three immutable candidates are retained:

1. `off`: known BF16 rollback;
2. `w8`: accuracy-first online quantization candidate;
3. `hybrid_w4`: score-first candidate.

Selection rules:

- Submit `hybrid_w4` only if both sampled accuracy deltas are at most 1%, every
  sampled SLA ratio is below 1.45, no sampled tier regresses from `w8`, and its
  directional projection is higher than `w8`.
- Otherwise submit `w8` if it passes the same fast accuracy/SLA limits and
  improves the weighted projection over the recorded `off` result.
- Otherwise retain `off`; a failed experiment must not replace the known
  66.8175 path.

W8 is expected to have the safer accuracy profile but may not reach 90. The
hybrid profile is the only selected profile with enough theoretical weight-
traffic reduction to approach 90, and it remains contingent on measured
accuracy and end-to-end results.

## 12. Planned file changes

| File | Responsibility |
|---|---|
| `csrc/fdu/gfx936_quant_gemv.hip` | Raw W8/W4 HIP kernels and C launchers |
| `scripts/build_gfx936_quant_jit.py` | Bounded runtime compilation and cache key |
| `scripts/preflight_gfx936_quant.py` | Symbol and synthetic GPU smoke gate |
| `vllm/model_executor/layers/gfx936_online_quant.py` | Loader, policy, packing, custom-op runtime |
| `vllm/model_executor/layers/linear.py` | Post-load conversion and apply dispatch |
| `vllm/model_executor/model_loader/utils.py` | One model-wide allocator-cache release |
| `launch.sh` | Compile/fallback startup contract |
| `scripts/rocm_env.sh` | Mode defaults and documented switches |
| `tests/fdu/` | Pure policy, builder, packing, dispatch, and contract tests |
| `scripts/bench_gfx936_quant.py` | Shape-level correctness/performance evidence |
| `docs/GFX936_HANDOFF.md`, `report.md`, `changelog.md` | Measured results only after SCNet gates |

No visual companion is needed for this work: the critical decisions are
runtime contracts, tensor formats, numerical gates, and fallback semantics,
all of which are more precise in the textual specification above.
