# gfx936 Online Quantization JIT Implementation Plan

> **Status (2026-07-14):** Tasks 1–7 have been implemented and merged. This
> file is retained as implementation history; do not copy its experiment-branch
> commands. The authoritative, failure-safe SCNet procedure is
> `docs/SCNET_RUN.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runtime-compiled gfx936 W8/W4 weight-only linear path that preserves BF16 prefill correctness, accelerates N=1 decode, and safely falls back to the current 66.8175-point BF16 candidate.

**Architecture:** `launch.sh` compiles one Torch-header-free HIP source into an ephemeral `/tmp` shared library under a 45-second timeout. vLLM converts only six exact Qwen3.5 linear shapes after checkpoint loading, invokes opaque Torch custom ops for quantized layers, uses shape-specialized HIP GEMV for N=1 decode, and reconstructs one temporary BF16 weight through a HIP dequantizer for prefill. W8 is the accuracy-first mode; `hybrid_w4` uses group-32 W4 only for the two MLP shapes and falls back shape-by-shape through W8 to the existing BF16/LLMM path.

**Tech Stack:** Python 3.10+ stdlib, PyTorch 2.10 ROCm APIs, vLLM custom-op registration, HIP C++17/`hipcc`, `ctypes`, `unittest`, bash, SCNet gfx936.

---

**Approved design:** `docs/superpowers/specs/2026-07-14-gfx936-online-quant-jit-design.md`

**Dedicated worktree:** `/private/tmp/fdu-sccscc26-gfx936`

**Baseline before Task 1:**

```bash
cd /private/tmp/fdu-sccscc26-gfx936
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
git status --short --branch
```

Expected: all existing tests pass; only the previously recorded changes in
`changelog.md`, `docs/GFX936_HANDOFF.md`, and `report.md` remain unstaged.

## File structure

| File | Responsibility |
|---|---|
| `vllm/model_executor/layers/gfx936_online_quant.py` | Torch-free policy at import time, reference formats, lazy HIP loader, packing, admission, and runtime dispatch |
| `csrc/fdu/gfx936_quant_gemv.hip` | W8/W4 decode GEMV plus W8/W4 BF16 reconstruction launchers |
| `scripts/build_gfx936_quant_jit.py` | Deterministic 45-second runtime compilation and ephemeral cache |
| `scripts/preflight_gfx936_quant.py` | Shared-library symbol validation and small real-K GPU smoke tests |
| `vllm/model_executor/layers/utils.py` | Opaque `torch.ops.vllm.gfx936_quant_linear` registration |
| `vllm/model_executor/layers/linear.py` | Post-load conversion hook and quantized apply branch |
| `vllm/model_executor/model_loader/utils.py` | One allocator-cache release after all layer conversions |
| `launch.sh`, `scripts/rocm_env.sh`, `Dockerfile` | Startup mode, builder, preflight, and fail-open BF16 contract |
| `scripts/bench_gfx936_quant.py` | Six-shape numerical/performance evidence |
| `scripts/scnet_ab_gfx936.sh` | Reproducible off/W8/hybrid server and evaluation modes |
| `tests/fdu/test_gfx936_quant_*.py` | Local policy, builder, source, integration, startup, and benchmark contracts |

### Task 1: Define the Torch-free policy and packed-format references

**Files:**
- Create: `vllm/model_executor/layers/gfx936_online_quant.py`
- Create: `tests/fdu/test_gfx936_quant_policy.py`

- [ ] **Step 1: Write the failing policy tests**

```python
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vllm/model_executor/layers/gfx936_online_quant.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "gfx936_online_quant_policy_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gfx936QuantPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quant = _load_module()

    def test_mode_parser_fails_closed(self) -> None:
        self.assertEqual(self.quant.parse_quant_mode(None), "off")
        self.assertEqual(self.quant.parse_quant_mode("W8"), "w8")
        self.assertEqual(self.quant.parse_quant_mode(" hybrid_w4 "), "hybrid_w4")
        self.assertEqual(self.quant.parse_quant_mode("unexpected"), "off")

    def test_exact_six_shapes_are_admitted(self) -> None:
        self.assertEqual(
            self.quant.QUANT_SHAPES,
            frozenset(
                {
                    (16384, 5120),
                    (96, 5120),
                    (14336, 5120),
                    (5120, 6144),
                    (34816, 5120),
                    (5120, 17408),
                }
            ),
        )
        self.assertFalse(self.quant.is_quant_shape(5120, 5120))

    def test_hybrid_tries_w4_then_w8_only_for_mlp(self) -> None:
        self.assertEqual(
            self.quant.candidate_kinds("hybrid_w4", 34816, 5120),
            ("w4", "w8"),
        )
        self.assertEqual(
            self.quant.candidate_kinds("hybrid_w4", 16384, 5120),
            ("w8",),
        )
        self.assertEqual(self.quant.candidate_kinds("off", 34816, 5120), ())

    def test_row_chunks_bound_original_bf16_bytes(self) -> None:
        chunks = list(self.quant.iter_row_chunks(34816, 5120, 64 << 20))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 34816)
        self.assertTrue(all((end - start) * 5120 * 2 <= 64 << 20 for start, end in chunks))

    def test_w8_reference_round_trip(self) -> None:
        packed, scale = self.quant.quantize_row_w8_reference([-1.0, 0.0, 1.0])
        self.assertEqual(packed, [-127, 0, 127])
        self.assertAlmostEqual(scale, 1.0 / 127.0)

    def test_w4_reference_packs_low_nibble_first(self) -> None:
        row = [-1.0, 1.0] + [0.0] * 30
        packed, scales = self.quant.quantize_group_w4_reference(row, group_size=32)
        unpacked = self.quant.unpack_group_w4_reference(packed, 32)
        self.assertEqual(unpacked[:2], [-7, 7])
        self.assertEqual(len(packed), 16)
        self.assertEqual(len(scales), 1)
        self.assertAlmostEqual(scales[0], 1.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the policy test and verify that it fails**

Run:

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_policy -v
```

Expected: FAIL because `gfx936_online_quant.py` does not exist.

- [ ] **Step 3: Add the minimal Torch-free policy implementation**

```python
from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


QuantMode = Literal["off", "w8", "hybrid_w4"]
QuantKind = Literal["w8", "w4"]

QUANT_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {
        (16384, 5120),
        (96, 5120),
        (14336, 5120),
        (5120, 6144),
        (34816, 5120),
        (5120, 17408),
    }
)
W4_MLP_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {(34816, 5120), (5120, 17408)}
)
PACK_CHUNK_BYTES = 64 << 20


def parse_quant_mode(value: str | None = None) -> QuantMode:
    normalized = (value if value is not None else os.getenv("FDU_GFX936_QUANT_MODE", "off"))
    normalized = normalized.strip().lower()
    if normalized in {"off", "w8", "hybrid_w4"}:
        return normalized  # type: ignore[return-value]
    return "off"


def is_quant_shape(m: int, k: int) -> bool:
    return (m, k) in QUANT_SHAPES


def candidate_kinds(mode: str, m: int, k: int) -> tuple[QuantKind, ...]:
    parsed = parse_quant_mode(mode)
    if parsed == "off" or not is_quant_shape(m, k):
        return ()
    if parsed == "hybrid_w4" and (m, k) in W4_MLP_SHAPES:
        return ("w4", "w8")
    return ("w8",)


def iter_row_chunks(
    rows: int, columns: int, byte_limit: int = PACK_CHUNK_BYTES
) -> Iterator[tuple[int, int]]:
    if rows <= 0 or columns <= 0 or byte_limit <= 0:
        raise ValueError("rows, columns, and byte_limit must be positive")
    rows_per_chunk = max(1, byte_limit // (columns * 2))
    for start in range(0, rows, rows_per_chunk):
        yield start, min(rows, start + rows_per_chunk)


def _symmetric_quantize(values: Sequence[float], limit: int) -> tuple[list[int], float]:
    maximum = max((abs(float(value)) for value in values), default=0.0)
    scale = maximum / limit if maximum > 0.0 else 1.0
    quantized = [max(-limit, min(limit, round(float(value) / scale))) for value in values]
    return quantized, scale


def quantize_row_w8_reference(values: Sequence[float]) -> tuple[list[int], float]:
    if not values:
        raise ValueError("W8 row must not be empty")
    return _symmetric_quantize(values, 127)


def quantize_group_w4_reference(
    values: Sequence[float], group_size: int = 32
) -> tuple[bytes, list[float]]:
    if group_size <= 0 or len(values) == 0 or len(values) % group_size != 0:
        raise ValueError("W4 row length must be a positive multiple of group_size")
    packed = bytearray()
    scales: list[float] = []
    for group_start in range(0, len(values), group_size):
        quantized, scale = _symmetric_quantize(
            values[group_start : group_start + group_size], 7
        )
        scales.append(scale)
        for index in range(0, group_size, 2):
            low = quantized[index] & 0x0F
            high = quantized[index + 1] & 0x0F
            packed.append(low | (high << 4))
    return bytes(packed), scales


def unpack_group_w4_reference(packed: bytes, value_count: int) -> list[int]:
    if value_count < 0 or value_count % 2 != 0 or len(packed) * 2 != value_count:
        raise ValueError("packed W4 length does not match value_count")
    values: list[int] = []
    for byte in packed:
        for nibble in (byte & 0x0F, byte >> 4):
            values.append(nibble - 16 if nibble >= 8 else nibble)
    return values
```

The module must have no runtime `torch` import.

- [ ] **Step 4: Run the policy and existing tests**

Run:

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_policy -v
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
```

Expected: the new policy tests and all existing tests PASS.

- [ ] **Step 5: Commit the policy boundary**

```bash
git add vllm/model_executor/layers/gfx936_online_quant.py tests/fdu/test_gfx936_quant_policy.py
git commit -m "feat: define gfx936 online quant policy"
```

### Task 2: Build a bounded, deterministic runtime HIP compiler

**Files:**
- Create: `scripts/build_gfx936_quant_jit.py`
- Create: `tests/fdu/test_gfx936_jit_builder.py`

- [ ] **Step 1: Write failing builder tests**

Create tests that load the script by path and exercise it with a fake compiler:

```python
from __future__ import annotations

import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_gfx936_quant_jit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gfx936_jit_builder_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gfx936JitBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load_module()

    def _fake_compiler(self, directory: Path, *, sleep: float = 0.0) -> Path:
        compiler = directory / "hipcc"
        compiler.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys, time\n"
            f"time.sleep({sleep!r})\n"
            "if '--version' in sys.argv:\n"
            "    print('fake hipcc 1.0')\n"
            "    raise SystemExit(0)\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "out.write_bytes(b'fake-so')\n"
        )
        compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
        return compiler

    def test_command_targets_only_gfx936_and_has_no_torch_include(self) -> None:
        command = self.builder.build_command(
            Path("/opt/rocm/bin/hipcc"), Path("kernel.hip"), Path("kernel.so"), "gfx936"
        )
        self.assertIn("--offload-arch=gfx936", command)
        self.assertIn("-shared", command)
        self.assertNotIn("torch", " ".join(command).lower())

    def test_cache_key_changes_with_source_or_compiler(self) -> None:
        first = self.builder.cache_key(b"one", "hipcc-a", "gfx936")
        second = self.builder.cache_key(b"two", "hipcc-a", "gfx936")
        third = self.builder.cache_key(b"one", "hipcc-b", "gfx936")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_compile_is_atomic_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            compiler = self._fake_compiler(directory)
            source = directory / "kernel.hip"
            source.write_text("extern \"C\" int kernel() { return 0; }\n")
            output = self.builder.compile_kernel(
                source=source,
                cache_root=directory / "cache",
                compiler=compiler,
                arch="gfx936",
                timeout_s=1.0,
            )
            self.assertEqual(output.read_bytes(), b"fake-so")
            self.assertEqual(
                output,
                self.builder.compile_kernel(source, directory / "cache", compiler, "gfx936", 1.0),
            )
            self.assertEqual(list((directory / "cache").glob("*.tmp.*")), [])

    def test_timeout_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            compiler = self._fake_compiler(directory, sleep=0.2)
            source = directory / "kernel.hip"
            source.write_text("extern \"C\" int kernel() { return 0; }\n")
            with self.assertRaises(self.builder.BuildError):
                self.builder.compile_kernel(
                    source, directory / "cache", compiler, "gfx936", 0.02
                )
            self.assertEqual(list((directory / "cache").glob("*")), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the builder tests and verify failure**

```bash
python3 -m unittest tests.fdu.test_gfx936_jit_builder -v
```

Expected: FAIL because `scripts/build_gfx936_quant_jit.py` is absent.

- [ ] **Step 3: Implement the builder with an exact 45-second default**

The script must expose the functions used above and a CLI. Use this complete
control flow:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_FLAGS = ("-O3", "-std=c++17", "-shared", "-fPIC")


class BuildError(RuntimeError):
    pass


def build_command(
    compiler: Path, source: Path, output: Path, arch: str
) -> list[str]:
    if arch != "gfx936":
        raise BuildError(f"unsupported architecture: {arch}")
    return [
        str(compiler),
        *DEFAULT_FLAGS,
        "--offload-arch=gfx936",
        "-o",
        str(output),
        str(source),
    ]


def compiler_identity(compiler: Path) -> str:
    try:
        completed = subprocess.run(
            [str(compiler), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BuildError(f"cannot query hipcc: {error}") from error
    return (completed.stdout + completed.stderr).strip()


def cache_key(source: bytes, identity: str, arch: str) -> str:
    digest = hashlib.sha256()
    digest.update(source)
    digest.update(identity.encode())
    digest.update(arch.encode())
    digest.update("\0".join(DEFAULT_FLAGS).encode())
    return digest.hexdigest()[:24]


def compile_kernel(
    source: Path,
    cache_root: Path,
    compiler: Path,
    arch: str = "gfx936",
    timeout_s: float = 45.0,
) -> Path:
    source = source.resolve(strict=True)
    compiler = compiler.resolve(strict=True)
    identity = compiler_identity(compiler)
    key = cache_key(source.read_bytes(), identity, arch)
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"gfx936_quant_{key}.so"
    if output.is_file() and output.stat().st_size > 0:
        return output
    temporary = cache_root / f".{output.name}.tmp.{os.getpid()}"
    temporary.unlink(missing_ok=True)
    command = build_command(compiler, source, temporary, arch)
    try:
        subprocess.run(
            command,
            check=True,
            timeout=timeout_s,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise BuildError("hipcc completed without producing a shared library")
        os.replace(temporary, output)
    except subprocess.TimeoutExpired as error:
        raise BuildError(f"hipcc exceeded {timeout_s:.1f}s") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuildError(f"hipcc failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return output


def find_hipcc(explicit: str | None) -> Path:
    candidates = [
        explicit,
        os.getenv("HIPCC"),
        shutil.which("hipcc"),
        "/opt/rocm/bin/hipcc",
        "/opt/dtk/bin/hipcc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    raise BuildError("hipcc was not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the gfx936 quant HIP library")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/fdu_gfx936_quant"))
    parser.add_argument("--hipcc")
    parser.add_argument("--arch", default="gfx936")
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = compile_kernel(
            args.source,
            args.cache_root,
            find_hipcc(args.hipcc),
            args.arch,
            args.timeout,
        )
    except BuildError as error:
        print(f"[gfx936-jit] {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run builder tests and the full local suite**

```bash
python3 -m unittest tests.fdu.test_gfx936_jit_builder -v
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
```

Expected: PASS; the timeout test completes in under one second.

- [ ] **Step 5: Commit the builder**

```bash
git add scripts/build_gfx936_quant_jit.py tests/fdu/test_gfx936_jit_builder.py
git commit -m "feat: add bounded gfx936 hip jit builder"
```

### Task 3: Add the raw HIP W8/W4 kernels and static contracts

**Files:**
- Create: `csrc/fdu/gfx936_quant_gemv.hip`
- Create: `tests/fdu/test_gfx936_quant_kernel_contract.py`

- [ ] **Step 1: Write failing source-contract tests**

```python
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "csrc/fdu/gfx936_quant_gemv.hip"


class Gfx936QuantKernelContractTest(unittest.TestCase):
    def test_source_is_torch_header_free(self) -> None:
        text = SOURCE.read_text()
        lowered = text.lower()
        self.assertNotIn("torch/", lowered)
        self.assertNotIn("aten/", lowered)
        self.assertNotIn("pybind", lowered)

    def test_exports_all_four_c_launchers(self) -> None:
        text = SOURCE.read_text()
        for symbol in (
            "fdu_gfx936_w8a16_gemv",
            "fdu_gfx936_w4a16_gemv",
            "fdu_gfx936_w8_dequant",
            "fdu_gfx936_w4_dequant",
        ):
            with self.subTest(symbol=symbol):
                self.assertRegex(text, rf'extern\s+"C"\s+int\s+{symbol}\s*\(')

    def test_decode_uses_fixed_threads_and_real_k_specializations(self) -> None:
        text = SOURCE.read_text()
        self.assertRegex(text, r"BLOCK_THREADS\s*=\s*256")
        self.assertRegex(text, r"ROWS_PER_BLOCK\s*=\s*4")
        for k in (5120, 6144, 17408):
            self.assertIn(f"case {k}:", text)
        self.assertNotIn("gfx942", text.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the kernel contract and verify failure**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_kernel_contract -v
```

Expected: ERROR because the HIP source is missing.

- [ ] **Step 3: Implement the complete Torch-header-free HIP source**

Use fixed 256-thread blocks, four output rows per block, FP32 reduction, and
compile-time K dispatch. The implementation must contain these complete kernel
and launcher structures; keep helper names unchanged so later ctypes code has
one stable ABI:

```cpp
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int BLOCK_THREADS = 256;
constexpr int WAVE_SIZE = 64;
constexpr int WAVES_PER_BLOCK = BLOCK_THREADS / WAVE_SIZE;
constexpr int ROWS_PER_BLOCK = 4;

__device__ __forceinline__ float bf16_to_float(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __hip_bfloat16 float_to_bf16(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ int decode_int4(uint8_t byte, int high) {
  int value = high ? (byte >> 4) : (byte & 0x0f);
  return value >= 8 ? value - 16 : value;
}

template <int K>
__global__ void w8a16_gemv_kernel(
    const int8_t* __restrict__ weight,
    const float* __restrict__ scale,
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int m) {
  extern __shared__ __hip_bfloat16 input_smem[];
  __shared__ float wave_sums[ROWS_PER_BLOCK][WAVES_PER_BLOCK];
  const int tid = threadIdx.x;
  const int wave = tid / WAVE_SIZE;
  const int lane = tid % WAVE_SIZE;
  const int row_base = blockIdx.x * ROWS_PER_BLOCK;
  for (int k = tid; k < K; k += BLOCK_THREADS) input_smem[k] = input[k];
  __syncthreads();

  float accumulators[ROWS_PER_BLOCK] = {};
#pragma unroll
  for (int row_offset = 0; row_offset < ROWS_PER_BLOCK; ++row_offset) {
    const int row = row_base + row_offset;
    if (row < m) {
      const int8_t* row_weight = weight + static_cast<int64_t>(row) * K;
      for (int k = tid; k < K; k += BLOCK_THREADS) {
        accumulators[row_offset] +=
            static_cast<float>(row_weight[k]) * bf16_to_float(input_smem[k]);
      }
    }
  }

#pragma unroll
  for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row_offset = 0; row_offset < ROWS_PER_BLOCK; ++row_offset) {
      accumulators[row_offset] += __shfl_down(accumulators[row_offset], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row_offset = 0; row_offset < ROWS_PER_BLOCK; ++row_offset) {
      wave_sums[row_offset][wave] = accumulators[row_offset];
    }
  }
  __syncthreads();
  if (wave == 0 && lane < ROWS_PER_BLOCK) {
    const int row = row_base + lane;
    if (row < m) {
      float total = 0.0f;
#pragma unroll
      for (int index = 0; index < WAVES_PER_BLOCK; ++index) {
        total += wave_sums[lane][index];
      }
      output[row] = float_to_bf16(total * scale[row]);
    }
  }
}

template <int K>
__global__ void w4a16_gemv_kernel(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ scale,
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int m) {
  extern __shared__ unsigned char smem[];
  auto* input_smem = reinterpret_cast<__hip_bfloat16*>(smem);
  auto* scale_smem = reinterpret_cast<__half*>(input_smem + K);
  __shared__ float wave_sums[ROWS_PER_BLOCK][WAVES_PER_BLOCK];
  constexpr int GROUPS = K / 32;
  constexpr int PACKED_K = K / 2;
  const int tid = threadIdx.x;
  const int wave = tid / WAVE_SIZE;
  const int lane = tid % WAVE_SIZE;
  const int row_base = blockIdx.x * ROWS_PER_BLOCK;
  for (int k = tid; k < K; k += BLOCK_THREADS) input_smem[k] = input[k];
  for (int index = tid; index < ROWS_PER_BLOCK * GROUPS; index += BLOCK_THREADS) {
    const int row_offset = index / GROUPS;
    const int group = index % GROUPS;
    const int row = row_base + row_offset;
    scale_smem[index] = row < m ? scale[static_cast<int64_t>(row) * GROUPS + group]
                                : __float2half(1.0f);
  }
  __syncthreads();

  float accumulators[ROWS_PER_BLOCK] = {};
#pragma unroll
  for (int row_offset = 0; row_offset < ROWS_PER_BLOCK; ++row_offset) {
    const int row = row_base + row_offset;
    if (row < m) {
      const uint8_t* row_weight = weight + static_cast<int64_t>(row) * PACKED_K;
      for (int k = tid; k < K; k += BLOCK_THREADS) {
        const uint8_t byte = row_weight[k >> 1];
        const int quant = decode_int4(byte, k & 1);
        const float group_scale = __half2float(scale_smem[row_offset * GROUPS + (k >> 5)]);
        accumulators[row_offset] +=
            static_cast<float>(quant) * group_scale * bf16_to_float(input_smem[k]);
      }
    }
  }

#pragma unroll
  for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row_offset = 0; row_offset < ROWS_PER_BLOCK; ++row_offset) {
      accumulators[row_offset] += __shfl_down(accumulators[row_offset], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row_offset = 0; row_offset < ROWS_PER_BLOCK; ++row_offset) {
      wave_sums[row_offset][wave] = accumulators[row_offset];
    }
  }
  __syncthreads();
  if (wave == 0 && lane < ROWS_PER_BLOCK) {
    const int row = row_base + lane;
    if (row < m) {
      float total = 0.0f;
#pragma unroll
      for (int index = 0; index < WAVES_PER_BLOCK; ++index) total += wave_sums[lane][index];
      output[row] = float_to_bf16(total);
    }
  }
}

__global__ void w8_dequant_kernel(
    const int8_t* weight, const float* scale, __hip_bfloat16* output,
    int64_t total, int k) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    const int row = static_cast<int>(index / k);
    output[index] = float_to_bf16(static_cast<float>(weight[index]) * scale[row]);
  }
}

__global__ void w4_dequant_kernel(
    const uint8_t* weight, const __half* scale, __hip_bfloat16* output,
    int64_t total, int k) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    const int row = static_cast<int>(index / k);
    const int column = static_cast<int>(index % k);
    const int groups = k / 32;
    const uint8_t byte = weight[static_cast<int64_t>(row) * (k / 2) + (column >> 1)];
    const int quant = decode_int4(byte, column & 1);
    const float group_scale = __half2float(scale[static_cast<int64_t>(row) * groups + (column >> 5)]);
    output[index] = float_to_bf16(static_cast<float>(quant) * group_scale);
  }
}

int last_error() {
  const hipError_t error = hipGetLastError();
  return error == hipSuccess ? 0 : static_cast<int>(error);
}

template <int K>
int launch_w8(const int8_t* weight, const float* scale,
              const __hip_bfloat16* input, __hip_bfloat16* output,
              int m, hipStream_t stream) {
  w8a16_gemv_kernel<K><<<(m + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
                          BLOCK_THREADS, K * sizeof(__hip_bfloat16), stream>>>(
      weight, scale, input, output, m);
  return last_error();
}

template <int K>
int launch_w4(const uint8_t* weight, const __half* scale,
              const __hip_bfloat16* input, __hip_bfloat16* output,
              int m, hipStream_t stream) {
  constexpr int shared_bytes =
      K * sizeof(__hip_bfloat16) + ROWS_PER_BLOCK * (K / 32) * sizeof(__half);
  w4a16_gemv_kernel<K><<<(m + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
                          BLOCK_THREADS, shared_bytes, stream>>>(
      weight, scale, input, output, m);
  return last_error();
}

}  // namespace

extern "C" int fdu_gfx936_w8a16_gemv(
    const void* weight, const void* scale, const void* input, void* output,
    int m, int k, hipStream_t stream) {
  if (!weight || !scale || !input || !output || m <= 0) return -1;
  switch (k) {
    case 5120: return launch_w8<5120>(static_cast<const int8_t*>(weight), static_cast<const float*>(scale), static_cast<const __hip_bfloat16*>(input), static_cast<__hip_bfloat16*>(output), m, stream);
    case 6144: return launch_w8<6144>(static_cast<const int8_t*>(weight), static_cast<const float*>(scale), static_cast<const __hip_bfloat16*>(input), static_cast<__hip_bfloat16*>(output), m, stream);
    case 17408: return launch_w8<17408>(static_cast<const int8_t*>(weight), static_cast<const float*>(scale), static_cast<const __hip_bfloat16*>(input), static_cast<__hip_bfloat16*>(output), m, stream);
    default: return -2;
  }
}

extern "C" int fdu_gfx936_w4a16_gemv(
    const void* weight, const void* scale, const void* input, void* output,
    int m, int k, hipStream_t stream) {
  if (!weight || !scale || !input || !output || m <= 0) return -1;
  switch (k) {
    case 5120: return launch_w4<5120>(static_cast<const uint8_t*>(weight), static_cast<const __half*>(scale), static_cast<const __hip_bfloat16*>(input), static_cast<__hip_bfloat16*>(output), m, stream);
    case 17408: return launch_w4<17408>(static_cast<const uint8_t*>(weight), static_cast<const __half*>(scale), static_cast<const __hip_bfloat16*>(input), static_cast<__hip_bfloat16*>(output), m, stream);
    default: return -2;
  }
}

extern "C" int fdu_gfx936_w8_dequant(
    const void* weight, const void* scale, void* output,
    int m, int k, hipStream_t stream) {
  if (!weight || !scale || !output || m <= 0 || k <= 0) return -1;
  const int64_t total = static_cast<int64_t>(m) * k;
  w8_dequant_kernel<<<static_cast<unsigned int>((total + BLOCK_THREADS - 1) / BLOCK_THREADS), BLOCK_THREADS, 0, stream>>>(
      static_cast<const int8_t*>(weight), static_cast<const float*>(scale),
      static_cast<__hip_bfloat16*>(output), total, k);
  return last_error();
}

extern "C" int fdu_gfx936_w4_dequant(
    const void* weight, const void* scale, void* output,
    int m, int k, hipStream_t stream) {
  if (!weight || !scale || !output || m <= 0 || k <= 0 || k % 32 != 0) return -1;
  const int64_t total = static_cast<int64_t>(m) * k;
  w4_dequant_kernel<<<static_cast<unsigned int>((total + BLOCK_THREADS - 1) / BLOCK_THREADS), BLOCK_THREADS, 0, stream>>>(
      static_cast<const uint8_t*>(weight), static_cast<const __half*>(scale),
      static_cast<__hip_bfloat16*>(output), total, k);
  return last_error();
}
```

- [ ] **Step 4: Run static contracts and local regression**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_kernel_contract -v
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
git diff --check
```

Expected: PASS locally without invoking `hipcc`.

- [ ] **Step 5: Commit the HIP ABI**

```bash
git add csrc/fdu/gfx936_quant_gemv.hip tests/fdu/test_gfx936_quant_kernel_contract.py
git commit -m "perf: add gfx936 w8 w4 hip kernels"
```

### Task 4: Add the lazy loader, bounded packers, and runtime dispatch

**Files:**
- Modify: `vllm/model_executor/layers/gfx936_online_quant.py`
- Create: `tests/fdu/test_gfx936_quant_runtime_contract.py`

- [ ] **Step 1: Write the failing runtime-contract tests**

```python
from __future__ import annotations

import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vllm/model_executor/layers/gfx936_online_quant.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "gfx936_online_quant_runtime_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Symbol:
    def __init__(self) -> None:
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return 0


class _Library:
    def __init__(self) -> None:
        self.fdu_gfx936_w8a16_gemv = _Symbol()
        self.fdu_gfx936_w4a16_gemv = _Symbol()
        self.fdu_gfx936_w8_dequant = _Symbol()
        self.fdu_gfx936_w4_dequant = _Symbol()


class Gfx936QuantRuntimeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = MODULE_PATH.read_text()
        cls.tree = ast.parse(cls.source)
        cls.quant = _load_module()

    def test_torch_is_not_imported_at_module_scope(self) -> None:
        imports = [
            node
            for node in self.tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(
            any(
                (isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
                or (isinstance(node, ast.ImportFrom) and node.module == "torch")
                for node in imports
            )
        )

    def test_exact_abi_symbols_are_required(self) -> None:
        self.assertEqual(
            self.quant.REQUIRED_SYMBOLS,
            (
                "fdu_gfx936_w8a16_gemv",
                "fdu_gfx936_w4a16_gemv",
                "fdu_gfx936_w8_dequant",
                "fdu_gfx936_w4_dequant",
            ),
        )

    def test_loader_binds_pointer_and_integer_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_path = Path(directory) / "kernel.so"
            library_path.write_bytes(b"fixture")
            fake = _Library()
            with mock.patch.object(self.quant.ctypes, "CDLL", return_value=fake):
                loaded = self.quant.load_kernel_library(library_path)
            self.assertIs(loaded.library, fake)
            self.assertEqual(len(fake.fdu_gfx936_w8a16_gemv.argtypes), 7)
            self.assertEqual(len(fake.fdu_gfx936_w8_dequant.argtypes), 6)

    def test_missing_library_fails_without_loading_torch(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FileNotFoundError):
                self.quant.load_kernel_library()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the new tests fail for missing runtime symbols**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_runtime_contract -v
```

Expected: FAIL on `REQUIRED_SYMBOLS` or `load_kernel_library`.

- [ ] **Step 3: Add the exact loader and call boundary**

Append the following imports and definitions. Keep every Torch import inside a
function so importing vLLM does not initialize ROCm early.

```python
import ctypes
from pathlib import Path
from typing import NamedTuple


KIND_W8 = 8
KIND_W4 = 4
REQUIRED_SYMBOLS = (
    "fdu_gfx936_w8a16_gemv",
    "fdu_gfx936_w4a16_gemv",
    "fdu_gfx936_w8_dequant",
    "fdu_gfx936_w4_dequant",
)


class LoadedKernels(NamedTuple):
    library: object
    w8_gemv: object
    w4_gemv: object
    w8_dequant: object
    w4_dequant: object


_LOADED_KERNELS: LoadedKernels | None = None


def load_kernel_library(path: str | Path | None = None) -> LoadedKernels:
    global _LOADED_KERNELS
    if _LOADED_KERNELS is not None and path is None:
        return _LOADED_KERNELS
    selected = Path(path or os.getenv("FDU_GFX936_QUANT_SO", ""))
    if not str(selected) or not selected.is_file():
        raise FileNotFoundError(f"gfx936 quant library not found: {selected}")
    library = ctypes.CDLL(str(selected))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    gemv_args = [pointer, pointer, pointer, pointer, integer, integer, pointer]
    dequant_args = [pointer, pointer, pointer, integer, integer, pointer]
    functions = [getattr(library, name) for name in REQUIRED_SYMBOLS]
    functions[0].argtypes = gemv_args
    functions[1].argtypes = gemv_args
    functions[2].argtypes = dequant_args
    functions[3].argtypes = dequant_args
    for function in functions:
        function.restype = integer
    loaded = LoadedKernels(library, *functions)
    if path is None:
        _LOADED_KERNELS = loaded
    return loaded


def _pointer(tensor: "torch.Tensor") -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_pointer() -> ctypes.c_void_p:
    import torch

    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with HIP status {status}")
```

- [ ] **Step 4: Add bounded W8 and group-32 W4 packing**

Use the original BF16 slice size, not temporary tensor size, to enforce the
64-MiB chunk contract:

```python
def pack_w8(weight: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    m, k = map(int, weight.shape)
    packed = torch.empty((m, k), dtype=torch.int8, device=weight.device)
    scales = torch.empty((m,), dtype=torch.float32, device=weight.device)
    for start, end in iter_row_chunks(m, k):
        source = weight[start:end].float()
        maximum = source.abs().amax(dim=1)
        scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        packed[start:end].copy_(
            torch.round(source / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        )
        scales[start:end].copy_(scale)
        del source, maximum, scale
    return packed.contiguous(), scales.contiguous()


def pack_w4(weight: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    m, k = map(int, weight.shape)
    if k % 32 != 0:
        raise ValueError(f"W4 requires K divisible by 32, got {k}")
    packed = torch.empty((m, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((m, k // 32), dtype=torch.float16, device=weight.device)
    for start, end in iter_row_chunks(m, k):
        source = weight[start:end].float().reshape(end - start, k // 32, 32)
        maximum = source.abs().amax(dim=2)
        scale = torch.where(maximum > 0, maximum / 7.0, torch.ones_like(maximum))
        quantized = torch.round(source / scale[:, :, None]).clamp_(-7, 7).to(torch.int8)
        pairs = quantized.reshape(end - start, k // 2, 2)
        packed[start:end].copy_(
            ((pairs[:, :, 0] & 0x0F) | ((pairs[:, :, 1] & 0x0F) << 4)).to(torch.uint8)
        )
        scales[start:end].copy_(scale.to(torch.float16))
        del source, maximum, scale, quantized, pairs
    return packed.contiguous(), scales.contiguous()


def pack_weight(
    weight: "torch.Tensor", kind: QuantKind
) -> tuple["torch.Tensor", "torch.Tensor"]:
    return pack_w4(weight) if kind == "w4" else pack_w8(weight)
```

- [ ] **Step 5: Add N=1 GEMV and N>1 dequantize-then-rocBLAS dispatch**

```python
def _dequantize_weight(
    packed: "torch.Tensor", scale: "torch.Tensor", kind: int, m: int, k: int
) -> "torch.Tensor":
    import torch

    output = torch.empty((m, k), dtype=torch.bfloat16, device=packed.device)
    kernels = load_kernel_library()
    function = kernels.w4_dequant if kind == KIND_W4 else kernels.w8_dequant
    status = function(
        _pointer(packed), _pointer(scale), _pointer(output), m, k, _stream_pointer()
    )
    _check_status(status, "w4_dequant" if kind == KIND_W4 else "w8_dequant")
    return output


def _dequantize_weight_reference(
    packed: "torch.Tensor", scale: "torch.Tensor", kind: int, m: int, k: int
) -> "torch.Tensor":
    import torch

    output = torch.empty((m, k), dtype=torch.bfloat16, device=packed.device)
    for start, end in iter_row_chunks(m, k):
        if kind == KIND_W8:
            output[start:end].copy_(
                packed[start:end].float() * scale[start:end].float()[:, None]
            )
            continue
        chunk = packed[start:end]
        low = (chunk & 0x0F).to(torch.int16)
        high = (chunk >> 4).to(torch.int16)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        values = torch.stack((low, high), dim=-1).reshape(end - start, k).float()
        expanded_scale = scale[start:end].float().repeat_interleave(32, dim=1)
        output[start:end].copy_(values * expanded_scale)
        del chunk, low, high, values, expanded_scale
    return output


def reconstruct_bf16_weight(
    packed: "torch.Tensor", scale: "torch.Tensor", kind: int, m: int, k: int
) -> "torch.Tensor":
    try:
        return _dequantize_weight(packed, scale, kind, m, k)
    except (FileNotFoundError, OSError, RuntimeError):
        return _dequantize_weight_reference(packed, scale, kind, m, k)


def run_quant_gemv(
    x: "torch.Tensor",
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    kind: int,
    m: int,
    k: int,
) -> "torch.Tensor":
    import torch

    flattened = x.reshape(-1, k).contiguous()
    if flattened.shape[0] != 1:
        raise ValueError("run_quant_gemv requires exactly one input row")
    output = torch.empty((1, m), dtype=torch.bfloat16, device=x.device)
    kernels = load_kernel_library()
    function = kernels.w4_gemv if kind == KIND_W4 else kernels.w8_gemv
    status = function(
        _pointer(packed),
        _pointer(scale),
        _pointer(flattened),
        _pointer(output),
        m,
        k,
        _stream_pointer(),
    )
    _check_status(status, "w4_gemv" if kind == KIND_W4 else "w8_gemv")
    return output.reshape(*x.shape[:-1], m)


def quant_linear_impl(
    x: "torch.Tensor",
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    kind: int,
    m: int,
    k: int,
    bias: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    import torch.nn.functional as functional

    if x.numel() // k == 1:
        try:
            output = run_quant_gemv(x, packed, scale, kind, m, k)
            return output if bias is None else output + bias
        except (FileNotFoundError, OSError, RuntimeError):
            pass
    bf16_weight = reconstruct_bf16_weight(packed, scale, kind, m, k)
    try:
        return functional.linear(x, bf16_weight, bias)
    finally:
        del bf16_weight
```

The N>1 branch must allocate only one reconstructed BF16 layer at a time and
must never cache it on the module. Add a module-level boolean guard around one
warning log when GEMV or HIP dequantization falls back; repeated decode tokens
must not flood the server log.

- [ ] **Step 6: Run contracts and commit**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_runtime_contract -v
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
git diff --check
git add vllm/model_executor/layers/gfx936_online_quant.py tests/fdu/test_gfx936_quant_runtime_contract.py
git commit -m "feat: add gfx936 quant runtime dispatch"
```

Expected: all local tests PASS without loading Torch or a DCU library.

### Task 5: Gate each shape and connect the path to vLLM

**Files:**
- Modify: `vllm/model_executor/layers/gfx936_online_quant.py`
- Modify: `vllm/model_executor/layers/utils.py`
- Modify: `vllm/model_executor/layers/linear.py`
- Modify: `vllm/model_executor/model_loader/utils.py`
- Create: `tests/fdu/test_gfx936_quant_integration_contract.py`

- [ ] **Step 1: Write failing integration contracts**

The test is static so it remains runnable on macOS:

```python
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT / "vllm/model_executor/layers/gfx936_online_quant.py"
UTILS = ROOT / "vllm/model_executor/layers/utils.py"
LINEAR = ROOT / "vllm/model_executor/layers/linear.py"
LOADER = ROOT / "vllm/model_executor/model_loader/utils.py"


class Gfx936QuantIntegrationContractTest(unittest.TestCase):
    def test_quant_module_defines_admission_thresholds(self) -> None:
        source = QUANT.read_text()
        self.assertIn("W8_NRMSE_LIMIT = 0.015", source)
        self.assertIn("W8_COSINE_LIMIT = 0.999", source)
        self.assertIn("W4_NRMSE_LIMIT = 0.08", source)
        self.assertIn("W4_COSINE_LIMIT = 0.995", source)
        self.assertIn("MIN_SPEEDUP = 1.10", source)
        ast.parse(source)

    def test_custom_op_has_real_and_fake_implementations(self) -> None:
        source = UTILS.read_text()
        self.assertIn('"gfx936_quant_linear"', source)
        self.assertIn("gfx936_quant_linear_impl", source)
        self.assertIn("gfx936_quant_linear_fake", source)

    def test_linear_converts_after_load_and_dispatches_before_bf16(self) -> None:
        source = LINEAR.read_text()
        self.assertIn("maybe_quantize_gfx936_layer(layer)", source)
        self.assertIn("is_gfx936_quantized_layer(layer)", source)
        self.assertIn("gfx936_quant_linear(", source)
        self.assertLess(
            source.index("is_gfx936_quantized_layer(layer)"),
            source.index("vllm_is_batch_invariant()"),
        )

    def test_loader_releases_allocator_cache_after_conversion_loop(self) -> None:
        source = LOADER.read_text()
        self.assertIn("online_quantization_active()", source)
        self.assertIn("torch.cuda.empty_cache()", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the contracts fail**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_integration_contract -v
```

Expected: FAIL because admission and vLLM hooks are absent.

- [ ] **Step 3: Implement deterministic numerical and speed admission**

Add these constants and one cached decision per `(M, K, kind)`:

```python
W8_NRMSE_LIMIT = 0.015
W8_COSINE_LIMIT = 0.999
W4_NRMSE_LIMIT = 0.08
W4_COSINE_LIMIT = 0.995
MIN_SPEEDUP = 1.10
WARMUP_REPETITIONS = 5
TIMED_REPETITIONS = 30


class AdmissionResult(NamedTuple):
    accepted: bool
    nrmse: float
    cosine: float
    baseline_ms: float
    candidate_ms: float
    speedup: float
    reason: str


_ADMISSION_CACHE: dict[tuple[int, int, QuantKind], AdmissionResult] = {}
_ACTIVE_LAYER_COUNT = 0


def admission_limits(kind: QuantKind) -> tuple[float, float]:
    if kind == "w4":
        return W4_NRMSE_LIMIT, W4_COSINE_LIMIT
    return W8_NRMSE_LIMIT, W8_COSINE_LIMIT


def evaluate_admission(
    kind: QuantKind,
    nrmse: float,
    cosine: float,
    baseline_ms: float,
    candidate_ms: float,
) -> AdmissionResult:
    import math

    nrmse_limit, cosine_limit = admission_limits(kind)
    speedup = baseline_ms / candidate_ms if candidate_ms > 0 else 0.0
    finite = all(math.isfinite(value) for value in (nrmse, cosine, speedup))
    accepted = (
        finite
        and nrmse <= nrmse_limit
        and cosine >= cosine_limit
        and speedup >= MIN_SPEEDUP
    )
    reason = "accepted" if accepted else (
        f"rejected:nrmse={nrmse:.6f},cosine={cosine:.6f},speedup={speedup:.3f}"
    )
    return AdmissionResult(
        accepted, nrmse, cosine, baseline_ms, candidate_ms, speedup, reason
    )
```

Implement `benchmark_candidate` with this exact measurement protocol:

1. Construct one BF16 input with `torch.linspace(-1, 1, k, device=weight.device)`.
2. Compute the numerical reference with `torch.nn.functional.linear` in FP32
   metric space.
3. Time the existing `rocm_unquantized_gemm` baseline and `run_quant_gemv`
   candidate with five warmups and 30 repetitions, synchronizing before and
   after each measured loop using `torch.cuda.Event(enable_timing=True)`.
4. Compute `NRMSE = sqrt(mean((candidate-reference)^2)) /
   max(sqrt(mean(reference^2)), 1e-12)` and FP32 cosine similarity.
5. Pass all five values to `evaluate_admission` and log the result once.

This uses the current LLMM1 path for its five whitelisted shapes and the stock
BF16 path for `(5120, 17408)`, because `rocm_unquantized_gemm` already owns that
selection.

- [ ] **Step 4: Convert layers one at a time without retaining BF16 weights**

Implement the following public boundary:

```python
def is_gfx936_runtime() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return "gfx936" in getattr(properties, "gcnArchName", "")


def is_gfx936_quantized_layer(layer: object) -> bool:
    return bool(getattr(layer, "_fdu_gfx936_quantized", False))


def online_quantization_active() -> bool:
    return _ACTIVE_LAYER_COUNT > 0
```

`maybe_quantize_gfx936_layer(layer)` must execute this ordered state machine:

1. Return unchanged unless mode is non-off, the library loads, runtime is
   gfx936, `layer.weight` is contiguous BF16 on HIP, `layer` has no bias,
   there is no sharding, and `(M, K)` is one of the six exact shapes.
2. Iterate `candidate_kinds(mode, M, K)`. Pack the current layer for that kind.
3. On the first layer of a `(M, K, kind)`, call `benchmark_candidate` and cache
   its `AdmissionResult`. For later layers reuse only the decision, but still
   pack each layer's own weight.
4. If W4 is rejected or raises, delete its tensors and try W8. If W8 is
   rejected or raises, leave the original BF16 parameter and current LLMM/stock
   dispatch untouched.
5. On acceptance remove `weight` from `layer._parameters`, register the packed
   tensor as non-persistent buffer `weight`, register `gfx936_scale`, set
   `_fdu_gfx936_quantized=True`, `_fdu_gfx936_quant_kind` to 4 or 8,
   `_fdu_gfx936_quant_m`, and `_fdu_gfx936_quant_k`; then increment the global
   active-layer count. Do not keep a reference to the BF16 parameter.
6. Catch and log every conversion failure at warning level and return the layer
   unchanged. A failed optimization must not abort model loading.

- [ ] **Step 5: Register and invoke the opaque custom op**

In `vllm/model_executor/layers/utils.py`, register the op beside the existing
ROCm unquantized op:

```python
def gfx936_quant_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kind: int,
    m: int,
    k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.model_executor.layers.gfx936_online_quant import quant_linear_impl

    return quant_linear_impl(x, weight, scale, kind, m, k, bias)


def gfx936_quant_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kind: int,
    m: int,
    k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], m))


direct_register_custom_op(
    op_name="gfx936_quant_linear",
    op_func=gfx936_quant_linear_impl,
    mutates_args=[],
    fake_impl=gfx936_quant_linear_fake,
)


def gfx936_quant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kind: int,
    m: int,
    k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.gfx936_quant_linear(x, weight, scale, kind, m, k, bias)
```

In `UnquantizedLinearMethod.process_weights_after_loading`, call
`maybe_quantize_gfx936_layer(layer)` after existing platform-specific weight
processing. In `UnquantizedLinearMethod.apply`, place the quantized-layer branch
before the current ROCm/LLMM and generic BF16 branches:

```python
if is_gfx936_quantized_layer(layer):
    return gfx936_quant_linear(
        x,
        layer.weight,
        layer.gfx936_scale,
        layer._fdu_gfx936_quant_kind,
        layer._fdu_gfx936_quant_m,
        layer._fdu_gfx936_quant_k,
        bias,
    )
```

Import the helper names at module scope from their vLLM modules; the quant
module itself remains Torch-free at import time.

- [ ] **Step 6: Release allocator cache once after model conversion**

In `vllm/model_executor/model_loader/utils.py`, immediately after the existing
`process_weights_after_loading` module loop and before attention postprocessing,
add:

```python
from vllm.model_executor.layers.gfx936_online_quant import online_quantization_active

if online_quantization_active():
    torch.cuda.empty_cache()
```

Do not call `empty_cache()` per layer.

- [ ] **Step 7: Run local contracts and commit**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_integration_contract -v
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
python3 -m py_compile \
  vllm/model_executor/layers/gfx936_online_quant.py \
  vllm/model_executor/layers/utils.py \
  vllm/model_executor/layers/linear.py \
  vllm/model_executor/model_loader/utils.py
git diff --check
git add \
  vllm/model_executor/layers/gfx936_online_quant.py \
  vllm/model_executor/layers/utils.py \
  vllm/model_executor/layers/linear.py \
  vllm/model_executor/model_loader/utils.py \
  tests/fdu/test_gfx936_quant_integration_contract.py
git commit -m "perf: route gfx936 linears through gated quant"
```

### Task 6: Make startup compile, validate, and fail open

**Files:**
- Create: `scripts/preflight_gfx936_quant.py`
- Create: `tests/fdu/test_gfx936_quant_startup_contract.py`
- Modify: `launch.sh`
- Modify: `scripts/rocm_env.sh`
- Modify: `Dockerfile`
- Modify: `tests/fdu/test_runtime_contract.py`

- [ ] **Step 1: Write failing startup contracts**

```python
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "scripts/preflight_gfx936_quant.py"


class Gfx936QuantStartupContractTest(unittest.TestCase):
    def test_preflight_imports_without_torch(self) -> None:
        spec = importlib.util.spec_from_file_location("quant_preflight", PREFLIGHT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader if spec else None)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.validate_symbols))

    def test_launch_has_bounded_builder_and_fail_open(self) -> None:
        source = (ROOT / "launch.sh").read_text()
        self.assertIn("build_gfx936_quant_jit.py", source)
        self.assertIn("--timeout 45", source)
        self.assertIn("preflight_gfx936_quant.py", source)
        self.assertIn("FDU_GFX936_QUANT_MODE=off", source)

    def test_defaults_are_safe(self) -> None:
        self.assertIn(
            'FDU_GFX936_QUANT_MODE="${FDU_GFX936_QUANT_MODE:-off}"',
            (ROOT / "scripts/rocm_env.sh").read_text(),
        )
        self.assertIn("FDU_GFX936_QUANT_MODE=off", (ROOT / "Dockerfile").read_text())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify startup contracts fail**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_startup_contract -v
```

- [ ] **Step 3: Implement symbol validation and real-K smoke tests**

`scripts/preflight_gfx936_quant.py` must import only stdlib at module scope. Its
`validate_symbols(path)` opens the `.so` with `ctypes.CDLL` and resolves the
four names in `REQUIRED_SYMBOLS`. Its CLI accepts `--library`, `--mode`, and
`--smoke`; `--smoke` lazily imports Torch and the quant module, then checks:

- W8 GEMV and dequant for K=5120, 6144, and 17408 with M=4.
- W4 GEMV and dequant for K=5120 and 17408 with M=4 only when mode is
  `hybrid_w4`.
- finite output, W8 NRMSE at most 0.015 and cosine at least 0.999, or W4 NRMSE
  at most 0.08 and cosine at least 0.995, against `torch.nn.functional.linear`.

The script exits 0 only when every requested check passes and prints one JSON
object containing `library`, `mode`, and a `checks` list. It exits nonzero on
missing symbols, HIP errors, non-finite values, or threshold failures.

- [ ] **Step 4: Run device preflight, compile, and smoke-test before `cd /tmp`**

Keep `unset PYTHONPATH`, but move the existing `preflight_rocm.py` invocation
before `cd /tmp`. After that native-gfx936 check succeeds, normalize the mode
and use this fail-open block before any vLLM import; run `cd /tmp` only after
the block:

```bash
case "${FDU_GFX936_QUANT_MODE:-off}" in
  off|w8|hybrid_w4) ;;
  *) export FDU_GFX936_QUANT_MODE=off ;;
esac

if [[ "$FDU_GFX936_QUANT_MODE" != "off" ]]; then
  if FDU_GFX936_QUANT_SO="$($PYTHON_BIN "$SCRIPT_DIR/scripts/build_gfx936_quant_jit.py" \
      --source "$SCRIPT_DIR/csrc/fdu/gfx936_quant_gemv.hip" \
      --arch gfx936 --timeout 45)" \
      && [[ -n "$FDU_GFX936_QUANT_SO" ]] \
      && "$PYTHON_BIN" "$SCRIPT_DIR/scripts/preflight_gfx936_quant.py" \
         --library "$FDU_GFX936_QUANT_SO" \
         --mode "$FDU_GFX936_QUANT_MODE" --smoke; then
    export FDU_GFX936_QUANT_SO
  else
    echo "[fdu] gfx936 quant JIT/preflight failed; keeping BF16 path" >&2
    export FDU_GFX936_QUANT_MODE=off
    unset FDU_GFX936_QUANT_SO
  fi
fi
```

Builder compilation itself must terminate by 45 seconds; the platform's
50-second compile ceiling retains five seconds of margin. Preflight time is
reported separately and is not included in the compiler timeout.

- [ ] **Step 5: Set safe defaults and extend the existing runtime contract**

Add these exact defaults:

```bash
# scripts/rocm_env.sh
export FDU_GFX936_QUANT_MODE="${FDU_GFX936_QUANT_MODE:-off}"
```

```dockerfile
ENV FDU_GFX936_QUANT_MODE=off
```

Extend `tests/fdu/test_runtime_contract.py` to assert that startup still unsets
submission-time `PYTHONPATH`, contains no forced FP8 activation, and defaults
the new path to `off`.

- [ ] **Step 6: Verify shell/Python startup and commit**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_startup_contract -v
python3 -m unittest tests.fdu.test_runtime_contract -v
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
python3 -m py_compile scripts/build_gfx936_quant_jit.py scripts/preflight_gfx936_quant.py
bash -n launch.sh scripts/rocm_env.sh
git diff --check
git add \
  scripts/preflight_gfx936_quant.py \
  tests/fdu/test_gfx936_quant_startup_contract.py \
  tests/fdu/test_runtime_contract.py \
  launch.sh scripts/rocm_env.sh Dockerfile
git commit -m "feat: gate gfx936 quant startup"
```

### Task 7: Add the six-shape benchmark and SCNet fast path

**Files:**
- Create: `scripts/bench_gfx936_quant.py`
- Create: `tests/fdu/test_gfx936_quant_benchmark.py`
- Modify: `scripts/scnet_ab_gfx936.sh`
- Modify: `tests/fdu/test_scnet_ab_gfx936_contract.py`

- [ ] **Step 1: Write the failing benchmark-policy test**

```python
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/bench_gfx936_quant.py"


class Gfx936QuantBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("quant_bench", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {SCRIPT}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_row_requires_accuracy_speed_and_finite_metrics(self) -> None:
        good = {"nrmse": 0.01, "cosine": 0.9995, "speedup": 1.11}
        self.assertTrue(self.module.row_is_admitted("w8", good))
        self.assertFalse(self.module.row_is_admitted("w8", {**good, "speedup": 1.09}))
        self.assertFalse(self.module.row_is_admitted("w8", {**good, "nrmse": float("nan")}))
        self.assertTrue(
            self.module.row_is_admitted(
                "w4", {"nrmse": 0.079, "cosine": 0.995, "speedup": 1.10}
            )
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement the benchmark report**

`scripts/bench_gfx936_quant.py` imports Torch only inside `main()`, accepts
`--mode {w8,hybrid_w4}`, `--library`, `--output`, `--warmup` (default 2), and
`--repetitions` (default 8). For each of the exact six `(M, K)` shapes it:

1. Allocates a deterministic BF16 weight and input on the DCU.
2. Packs the requested kind, runs `benchmark_candidate`, and records packing
   seconds, NRMSE, cosine, BF16 milliseconds, candidate milliseconds, speedup,
   peak allocated bytes, and admitted status.
3. For `hybrid_w4`, uses W4 only on `(34816,5120)` and `(5120,17408)`; if W4
   rejects, measures W8 and records `selected_kind="w8"` only if W8 admits.
4. Writes one JSON document with arch, ROCm/Torch versions, mode, compile path,
   all rows, and `passed=all(row["admitted"] for row in rows)`.
5. Exits nonzero if any selected shape fails, ensuring a shell pipeline cannot
   accidentally promote a partial candidate.

Keep `row_is_admitted` Torch-free and use the same thresholds exported by the
runtime module.

- [ ] **Step 3: Make SCNet commands start the actual submission entrypoint**

Update `scripts/scnet_ab_gfx936.sh` so `start_server()` invokes `launch.sh`
instead of calling `api_server` directly. Preserve existing baseline commands
and add:

```text
quant-bench-w8
quant-bench-hybrid
sync-candidate-python
start-candidate-off
start-candidate-w8
start-candidate-hybrid
probe-candidate-off
probe-candidate-w8
probe-candidate-hybrid
```

`sync-candidate-python` copies only the changed Python runtime files plus
`rocm_skinny_policy.py`, `rocm_skinny_shapes.py`, and `envs.py` from
`$SOURCE_ROOT/vllm/` into the existing candidate venv's installed `vllm/`
package, then runs `py_compile` on every copied file. This avoids rebuilding
the existing LLMM-enabled wheel for a Python-only experiment; if the venv lacks
the compiled `LLMM1` op, run the existing `build-candidate` command once.

Each start command exports exactly one `FDU_GFX936_QUANT_MODE`, passes the
existing model path/port through the launch environment, records the server PID,
and tees logs into `/tmp/fdu_gfx936_<mode>.log`. Each probe waits for health,
runs one 16-token request, and fails if the server log contains traceback,
non-finite admission metrics, or a quant mode different from the requested one.

Extend `tests/fdu/test_scnet_ab_gfx936_contract.py` to assert all nine commands,
the use of `launch.sh`, the three mode exports, and distinct log names.

- [ ] **Step 4: Run local tests and commit**

```bash
python3 -m unittest tests.fdu.test_gfx936_quant_benchmark -v
python3 -m unittest tests.fdu.test_scnet_ab_gfx936_contract -v
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
python3 -m py_compile scripts/bench_gfx936_quant.py
bash -n scripts/scnet_ab_gfx936.sh
git diff --check
git add \
  scripts/bench_gfx936_quant.py \
  scripts/scnet_ab_gfx936.sh \
  tests/fdu/test_gfx936_quant_benchmark.py \
  tests/fdu/test_scnet_ab_gfx936_contract.py
git commit -m "perf: add gfx936 quant ab workflow"
```

### Task 8: Run the speed-first SCNet gate and choose the candidate

**Files:**
- Runtime evidence: `/tmp/fdu_gfx936_quant_*.json`, `/tmp/fdu_gfx936_*.log`
- Modify after evidence: `scripts/rocm_env.sh`
- Modify after evidence: `Dockerfile`
- Modify after evidence: `docs/GFX936_HANDOFF.md`
- Modify after evidence: `report.md`
- Modify after evidence: `changelog.md`

- [ ] **Step 1: Sync the candidate and verify the 45-second compile budget**

Publish the non-submission experiment branch, then on SCNet sync its Python
overlay into the already-built LLMM candidate environment:

```bash
cd /private/tmp/fdu-sccscc26-gfx936
git push origin HEAD:codex/gfx936-bf16

cd /public/home/xdzs2026_c415/vllm_cscc
git fetch origin codex/gfx936-bf16
git checkout codex/gfx936-bf16
bash scripts/scnet_ab_gfx936.sh init
bash scripts/scnet_ab_gfx936.sh sync-candidate-python
/usr/bin/time -f 'compile_wall_s=%e' \
  python3 scripts/build_gfx936_quant_jit.py \
    --source csrc/fdu/gfx936_quant_gemv.hip \
    --arch gfx936 --timeout 45 \
  2>&1 | tee /tmp/fdu_gfx936_quant_compile.log
```

Pass condition: exit 0, exactly one `.so` path is printed, all four symbols are
present, `compile_wall_s <= 50`, and a second invocation reports the same cache
path without compiling. Otherwise keep mode `off` and stop this route.

- [ ] **Step 2: Gate W8 first**

```bash
bash scripts/scnet_ab_gfx936.sh quant-bench-w8
python3 -m json.tool /tmp/fdu_gfx936_quant_w8.json >/dev/null
```

Full admission requires finite metrics, NRMSE <= 0.015, cosine >= 0.999, and
speedup >= 1.10 on all six shapes. The fast end-to-end test may still proceed
with a partial profile only when both MLP shapes `(34816,5120)` and
`(5120,17408)` plus at least two other shapes admit; every rejected shape must
remain BF16 and the server must start without OOM.

- [ ] **Step 3: Run one short highest-weight-tier test without repeating `off`**

```bash
cd /public/home/xdzs2026_c415/vllm_cscc
bash scripts/scnet_ab_gfx936.sh start-candidate-w8
bash scripts/scnet_ab_gfx936.sh probe-candidate-w8
bash scripts/scnet_ab_gfx936.sh throughput 8-16K 3 w8-fast
```

Do not spend another server load on `off`: use the already measured dx_branch
8-16K value of 12.00 tok/s as the comparison. Advance W8 only when the short run
is at least 12.60 tok/s, TTFT P99 is no more than 1.45x the official baseline,
TPOT P99 is no more than 1.45x baseline, the log confirms admitted quantized
layers for the high-byte shapes, and there is no OOM or non-finite value. The
three-case result is directional evidence with sampling risk, not a final score.

- [ ] **Step 4: Try hybrid W4 only after W8 proves useful**

Run the hybrid microbenchmark if W8 passes Step 3 but its directional projected
score is below 90. Start the hybrid server only when both W4 MLP rows pass their
numerical gates and each is at least 1.05x faster than its W8 row:

```bash
bash scripts/scnet_ab_gfx936.sh quant-bench-hybrid
bash scripts/scnet_ab_gfx936.sh start-candidate-hybrid
bash scripts/scnet_ab_gfx936.sh probe-candidate-hybrid
bash scripts/scnet_ab_gfx936.sh throughput 8-16K 3 hybrid-fast
```

Advance hybrid only when both W4 shapes meet their W4 numerical limits and the
tier throughput is at least 1.03x W8 with the same 1.45x SLA safety margin.
Otherwise W8 remains the candidate. A rejected W4 shape must be logged as W8
fallback, never silently omitted.

- [ ] **Step 5: Run a short three-tier check and two representative accuracy checks**

Set `WINNER=w8` or `WINNER=hybrid_w4`, start that server, then run:

```bash
cd /public/home/xdzs2026_c415/vllm_cscc
for tier in 8-16K 16-32K 4-8K; do
  bash scripts/scnet_ab_gfx936.sh throughput "$tier" 3 "$WINNER-fast"
done
bash scripts/scnet_ab_gfx936.sh accuracy hotpotqa 3 "$WINNER-fast"
bash scripts/scnet_ab_gfx936.sh accuracy retrieval_multi_point 3 "$WINNER-fast"
```

Fast local gate: every throughput tier has TTFT/TPOT under 1.45x baseline, both
sampled accuracy tasks show no degradation beyond 1%, no traceback/OOM occurs
after admission, and the weighted throughput projection exceeds 66.8175. At the
user's direction, the platform run is the final statistical and four-task
accuracy validation. If this short gate fails, revert selection to `off`.

- [ ] **Step 6: Select the measured default and record only observed values**

After the gate, change both defaults from `off` to the exact winning value:

```bash
# scripts/rocm_env.sh
export FDU_GFX936_QUANT_MODE="${FDU_GFX936_QUANT_MODE:-w8}"
```

```dockerfile
ENV FDU_GFX936_QUANT_MODE=w8
```

Use `hybrid_w4` in both places instead if it won. If no candidate passed, leave
both at `off`. Update `docs/GFX936_HANDOFF.md`, `report.md`, and `changelog.md`
with compile seconds, six-shape admission rows, the three short-run throughputs,
SLA P99 values, two sampled accuracy deltas, and the selection rationale. Label
all three-case results `SCNet fast gate`; do not present them as platform scores
or enter estimated/projected numbers as measurements.

- [ ] **Step 7: Commit the evidence-backed selection**

```bash
python3 -m unittest discover -s tests/fdu -p 'test_*.py'
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_ab_gfx936.sh
git diff --check
git add scripts/rocm_env.sh Dockerfile docs/GFX936_HANDOFF.md report.md changelog.md
git commit -m "perf: select gated gfx936 online quant mode"
```

### Task 9: Final verification and submission handoff

**Files:**
- Verify: all files changed by Tasks 1-8
- Verify: SCNet logs named in Task 8

- [ ] **Step 1: Run the complete local verification from a clean process**

```bash
cd /private/tmp/fdu-sccscc26-gfx936
python3 -m unittest discover -s tests/fdu -p 'test_*.py' -v
python3 -m py_compile \
  scripts/build_gfx936_quant_jit.py \
  scripts/preflight_gfx936_quant.py \
  scripts/bench_gfx936_quant.py \
  vllm/model_executor/layers/gfx936_online_quant.py
bash -n launch.sh scripts/rocm_env.sh scripts/scnet_ab_gfx936.sh
git diff --check
git status --short --branch
```

Expected: tests PASS, Python and shell syntax checks exit 0, no whitespace
errors, and only explicitly documented evidence files remain uncommitted.

- [ ] **Step 2: Audit the competition-safety boundary**

```bash
rg -n "(save|torch\.save|safetensors|download|speculative|layer.?skip|prun)" \
  vllm/model_executor/layers/gfx936_online_quant.py \
  scripts/build_gfx936_quant_jit.py \
  scripts/preflight_gfx936_quant.py \
  launch.sh
find /tmp -maxdepth 3 -name 'gfx936_quant_*.so' -print
```

Manually confirm that quantized weights exist only in process memory, the `.so`
contains code only, no model tensor is written to disk, no evaluator-locked
scheduler or generation parameter is changed, and every failure path returns to
the existing BF16 behavior.

- [ ] **Step 3: Review the evidence and diff before any push**

```bash
git log --oneline --decorate -10
git diff origin/dx_branch...HEAD --stat
git diff origin/dx_branch...HEAD -- \
  launch.sh scripts/rocm_env.sh Dockerfile \
  vllm/model_executor/layers/gfx936_online_quant.py \
  vllm/model_executor/layers/linear.py \
  vllm/model_executor/layers/utils.py
```

Invoke `superpowers:requesting-code-review`, resolve any correctness or
competition-rule findings, then invoke `superpowers:verification-before-completion`
and rerun Step 1. The experiment branch may be pushed for SCNet synchronization;
do not update or bind the platform submission branch until the user reviews the
measured winner and explicitly authorizes submission.

## Execution order under the time limit

The critical path is Tasks 1-7, then Task 8 Steps 1-3. W8 gets the first server
and a three-case 8-16K run because that tier carries 50% of the score. There is
no repeated `off` run. Hybrid W4 runs only if W8 is useful but still short of the
target; the winner receives one three-case all-tier run plus two three-case
accuracy probes. The BF16 `off` mode remains runnable at every checkpoint, and
the platform evaluation supplies the exhaustive final evidence.
