#!/usr/bin/env python3
"""Benchmark online gfx936 W8/W4 quantization for the six model shapes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
QUANT_SOURCE = ROOT / "vllm/model_executor/layers/gfx936_online_quant.py"
HIP_SOURCE = ROOT / "csrc/fdu/gfx936_quant_gemv.hip"
QUANT_SHAPES = (
    (16384, 5120),
    (96, 5120),
    (14336, 5120),
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
)


def _load_quant_module():
    spec = importlib.util.spec_from_file_location(
        "_fdu_gfx936_online_quant_benchmark", QUANT_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load quant source: {QUANT_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_QUANT = _load_quant_module()
if frozenset(QUANT_SHAPES) != _QUANT.QUANT_SHAPES:
    raise RuntimeError("benchmark shapes do not match the runtime quant policy")
W4_MLP_SHAPES = frozenset(_QUANT.W4_MLP_SHAPES)


def row_is_admitted(kind: str, row: Mapping[str, object]) -> bool:
    """Apply the exact runtime correctness and speed gates to one row."""
    if kind == "w8":
        nrmse_limit = float(_QUANT.W8_NRMSE_LIMIT)
        cosine_limit = float(_QUANT.W8_COSINE_LIMIT)
    elif kind == "w4":
        nrmse_limit = float(_QUANT.W4_NRMSE_LIMIT)
        cosine_limit = float(_QUANT.W4_COSINE_LIMIT)
    else:
        return False
    try:
        nrmse = float(row["nrmse"])
        cosine = float(row["cosine"])
        speedup = float(row["speedup"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in (nrmse, cosine, speedup))
        and nrmse <= nrmse_limit
        and cosine >= cosine_limit
        and speedup >= float(_QUANT.MIN_SPEEDUP)
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("w8", "hybrid_w4"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--repetitions", type=_positive_int, default=8)
    return parser.parse_args(argv)


def _json_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rejected_row(m: int, k: int, requested_kind: str, reason: str) -> dict[str, Any]:
    return {
        "M": m,
        "K": k,
        "requested_kind": requested_kind,
        "selected_kind": None,
        "packing_seconds": None,
        "nrmse": None,
        "cosine": None,
        "baseline_ms": None,
        "candidate_ms": None,
        "speedup": None,
        "peak_allocated_bytes": 0,
        "admitted": False,
        "reason": reason,
    }


def _measure_kind(
    torch: Any,
    quant: Any,
    weight: Any,
    device: Any,
    m: int,
    k: int,
    requested_kind: str,
    kind: str,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    row = _rejected_row(m, k, requested_kind, "rejected:not measured")
    packed = scale = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        packed, scale = quant.pack_weight(weight, kind)
        torch.cuda.synchronize(device)
        packing_seconds = time.perf_counter() - started
        decision = quant.benchmark_candidate(
            weight,
            packed,
            scale,
            kind,
            warmup_repetitions=warmup,
            timed_repetitions=repetitions,
        )
        raw = {
            "nrmse": float(decision.nrmse),
            "cosine": float(decision.cosine),
            "speedup": float(decision.speedup),
        }
        admitted = row_is_admitted(kind, raw)
        reason = (
            "accepted"
            if admitted
            else (
                f"rejected:nrmse={raw['nrmse']:.6f},"
                f"cosine={raw['cosine']:.6f},speedup={raw['speedup']:.3f}"
            )
        )
        row.update(
            selected_kind=kind if admitted else None,
            packing_seconds=_json_number(packing_seconds),
            nrmse=_json_number(decision.nrmse),
            cosine=_json_number(decision.cosine),
            baseline_ms=_json_number(decision.baseline_ms),
            candidate_ms=_json_number(decision.candidate_ms),
            speedup=_json_number(decision.speedup),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            admitted=admitted,
            reason=reason,
        )
    except Exception as error:
        row["reason"] = f"rejected:error={type(error).__name__}:{error}"
        try:
            row["peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
        except Exception:
            pass
    finally:
        del packed, scale
        torch.cuda.empty_cache()
    return row


def _run_shape(
    torch: Any,
    quant: Any,
    device: Any,
    shape_index: int,
    m: int,
    k: int,
    mode: str,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    requested_kind = (
        "w4" if mode == "hybrid_w4" and (m, k) in W4_MLP_SHAPES else "w8"
    )
    weight = None
    row = _rejected_row(m, k, requested_kind, "rejected:not measured")
    try:
        torch.manual_seed(20260714 + shape_index)
        weight = torch.empty(
            (m, k), dtype=torch.bfloat16, device=device
        ).uniform_(-1.0, 1.0).contiguous()
        kinds = ("w4", "w8") if requested_kind == "w4" else ("w8",)
        for kind in kinds:
            row = _measure_kind(
                torch,
                quant,
                weight,
                device,
                m,
                k,
                requested_kind,
                kind,
                warmup,
                repetitions,
            )
            if row["admitted"]:
                break
    except Exception as error:
        row = _rejected_row(
            m,
            k,
            requested_kind,
            f"rejected:error={type(error).__name__}:{error}",
        )
    finally:
        del weight
        torch.cuda.empty_cache()
    return row


def _git_commit() -> str | None:
    explicit = os.getenv("FDU_SOURCE_COMMIT", "").strip().lower()
    if len(explicit) == 40 and all(
        character in "0123456789abcdef" for character in explicit
    ):
        return explicit
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import torch

        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("PyTorch ROCm/HIP device is unavailable")
        if os.getenv("HSA_OVERRIDE_GFX_VERSION"):
            raise RuntimeError("HSA_OVERRIDE_GFX_VERSION must be unset for native gfx936")
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
        if arch != "gfx936":
            raise RuntimeError(f"native gfx936 is required, detected {arch!r}")
        library = args.library.expanduser().resolve(strict=True)
        if not library.is_file():
            raise RuntimeError(f"quant library is not a file: {library}")
        os.environ["FDU_GFX936_QUANT_SO"] = str(library)
        quant = _QUANT
        quant.load_kernel_library()
        hip_source_hash = hashlib.sha256(HIP_SOURCE.read_bytes()).hexdigest()
        device = torch.device("cuda", device_index)
    except Exception as error:
        print(f"[gfx936-quant-bench] fatal: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for shape_index, (m, k) in enumerate(QUANT_SHAPES):
        print(
            f"[gfx936-quant-bench] start M={m} K={k} mode={args.mode}",
            flush=True,
        )
        row = _run_shape(
            torch,
            quant,
            device,
            shape_index,
            m,
            k,
            args.mode,
            args.warmup,
            args.repetitions,
        )
        rows.append(row)
        print(
            f"[gfx936-quant-bench] done M={m} K={k} "
            f"selected={row['selected_kind']} admitted={row['admitted']} "
            f"reason={row['reason']}",
            flush=True,
        )

    document = {
        "git_commit": _git_commit(),
        "hip_source_sha256": hip_source_hash,
        "arch": arch,
        "torch_version": str(torch.__version__),
        "rocm_version": str(torch.version.hip),
        "mode": args.mode,
        "library": str(library),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "rows": rows,
        "passed": all(bool(row["admitted"]) for row in rows),
    }
    try:
        _write_json(args.output, document)
    except (OSError, TypeError, ValueError) as error:
        print(f"[gfx936-quant-bench] cannot write {args.output}: {error}", file=sys.stderr)
        return 2
    if not document["passed"]:
        print(f"gfx936 quant gate failed; see {args.output}", file=sys.stderr)
        return 2
    print(f"gfx936 quant gate passed; see {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
