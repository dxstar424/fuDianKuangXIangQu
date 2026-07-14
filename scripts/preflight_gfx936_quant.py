#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import pathlib
import sys


REQUIRED_SYMBOLS = (
    "fdu_gfx936_w8a16_gemv",
    "fdu_gfx936_w4a16_gemv",
    "fdu_gfx936_w8_dequant",
    "fdu_gfx936_w4_dequant",
)

_KIND_VALUES = {"w8": 8, "w4": 4}
_LIMITS = {
    "w8": (0.015, 0.999),
    "w4": (0.08, 0.995),
}


class PreflightError(RuntimeError):
    pass


def validate_symbols(path: str | pathlib.Path) -> pathlib.Path:
    selected = pathlib.Path(path).expanduser()
    try:
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise PreflightError(
            f"gfx936 quant library is not a readable regular file: {selected}: {error}"
        ) from error
    try:
        valid_file = resolved.is_file() and resolved.stat().st_size > 0
    except OSError as error:
        raise PreflightError(
            f"cannot inspect gfx936 quant library {resolved}: {error}"
        ) from error
    if not valid_file:
        raise PreflightError(
            f"gfx936 quant library must be a nonempty regular file: {resolved}"
        )
    try:
        library = ctypes.CDLL(str(resolved))
    except OSError as error:
        raise PreflightError(
            f"cannot load gfx936 quant library {resolved}: {error}"
        ) from error
    for name in REQUIRED_SYMBOLS:
        try:
            getattr(library, name)
        except AttributeError as error:
            raise PreflightError(
                f"gfx936 quant library {resolved} missing required ABI symbol: {name}"
            ) from error
    return resolved


def load_quant_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    module_path = root / "vllm/model_executor/layers/gfx936_online_quant.py"
    spec = importlib.util.spec_from_file_location(
        "_fdu_gfx936_online_quant_preflight", module_path
    )
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot create import spec for quant runtime: {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise PreflightError(
            f"cannot load quant runtime directly from {module_path}: {error}"
        ) from error
    return module


def compute_metrics(reference, candidate) -> tuple[float, float]:
    reference_values = [float(value) for value in reference]
    candidate_values = [float(value) for value in candidate]
    if not reference_values or len(reference_values) != len(candidate_values):
        raise PreflightError("metric inputs must have the same nonzero length")
    if not all(
        math.isfinite(value) for value in reference_values + candidate_values
    ):
        raise PreflightError("metric input contains a non-finite value")

    count = len(reference_values)
    squared_error = math.fsum(
        (actual - expected) ** 2
        for expected, actual in zip(reference_values, candidate_values)
    )
    reference_energy = math.fsum(value * value for value in reference_values)
    candidate_energy = math.fsum(value * value for value in candidate_values)
    dot_product = math.fsum(
        expected * actual
        for expected, actual in zip(reference_values, candidate_values)
    )
    rmse = math.sqrt(squared_error / count)
    reference_rms = math.sqrt(reference_energy / count)
    nrmse = rmse / max(reference_rms, 1e-12)
    cosine = dot_product / max(
        math.sqrt(reference_energy * candidate_energy), 1e-12
    )
    if not math.isfinite(nrmse) or not math.isfinite(cosine):
        raise PreflightError("computed a non-finite smoke metric")
    return nrmse, cosine


def _record_check(checks, kind: str, k: int, operation: str, reference, output):
    nrmse, cosine = compute_metrics(reference, output)
    nrmse_limit, cosine_limit = _LIMITS[kind]
    checks.append(
        {
            "kind": kind,
            "k": k,
            "operation": operation,
            "nrmse": nrmse,
            "cosine": cosine,
            "passed": nrmse <= nrmse_limit and cosine >= cosine_limit,
        }
    )


def run_smoke(library: str | pathlib.Path, mode: str) -> list[dict[str, object]]:
    if mode not in {"off", "w8", "hybrid_w4"}:
        raise PreflightError(f"unsupported gfx936 quant mode: {mode}")
    resolved = pathlib.Path(library).expanduser().resolve()
    os.environ["FDU_GFX936_QUANT_SO"] = str(resolved)
    if mode == "off":
        return []

    try:
        import torch
        import torch.nn.functional as functional
    except Exception as error:
        raise PreflightError(f"cannot import PyTorch for HIP smoke test: {error}") from error
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise PreflightError("PyTorch ROCm/HIP device is unavailable")

    quant = load_quant_module()
    device = torch.device("cuda", torch.cuda.current_device())
    cases = [("w8", k) for k in (5120, 6144, 17408)]
    if mode == "hybrid_w4":
        cases.extend(("w4", k) for k in (5120, 17408))

    checks: list[dict[str, object]] = []
    for kind, k in cases:
        axis = weight = input_tensor = packed = scale = None
        reference = gemv_output = dequantized_weight = dequant_output = None
        try:
            axis = torch.linspace(-1.0, 1.0, k, dtype=torch.float32, device=device)
            weight = torch.stack(
                (
                    axis,
                    -0.75 * axis + 0.125,
                    0.5 * axis - 0.25,
                    -0.25 * axis - 0.5,
                )
            ).to(dtype=torch.bfloat16).contiguous()
            input_tensor = axis.to(dtype=torch.bfloat16).reshape(1, k).contiguous()
            packed, scale = quant.pack_weight(weight, kind)
            kernel_kind = _KIND_VALUES[kind]
            reference = functional.linear(input_tensor, weight)
            gemv_output = quant.run_quant_gemv(
                input_tensor, packed, scale, kernel_kind, 4, k
            )
            dequantized_weight = quant._dequantize_weight(
                packed, scale, kernel_kind, 4, k
            )
            dequant_output = functional.linear(input_tensor, dequantized_weight)
            torch.cuda.synchronize()

            for operation, tensor in (
                ("reference", reference),
                ("gemv", gemv_output),
                ("dequant", dequant_output),
                ("dequant_weight", dequantized_weight),
            ):
                if not bool(torch.isfinite(tensor).all().item()):
                    raise PreflightError(
                        f"{kind} K={k} {operation} produced non-finite output"
                    )

            reference_values = reference.float().reshape(-1).cpu().tolist()
            _record_check(
                checks,
                kind,
                k,
                "gemv",
                reference_values,
                gemv_output.float().reshape(-1).cpu().tolist(),
            )
            _record_check(
                checks,
                kind,
                k,
                "dequant",
                reference_values,
                dequant_output.float().reshape(-1).cpu().tolist(),
            )
        except PreflightError:
            raise
        except Exception as error:
            raise PreflightError(f"{kind} K={k} HIP smoke failed: {error}") from error
        finally:
            del axis, weight, input_tensor, packed, scale
            del reference, gemv_output, dequantized_weight, dequant_output
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the gfx936 quant library")
    parser.add_argument("--library", required=True)
    parser.add_argument("--mode", choices=("off", "w8", "hybrid_w4"), required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "library": str(pathlib.Path(args.library).expanduser().resolve()),
        "mode": args.mode,
        "checks": [],
    }
    try:
        resolved = validate_symbols(args.library)
        report["library"] = str(resolved)
        if args.smoke:
            report["checks"] = run_smoke(resolved, args.mode)
    except (PreflightError, OSError, RuntimeError) as error:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        print(f"[gfx936-quant-preflight] {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, sort_keys=True, allow_nan=False))
    failed = [check for check in report["checks"] if not check["passed"]]
    if failed:
        details = ", ".join(
            f"{check['kind']}/K={check['k']}/{check['operation']}"
            for check in failed
        )
        print(
            f"[gfx936-quant-preflight] smoke thresholds failed: {details}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
