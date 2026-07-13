#!/usr/bin/env python3
"""Benchmark the native gfx936 skinny GEMM and generate its safe whitelist."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence


DEFAULT_MODEL_CONFIG = Path(
    "/public/home/xdzs2026_c415/Qwen3.5-27B/config.json"
)
DEFAULT_OUTPUT = Path(
    "/public/home/xdzs2026_c415/results/gfx936_skinny/microbench.json"
)
DECODE_BATCH_SIZES = (1, 2, 4)
EXPECTED_FAMILIES = frozenset(
    {
        "gdn_qkvz",
        "gdn_ba",
        "full_attention_qkv_gate",
        "attention_output",
        "mlp_gate_up",
        "mlp_down",
    }
)


def _positive_int(config: dict[str, Any], name: str, default: int | None = None) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def derive_qwen35_shapes(config: dict[str, Any]) -> list[dict[str, int | str]]:
    """Derive the six dominant TP=1 linear families from Qwen3.5 config."""
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be an object")

    hidden = _positive_int(text, "hidden_size")
    intermediate = _positive_int(text, "intermediate_size")
    layers = _positive_int(text, "num_hidden_layers")
    attention_heads = _positive_int(text, "num_attention_heads")
    kv_heads = _positive_int(text, "num_key_value_heads")
    head_dim = _positive_int(text, "head_dim", hidden // attention_heads)
    linear_key_dim = _positive_int(text, "linear_key_head_dim", 128)
    linear_value_dim = _positive_int(text, "linear_value_head_dim", 128)
    linear_key_heads = _positive_int(text, "linear_num_key_heads", attention_heads)
    linear_value_heads = _positive_int(
        text, "linear_num_value_heads", attention_heads * 2
    )

    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = _positive_int(text, "full_attention_interval", 4)
        layer_types = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(layers)
        ]
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise ValueError("layer_types must contain one entry per hidden layer")
    unknown = set(layer_types) - {"linear_attention", "full_attention"}
    if unknown:
        raise ValueError(f"unsupported layer types: {sorted(unknown)!r}")

    linear_layers = layer_types.count("linear_attention")
    full_layers = layer_types.count("full_attention")
    if not linear_layers or not full_layers:
        raise ValueError("Qwen3.5 benchmark requires both linear and full attention")

    key_dim = linear_key_heads * linear_key_dim
    value_dim = linear_value_heads * linear_value_dim
    gate_multiplier = 2 if bool(text.get("attn_output_gate", True)) else 1
    full_qkv_gate = (
        attention_heads * head_dim * gate_multiplier + 2 * kv_heads * head_dim
    )

    return [
        {
            "family": "gdn_qkvz",
            "m": 2 * key_dim + 2 * value_dim,
            "k": hidden,
            "layer_count": linear_layers,
        },
        {
            "family": "gdn_ba",
            "m": 2 * linear_value_heads,
            "k": hidden,
            "layer_count": linear_layers,
        },
        {
            "family": "full_attention_qkv_gate",
            "m": full_qkv_gate,
            "k": hidden,
            "layer_count": full_layers,
        },
        {
            "family": "attention_output",
            "m": hidden,
            "k": attention_heads * head_dim,
            "layer_count": layers,
        },
        {
            "family": "mlp_gate_up",
            "m": 2 * intermediate,
            "k": hidden,
            "layer_count": layers,
        },
        {
            "family": "mlp_down",
            "m": hidden,
            "k": intermediate,
            "layer_count": layers,
        },
    ]


def _paired_vectors(reference: Sequence[float], actual: Sequence[float]) -> None:
    if not reference or len(reference) != len(actual):
        raise ValueError("vectors must be non-empty and have equal length")


def cosine_similarity(reference: Sequence[float], actual: Sequence[float]) -> float:
    _paired_vectors(reference, actual)
    dot = sum(a * b for a, b in zip(reference, actual))
    norm_reference = math.sqrt(sum(value * value for value in reference))
    norm_actual = math.sqrt(sum(value * value for value in actual))
    if norm_reference == 0.0 and norm_actual == 0.0:
        return 1.0
    if norm_reference == 0.0 or norm_actual == 0.0:
        return 0.0
    return dot / (norm_reference * norm_actual)


def relative_l2(reference: Sequence[float], actual: Sequence[float]) -> float:
    _paired_vectors(reference, actual)
    denominator = math.sqrt(sum(value * value for value in reference))
    numerator = math.sqrt(
        sum((a - b) * (a - b) for a, b in zip(reference, actual))
    )
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def row_is_admitted(row: dict[str, Any]) -> bool:
    required_numbers = (
        "cosine_similarity",
        "relative_l2",
        "stock_median_ms",
        "candidate_median_ms",
        "stock_p99_ms",
        "candidate_p99_ms",
        "speedup",
    )
    try:
        finite_numbers = all(math.isfinite(float(row[key])) for key in required_numbers)
        return (
            bool(row["finite"])
            and bool(row["assert_close"])
            and finite_numbers
            and float(row["cosine_similarity"]) >= 0.999
            and float(row["relative_l2"]) <= 0.01
            and float(row["stock_median_ms"]) > 0.0
            and float(row["candidate_median_ms"]) > 0.0
            and float(row["speedup"]) >= 1.15
            and float(row["candidate_p99_ms"]) <= float(row["stock_p99_ms"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def project_linear_speedup(rows: Iterable[dict[str, Any]]) -> dict[int, float]:
    admitted: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not bool(row.get("admitted", row_is_admitted(row))):
            continue
        key = (int(row["n"]), str(row["family"]))
        if key in admitted:
            raise ValueError(f"duplicate benchmark row: {key!r}")
        admitted[key] = row

    projected: dict[int, float] = {}
    for n in DECODE_BATCH_SIZES:
        selected = [admitted.get((n, family)) for family in EXPECTED_FAMILIES]
        if any(row is None for row in selected):
            projected[n] = 0.0
            continue
        stock_total = sum(
            float(row["stock_median_ms"]) * int(row["layer_count"])
            for row in selected
            if row is not None
        )
        candidate_total = sum(
            float(row["candidate_median_ms"]) * int(row["layer_count"])
            for row in selected
            if row is not None
        )
        projected[n] = stock_total / candidate_total if candidate_total > 0 else 0.0
    return projected


def render_whitelist_module(rows: Iterable[dict[str, Any]]) -> str:
    shapes = sorted(
        {
            (
                int(row["n"]),
                int(row["m"]),
                int(row["k"]),
                str(row["dtype"]),
                bool(row.get("bias_present", False)),
            )
            for row in rows
            if row_is_admitted(row)
        }
    )
    header = (
        "from __future__ import annotations\n\n\n"
        "SkinnyShape = tuple[int, int, int, str, bool]\n\n"
        "# Generated by scripts/bench_gfx936_skinny.py after all gates pass.\n"
    )
    if not shapes:
        return header + "VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset()\n"
    rendered = "\n".join(
        f'    ({n}, {m}, {k}, "{dtype_name}", {bias_present}),' 
        for n, m, k, dtype_name, bias_present in shapes
    )
    return (
        header
        + "VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset({\n"
        + rendered
        + "\n})\n"
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def _effective_bandwidth_gbps(n: int, m: int, k: int, ms: float) -> float:
    bf16_bytes = 2 * (m * k + n * k + n * m)
    return bf16_bytes / (ms * 1_000_000.0)


def run_gpu_benchmark(
    *,
    shapes: list[dict[str, int | str]],
    dtype_name: str,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[str, list[dict[str, Any]]]:
    import torch
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("ROCm GPU is required")
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if arch != "gfx936":
        raise RuntimeError(f"native gfx936 is required, detected {arch!r}")
    dtype = getattr(torch, dtype_name)
    compute_units = int(num_compute_units(0))
    rows: list[dict[str, Any]] = []

    def stock(weight: Any, activation: Any) -> Any:
        return torch.nn.functional.linear(activation, weight)

    def candidate(weight: Any, activation: Any) -> Any:
        return ops.wvSplitK(weight, activation, compute_units, None)

    def measure(function: Any, weight: Any, activation: Any) -> list[float]:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            function(weight, activation)
            end.record()
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]

    for n in DECODE_BATCH_SIZES:
        for shape_index, shape in enumerate(shapes):
            family = str(shape["family"])
            m, k = int(shape["m"]), int(shape["k"])
            print(
                f"[gfx936:bench] start n={n} family={family} m={m} k={k}",
                flush=True,
            )
            row: dict[str, Any] = {
                **shape,
                "n": n,
                "dtype": dtype_name,
                "bias_present": False,
                "finite": False,
                "assert_close": False,
                "admitted": False,
            }
            try:
                torch.manual_seed(20260714 + n * 100 + shape_index)
                activation = torch.randn((n, k), device="cuda", dtype=dtype).contiguous()
                weight = torch.randn((m, k), device="cuda", dtype=dtype).contiguous()
                reference = stock(weight, activation)
                actual = candidate(weight, activation)
                torch.cuda.synchronize()
                reference32 = reference.float()
                actual32 = actual.float()
                row["finite"] = bool(torch.isfinite(actual32).all().item())
                ref_flat = reference32.flatten()
                actual_flat = actual32.flatten()
                denominator = float(torch.linalg.vector_norm(ref_flat).item())
                difference = float(torch.linalg.vector_norm(ref_flat - actual_flat).item())
                row["relative_l2"] = (
                    difference / denominator
                    if denominator > 0.0
                    else (0.0 if difference == 0.0 else math.inf)
                )
                ref_norm = float(torch.linalg.vector_norm(ref_flat).item())
                actual_norm = float(torch.linalg.vector_norm(actual_flat).item())
                row["cosine_similarity"] = (
                    float(torch.dot(ref_flat, actual_flat).item()) / (ref_norm * actual_norm)
                    if ref_norm and actual_norm
                    else (1.0 if ref_norm == actual_norm == 0.0 else 0.0)
                )
                try:
                    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.5)
                    row["assert_close"] = True
                except AssertionError as error:
                    row["close_error"] = str(error)

                for _ in range(warmup):
                    stock(weight, activation)
                    candidate(weight, activation)
                torch.cuda.synchronize()
                stock_times: list[float] = []
                candidate_times: list[float] = []
                for repeat in range(repeats):
                    pair = (
                        ((stock, stock_times), (candidate, candidate_times))
                        if repeat % 2 == 0
                        else ((candidate, candidate_times), (stock, stock_times))
                    )
                    for function, destination in pair:
                        destination.extend(measure(function, weight, activation))

                row["stock_median_ms"] = statistics.median(stock_times)
                row["candidate_median_ms"] = statistics.median(candidate_times)
                row["stock_p99_ms"] = percentile(stock_times, 0.99)
                row["candidate_p99_ms"] = percentile(candidate_times, 0.99)
                row["speedup"] = (
                    row["stock_median_ms"] / row["candidate_median_ms"]
                )
                row["stock_effective_bandwidth_gbps"] = _effective_bandwidth_gbps(
                    n, m, k, row["stock_median_ms"]
                )
                row["candidate_effective_bandwidth_gbps"] = _effective_bandwidth_gbps(
                    n, m, k, row["candidate_median_ms"]
                )
                row["admitted"] = row_is_admitted(row)
                if not row["admitted"]:
                    row["rejection"] = "correctness or latency gate failed"
                del reference, actual, reference32, actual32, activation, weight
                torch.cuda.empty_cache()
            except Exception as error:  # keep every failed shape in the artifact
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            print(
                f"[gfx936:bench] done n={n} family={family} "
                f"admitted={row['admitted']} speedup={row.get('speedup')} "
                f"error={row.get('error')}",
                flush=True,
            )
    return arch, rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--write-whitelist", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        raise SystemExit("warmup, iterations, and repeats must be positive")
    artifact: dict[str, Any] = {
        "model_config": str(args.model_config),
        "dtype": args.dtype,
        "settings": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "batch_sizes": list(DECODE_BATCH_SIZES),
        },
        "rows": [],
        "projected_linear_speedup": {},
        "passed": False,
    }
    try:
        config = json.loads(args.model_config.read_text())
        shapes = derive_qwen35_shapes(config)
        arch, rows = run_gpu_benchmark(
            shapes=shapes,
            dtype_name=args.dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        projections = project_linear_speedup(rows)
        passed = (
            len(rows) == len(shapes) * len(DECODE_BATCH_SIZES)
            and all(bool(row.get("admitted")) for row in rows)
            and all(projections[n] >= 1.6 for n in DECODE_BATCH_SIZES)
        )
        artifact.update(
            arch=arch,
            shapes=shapes,
            rows=rows,
            projected_linear_speedup=projections,
            passed=passed,
        )
        if passed and args.write_whitelist is not None:
            _atomic_write(args.write_whitelist, render_whitelist_module(rows))
            artifact["whitelist"] = str(args.write_whitelist)
    except Exception as error:
        artifact["fatal_error"] = f"{type(error).__name__}: {error}"
    _atomic_write(args.output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if not artifact["passed"]:
        print(f"gfx936 skinny GEMM gate failed; see {args.output}", file=sys.stderr)
        return 2
    print(f"gfx936 skinny GEMM gate passed; see {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
