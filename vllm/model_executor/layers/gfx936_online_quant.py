from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    import torch


QuantMode = Literal["off", "w8", "hybrid_w4"]
QuantKind = Literal["w8", "w4"]

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


class AdmissionResult(NamedTuple):
    accepted: bool
    nrmse: float
    cosine: float
    baseline_ms: float
    candidate_ms: float
    speedup: float
    reason: str


_LOADED_KERNELS: LoadedKernels | None = None
_RUNTIME_FALLBACK_WARNED = False
_ADMISSION_CACHE: dict[tuple[int, int, QuantKind], AdmissionResult] = {}
_ACTIVE_LAYER_COUNT = 0
logger = logging.getLogger(__name__)

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
W8_NRMSE_LIMIT = 0.015
W8_COSINE_LIMIT = 0.999
W4_NRMSE_LIMIT = 0.08
W4_COSINE_LIMIT = 0.995
MIN_SPEEDUP = 1.10
WARMUP_REPETITIONS = 5
TIMED_REPETITIONS = 30


def parse_quant_mode(value: str | None = None) -> QuantMode:
    normalized = value if value is not None else os.getenv(
        "FDU_GFX936_QUANT_MODE", "w8"
    )
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


def admission_limits(kind: QuantKind) -> tuple[float, float]:
    return (W4_NRMSE_LIMIT, W4_COSINE_LIMIT) if kind == "w4" else (
        W8_NRMSE_LIMIT,
        W8_COSINE_LIMIT,
    )


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
    reason = (
        "accepted"
        if accepted
        else (
            f"rejected:nrmse={nrmse:.6f},cosine={cosine:.6f},"
            f"speedup={speedup:.3f}"
        )
    )
    return AdmissionResult(
        accepted,
        nrmse,
        cosine,
        baseline_ms,
        candidate_ms,
        speedup,
        reason,
    )


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


def require_online_quantization() -> None:
    mode = parse_quant_mode()
    if mode == "off" or not is_gfx936_runtime():
        return
    if not online_quantization_active():
        raise RuntimeError(
            f"gfx936 quant mode {mode!r} requested but zero layers were activated"
        )
    logger.info(
        "gfx936 quant activation complete mode=%s layers=%d",
        mode,
        _ACTIVE_LAYER_COUNT,
    )


def iter_row_chunks(
    rows: int, columns: int, byte_limit: int = PACK_CHUNK_BYTES
) -> Iterator[tuple[int, int]]:
    if rows <= 0 or columns <= 0 or byte_limit <= 0:
        raise ValueError("rows, columns, and byte_limit must be positive")
    rows_per_chunk = max(1, byte_limit // (columns * 2))
    for start in range(0, rows, rows_per_chunk):
        yield start, min(rows, start + rows_per_chunk)


def resolve_kernel_library_path(path: str | Path | None = None) -> Path:
    explicit = path if path is not None else os.getenv("FDU_GFX936_QUANT_SO")
    if explicit:
        selected = Path(explicit).expanduser()
        try:
            resolved = selected.resolve(strict=True)
        except OSError as error:
            raise FileNotFoundError(
                f"gfx936 quant library not found: {selected}"
            ) from error
        if not resolved.is_file():
            raise FileNotFoundError(f"gfx936 quant library not found: {resolved}")
        return resolved

    spec = importlib.util.find_spec("vllm")
    locations = () if spec is None else (spec.submodule_search_locations or ())
    for location in locations:
        package = Path(location)
        for candidate in sorted(package.glob("_rocm_C*.so")):
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        "bundled gfx936 quant library was not found in vllm._rocm_C"
    )


def load_kernel_library(path: str | Path | None = None) -> LoadedKernels:
    global _LOADED_KERNELS
    if _LOADED_KERNELS is not None and path is None:
        return _LOADED_KERNELS
    selected = resolve_kernel_library_path(path)
    library = ctypes.CDLL(str(selected))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    gemv_args = [pointer, pointer, pointer, pointer, integer, integer, pointer]
    dequant_args = [pointer, pointer, pointer, integer, integer, pointer]
    functions = []
    for name in REQUIRED_SYMBOLS:
        try:
            functions.append(getattr(library, name))
        except AttributeError as error:
            raise RuntimeError(
                f"gfx936 quant library missing required ABI symbol: {name}"
            ) from error
    signatures = (gemv_args, gemv_args, dequant_args, dequant_args)
    for name, function, arguments in zip(REQUIRED_SYMBOLS, functions, signatures):
        try:
            function.argtypes = arguments
            function.restype = integer
        except (AttributeError, TypeError, ValueError, ctypes.ArgumentError) as error:
            raise RuntimeError(
                f"failed to bind gfx936 quant ABI symbol {name}: {error}"
            ) from error
    loaded = LoadedKernels(library, *functions)
    if path is None:
        _LOADED_KERNELS = loaded
    return loaded


def _pointer(tensor: "torch.Tensor") -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_pointer() -> ctypes.c_void_p:
    import torch

    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _call_kernel(function, operation: str, *args):
    try:
        return function(*args)
    except ctypes.ArgumentError as error:
        raise RuntimeError(f"{operation} ABI invocation failed: {error}") from error


def _check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with HIP status {status}")


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
            ((pairs[:, :, 0] & 0x0F) | ((pairs[:, :, 1] & 0x0F) << 4)).to(
                torch.uint8
            )
        )
        scales[start:end].copy_(scale.to(torch.float16))
        del source, maximum, scale, quantized, pairs
    return packed.contiguous(), scales.contiguous()


def pack_weight(
    weight: "torch.Tensor", kind: QuantKind
) -> tuple["torch.Tensor", "torch.Tensor"]:
    return pack_w4(weight) if kind == "w4" else pack_w8(weight)


def _dequantize_weight(packed, scale, kind: int, m: int, k: int):
    import torch

    output = torch.empty((m, k), dtype=torch.bfloat16, device=packed.device)
    kernels = load_kernel_library()
    function = kernels.w4_dequant if kind == KIND_W4 else kernels.w8_dequant
    operation = "w4_dequant" if kind == KIND_W4 else "w8_dequant"
    status = _call_kernel(
        function,
        operation,
        _pointer(packed),
        _pointer(scale),
        _pointer(output),
        m,
        k,
        _stream_pointer(),
    )
    _check_status(status, operation)
    return output


def _dequantize_weight_reference(packed, scale, kind: int, m: int, k: int):
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


def _warn_runtime_fallback_once(operation: str, error: BaseException) -> None:
    global _RUNTIME_FALLBACK_WARNED
    if not _RUNTIME_FALLBACK_WARNED:
        logger.warning(
            "gfx936 quant %s failed; using BF16 reconstruction: %s",
            operation,
            error,
        )
        _RUNTIME_FALLBACK_WARNED = True


def reconstruct_bf16_weight(packed, scale, kind: int, m: int, k: int):
    try:
        return _dequantize_weight(packed, scale, kind, m, k)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        _warn_runtime_fallback_once("dequant", error)
        return _dequantize_weight_reference(packed, scale, kind, m, k)


def run_quant_gemv(x, packed, scale, kind: int, m: int, k: int):
    import torch

    flattened = x.reshape(-1, k).contiguous()
    if flattened.shape[0] != 1:
        raise ValueError("run_quant_gemv requires exactly one input row")
    output = torch.empty((1, m), dtype=torch.bfloat16, device=x.device)
    kernels = load_kernel_library()
    function = kernels.w4_gemv if kind == KIND_W4 else kernels.w8_gemv
    operation = "w4_gemv" if kind == KIND_W4 else "w8_gemv"
    status = _call_kernel(
        function,
        operation,
        _pointer(packed),
        _pointer(scale),
        _pointer(flattened),
        _pointer(output),
        m,
        k,
        _stream_pointer(),
    )
    _check_status(status, operation)
    return output.reshape(*x.shape[:-1], m)


def _median_cuda_ms(operation, repetitions: int) -> float:
    import statistics

    import torch

    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del output
    return float(statistics.median(samples))


def benchmark_candidate(
    weight: "torch.Tensor",
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    kind: QuantKind,
    *,
    warmup_repetitions: int = WARMUP_REPETITIONS,
    timed_repetitions: int = TIMED_REPETITIONS,
) -> AdmissionResult:
    import torch
    import torch.nn.functional as functional

    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    if warmup_repetitions <= 0 or timed_repetitions <= 0:
        raise ValueError("benchmark repetition counts must be positive")

    m, k = map(int, weight.shape)
    kernel_kind = KIND_W4 if kind == "w4" else KIND_W8
    x = torch.linspace(
        -1,
        1,
        k,
        dtype=torch.bfloat16,
        device=weight.device,
    ).reshape(1, k)

    reference = functional.linear(x, weight)
    candidate = run_quant_gemv(x, packed, scale, kernel_kind, m, k)
    reference_fp32 = reference.float()
    candidate_fp32 = candidate.float()
    difference = candidate_fp32 - reference_fp32
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(reference_fp32.square())).item())
    nrmse = rmse / max(reference_rms, 1e-12)
    cosine = float(
        functional.cosine_similarity(
            candidate_fp32.reshape(-1),
            reference_fp32.reshape(-1),
            dim=0,
            eps=1e-12,
        ).item()
    )
    del reference, candidate, reference_fp32, candidate_fp32, difference

    for _ in range(warmup_repetitions):
        baseline_output = rocm_unquantized_gemm_impl(x, weight, None)
        candidate_output = run_quant_gemv(
            x, packed, scale, kernel_kind, m, k
        )
        del baseline_output, candidate_output
    torch.cuda.synchronize()

    baseline_ms = _median_cuda_ms(
        lambda: rocm_unquantized_gemm_impl(x, weight, None),
        timed_repetitions,
    )
    candidate_ms = _median_cuda_ms(
        lambda: run_quant_gemv(x, packed, scale, kernel_kind, m, k),
        timed_repetitions,
    )
    result = evaluate_admission(
        kind,
        nrmse,
        cosine,
        baseline_ms,
        candidate_ms,
    )
    logger.info(
        "gfx936_quant_admission m=%d k=%d kind=%s nrmse=%.6f "
        "cosine=%.6f baseline_ms=%.6f candidate_ms=%.6f "
        "speedup=%.3f decision=%s",
        m,
        k,
        kind,
        result.nrmse,
        result.cosine,
        result.baseline_ms,
        result.candidate_ms,
        result.speedup,
        result.reason,
    )
    return result


def quant_linear_impl(x, packed, scale, kind: int, m: int, k: int, bias=None):
    import torch.nn.functional as functional

    if x.numel() // k == 1:
        try:
            output = run_quant_gemv(x, packed, scale, kind, m, k)
            return output if bias is None else output + bias
        except (FileNotFoundError, OSError, RuntimeError) as error:
            _warn_runtime_fallback_once("gemv", error)
    bf16_weight = reconstruct_bf16_weight(packed, scale, kind, m, k)
    try:
        return functional.linear(x, bf16_weight, bias)
    finally:
        del bf16_weight


def _conversion_error_result(error: Exception) -> AdmissionResult:
    not_a_number = float("nan")
    return AdmissionResult(
        False,
        not_a_number,
        not_a_number,
        0.0,
        0.0,
        0.0,
        f"rejected:error={type(error).__name__}:{error}",
    )


def _install_quantized_weight(layer, packed, scale, kind: QuantKind, m: int, k: int):
    global _ACTIVE_LAYER_COUNT

    original_parameter = layer._parameters.pop("weight")
    try:
        layer.register_buffer("weight", packed, persistent=False)
        layer.register_buffer("gfx936_scale", scale, persistent=False)
        layer._fdu_gfx936_quantized = True
        layer._fdu_gfx936_quant_kind = KIND_W4 if kind == "w4" else KIND_W8
        layer._fdu_gfx936_quant_m = m
        layer._fdu_gfx936_quant_k = k
    except Exception:
        layer._buffers.pop("weight", None)
        layer._buffers.pop("gfx936_scale", None)
        non_persistent = getattr(layer, "_non_persistent_buffers_set", set())
        non_persistent.discard("weight")
        non_persistent.discard("gfx936_scale")
        for attribute in (
            "_fdu_gfx936_quantized",
            "_fdu_gfx936_quant_kind",
            "_fdu_gfx936_quant_m",
            "_fdu_gfx936_quant_k",
        ):
            layer.__dict__.pop(attribute, None)
        layer._parameters["weight"] = original_parameter
        raise
    _ACTIVE_LAYER_COUNT += 1
    del original_parameter


def maybe_quantize_gfx936_layer(layer: object) -> object:
    if is_gfx936_quantized_layer(layer):
        return layer

    mode = parse_quant_mode()
    if mode == "off":
        return layer

    try:
        if not is_gfx936_runtime():
            return layer
        load_kernel_library()

        import torch

        weight = getattr(layer, "weight", None)
        if (
            weight is None
            or getattr(weight, "ndim", None) != 2
            or not weight.is_contiguous()
            or weight.dtype != torch.bfloat16
            or getattr(weight.device, "type", None) != "cuda"
            or getattr(layer, "bias", None) is not None
        ):
            return layer
        m, k = map(int, weight.shape)
        if not is_quant_shape(m, k):
            return layer
    except Exception as error:
        logger.warning("gfx936 quant layer eligibility failed open: %s", error)
        return layer

    for kind in candidate_kinds(mode, m, k):
        cache_key = (m, k, kind)
        decision = _ADMISSION_CACHE.get(cache_key)
        if decision is not None and not decision.accepted:
            continue

        packed = None
        scale = None
        try:
            packed, scale = pack_weight(weight, kind)
            if decision is None:
                decision = benchmark_candidate(weight, packed, scale, kind)
                _ADMISSION_CACHE[cache_key] = decision
            if not decision.accepted:
                del packed, scale
                continue

            _install_quantized_weight(layer, packed, scale, kind, m, k)
        except Exception as error:
            _ADMISSION_CACHE[cache_key] = _conversion_error_result(error)
            logger.warning(
                "gfx936 quant conversion failed open m=%d k=%d kind=%s: %s",
                m,
                k,
                kind,
                error,
            )
            del packed, scale
            continue

        logger.info("gfx936 quant selected m=%d k=%d kind=%s", m, k, kind)
        del weight
        return layer

    return layer


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
