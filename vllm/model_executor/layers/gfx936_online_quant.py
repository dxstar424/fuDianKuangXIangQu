from __future__ import annotations

import ctypes
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


_LOADED_KERNELS: LoadedKernels | None = None
_RUNTIME_FALLBACK_WARNED = False
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
