"""
DCU-Optimized Attention Backend for vLLM (ROCm/HIP).

两套执行路径：
  A. HIP Kernel（DCU 实机）— hipcc 预编译 .so → ctypes host wrapper
  B. PyTorch Fallback（本地）— F.scaled_dot_product_attention + GQA

编译: bash scripts/compile_kernels.sh
启用: export VLLM_ATTENTION_BACKEND=dcu_optimized
"""

import ctypes
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


class DCUAttentionBackend:
    """DCU 优化的 Attention 后端。通过 ctypes 调用预编译的 HIP host wrapper。"""

    WAVEFRONT_SIZE = 64
    TILE_Q = 128
    TILE_KV = 64

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        use_fp8: bool = False,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (head_dim ** -0.5)
        self.use_fp8 = use_fp8
        self.num_queries_per_kv = num_heads // num_kv_heads
        self._kernel_lib: Optional[ctypes.CDLL] = None

    @staticmethod
    def is_rocm() -> bool:
        if not torch.cuda.is_available():
            return False
        return hasattr(torch.version, "hip") and torch.version.hip is not None

    @property
    def kernel_loaded(self) -> bool:
        return self._kernel_lib is not None

    def load_kernel(self, verbose: bool = True) -> bool:
        """加载预编译 HIP kernel .so，绑定 host wrapper 函数签名。"""
        if not self.is_rocm():
            if verbose:
                print("[DCUAttn] Not ROCm — skip kernel load.")
            return False

        search_paths = [
            Path(__file__).parent.parent.parent / "build" / "kernels" / "dcu_flash_attn.so",
            Path(__file__).parent / "hip_kernels" / "dcu_flash_attn.so",
        ]
        so_path = None
        for p in search_paths:
            if p.exists():
                so_path = str(p.resolve())
                break

        if so_path is None:
            if verbose:
                print("[DCUAttn] Kernel .so not found. Run: bash scripts/compile_kernels.sh")
            return False

        try:
            lib = ctypes.CDLL(so_path)
            # 绑定 host wrapper: 返回 int，接受 10 个参数 + stream
            lib.dcu_flash_attn_forward.argtypes = [
                ctypes.c_void_p,  # Q
                ctypes.c_void_p,  # K
                ctypes.c_void_p,  # V
                ctypes.c_void_p,  # O
                ctypes.c_float,   # scale
                ctypes.c_int,     # seq_len
                ctypes.c_int,     # num_heads
                ctypes.c_int,     # num_kv_heads
                ctypes.c_int,     # head_dim
                ctypes.c_void_p,  # hipStream_t
            ]
            lib.dcu_flash_attn_forward.restype = ctypes.c_int
            self._kernel_lib = lib
            if verbose:
                print(f"[DCUAttn] Kernel loaded: {so_path}")
            return True
        except OSError as e:
            if verbose:
                print(f"[DCUAttn] Failed to load {so_path}: {e}")
            return False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_tables=None,
        context_lens=None,
        max_context_len: int = 0,
        alibi_slopes=None,
    ) -> torch.Tensor:
        if self.kernel_loaded:
            return self._forward_hip(query, key, value)
        return self._forward_torch(query, key, value)

    def _forward_hip(self, query, key, value) -> torch.Tensor:
        """通过 ctypes host wrapper 调用 HIP kernel。"""
        num_tokens = query.shape[0]
        output = torch.empty_like(query)

        # 获取当前 CUDA/HIP stream
        stream = torch.cuda.current_stream().cuda_stream

        ret = self._kernel_lib.dcu_flash_attn_forward(
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            output.data_ptr(),
            ctypes.c_float(self.scale),
            ctypes.c_int(num_tokens),
            ctypes.c_int(self.num_heads),
            ctypes.c_int(self.num_kv_heads),
            ctypes.c_int(self.head_dim),
            ctypes.c_void_p(stream),
        )

        if ret != 0:
            raise RuntimeError(f"[DCUAttn] HIP kernel returned error code {ret}")

        torch.cuda.synchronize()
        return output

    def _forward_torch(self, query, key, value) -> torch.Tensor:
        """PyTorch fallback with GQA support."""
        if query.shape[1] != key.shape[1]:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)
        return F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            scale=self.scale,
        ).transpose(0, 1)
