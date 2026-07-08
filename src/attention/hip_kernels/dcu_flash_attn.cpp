/**
 * DCU FlashAttention HIP Kernel
 * ==============================
 * CDNA2/CDNA3 优化的 FlashAttention 前向 kernel。
 *
 * 编译:
 *   bash scripts/compile_kernels.sh
 *   手动: hipcc -O3 --offload-arch=gfx942 -std=c++17 -fPIC -shared
 *         -o dcu_flash_attn.so dcu_flash_attn.cpp
 *
 * 架构:
 *   Wavefront=64, LDS/CU=64KB, MFMA=16×16×16, HBM burst=128B
 *
 * 注意: __global__ kernel 不能有 C linkage。Python 通过 C-linkage
 * host wrapper 函数调用（见文件末尾的 extern "C" 函数）。
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>

#define WAVEFRONT_SIZE 64
#define TILE_Q 128
#define TILE_KV 64
#define HEAD_DIM 128

// ---- bf16 ↔ float ----

__device__ __forceinline__ float bf16_to_float(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __hip_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

// ---- FlashAttention kernel (C++ name mangling, no extern "C") ----

__global__ void dcu_flash_attn_forward_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const __hip_bfloat16* __restrict__ K,
    const __hip_bfloat16* __restrict__ V,
    __hip_bfloat16* __restrict__ O,
    const float scale,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    __shared__ float Q_lds[TILE_Q][HEAD_DIM];
    __shared__ float K_lds[TILE_KV][HEAD_DIM];
    __shared__ float V_lds[TILE_KV][HEAD_DIM];

    int tid = threadIdx.x;
    int gid = blockIdx.x * TILE_Q + tid;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    static constexpr int ACC_ELEMS = HEAD_DIM / WAVEFRONT_SIZE;
    float acc[ACC_ELEMS];
    #pragma unroll
    for (int d = 0; d < ACC_ELEMS; d++) acc[d] = 0.0f;

    // Load Q → LDS (coalesced)
    if (gid < seq_len) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            Q_lds[tid][d] = bf16_to_float(Q[gid * HEAD_DIM + d]);
    } else {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            Q_lds[tid][d] = 0.0f;
    }
    __syncthreads();

    // Main loop: iterate KV tiles
    for (int kv_start = 0; kv_start < seq_len; kv_start += TILE_KV) {

        // Load K tile
        int kv_tid = kv_start + tid;
        if (kv_tid < seq_len) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                K_lds[tid][d] = bf16_to_float(K[kv_tid * HEAD_DIM + d]);
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                K_lds[tid][d] = 0.0f;
        }
        __syncthreads();

        // QK^T
        float S[TILE_KV];
        #pragma unroll
        for (int j = 0; j < TILE_KV; j++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                dot += Q_lds[tid][d] * K_lds[j][d];
            S[j] = dot * scale;
        }

        // Online softmax
        float m_new = m_i;
        #pragma unroll
        for (int j = 0; j < TILE_KV; j++)
            m_new = fmaxf(m_new, S[j]);

        float l_new = expf(m_i - m_new) * l_i;
        #pragma unroll
        for (int j = 0; j < TILE_KV; j++)
            l_new += expf(S[j] - m_new);

        // Load V tile
        __syncthreads();
        if (kv_tid < seq_len) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                V_lds[tid][d] = bf16_to_float(V[kv_tid * HEAD_DIM + d]);
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                V_lds[tid][d] = 0.0f;
        }
        __syncthreads();

        // Rescale old accumulator
        float rescale = expf(m_i - m_new);
        #pragma unroll
        for (int d = 0; d < ACC_ELEMS; d++)
            acc[d] *= rescale;

        // PV
        #pragma unroll
        for (int j = 0; j < TILE_KV; j++) {
            float P_ij = expf(S[j] - m_new);
            #pragma unroll
            for (int d = 0; d < ACC_ELEMS; d++)
                acc[d] += P_ij * V_lds[j][tid * ACC_ELEMS + d];
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    // Write O = acc / l_i
    if (gid < seq_len) {
        #pragma unroll
        for (int d = 0; d < ACC_ELEMS; d++)
            O[gid * HEAD_DIM + tid * ACC_ELEMS + d] = float_to_bf16(acc[d] / l_i);
    }
}


// ============================================================
// C-linkage host wrapper（供 Python ctypes 调用）
// ============================================================

extern "C" int dcu_flash_attn_forward(
    void* Q_ptr,
    void* K_ptr,
    void* V_ptr,
    void* O_ptr,
    float scale,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    hipStream_t stream
) {
    if (seq_len <= 0) return -1;

    // Grid: one block per TILE_Q rows
    int grid = (seq_len + TILE_Q - 1) / TILE_Q;
    int block = WAVEFRONT_SIZE;  // one wavefront per block

    dcu_flash_attn_forward_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __hip_bfloat16*>(Q_ptr),
        static_cast<const __hip_bfloat16*>(K_ptr),
        static_cast<const __hip_bfloat16*>(V_ptr),
        static_cast<__hip_bfloat16*>(O_ptr),
        scale, seq_len, num_heads, num_kv_heads, head_dim
    );

    return hipGetLastError() == hipSuccess ? 0 : -1;
}
