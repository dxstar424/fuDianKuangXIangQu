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
  for (int index = tid; index < ROWS_PER_BLOCK * GROUPS;
       index += BLOCK_THREADS) {
    const int row_offset = index / GROUPS;
    const int group = index % GROUPS;
    const int row = row_base + row_offset;
    scale_smem[index] =
        row < m ? scale[static_cast<int64_t>(row) * GROUPS + group]
                : __float2half(1.0f);
  }
  __syncthreads();

  float accumulators[ROWS_PER_BLOCK] = {};
#pragma unroll
  for (int row_offset = 0; row_offset < ROWS_PER_BLOCK; ++row_offset) {
    const int row = row_base + row_offset;
    if (row < m) {
      const uint8_t* row_weight =
          weight + static_cast<int64_t>(row) * PACKED_K;
      for (int k = tid; k < K; k += BLOCK_THREADS) {
        const uint8_t byte = row_weight[k >> 1];
        const int quant = decode_int4(byte, k & 1);
        const float group_scale =
            __half2float(scale_smem[row_offset * GROUPS + (k >> 5)]);
        accumulators[row_offset] += static_cast<float>(quant) * group_scale *
                                    bf16_to_float(input_smem[k]);
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
      output[row] = float_to_bf16(total);
    }
  }
}

__global__ void w8_dequant_kernel(
    const int8_t* weight,
    const float* scale,
    __hip_bfloat16* output,
    int64_t total,
    int k) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    const int row = static_cast<int>(index / k);
    output[index] =
        float_to_bf16(static_cast<float>(weight[index]) * scale[row]);
  }
}

__global__ void w4_dequant_kernel(
    const uint8_t* weight,
    const __half* scale,
    __hip_bfloat16* output,
    int64_t total,
    int k) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    const int row = static_cast<int>(index / k);
    const int column = static_cast<int>(index % k);
    const int groups = k / 32;
    const uint8_t byte =
        weight[static_cast<int64_t>(row) * (k / 2) + (column >> 1)];
    const int quant = decode_int4(byte, column & 1);
    const float group_scale = __half2float(
        scale[static_cast<int64_t>(row) * groups + (column >> 5)]);
    output[index] =
        float_to_bf16(static_cast<float>(quant) * group_scale);
  }
}

int last_error() {
  const hipError_t error = hipGetLastError();
  return error == hipSuccess ? 0 : static_cast<int>(error);
}

template <int K>
int launch_w8(
    const int8_t* weight,
    const float* scale,
    const __hip_bfloat16* input,
    __hip_bfloat16* output,
    int m,
    hipStream_t stream) {
  w8a16_gemv_kernel<K>
      <<<(m + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, BLOCK_THREADS,
         K * sizeof(__hip_bfloat16), stream>>>(weight, scale, input, output, m);
  return last_error();
}

template <int K>
int launch_w4(
    const uint8_t* weight,
    const __half* scale,
    const __hip_bfloat16* input,
    __hip_bfloat16* output,
    int m,
    hipStream_t stream) {
  constexpr int shared_bytes = K * sizeof(__hip_bfloat16) +
                               ROWS_PER_BLOCK * (K / 32) * sizeof(__half);
  w4a16_gemv_kernel<K>
      <<<(m + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, BLOCK_THREADS,
         shared_bytes, stream>>>(weight, scale, input, output, m);
  return last_error();
}

}  // namespace

extern "C" int fdu_gfx936_w8a16_gemv(
    const void* weight,
    const void* scale,
    const void* input,
    void* output,
    int m,
    int k,
    hipStream_t stream) {
  if (!weight || !scale || !input || !output || m <= 0 || k <= 0) return -1;
  switch (k) {
    case 5120:
      return launch_w8<5120>(static_cast<const int8_t*>(weight),
                             static_cast<const float*>(scale),
                             static_cast<const __hip_bfloat16*>(input),
                             static_cast<__hip_bfloat16*>(output), m, stream);
    case 6144:
      return launch_w8<6144>(static_cast<const int8_t*>(weight),
                             static_cast<const float*>(scale),
                             static_cast<const __hip_bfloat16*>(input),
                             static_cast<__hip_bfloat16*>(output), m, stream);
    case 17408:
      return launch_w8<17408>(static_cast<const int8_t*>(weight),
                              static_cast<const float*>(scale),
                              static_cast<const __hip_bfloat16*>(input),
                              static_cast<__hip_bfloat16*>(output), m, stream);
    default:
      return -2;
  }
}

extern "C" int fdu_gfx936_w4a16_gemv(
    const void* weight,
    const void* scale,
    const void* input,
    void* output,
    int m,
    int k,
    hipStream_t stream) {
  if (!weight || !scale || !input || !output || m <= 0 || k <= 0) return -1;
  switch (k) {
    case 5120:
      return launch_w4<5120>(static_cast<const uint8_t*>(weight),
                             static_cast<const __half*>(scale),
                             static_cast<const __hip_bfloat16*>(input),
                             static_cast<__hip_bfloat16*>(output), m, stream);
    case 17408:
      return launch_w4<17408>(static_cast<const uint8_t*>(weight),
                              static_cast<const __half*>(scale),
                              static_cast<const __hip_bfloat16*>(input),
                              static_cast<__hip_bfloat16*>(output), m, stream);
    default:
      return -2;
  }
}

extern "C" int fdu_gfx936_w8_dequant(
    const void* weight,
    const void* scale,
    void* output,
    int m,
    int k,
    hipStream_t stream) {
  if (!weight || !scale || !output || m <= 0 || k <= 0) return -1;
  const int64_t total = static_cast<int64_t>(m) * k;
  w8_dequant_kernel
      <<<static_cast<unsigned int>((total + BLOCK_THREADS - 1) /
                                   BLOCK_THREADS),
         BLOCK_THREADS, 0, stream>>>(static_cast<const int8_t*>(weight),
                                     static_cast<const float*>(scale),
                                     static_cast<__hip_bfloat16*>(output), total,
                                     k);
  return last_error();
}

extern "C" int fdu_gfx936_w4_dequant(
    const void* weight,
    const void* scale,
    void* output,
    int m,
    int k,
    hipStream_t stream) {
  if (!weight || !scale || !output || m <= 0 || k <= 0 || k % 32 != 0) {
    return -1;
  }
  const int64_t total = static_cast<int64_t>(m) * k;
  w4_dequant_kernel
      <<<static_cast<unsigned int>((total + BLOCK_THREADS - 1) /
                                   BLOCK_THREADS),
         BLOCK_THREADS, 0, stream>>>(static_cast<const uint8_t*>(weight),
                                     static_cast<const __half*>(scale),
                                     static_cast<__hip_bfloat16*>(output), total,
                                     k);
  return last_error();
}
