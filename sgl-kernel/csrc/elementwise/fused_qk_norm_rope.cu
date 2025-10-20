/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "fused_qk_norm_rope.h"
#include "utils.h"

namespace {

template <int VecSize>
struct PackedUint;

template <>
struct PackedUint<1> {
  using type = uint32_t;
};

template <>
struct PackedUint<2> {
  using type = uint2;
};

template <>
struct PackedUint<4> {
  using type = uint4;
};

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

inline constexpr int div_up(int a, int b) {
  return (a + b - 1) / b;
}

template <int HeadDim, bool Interleave>
__global__ void fused_qk_norm_rope_kernel(
    __nv_bfloat16* qkv,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    float base,
    const int* position_ids,
    int num_tokens,
    float factor,
    float low,
    float high,
    float attention_factor) {
  int const warps_per_block = blockDim.x / 32;
  int const warp_id = threadIdx.x / 32;
  int const lane_id = threadIdx.x % 32;

  int const global_warp_idx = blockIdx.x * warps_per_block + warp_id;
  int const total_qk_heads = num_heads_q + num_heads_k;

  int const token_idx = global_warp_idx / total_qk_heads;
  int const local_head_idx = global_warp_idx % total_qk_heads;

  if (token_idx >= num_tokens) {
    return;
  }

  bool const is_q = local_head_idx < num_heads_q;
  int const head_idx = is_q ? local_head_idx : local_head_idx - num_heads_q;
  int const num_heads = num_heads_q + num_heads_k + num_heads_v;

  static_assert(
      HeadDim % (32 * 2) == 0,
      "HeadDim must be divisible by 64 so each thread handles an even number of elements.");
  constexpr int elements_per_thread = HeadDim / 32;
  float elements[elements_per_thread];
  constexpr int element_bytes = elements_per_thread * sizeof(__nv_bfloat16);
  static_assert(element_bytes % 4 == 0, "Packed vector size must be a multiple of 4 bytes.");
  constexpr int vec_size = element_bytes / 4;
  using VecT = typename PackedUint<vec_size>::type;

  int offset_warp;
  if (is_q) {
    offset_warp = token_idx * num_heads * HeadDim + head_idx * HeadDim;
  } else {
    offset_warp = token_idx * num_heads * HeadDim + num_heads_q * HeadDim + head_idx * HeadDim;
  }
  int const offset_thread = offset_warp + lane_id * elements_per_thread;

  float sum_squares = 0.0f;
  {
    VecT vec = *reinterpret_cast<const VecT*>(&qkv[offset_thread]);
    for (int i = 0; i < vec_size; ++i) {
      auto* raw = reinterpret_cast<const uint32_t*>(&vec) + i;
      float2 vals = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(raw));
      sum_squares += vals.x * vals.x;
      sum_squares += vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  sum_squares = warp_reduce_sum(sum_squares);
  float rms_rcp = rsqrtf(sum_squares / static_cast<float>(HeadDim) + eps);

  for (int i = 0; i < elements_per_thread; ++i) {
    int dim = lane_id * elements_per_thread + i;
    float weight = is_q ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
    elements[i] *= rms_rcp * weight;
  }

  float rotated[elements_per_thread];
  float cos_vals[elements_per_thread];
  float sin_vals[elements_per_thread];
  float const pos_id = static_cast<float>(position_ids[token_idx]);

  if constexpr (Interleave) {
    for (int i = 0; i < elements_per_thread; ++i) {
      if ((i & 1) == 0) {
        rotated[i] = -elements[i + 1];
      } else {
        rotated[i] = elements[i - 1];
      }

      int const dim_idx = lane_id * elements_per_thread + i;
      int const half_dim = dim_idx / 2;
      float freq = powf(base, -2.0f * half_dim / static_cast<float>(HeadDim));

      if (factor != 1.0f) {
        float inv_freq_extrapolation = freq;
        float inv_freq_interpolation = freq / factor;

        if (fabsf(low - high) <= 1e-6f) {
          high += 0.001f;
        }
        float linear_func = (static_cast<float>(half_dim) - low) / (high - low);
        float ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
        float inv_freq_extrapolation_factor = 1.0f - ramp_func;
        freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor;
      }

      float theta = pos_id * freq;
      __sincosf(theta, &sin_vals[i], &cos_vals[i]);
    }
  } else {
    __syncwarp();
    for (int i = 0; i < elements_per_thread; ++i) {
      rotated[i] = __shfl_xor_sync(0xffffffff, elements[i], 16);
      if (lane_id < 16) {
        rotated[i] = -rotated[i];
      }

      int dim_idx = lane_id * elements_per_thread + i;
      dim_idx = (dim_idx * 2) % HeadDim;
      int half_dim = dim_idx / 2;
      float freq = powf(base, -2.0f * half_dim / static_cast<float>(HeadDim));

      if (factor != 1.0f) {
        float inv_freq_extrapolation = freq;
        float inv_freq_interpolation = freq / factor;

        if (fabsf(low - high) <= 1e-6f) {
          high += 0.001f;
        }
        float linear_func = (static_cast<float>(half_dim) - low) / (high - low);
        float ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
        float inv_freq_extrapolation_factor = 1.0f - ramp_func;
        freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor;
      }

      float theta = pos_id * freq;
      __sincosf(theta, &sin_vals[i], &cos_vals[i]);
    }
    __syncwarp();
  }

  for (int i = 0; i < elements_per_thread; ++i) {
    elements[i] = (elements[i] * cos_vals[i] + rotated[i] * sin_vals[i]) * attention_factor;
  }

  {
    VecT vec;
    for (int i = 0; i < vec_size; ++i) {
      auto* raw = reinterpret_cast<uint32_t*>(&vec) + i;
      __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
      *reinterpret_cast<__nv_bfloat162*>(raw) = packed;
    }
    auto* output_ptr = reinterpret_cast<VecT*>(&qkv[offset_thread]);
    *output_ptr = vec;
  }
}

#define DISPATCH_INTERLEAVE(flag, NAME, ...) \
  if (flag) {                                \
    constexpr bool NAME = true;              \
    __VA_ARGS__                              \
  } else {                                   \
    constexpr bool NAME = false;             \
    __VA_ARGS__                              \
  }

}  // namespace

void launchFusedQKNormRope(
    void* qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int head_dim,
    float eps,
    const void* q_weight,
    const void* k_weight,
    float base,
    bool interleave,
    const int* position_ids,
    float factor,
    float low,
    float high,
    float attention_factor,
    cudaStream_t stream) {
  if (factor == 1.0f) {
    TORCH_CHECK(
        attention_factor == 1.0f,
        "When rope scaling factor is 1.0, attention_factor must also be 1.0 to keep numerical parity.");
  }

  constexpr int block_size = 256;
  int const warps_per_block = block_size / 32;
  int const total_qk_heads = num_heads_q + num_heads_k;
  int const total_warps = num_tokens * total_qk_heads;
  int const grid_size = div_up(total_warps, warps_per_block);

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);

  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, kInterleave, {
        fused_qk_norm_rope_kernel<64, kInterleave><<<grid_dim, block_dim, 0, stream>>>(
            static_cast<__nv_bfloat16*>(qkv),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            eps,
            static_cast<const __nv_bfloat16*>(q_weight),
            static_cast<const __nv_bfloat16*>(k_weight),
            base,
            position_ids,
            num_tokens,
            factor,
            low,
            high,
            attention_factor);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, kInterleave, {
        fused_qk_norm_rope_kernel<128, kInterleave><<<grid_dim, block_dim, 0, stream>>>(
            static_cast<__nv_bfloat16*>(qkv),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            eps,
            static_cast<const __nv_bfloat16*>(q_weight),
            static_cast<const __nv_bfloat16*>(k_weight),
            base,
            position_ids,
            num_tokens,
            factor,
            low,
            high,
            attention_factor);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, kInterleave, {
        fused_qk_norm_rope_kernel<256, kInterleave><<<grid_dim, block_dim, 0, stream>>>(
            static_cast<__nv_bfloat16*>(qkv),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            eps,
            static_cast<const __nv_bfloat16*>(q_weight),
            static_cast<const __nv_bfloat16*>(k_weight),
            base,
            position_ids,
            num_tokens,
            factor,
            low,
            high,
            attention_factor);
      });
      break;
    default:
      TORCH_CHECK(
          false, "Unsupported head dimension for fused_qk_norm_rope kernel: ", head_dim, ". Expected 64, 128, or 256.");
  }
  CHECK_CUDA_SUCCESS(cudaGetLastError());
}

void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    bool is_neox,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(position_ids);

  TORCH_CHECK(qkv.dim() == 2, "fused_qk_norm_rope expects qkv to be rank-2 [tokens, hidden].");
  TORCH_CHECK(
      qkv.scalar_type() == at::kBFloat16,
      "fused_qk_norm_rope currently supports qkv tensors in bfloat16 for best performance.");
  TORCH_CHECK(
      q_weight.scalar_type() == at::kBFloat16 && k_weight.scalar_type() == at::kBFloat16,
      "fused_qk_norm_rope expects q/k RMSNorm weights in bfloat16.");
  TORCH_CHECK(
      position_ids.scalar_type() == at::kInt,
      "fused_qk_norm_rope expects position_ids to be int32 tensor (torch.int32).");

  auto device = qkv.device();
  TORCH_CHECK(q_weight.device() == device, "q_weight must be on the same device as qkv.");
  TORCH_CHECK(k_weight.device() == device, "k_weight must be on the same device as qkv.");
  TORCH_CHECK(position_ids.device() == device, "position_ids must be on the same device as qkv.");

  TORCH_CHECK(position_ids.dim() == 1, "fused_qk_norm_rope expects position_ids to be 1D.");

  int64_t const num_tokens = qkv.size(0);
  int64_t const hidden_size = qkv.size(1);
  int64_t const expected_hidden = (num_heads_q + num_heads_k + num_heads_v) * head_dim;
  TORCH_CHECK(
      hidden_size == expected_hidden,
      "fused_qk_norm_rope expects qkv hidden dimension ",
      expected_hidden,
      ", but got ",
      hidden_size,
      ".");
  TORCH_CHECK(
      position_ids.size(0) == num_tokens,
      "fused_qk_norm_rope expects position_ids to have length ",
      num_tokens,
      ", but got ",
      position_ids.size(0),
      ".");
  TORCH_CHECK(q_weight.numel() == head_dim, "q_weight must have ", head_dim, " elements.");
  TORCH_CHECK(k_weight.numel() == head_dim, "k_weight must have ", head_dim, " elements.");

  TORCH_CHECK(
      head_dim == 64 || head_dim == 128 || head_dim == 256,
      "fused_qk_norm_rope supports head_dim values 64, 128, or 256. Got ",
      head_dim,
      ".");

  bool const interleave = !is_neox;
  auto stream = at::cuda::getCurrentCUDAStream();
  launchFusedQKNormRope(
      qkv.data_ptr(),
      static_cast<int>(num_tokens),
      static_cast<int>(num_heads_q),
      static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v),
      static_cast<int>(head_dim),
      static_cast<float>(eps),
      q_weight.data_ptr(),
      k_weight.data_ptr(),
      static_cast<float>(base),
      interleave,
      position_ids.data_ptr<int>(),
      static_cast<float>(factor),
      static_cast<float>(low),
      static_cast<float>(high),
      static_cast<float>(attention_factor),
      stream);
}

