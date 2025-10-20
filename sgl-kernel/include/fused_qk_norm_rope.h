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

#pragma once

#include <cuda_runtime.h>

// Perform fused QK RMS normalization and RoPE in a single CUDA kernel.
// qkv: [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim] in bf16.
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
    cudaStream_t stream);

