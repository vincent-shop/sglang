# Fused MoE Triton System Documentation

## Introduction

The fused MoE (Mixture of Experts) Triton implementation in SGLang provides high-performance GPU kernels for executing MoE layers in transformer models. This system handles token routing, expert computation, and result aggregation through optimized Triton kernels that support multiple quantization schemes and hardware backends.

### Major Subsystems

1. **Kernel Layer** (`fused_moe_triton_kernels.py`, `triton_kernels_moe.py`) - Core Triton GPU kernels for matrix multiplication and reduction
2. **Configuration System** (`fused_moe_triton_config.py`) - Performance tuning and kernel parameter selection
3. **High-Level API** (`fused_moe.py`) - Entry points and orchestration for MoE computation
4. **Layer Integration** (`layer.py`) - PyTorch module integration with weight loading and model interfaces
5. **Token Alignment** (`moe_align_block_size.py`) - Token sorting and padding for efficient block-wise computation

---

## Architecture Overview

### Data Flow

```
Input Tokens (hidden_states)
    ↓
[Token Sorting & Alignment] ← moe_align_block_size
    ↓
[GEMM1: tokens × w1] ← fused_moe_kernel
    ↓
[Activation (SiLU/GELU)] ← silu_and_mul / gelu_and_mul
    ↓
[GEMM2: intermediate × w2] ← fused_moe_kernel
    ↓
[Reduction across experts] ← moe_sum_reduce_triton
    ↓
Output Tokens
```

### Key Data Structures

**TopK Output** (`topk_output: StandardTopKOutput`)
- `topk_weights`: Expert routing weights per token, shape `[num_tokens, top_k]`
- `topk_ids`: Selected expert indices per token, shape `[num_tokens, top_k]`
- Used to determine which experts process which tokens

**Sorted Token Mapping**
- `sorted_token_ids`: Token indices sorted by expert assignment
- `expert_ids`: Expert ID for each block of tokens
- `num_tokens_post_padded`: Total tokens after padding to block size

**Weight Tensors**
- `w1`: First expert weights (gate/up projection), shape `[E, N, K]`
- `w2`: Second expert weights (down projection), shape `[E, N', K']`
- Stored in transposed format for Triton kernels: `[E, K, N]` layout

### Quantization Support

The system supports multiple quantization schemes:

1. **FP8 W8A8** (8-bit floating point weights and activations)
   - Tensor-wise, channel-wise, or block-wise scaling
   - Dynamic per-token activation quantization
   - Static or dynamic weight scales

2. **INT8 W8A8** (8-bit integer weights and activations)
   - Channel-wise or block-wise quantization
   - Per-token activation scaling

3. **INT8 W8A16** (8-bit weights, 16-bit activations)
   - GPTQ/AWQ-style quantization
   - Per-channel weight scales with optional zero points
   - Group quantization support

4. **INT4 W4A16** (4-bit weights, 16-bit activations)
   - Two weights packed per byte
   - Group quantization with scales and zero points

---

## Component Analysis

### 1. Token Alignment (`moe_align_block_size`)

**Location**: `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py:17`

**Purpose**: Reorganizes tokens for efficient block matrix multiplication by sorting tokens by assigned expert and padding to ensure divisibility by block size.

**Algorithm**:
```python
def moe_align_block_size(topk_ids, block_size, num_experts):
    # Input: topk_ids [num_tokens, top_k]
    # Each token repeated top_k times, one per expert assignment

    # 1. Allocate output buffers
    max_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(max_tokens_padded, dtype=int32)
    expert_ids = torch.empty(triton.cdiv(max_tokens_padded, block_size), dtype=int32)

    # 2. Call CUDA kernel (sgl_moe_align_block_size)
    # Kernel performs:
    #   - Count tokens per expert (histogram)
    #   - Compute cumulative sum for expert token offsets
    #   - Pad each expert's token count to block_size
    #   - Sort tokens by expert and write to sorted_ids
    #   - Assign expert IDs to each block

    return sorted_ids, expert_ids, num_tokens_post_padded
```

**Example**:
```python
# Input: 4 tokens, top_k=3, 4 experts, block_size=4
topk_ids = [[2,3,4], [1,2,4], [1,3,4], [1,2,3]]
# Flattened: [2,3,4,1,2,4,1,3,4,1,2,3] (12 tokens)

# Expert token counts: E1=3, E2=3, E3=3, E4=3
# After padding to 4: E1=4, E2=4, E3=4, E4=4 (16 total)

# sorted_ids: [3,6,9,12, 0,4,10,12, 1,7,11,12, 2,5,8,12]
#              └─ E1 ─┘  └─ E2 ──┘  └─ E3 ──┘  └─ E4 ──┘
# Token 12 is padding (ignored in computation)
```

**Performance Considerations**:
- Fuses sorting and padding for tokens ≤ 4096 (`fuse_sorted_ids_padding`)
- Uses cumsum buffer to track expert token offsets
- Padding overhead is amortized across batch size

---

### 2. Fused MoE Kernel (`fused_moe_kernel`)

**Location**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:296`

**Purpose**: Core Triton kernel implementing batched matrix multiplication across experts with quantization support.

**Kernel Structure**:

```python
@triton.jit
def fused_moe_kernel(
    a_ptr,           # Input activations [M, K]
    b_ptr,           # Expert weights [E, N, K] (transposed: [E, K, N])
    c_ptr,           # Output [M, N]
    # ... scale/bias pointers
    # ... stride parameters
    BLOCK_SIZE_M,    # Tile size in M dimension
    BLOCK_SIZE_N,    # Tile size in N dimension
    BLOCK_SIZE_K,    # Tile size in K dimension
    GROUP_SIZE_M,    # Number of M blocks processed together
    # ... quantization flags
):
    # 1. Compute block ID and expert assignment
    pid = tl.program_id(0)
    pid_m, pid_n = compute_2d_block_id(pid, GROUP_SIZE_M)

    # Load expert ID for this block
    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id == -1:
        write_zeros_to_output(...)  # Expert not in this rank
        return

    # 2. Load token IDs for this block
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M : ...)

    # 3. Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)

    # 4. Iterate over K dimension
    for k in range(0, cdiv(K, BLOCK_SIZE_K)):
        # Load activation tile [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a = tl.load(a_ptrs, mask=...)

        # Load weight tile [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b = tl.load(b_ptrs)

        # Dequantize if needed
        if use_fp8_w8a8 or use_int8_w8a8:
            if block_wise_quant:
                # Load per-block scales
                a_scale = tl.load(a_scale_ptrs + block_offset)
                b_scale = tl.load(b_scale_ptrs + block_offset)
                accumulator += tl.dot(a, b) * a_scale * b_scale
            elif channel_wise_quant:
                # Load per-channel scales
                a_scale = tl.load(a_scale_ptrs)[:, None]
                b_scale = tl.load(b_scale_ptrs)[None, :]
                # Applied after accumulation

        # Matrix multiply and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. Apply scales (for quantization)
    if use_int8_w8a16:
        accumulator *= b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not block_wise_quant:
        accumulator *= a_scale * b_scale

    # 6. Apply router weights (optional)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token)
        accumulator *= moe_weight[:, None]

    # 7. Store result
    tl.store(c_ptrs, accumulator.to(compute_type), mask=...)
```

**Grouped Ordering for Data Reuse**:
The kernel uses a grouped 2D tiling strategy to maximize L2 cache hit rate:

```
# Standard row-major ordering (poor L2 reuse):
# Process blocks: (0,0), (0,1), (0,2), ..., (1,0), (1,1), ...

# Grouped ordering (better L2 reuse):
# GROUP_SIZE_M = 2
# Group 0: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2), ...
#          └───────┘  └───────┘  └───────┘
#          Same weight tiles reused
```

**Quantization Paths**:

1. **FP8/INT8 Block-wise** (`block_shape=[block_n, block_k]`):
   ```python
   # Scales loaded per K-block
   k_start = k * BLOCK_SIZE_K
   offs_ks = k_start // block_k
   a_scale = load(a_scale_ptrs + offs_ks)
   b_scale = load(b_scale_ptrs + offs_ks)
   accumulator += dot(a, b) * a_scale[:, None] * b_scale[None, :]
   ```

2. **FP8/INT8 Channel-wise** (`per_channel_quant=True`):
   ```python
   # Scales loaded once per output channel
   b_scale = load(b_scale_ptrs + expert_id * stride_bse + offs_bn)
   a_scale = load(a_scale_ptrs + offs_token)[:, None]
   # Applied after accumulation loop
   accumulator *= a_scale * b_scale
   ```

3. **INT8/INT4 Group-wise** (`block_shape[1] > 0`):
   ```python
   # Uses separate kernel: fused_moe_kernel_gptq_awq
   # Scales loaded per group in K dimension
   group_id = (offs_k + k * BLOCK_SIZE_K) // group_size
   b_scale = load(b_scale_ptrs + group_id)
   b_zp = load(b_zp_ptrs + group_id) if has_zp else constant
   b_dequant = (b.to(float32) - b_zp) * b_scale
   ```

---

### 3. Configuration System (`try_get_optimal_moe_config`)

**Location**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py:172`

**Purpose**: Selects optimal kernel parameters based on problem size, hardware, and quantization scheme.

**Configuration Loading Hierarchy**:

```python
def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M, is_marlin, block_shape):
    # 1. Check for manual override
    override_config = get_config()
    if override_config:
        return override_config

    # 2. Try to load pre-tuned config
    E, _, N = w2_shape
    configs = get_moe_configs(E, N, dtype, block_shape[0], block_shape[1])
    if configs:
        # configs is a dict: {batch_size -> config}
        # Pick closest batch size
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        return config

    # 3. Fall back to heuristic default
    return get_default_config(M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape)
```

**Config File Format**:
```
configs/triton_3_2_0/E=8,N=14336,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
```

Contains:
```json
{
  "1": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
  "128": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
          "GROUP_SIZE_M": 32, "num_warps": 8, "num_stages": 4},
  ...
}
```

**Default Configuration Logic** (`get_default_config:115`):

```python
def get_default_config(M, E, N, K, topk, dtype, is_marlin, block_shape):
    if dtype == "fp8_w8a8":
        if block_shape is None:
            # Tensor/channel-wise quantization
            if M <= E:
                # Small batch: optimize for latency
                return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}
            else:
                # Large batch: optimize for throughput
                return {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
                        "GROUP_SIZE_M": 32, "num_warps": 8, "num_stages": 4}
        else:
            # Block-wise quantization: K must be divisible by block_k
            return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": block_shape[0],
                    "BLOCK_SIZE_K": block_shape[1], "GROUP_SIZE_M": 32,
                    "num_warps": 4, "num_stages": 3}
    else:
        # BF16/FP16
        if M <= E or (is_marlin and M <= 32):
            # Small batch config
            return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1}
        else:
            return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8}
```

**Key Tuning Parameters**:
- `BLOCK_SIZE_M/N/K`: Tile dimensions for matrix multiply
- `GROUP_SIZE_M`: Number of M-blocks processed together (affects L2 reuse)
- `num_warps`: Number of parallel warps (4-16, higher for larger tiles)
- `num_stages`: Software pipelining depth (2-4, enables memory/compute overlap)

---

### 4. High-Level API (`fused_experts`)

**Location**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:244`

**Purpose**: Orchestrates the complete MoE computation including chunking, activation, and result combination.

**Implementation Flow**:

```python
def fused_experts(hidden_states, w1, w2, topk_output, moe_runner_config, ...):
    topk_weights, topk_ids, _ = topk_output

    # Dispatch to inplace or outplace implementation
    if moe_runner_config.inplace:
        torch.ops.sglang.inplace_fused_experts(...)
        return hidden_states
    else:
        return torch.ops.sglang.outplace_fused_experts(...)
```

**Core Implementation** (`fused_experts_impl:339`):

```python
def fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, ...):
    num_tokens, hidden_size = hidden_states.shape
    E, N, _ = w1.shape  # num_experts, intermediate_size, hidden_size

    # 1. Determine compute configuration
    CHUNK_SIZE = 64 * 1024  # Process tokens in chunks to avoid OOM
    M = min(num_tokens, CHUNK_SIZE)
    config = get_config_func(M)

    # 2. Allocate intermediate buffers (reused across chunks)
    cache = torch.empty(M * topk_ids.shape[1] * max(N, w2.shape[1]), ...)
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(M, topk_ids.shape[1], N)
    intermediate_cache2 = torch.empty(M * topk_ids.shape[1], N // 2, ...)
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(...)

    # 3. Allocate output
    if no_combine:
        out = torch.empty(num_tokens, topk_ids.shape[1], w2.shape[1], ...)
    elif inplace:
        out = hidden_states
    else:
        out = torch.empty_like(hidden_states)

    # 4. Process tokens in chunks
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        curr_hidden_states = hidden_states[chunk*CHUNK_SIZE : (chunk+1)*CHUNK_SIZE]
        curr_topk_ids = topk_ids[chunk*CHUNK_SIZE : ...]
        curr_topk_weights = topk_weights[chunk*CHUNK_SIZE : ...]

        # 4a. Sort and pad tokens for this chunk
        sorted_token_ids, expert_ids, num_tokens_post_padded = \
            moe_align_block_size(curr_topk_ids, config["BLOCK_SIZE_M"], E)

        # 4b. First GEMM: hidden_states @ w1 -> intermediate_cache1
        invoke_fused_moe_kernel(
            curr_hidden_states, w1, b1, intermediate_cache1,
            a1_scale, w1_scale, w1_zp,
            curr_topk_weights, curr_topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=apply_router_weight_on_input,
            top_k=topk_ids.shape[1],
            config=config,
            ...
        )
        # Output shape: [tokens_in_chunk, top_k, N]

        # 4c. Activation function
        if activation == "silu":
            if gemm1_alpha is not None:
                # Specialized SwiGLU with clamping
                intermediate_cache2 = swiglu_with_alpha_and_limit(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            else:
                # Standard SiLU: silu(gate) * up
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        elif activation == "gelu":
            gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        # Output shape: [tokens_in_chunk * top_k, N // 2]

        # 4d. Second GEMM: intermediate @ w2 -> intermediate_cache3 or output
        invoke_fused_moe_kernel(
            intermediate_cache2, w2, b2,
            intermediate_cache3 if topk_ids.shape[1] != 1 else out[chunk*CHUNK_SIZE : ...],
            a2_scale, w2_scale, w2_zp,
            curr_topk_weights, curr_topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=not apply_router_weight_on_input,
            top_k=1,  # Each token now has single output per expert
            config=config,
            ...
        )

        # 4e. Combine expert outputs (if top_k > 1)
        if no_combine:
            pass  # Keep separate expert outputs
        elif topk_ids.shape[1] == 1:
            pass  # Already written to output
        elif topk_ids.shape[1] == 2:
            # Optimized path for top_k=2
            torch.add(intermediate_cache3[:, 0], intermediate_cache3[:, 1],
                     out=out[chunk*CHUNK_SIZE : ...])
        else:
            # General reduction
            if tokens_in_chunk <= 32:
                moe_sum_reduce_torch_compile(intermediate_cache3,
                                             out[chunk*CHUNK_SIZE : ...],
                                             routed_scaling_factor)
            else:
                moe_sum_reduce_triton(intermediate_cache3,
                                     out[chunk*CHUNK_SIZE : ...],
                                     routed_scaling_factor)

    return out
```

**Memory Management**:
- **Cache reuse**: Single allocation for `intermediate_cache1` and `intermediate_cache3` via views
- **Chunking**: Processes tokens in 64K chunks to bound memory usage
- **Inplace option**: Writes directly to input buffer when possible

**Router Weight Application**:
Two modes controlled by `apply_router_weight_on_input`:
1. **On input** (`True`): Multiply topk_weights before first GEMM
   - Used when weights should affect intermediate computations
2. **On output** (`False`): Multiply topk_weights after second GEMM (default)
   - More numerically stable, standard MoE behavior

---

### 5. Reduction Kernel (`moe_sum_reduce_triton`)

**Location**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:766`

**Purpose**: Efficiently sums outputs from multiple experts per token.

**Kernel Implementation**:

```python
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,      # [num_tokens, top_k, hidden_dim]
    output_ptr,     # [num_tokens, hidden_dim]
    token_num, topk_num, hidden_dim,
    routed_scaling_factor,  # Optional scaling factor
    BLOCK_M, BLOCK_DIM, NUM_STAGE
):
    # 1. Get block coordinates
    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    # 2. Accumulate across experts
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=float32)
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(input_ptr + offs_token[:, None] * stride_0
                                  + i * stride_1
                                  + offs_dim[None, :],
                      mask=...)
        accumulator += tile.to(float32)

    # 3. Apply scaling factor
    accumulator *= routed_scaling_factor

    # 4. Store result
    tl.store(output_ptr + offs_token[:, None] * stride_0 + offs_dim[None, :],
            accumulator.to(input_dtype), mask=...)
```

**Configuration** (`moe_sum_reduce_triton:766`):
```python
BLOCK_M = 1         # Process 1 token at a time
BLOCK_DIM = 2048    # Process 2048 features at a time
NUM_STAGE = 1       # Pipeline depth
num_warps = 16      # Use 16 warps for high occupancy
```

**Performance Path Selection** (`fused_experts_impl:554`):
```python
if tokens_in_chunk <= 32:
    # Small batch: torch.compile is faster due to fusion
    moe_sum_reduce_torch_compile(...)
else:
    # Large batch: Triton kernel amortizes launch overhead
    moe_sum_reduce_triton(...)
```

---

### 6. Layer Integration (`FusedMoE`)

**Location**: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:102`

**Purpose**: PyTorch `nn.Module` that integrates MoE computation into model architectures.

**Class Structure**:

```python
class FusedMoE(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, layer_id,
                 top_k=None, quant_config=None, ...):
        # 1. Distributed setup
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.moe_ep_rank = get_moe_expert_parallel_rank()
        self.num_local_experts = num_experts // self.moe_ep_size

        # 2. Backend selection
        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernels()
        self.enable_flashinfer_cutlass_moe = ...

        # 3. Quantization method
        self.quant_method = quant_config.get_quant_method(self, prefix) \
                           if quant_config else UnquantizedFusedMoEMethod()

        # 4. Create weight parameters (via quant_method.create_weights)
        # For unquantized:
        #   self.experts.w13_weight: [num_local_experts, 2*intermediate_size, hidden_size]
        #   self.experts.w2_weight: [num_local_experts, hidden_size, intermediate_size]

        # 5. Create MoE runner
        self.quant_method.create_moe_runner(self, moe_runner_config)

        # 6. Create token dispatcher
        self.dispatcher = create_moe_dispatcher(moe_runner_config)

    def forward(self, hidden_states, topk_output):
        # 1. Dispatch tokens to experts (may involve all2all)
        dispatch_output = self.dispatcher.dispatch(hidden_states, topk_output)

        # 2. Run MoE computation
        combine_input = self.run_moe_core(dispatch_output)

        # 3. Combine results (may involve all2all)
        final_hidden_states = self.dispatcher.combine(combine_input)

        # 4. Reduce across tensor parallel ranks if needed
        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    def run_moe_core(self, dispatch_output):
        return self.quant_method.apply(layer=self, dispatch_output=dispatch_output)
```

**Weight Loading** (`weight_loader:472`):

The weight loader handles various quantization formats and expert parallelism:

```python
def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
    # shard_id: "w1" (gate), "w2" (down), "w3" (up)
    # expert_id: global expert index (0 to num_experts-1)

    # 1. Map global to local expert
    local_expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
    if local_expert_id == -1:
        return  # This expert not in current EP rank

    # 2. Handle different weight types
    if "input_scale" in weight_name:
        # FP8 input scales
        self._load_single_value(param, loaded_weight, local_expert_id)

    elif "scale" in weight_name:
        # Weight scales for quantization
        quant_method = getattr(param, "quant_method", None)
        if quant_method == "channel":
            self._load_per_channel_weight_scale(...)
        elif quant_method in ["group", "block"]:
            self._load_model_weight_or_group_weight_scale(...)
        elif quant_method == "tensor":
            self._load_per_tensor_weight_scale(...)

    elif "weight" in weight_name:
        # Model weights
        expert_data = param.data[local_expert_id]

        # Determine shard dimension (TP sharding)
        shard_dim = {"w1": 0, "w2": 1, "w3": 0}[shard_id]
        if self.use_triton_kernels:
            # Triton expects transposed layout
            shard_dim = int(not shard_dim)

        # Load and shard
        if shard_id == "w2":
            self._load_w2(expert_data, shard_dim, loaded_weight, tp_rank)
        elif shard_id in ("w1", "w3"):
            self._load_w13(expert_data, shard_dim, loaded_weight, tp_rank)
```

**Weight Layout for Triton**:

Standard PyTorch layout:
- `w1/w3` (up/gate): `[num_experts, intermediate_size, hidden_size]`
- `w2` (down): `[num_experts, hidden_size, intermediate_size]`

Triton kernel layout (transposed):
- `w1/w3`: `[num_experts, hidden_size, intermediate_size]`
- `w2`: `[num_experts, intermediate_size, hidden_size]`

Transposition happens during weight loading (`_load_w13:357`, `_load_w2:426`):
```python
if self.use_triton_kernels:
    loaded_weight = loaded_weight.transpose(-2, -1)
```

---

### 7. Alternative Triton Implementation (`triton_kernels_moe.py`)

**Location**: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`

**Purpose**: Newer Triton kernel implementation using the `triton_kernels` library for matmul with routing.

**Key Differences from `fused_moe_triton_kernels.py`**:

1. **Routing Integration**: Uses `RoutingData`, `GatherIndx`, `ScatterIndx` primitives
2. **matmul_ogs Kernel**: Optimized grouped scatter kernel from `triton_kernels` library
3. **Simplified Interface**: Abstracts token sorting and expert assignment

**Implementation**:

```python
def triton_kernel_fused_experts(hidden_states, w1, w2, routing_data,
                                gather_indx, scatter_indx, ...):
    # 1. First GEMM with gather (input routing)
    intermediate_cache1 = matmul_ogs(
        hidden_states,
        w1,
        bias=None,
        routing_data=routing_data,
        gather_indx=gather_indx,  # Routes tokens to experts
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
    )
    # matmul_ogs performs:
    #   - Token gathering based on gather_indx
    #   - Batched GEMM across experts
    #   - Optional gamma (weight) scaling

    # 2. Activation
    if activation == "silu":
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
    elif activation == "gelu":
        gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

    # 3. Second GEMM with scatter (output routing)
    intermediate_cache3 = matmul_ogs(
        intermediate_cache2,
        w2,
        bias=None,
        routing_data=routing_data,
        scatter_indx=scatter_indx,  # Routes outputs back to tokens
        gammas=None if apply_router_weight_on_input else routing_data.gate_scal,
    )
    # matmul_ogs performs:
    #   - Batched GEMM across experts
    #   - Output scattering and accumulation
    #   - Optional gamma scaling

    return intermediate_cache3
```

**RoutingData Structure**:
```python
@dataclass
class RoutingData:
    n_expts_act: int           # Number of experts to activate per token
    gate_scal: torch.Tensor    # Router weights [num_tokens * n_expts_act]
    # ... other routing metadata
```

**Limitations** (as of current implementation):
- No quantization support (`use_fp8_w8a8=False`, `block_shape=None`)
- BF16 only (`assert hidden_states.dtype == torch.bfloat16`)
- No inplace operation

---

## Advanced Topics

### Expert Parallelism (EP)

The system supports distributing experts across multiple GPUs:

**Setup** (`layer.py:165`):
```python
self.moe_ep_size = get_moe_expert_parallel_world_size()
self.moe_ep_rank = get_moe_expert_parallel_rank()
self.num_local_experts = num_experts // self.moe_ep_size
```

**Weight Loading** (`weight_loader:464`):
```python
def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
    start_idx = self.moe_ep_rank * self.num_local_experts
    end_idx = (self.moe_ep_rank + 1) * self.num_local_experts
    if start_idx <= expert_id < end_idx:
        return expert_id - start_idx
    else:
        return -1  # Expert not in this rank
```

**Kernel Handling** (`fused_moe_kernel:404`):
```python
expert_id = tl.load(expert_ids_ptr + pid_m)
if expert_id == -1:
    # Expert filtered out by EP
    write_zeros_to_output(...)
    return
```

### Tensor Parallelism (TP)

Experts can be sharded across tensor parallel ranks:

**Weight Sharding**:
- `w1/w3` (gate/up): Shard along output dimension (intermediate_size)
- `w2` (down): Shard along input dimension (intermediate_size)

**All-Reduce** (`layer.py:851`):
```python
if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

### FP8 Dynamic Quantization

The system performs per-token FP8 quantization at runtime:

**Activation Quantization** (`invoke_fused_moe_kernel:571`):
```python
if use_fp8_w8a8:
    if block_shape is None:
        # Per-token quantization
        A, A_scale = scaled_fp8_quant(A, A_scale,
                                     use_per_token_if_dynamic=per_channel_quant)
    else:
        # Block-wise quantization
        A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
```

**Dequantization in Kernel** (`fused_moe_kernel:502`):
```python
# Channel-wise dequantization
accumulator = tl.dot(a, b, acc=accumulator)  # FP8 -> FP32 accumulator
# ...after K loop
accumulator *= a_scale * b_scale  # Scale to original range
```

### Optimization: No-Combine Mode

When `no_combine=True`, expert outputs are kept separate:

**Use Case**: Allows downstream processing of individual expert outputs

**Implementation** (`fused_experts_impl:428`):
```python
if no_combine:
    out_hidden_states = torch.empty(
        (num_tokens, topk_ids.shape[1], w2.shape[1]), ...
    )
    # Skip reduction, return [num_tokens, top_k, hidden_size]
```

### Chunking Strategy

To handle large batches without OOM:

**Chunk Size** (`fused_experts_impl:388`):
```python
CHUNK_SIZE = 64 * 1024
M = min(num_tokens, CHUNK_SIZE)
```

**Dynamic Config Update** (`fused_experts_impl:451`):
```python
if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
    # Last chunk may be smaller
    # Recompute config for actual chunk size
    config = get_config_func(tokens_in_chunk)
    # Resize intermediate buffers
    intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
    ...
```

---

## Performance Characteristics

### Key Optimization Techniques

1. **Grouped Tiling** (`fused_moe_kernel:376-386`)
   - Processes multiple M-blocks for same N-block
   - Maximizes weight reuse in L2 cache

2. **Software Pipelining** (`num_stages` parameter)
   - Overlaps memory loads with computation
   - Typical values: 2-4 stages

3. **Quantization-Aware Layout**
   - Block-wise quantization aligns scales with tile boundaries
   - `BLOCK_SIZE_K` divisible by `block_k`

4. **Zero-Copy for Small Batches**
   - Direct write to output for `top_k=1` case
   - Avoids reduction overhead

5. **Hybrid Reduction**
   - Torch.compile for small batches (≤32 tokens)
   - Triton kernel for large batches

### Bottlenecks and Tuning

**Memory Bandwidth Limited**: FP16/BF16 computation
- Increase tile sizes (`BLOCK_SIZE_M/N/K`)
- Reduce `num_stages` to minimize register pressure

**Compute Limited**: Quantized computation
- Smaller tile sizes may improve occupancy
- Higher `num_warps` for compute-heavy workloads

**Launch Overhead**: Small batch inference
- Use larger `BLOCK_SIZE_M` to reduce grid size
- Consider fused activation paths

---

## Usage Examples

### Basic Usage

```python
from sglang.srt.layers.moe.fused_moe_triton import fused_experts
from sglang.srt.layers.moe import MoeRunnerConfig

# Setup
hidden_states = torch.randn(512, 4096, dtype=torch.bfloat16, device='cuda')
w1 = torch.randn(8, 14336, 4096, dtype=torch.bfloat16, device='cuda')  # 8 experts
w2 = torch.randn(8, 4096, 14336, dtype=torch.bfloat16, device='cuda')
topk_weights = torch.randn(512, 2, device='cuda')  # top-2 routing
topk_ids = torch.randint(0, 8, (512, 2), device='cuda')

config = MoeRunnerConfig(
    num_experts=8,
    top_k=2,
    activation="silu",
    inplace=False,
)

# Run
output = fused_experts(
    hidden_states=hidden_states,
    w1=w1,
    w2=w2,
    topk_output=(topk_weights, topk_ids, None),
    moe_runner_config=config,
)
```

### FP8 Quantization

```python
# Prepare quantized weights
w1_fp8 = w1.to(torch.float8_e4m3fn)
w2_fp8 = w2.to(torch.float8_e4m3fn)

# Per-tensor scales
w1_scale = torch.tensor([0.01], device='cuda').expand(8)
w2_scale = torch.tensor([0.02], device='cuda').expand(8)

output = fused_experts(
    hidden_states=hidden_states,
    w1=w1_fp8,
    w2=w2_fp8,
    topk_output=(topk_weights, topk_ids, None),
    moe_runner_config=config,
    use_fp8_w8a8=True,
    w1_scale=w1_scale,
    w2_scale=w2_scale,
)
```

### Block-wise Quantization

```python
block_shape = [128, 128]  # [block_n, block_k]

# Block-wise scales: [num_experts, cdiv(N, block_n), cdiv(K, block_k)]
w1_scale = torch.randn(8, 14336//128, 4096//128, device='cuda')
w2_scale = torch.randn(8, 4096//128, 14336//128, device='cuda')

output = fused_experts(
    hidden_states=hidden_states,
    w1=w1_fp8,
    w2=w2_fp8,
    topk_output=(topk_weights, topk_ids, None),
    moe_runner_config=config,
    use_fp8_w8a8=True,
    w1_scale=w1_scale,
    w2_scale=w2_scale,
    block_shape=block_shape,
)
```

### Custom Configuration

```python
from sglang.srt.layers.moe.fused_moe_triton import override_config

custom_config = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 16,
    "num_warps": 8,
    "num_stages": 4,
}

with override_config(custom_config):
    output = fused_experts(...)
```

---

## Configuration Tuning Guide

### Tuning Process

1. **Benchmark Baseline**:
   ```bash
   python benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py
   ```

2. **Profile Bottlenecks**:
   ```bash
   nsys profile --stats=true python your_script.py
   ```

3. **Tune Parameters**:
   ```bash
   python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
       --num_experts 8 --intermediate_size 14336 --dtype fp8_w8a8
   ```

4. **Save Optimal Config**:
   Configs saved to `configs/triton_{version}/E={E},N={N},...json`

### Parameter Guidelines

**BLOCK_SIZE_M** (Token tile size):
- Small (16-64): Better for low batch sizes
- Large (128-256): Better for high batch sizes
- Must divide into sorted token count efficiently

**BLOCK_SIZE_N** (Output feature tile size):
- Small (32-64): Lower register pressure
- Large (128-256): Better memory coalescing
- Should align with hardware memory transaction size (128 bytes)

**BLOCK_SIZE_K** (Input feature tile size):
- Small (32-64): More flexible scheduling
- Large (128-256): Better dot product efficiency
- Must be divisible by quantization block size

**GROUP_SIZE_M**:
- Small (1-8): Less L2 reuse, better load balance
- Large (16-32): More L2 reuse, potential tail effects
- Set to 1 for small batch inference

**num_warps**:
- 4: Small tiles (< 64x64)
- 8: Medium tiles (64x128 to 128x256)
- 16: Large tiles (> 128x256)

**num_stages**:
- 2: Low register pressure, simpler scheduling
- 3-4: Better memory/compute overlap, higher registers

---

## Debugging and Troubleshooting

### Common Issues

**1. Shape Mismatch Errors**

```python
AssertionError: hidden_states shape[-1] must equal w1 shape[2]
```

**Cause**: Weight dimensions inconsistent with hidden size
**Fix**: Verify weight shapes match expected `[E, N, K]` layout (transposed for Triton)

**2. Quantization Scale Shape Errors**

```python
AssertionError: triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
```

**Cause**: Scale tensor dimensions don't match quantization block configuration
**Fix**: Ensure scale shape is `[num_tokens, cdiv(K, block_k)]` for block-wise quant

**3. Expert Filtering in EP Mode**

Symptom: All zeros output for some experts

**Cause**: Expert not assigned to current EP rank
**Debug**:
```python
print(f"EP rank {self.moe_ep_rank}, local experts {self.num_local_experts}")
print(f"Global expert {expert_id}, local {self._map_global_expert_id_to_local_expert_id(expert_id)}")
```

**4. OOM Errors**

**Cause**: Batch size too large for available memory
**Fix**: Decrease `CHUNK_SIZE` in `fused_experts_impl:388`:
```python
CHUNK_SIZE = 32 * 1024  # Reduce from 64K
```

### Debugging Tools

**Enable Triton Kernel Logging**:
```python
import os
os.environ["TRITON_INTERPRET"] = "1"  # Run in interpreter mode
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # Show autotuning
```

**Profile Memory**:
```python
torch.cuda.memory._record_memory_history(enabled='all')
# ... run code
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
```

**Validate Kernel Output**:
```python
# Compare against reference implementation
output_ref = reference_fused_moe(hidden_states, w1, w2, topk_output)
output_triton = fused_experts(...)
torch.testing.assert_close(output_triton, output_ref, rtol=1e-2, atol=1e-2)
```

---

## Future Improvements

### Planned Enhancements

1. **Async Pipelining**: Overlap token dispatch with computation
2. **Persistent Kernels**: Keep experts resident on GPU across calls
3. **Mixed-Precision Accumulation**: FP32 accumulation for FP16/BF16 inputs
4. **Adaptive Quantization**: Dynamic selection of quantization granularity
5. **Multi-GPU Optimization**: Optimized all2all for expert parallelism

### Experimental Features

- **Grouped Query Experts**: Support for MoE with grouped attention patterns
- **Sparse Expert Weights**: Structured sparsity within expert weights
- **Expert Quantization Heterogeneity**: Different quantization per expert

---

## References

### Key Files

- `fused_moe_triton_kernels.py:296` - `fused_moe_kernel`: Main computation kernel
- `fused_moe_triton_kernels.py:715` - `_moe_sum_reduce_kernel`: Expert output reduction
- `fused_moe.py:339` - `fused_experts_impl`: Orchestration logic
- `fused_moe_triton_config.py:172` - `try_get_optimal_moe_config`: Config selection
- `layer.py:102` - `FusedMoE`: PyTorch module integration
- `moe_align_block_size.py:17` - Token sorting and padding

### External Dependencies

- **Triton**: GPU kernel language and compiler
- **sgl_kernel**: CUDA kernels for activation functions and utilities
- **flashinfer**: Optional high-performance MoE kernels (CUTLASS-based)

### Related Documentation

- Triton documentation: https://triton-lang.org/
- vLLM MoE implementation: https://github.com/vllm-project/vllm
- MoE tuning guide: `benchmark/kernels/fused_moe_triton/README.md`
