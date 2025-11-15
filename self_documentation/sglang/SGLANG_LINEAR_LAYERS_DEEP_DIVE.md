# SGLang Linear Layers Deep Dive

This document provides a comprehensive explanation of SGLang's linear layer system, which is fundamental to how SGLang executes large language models with tensor parallelism, quantization, and efficient weight loading.

## Overview

SGLang's linear layer system (in `python/sglang/srt/layers/linear.py`) is a sophisticated abstraction that handles:

1. **Tensor Model Parallelism**: Distributing weights across multiple GPUs
2. **Quantization**: Supporting various quantization methods (FP8, INT4, INT8, AWQ, GPTQ, etc.)
3. **Weight Loading**: Complex weight loading logic for sharded, fused, and quantized weights
4. **Efficient Computation**: Leveraging custom CUDA kernels and optimized operations

The system is adapted from vLLM but with SGLang-specific optimizations and extensions.

## Core Architecture

### Base Classes

#### LinearBase (`linear.py:134`)

The foundation for all linear layers. Key responsibilities:

- Stores layer dimensions (`input_size`, `output_size`)
- Manages data type (`params_dtype`)
- Holds a reference to the quantization method (`quant_method`)
- Provides `skip_bias_add` functionality for fusing bias with subsequent operations

**Quantization Method Assignment**: Every LinearBase layer gets a quantization method. If no quantization is specified, it defaults to `UnquantizedLinearMethod()`. This means ALL linear operations go through the same interface, whether quantized or not.

### Main Linear Layer Types

SGLang provides four main linear layer types, each serving different parallelism strategies:

#### 1. ReplicatedLinear (`linear.py:173`)

**What it does**: Standard linear layer without any tensor parallelism. The weight matrix is fully replicated on each GPU.

**Use case**:
- Small models that fit on a single GPU
- Layers that shouldn't be sharded (rare)

**Weight shape**: `[output_size, input_size]` (full matrix on each device)

**Forward pass**:
```python
Y = XA^T + b  # Standard matrix multiplication
```

**Key characteristics**:
- No communication between GPUs
- Simple weight loading (copy entire weight)
- Most straightforward implementation

#### 2. ColumnParallelLinear (`linear.py:263`)

**What it does**: Implements column parallelism by splitting the weight matrix along the output dimension.

**Mathematical formulation**:
```
Given: Y = XA + b
Where A = [A_1, A_2, ..., A_p]  (split along columns)

Each GPU i computes: Y_i = XA_i
If gather_output=True: Y = AllGather([Y_1, Y_2, ..., Y_p])
If gather_output=False: Return Y_i (remains sharded)
```

**Weight shape per GPU**: `[output_size / tp_size, input_size]`

**Use case**:
- First layer in attention (QKV projection)
- First layer in MLP (gate_up_proj)

**Key parameters**:
- `gather_output`: Whether to all-gather results across GPUs
- `output_sizes`: For fused layers (e.g., QKV), specifies size of each logical matrix

**Weight loading complexity**:
```python
# For rank i, shard offset calculation:
shard_size = param.shape[output_dim]
start_idx = tp_rank * shard_size
loaded_weight = full_weight.narrow(output_dim, start_idx, shard_size)
```

**Special handling**:
- Marlin quantization: Adjusts shard size by `marlin_tile_size`
- BitsAndBytes: Different quantization ratio per shard
- GGUF: Lazy weight materialization
- Pre-sharded weights: Skip narrowing if weights already sharded on disk

#### 3. RowParallelLinear (`linear.py:1194`)

**What it does**: Implements row parallelism by splitting the weight matrix along the input dimension.

**Mathematical formulation**:
```
Given: Y = XA + b
Where:
    A = [A_1; A_2; ...; A_p]  (split along rows)
    X = [X_1, X_2, ..., X_p]  (split along columns)

Each GPU i computes: Y_i = X_i @ A_i
If reduce_results=True: Y = AllReduce(Y_1 + Y_2 + ... + Y_p)
```

**Weight shape per GPU**: `[output_size, input_size / tp_size]`

**Use case**:
- Second layer in attention (output projection)
- Second layer in MLP (down_proj)

**Key parameters**:
- `input_is_parallel`: Expects input already split (common case)
- `reduce_results`: Whether to all-reduce across GPUs

**Communication pattern**:
```python
# Forward pass communication
if not input_is_parallel:
    input_parallel = split_tensor_along_last_dim(input_)

output_parallel = matmul(input_parallel, weight)

if reduce_results and tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

**Optimization**: Uses symmetric memory allocation via `use_symmetric_memory()` context manager for efficient NCCL operations.

#### 4. MergedColumnParallelLinear (`linear.py:445`)

**What it does**: Special case of ColumnParallelLinear where multiple logical matrices are concatenated into one physical matrix.

**Example - MLP layers**:
```
Original: gate_proj and up_proj as separate matrices
Merged: gate_up_proj = concat([gate_proj, up_proj], dim=0)

Shape: [gate_size + up_size, hidden_size]
```

**Why merge**:
- Single GPU kernel call instead of two
- Better memory locality
- Reduced kernel launch overhead

**Weight loading challenges**:

The complexity arises when loading weights from disk that might be:
1. Already fused on disk (e.g., Phi-3)
2. Separate files that need to be loaded into different "shards" of the fused matrix
3. Quantized with different packing factors

**Shard mapping**:
```python
# For gate_up_proj with output_sizes=[gate_size, up_size]
shard_offsets = [
    (0, 0, gate_size),           # gate shard
    (1, gate_size, up_size),     # up shard
]

# Each shard loaded separately:
for shard_id, shard_offset, shard_size in shard_offsets:
    param_slice = param.narrow(output_dim, shard_offset, shard_size)
    # Then apply column parallel logic to this slice
```

#### 5. QKVParallelLinear (`linear.py:761`)

**What it does**: Specialized MergedColumnParallelLinear for attention QKV projections with multi-query/grouped-query attention support.

**Complexity**: Handles different numbers of Q, K, V heads:
```
MHA (Multi-Head Attention):     num_kv_heads = num_q_heads
MQA (Multi-Query Attention):    num_kv_heads = 1
GQA (Grouped-Query Attention):  1 < num_kv_heads < num_q_heads
```

**Head distribution across GPUs**:

```python
# For GQA with tp_size=8, num_q_heads=32, num_kv_heads=8
num_heads_per_gpu = 32 / 8 = 4
num_kv_heads_per_gpu = 8 / 8 = 1

# Each GPU gets:
# - 4 query heads
# - 1 key head
# - 1 value head
```

**When tp_size > num_kv_heads** (MQA case):
```python
# tp_size=8, num_kv_heads=1
num_kv_heads_per_gpu = 1
num_kv_head_replicas = 8 / 1 = 8

# Key and value heads are replicated across GPUs
# Each GPU has the SAME key/value head
```

**Weight layout per GPU**:
```
[Q heads (num_heads * head_size),
 K heads (num_kv_heads * head_size),
 V heads (num_kv_heads * head_size)]
```

**Shard ID mapping**:
```python
shard_offsets = {
    "q": 0,
    "k": num_heads * head_size,
    "v": (num_heads + num_kv_heads) * head_size,
}
```

**Load path determination**:
```python
if loaded_shard_id == "q":
    shard_id = tp_rank  # Each GPU gets different Q heads
else:  # "k" or "v"
    shard_id = tp_rank // num_kv_head_replicas  # Replicated K/V
```

## Weight Loading System

The weight loading system is one of the most complex parts of SGLang's linear layers. It handles:

1. Tensor parallelism sharding
2. Quantization format conversions
3. Fused weight separation
4. Pre-sharded weight detection

### Weight Loader Versions

#### V1 Weight Loader (Legacy)

Used by older quantization methods. Characteristics:
- Operates on raw `torch.nn.Parameter`
- Uses attributes like `output_dim`, `input_dim`, `packed_dim`
- Manual narrowing and copying logic
- Less abstraction

**Example flow** (`weight_loader` in ColumnParallelLinear):
```python
def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    output_dim = getattr(param, "output_dim", None)

    if output_dim is not None:
        shard_size = param.shape[output_dim]
        start_idx = self.tp_rank * shard_size

        if not self.use_presharded_weights:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

    assert param.shape == loaded_weight.shape
    param.data.copy_(loaded_weight)
```

#### V2 Weight Loader (Modern)

Used by newer quantization methods (FP8, AWQ, GPTQ, CompressedTensors, etc.). Characteristics:
- Operates on `BasevLLMParameter` subclasses
- Encapsulates sharding logic in parameter classes
- Cleaner separation of concerns
- Easier to extend

**Example flow** (`weight_loader_v2` in ColumnParallelLinear):
```python
def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
    param.load_column_parallel_weight(
        loaded_weight,
        tp_rank=self.tp_rank,
        use_presharded_weights=self.use_presharded_weights,
    )
```

The heavy lifting happens inside the parameter class:
```python
# In _ColumnvLLMParameter
def load_column_parallel_weight(self, loaded_weight, tp_rank, use_presharded_weights):
    if not use_presharded_weights:
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim, tp_rank * shard_size, shard_size
        )

    assert self.data.shape == loaded_weight.shape
    self.data.copy_(loaded_weight)
```

### Custom Parameter Classes

SGLang defines custom parameter classes (in `python/sglang/srt/layers/parameter.py`) that encapsulate weight loading logic:

#### BasevLLMParameter (`parameter.py:30`)

Base class for all custom parameters. Provides:
- `load_column_parallel_weight()`
- `load_row_parallel_weight()`
- `load_merged_column_weight()`
- `load_qkv_weight()`

Default implementations just copy weights, subclasses override for specific sharding logic.

#### _ColumnvLLMParameter (`parameter.py:74`)

For parameters with column parallelism. Adds:
- `output_dim` attribute
- Sharding logic in `load_column_parallel_weight()`
- Fused weight handling in `load_merged_column_weight()`

#### RowvLLMParameter (`parameter.py:226`)

For parameters with row parallelism. Adds:
- `input_dim` attribute
- Sharding logic in `load_row_parallel_weight()`

#### ModelWeightParameter (`parameter.py:289`)

Combines both column and row parallelism. Can be sharded either way depending on the layer.

#### PackedvLLMParameter (`parameter.py:433`)

For quantized weights that are packed (e.g., INT4 packed into INT32). Adds:
- `packed_factor`: How many values packed per storage element
- `packed_dim`: Which dimension is packed
- `marlin_tile_size`: Optional Marlin kernel tile size

**Shard adjustment for packing**:
```python
def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
    shard_size = shard_size // self.packed_factor
    shard_offset = shard_offset // self.packed_factor

    if self.marlin_tile_size:
        shard_size *= self.marlin_tile_size
        shard_offset *= self.marlin_tile_size

    return shard_size, shard_offset
```

#### PerTensorScaleParameter (`parameter.py:325`)

For per-tensor quantization scales in fused layers (e.g., QKV has 3 scales).

**Scalar to array mapping**:
```python
# For QKV, we have 3 scales in an array [scale_q, scale_k, scale_v]
# When loading "k" scale:
shard_id = "k"  # Convert to index 1
param[1] = loaded_scale  # Load into correct position
```

### Special Cases in Weight Loading

#### Pre-sharded Weights

Some model formats save weights already sharded for tensor parallelism:
```python
# Checkpoint has separate files per TP rank:
# model-rank0.safetensors
# model-rank1.safetensors
# ...

# Weight loader detects this:
if self.use_presharded_weights:
    # Don't narrow, weight is already the right size
    assert param.shape == loaded_weight.shape
else:
    # Need to extract our shard from full weight
    loaded_weight = loaded_weight.narrow(dim, offset, size)
```

#### Fused Weights on Disk

Some models (e.g., Phi-3, Qwen) save fused QKV or MLP weights as single tensors:
```python
# On disk: qkv_proj.weight with shape [3 * hidden, hidden]
# In memory: Need to split into Q, K, V shards

def _load_fused_module_from_checkpoint(self, param, loaded_weight):
    shard_offsets = [
        ("q", 0, q_size),
        ("k", q_size, k_size),
        ("v", q_size + k_size, v_size),
    ]

    for shard_id, shard_offset, shard_size in shard_offsets:
        weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
        self.weight_loader(param, weight_shard, shard_id)
```

#### GGUF Format

GGUF uses uninitialized parameters that are materialized lazily:
```python
# Parameter starts as UninitializedParameter
if isinstance(param, UninitializedParameter):
    # Materialize with actual shape and dtype
    param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)

# Track shard IDs in a list
param.shard_id.append(loaded_shard_id)
param.data_container.append(loaded_weight)
```

#### BitsAndBytes 4-bit

BitsAndBytes quantizes to non-uniform sizes:
```python
# Original sizes: [8192, 4096, 4096] (Q, K, V)
# Quantized: Different ratio per shard

def adjust_bitsandbytes_4bit_shard(param, shard_offsets, loaded_shard_id):
    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset
```

#### Scalar Scales to Fused Arrays

AutoFP8 saves per-tensor scales as scalars, but fused layers need arrays:
```python
# On disk: scale (scalar)
# In memory: scales = [scale_gate, scale_up] (array)

def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    # Convert shard_id "gate"/"up" or 0/1 to index
    idx = qkv_idxs.get(shard_id, shard_id)

    # Extract scalar from potential shape [1]
    if len(loaded_weight.shape) != 0:
        loaded_weight = loaded_weight[0]

    return param[idx], loaded_weight
```

#### CPU and NPU Padding

CPUs and NPUs may require special padding for alignment:
```python
if _is_cpu:
    # Use special padding-aware narrowing
    param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
        param_data, loaded_weight, 0, start_idx, dim, shard_size
    )
else:
    # Handle unaligned dimensions (e.g., Qwen2.5-VL MLP)
    end_idx = start_idx + shard_size
    if end_idx > loaded_weight.shape[dim]:
        loaded_weight = pad_or_narrow_weight(
            loaded_weight, dim, start_idx, shard_size
        )
```

## Quantization System

The quantization system uses a plugin architecture where each quantization method implements the same interface.

### QuantizeMethodBase Interface

All quantization methods implement (`base_config.py:16`):

```python
class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        # Create parameter(s) for the layer
        pass

    @abstractmethod
    def apply(self, layer, *args, **kwargs) -> torch.Tensor:
        # Apply the layer (forward pass)
        pass

    def process_weights_after_loading(self, layer) -> None:
        # Optional post-processing (e.g., transpose, repack)
        pass
```

### LinearMethodBase

Specialized for linear layers (`base_config.py:43`):

```python
class LinearMethodBase(QuantizeMethodBase):
    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Create weights with full shape information
        pass
```

### UnquantizedLinearMethod Example

The simplest implementation (`unquant.py:83`):

```python
class UnquantizedLinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size_per_partition,
                      output_partition_sizes, ...):
        # Create standard PyTorch parameter
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        # Mark dimensions for weight loading
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)

    def apply(self, layer, x, bias=None):
        # Standard F.linear call
        return F.linear(x, layer.weight, bias)
```

### Quantized Method Examples

#### FP8 Quantization

FP8 stores:
- `weight`: INT8 packed FP8 values
- `weight_scale`: Per-tensor or per-channel scales

Forward pass:
```python
def apply(self, layer, x, bias):
    # Use custom CUDA kernel
    output = torch.ops.sgl_kernel.fp8_gemm(
        x, layer.weight, layer.weight_scale, ...
    )
    if bias:
        output += bias
    return output
```

#### AWQ (Activation-aware Weight Quantization)

AWQ stores:
- `qweight`: INT4 packed into INT32
- `qzeros`: Quantized zero points
- `scales`: Per-group scales

Forward pass uses Marlin kernel:
```python
def apply(self, layer, x, bias):
    output = torch.ops.sgl_kernel.awq_marlin_gemm(
        x, layer.qweight, layer.qzeros, layer.scales, ...
    )
    if bias:
        output += bias
    return output
```

#### GPTQ (Generalized Post-Training Quantization)

Similar to AWQ but different packing format and kernel.

### Quantization Weight Creation Flow

```python
# In LinearBase.__init__:
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
else:
    self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

# In ColumnParallelLinear.__init__:
self.quant_method.create_weights(
    layer=self,
    input_size_per_partition=self.input_size,
    output_partition_sizes=self.output_partition_sizes,
    input_size=self.input_size,
    output_size=self.output_size,
    params_dtype=self.params_dtype,
    weight_loader=self.weight_loader_v2,  # or weight_loader
)
```

The quantization method can create:
- Single `weight` parameter (unquantized)
- Multiple parameters (`qweight`, `scales`, `zeros`, etc.)
- Custom parameter types (e.g., `PackedvLLMParameter`)

### Weight Loader Selection

```python
WEIGHT_LOADER_V2_SUPPORTED = [
    "CompressedTensorsLinearMethod",
    "AWQMarlinLinearMethod",
    "Fp8LinearMethod",
    "GPTQMarlinLinearMethod",
    # ... and more
]

# In create_weights call:
weight_loader = (
    self.weight_loader_v2
    if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
    else self.weight_loader
)
```

## Forward Pass Flow

### Simple Case: ReplicatedLinear

```python
def forward(self, x: torch.Tensor):
    bias = self.bias if not self.skip_bias_add else None
    output = self.quant_method.apply(self, x, bias)
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```

Returns tuple `(output, bias)` to support `skip_bias_add` optimization.

### Column Parallel Case

```python
def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None

    # Each GPU computes its partition
    output_parallel = self.quant_method.apply(self, input_, bias)

    if self.gather_output:
        # Concatenate results across GPUs
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        # Keep sharded
        output = output_parallel

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```

**Communication**: Only happens if `gather_output=True`. Uses NCCL all-gather.

### Row Parallel Case

```python
def forward(self, input_, skip_all_reduce=False):
    if self.input_is_parallel:
        input_parallel = input_
    else:
        # Split input across GPUs
        splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
        input_parallel = splitted_input[self.tp_rank].contiguous()

    # Each GPU computes its contribution
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

    with use_symmetric_memory(parallel_state.get_tp_group()) as sm:
        output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
        sm.tag(output_parallel)  # Mark for symmetric memory

    if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
        # Sum contributions across GPUs
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```

**Key optimizations**:
- Bias only added on rank 0 (to avoid duplicate addition)
- `use_symmetric_memory`: Enables efficient NCCL by pre-allocating communication buffers
- `skip_all_reduce`: Used for gradient accumulation in training

## Common Patterns in Models

### Standard Transformer Layer

```python
class TransformerLayer(nn.Module):
    def __init__(self, config, quant_config):
        # Attention
        self.self_attn = Attention(config, quant_config)

        # MLP
        self.mlp = MLP(config, quant_config)

        # Layer norms
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)

    def forward(self, hidden_states, ...):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

### Attention Layer Parallelism

```python
class Attention(nn.Module):
    def __init__(self, config, quant_config):
        # QKV: Column parallel with gather_output=False
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=config.head_size,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            gather_output=False,  # Keep sharded for attention
            quant_config=quant_config,
        )

        # O: Row parallel with reduce_results=True
        self.o_proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            reduce_results=True,  # All-reduce output
            quant_config=quant_config,
        )

    def forward(self, hidden_states, ...):
        # QKV projection: [B, T, H] -> [B, T, (Nq + 2*Nkv) * D]
        qkv, _ = self.qkv_proj(hidden_states)

        # Split into Q, K, V
        q, k, v = qkv.split([...], dim=-1)

        # Attention computation (each GPU on its heads)
        attn_output = self.attn(q, k, v, ...)

        # Output projection: [B, T, H] -> [B, T, H]
        # All-reduce happens inside o_proj
        output, _ = self.o_proj(attn_output)

        return output
```

**Communication pattern**:
1. QKV: No communication (keep sharded)
2. Attention: No communication (each GPU processes its heads)
3. O_proj: All-reduce (sum partial results)

### MLP Layer Parallelism

```python
class MLP(nn.Module):
    def __init__(self, config, quant_config):
        # Gate + Up: Fused column parallel
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size, config.intermediate_size],
            gather_output=False,
            quant_config=quant_config,
        )

        # Down: Row parallel
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            reduce_results=True,
            quant_config=quant_config,
        )

    def forward(self, hidden_states):
        # Gate + Up: [B, T, H] -> [B, T, 2*I]
        gate_up, _ = self.gate_up_proj(hidden_states)

        # Split and activation
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up  # SwiGLU

        # Down: [B, T, I] -> [B, T, H]
        output, _ = self.down_proj(intermediate)

        return output
```

**Communication pattern**:
1. Gate_up: No communication (keep sharded)
2. Activation: No communication (element-wise, independent)
3. Down: All-reduce (sum partial results)

## Efficiency Optimizations

### 1. Fused Layers

Combining multiple matrix multiplications:
```python
# Instead of:
gate = linear1(x)
up = linear2(x)

# Use fused:
gate_up = fused_linear(x)  # Single kernel call
gate, up = gate_up.split(...)
```

**Benefits**:
- Reduced kernel launch overhead
- Better memory locality
- Fewer memory accesses

### 2. Skip Bias Add

Deferring bias addition to fuse with activation:
```python
# Instead of:
x = linear(x) + bias
x = activation(x)

# Use:
x, bias = linear(x)  # skip_bias_add=True
x = activation(x + bias)  # Fused kernel
```

### 3. Symmetric Memory

NCCL optimization for all-reduce:
```python
with use_symmetric_memory(tp_group) as sm:
    output = matmul(...)
    sm.tag(output)  # Mark for symmetric allocation

# All-reduce uses pre-allocated symmetric buffers
output = all_reduce(output)
```

**Benefits**:
- Reduced memory copies
- Faster NCCL operations
- Better for repeated communication patterns

### 4. Pre-sharded Weights

Skip runtime sharding when weights are already split:
```python
# Model checkpoint saved with TP=8
# Loading on TP=8 system
if use_presharded_weights:
    # Directly load rank-specific weight file
    param.copy_(loaded_weight)  # No narrowing needed
```

### 5. Custom CUDA Kernels

SGLang uses optimized kernels for:
- FP8 GEMM
- INT4/INT8 quantized GEMM
- Fused MoE
- Fused attention
- Activation functions

Example kernel dispatch:
```python
if use_fp8_quantization:
    output = torch.ops.sgl_kernel.fp8_gemm(...)
elif use_awq_quantization:
    output = torch.ops.sgl_kernel.awq_marlin_gemm(...)
else:
    output = F.linear(...)
```

## Key Takeaways

1. **Unified Interface**: All linear operations go through the same `quant_method.apply()` interface, whether quantized or not. This enables seamless quantization support.

2. **Two-Phase Initialization**:
   - Phase 1: `create_weights()` sets up parameters
   - Phase 2: Weight loading fills in values
   - Phase 3 (optional): `process_weights_after_loading()` for repacking

3. **Flexible Parallelism**: Column, row, and replicated parallelism can be mixed in a single model. The combination determines communication patterns.

4. **Complex Weight Loading**: Handles sharding, quantization packing, fused weights, pre-sharded weights, and various edge cases in a systematic way.

5. **Custom Parameters**: `BasevLLMParameter` subclasses encapsulate sharding logic, making the system extensible.

6. **V2 Weight Loader**: Modern quantization methods use the cleaner V2 interface that delegates to parameter classes.

7. **Communication Optimization**: Strategic use of `gather_output`, `reduce_results`, and symmetric memory minimizes GPU communication.

8. **Quantization Plugin System**: New quantization methods can be added by implementing `LinearMethodBase` interface.

## Debugging Tips

### Check Weight Shapes

```python
# Expected shape for ColumnParallelLinear with TP=8:
# output_size=8192, input_size=4096, tp_size=8
weight.shape == [8192 // 8, 4096]  # [1024, 4096]

# Expected shape for RowParallelLinear with TP=8:
# output_size=4096, input_size=8192, tp_size=8
weight.shape == [4096, 8192 // 8]  # [4096, 1024]
```

### Check Shard Offsets

```python
# For MergedColumnParallelLinear with output_sizes=[4096, 4096]
# TP rank 2 of 8:
shard_offset = sum(output_sizes[:shard_id]) // tp_size
shard_size = output_sizes[shard_id] // tp_size

# Gate: shard_offset = 0, shard_size = 512
# Up: shard_offset = 512, shard_size = 512
```

### Verify Communication

```python
# Check if all-reduce is happening:
# Row parallel should have reduce_results=True
assert row_parallel.reduce_results == True

# Column parallel output projection should gather:
assert attn.o_proj.reduce_results == True

# QKV should NOT gather:
assert attn.qkv_proj.gather_output == False
```

### Quantization Format

```python
# Check quantization method:
print(layer.quant_method.__class__.__name__)

# Check parameter types:
for name, param in layer.named_parameters():
    print(f"{name}: {type(param).__name__} shape={param.shape}")

# For quantized layers, check auxiliary parameters:
if hasattr(layer, 'scales'):
    print(f"scales shape: {layer.scales.shape}")
```

## References

- Original implementation: vLLM `vllm/model_executor/layers/linear.py`
- SGLang extensions: Symmetric memory, weight loader V2, additional quantization methods
- Tensor parallelism: Megatron-LM paper and implementations
- Quantization methods: AWQ, GPTQ, SmoothQuant papers
