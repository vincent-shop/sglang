# Hybrid Attention and Mamba Backend Architecture

## Overview

SGLang implements a sophisticated hybrid attention system that supports models combining traditional attention mechanisms with linear attention variants like Mamba and Gated Delta Networks (GDN). This architecture enables efficient serving of models like Falcon-H1 and NemotronH that use different attention mechanisms in different layers.

## Architecture Components

### 1. Base Abstraction Layer

#### AttentionBackend (`base_attn_backend.py`)

Base abstract class defining the attention backend interface:

- **Core Methods**:
  - `init_forward_metadata(forward_batch)`: Initialize metadata for forward pass
  - `forward_decode(q, k, v, layer, forward_batch)`: Handle decode/generation phase
  - `forward_extend(q, k, v, layer, forward_batch)`: Handle prefill/extend phase
  - `forward(q, k, v, layer, forward_batch)`: Main entry point dispatching to decode/extend

- **CUDA Graph Support**:
  - `init_cuda_graph_state(max_bs, max_num_tokens)`: Initialize shared state
  - `init_forward_metadata_capture_cuda_graph()`: Capture phase metadata
  - `init_forward_metadata_replay_cuda_graph()`: Replay phase metadata
  - `get_cuda_graph_seq_len_fill_value()`: Padding value for sequence lengths

### 2. Hybrid Attention Backend

#### HybridAttnBackend (`hybrid_attn_backend.py`)

Manages separate backends for prefill and decode phases, enabling different kernel implementations optimized for each workload pattern.

**Design Pattern**: Strategy pattern with mode-based backend selection

**Key Features**:

- **Dual Backend Architecture**:
  - `prefill_backend`: Optimized for long sequences during prompt processing
  - `decode_backend`: Optimized for single-token generation

- **Backend Selection Logic** (`_select_backend`):
  ```
  decode_or_idle → decode_backend
  target_verify/draft_extend → decode_backend (if speculative_attention_mode="decode")
                             → prefill_backend (otherwise)
  prefill → prefill_backend
  ```

- **Delegation Pattern**: Routes all operations through selected backend transparently
- **Speculative Decoding Support**: Handles both draft and verification phases

**Usage Pattern**:
```python
hybrid_backend = HybridAttnBackend(
    model_runner=runner,
    prefill_backend=flashinfer_backend,
    decode_backend=triton_backend
)
```

### 3. Linear Attention Backends

#### MambaAttnBackendBase (`hybrid_linear_attn_backend.py`)

Base class for Mamba-style linear attention mechanisms, providing shared infrastructure for state management and metadata handling.

**Core Responsibilities**:

1. **State Management**:
   - `pad_slot_id`: Sentinel value for padding slots
   - `req_to_token_pool`: Hybrid memory pool managing both KV cache and Mamba states
   - `state_indices_list`: Pre-allocated tensors for CUDA graph capture
   - `query_start_loc_list`: Pre-allocated query position tracking

2. **Metadata Construction** (`_forward_metadata`):
   - **Decode Mode**: Simple sequential query locations `[0, 1, 2, ..., bs]`
   - **Extend Mode**:
     - Target verify: Stepped locations by draft token count
     - Standard extend: Custom locations from `extend_start_loc`
   - Retrieves Mamba cache indices from request pool

3. **CUDA Graph Infrastructure**:
   - Pre-allocates metadata buffers up to `max_bs`
   - Caches common query location patterns
   - Handles padding requests during replay

4. **Key Difference from Traditional Attention**:
   - Returns `1` for `get_cuda_graph_seq_len_fill_value()` (doesn't index by sequence length)
   - Maintains recurrent states instead of KV cache

#### GDNAttnBackend (Gated Delta Network)

Implements Gated Delta Rule attention using fused kernels for efficient linear attention.

**Architecture**:

```
Input → Conv1D → Split(Q,K,V) → Sigmoid Gating → Delta Rule → Output
         ↓                          ↓
    Conv State               SSM State (Recurrent)
```

**Forward Decode** (`forward_decode`):
1. Apply causal conv1d update with cached conv states
2. Split mixed QKV into query, key, value components
3. Reshape to `[1, seq_len, num_heads, head_dim]`
4. Run `fused_sigmoid_gating_delta_rule_update` kernel:
   - Applies sigmoid gating with learnable parameters `a`, `b`
   - Updates SSM recurrent state
   - Computes attention output with QK L2 normalization

**Forward Extend** (`forward_extend`):
1. **Target Verify Mode** (Speculative Decoding):
   - Reshape for batch processing draft tokens
   - Use `causal_conv1d_update` with intermediate cache
   - Run `fused_recurrent_gated_delta_rule_update` with state caching

2. **Standard Extend Mode**:
   - Apply full `causal_conv1d_fn` over sequence
   - Compute gating: `beta = sigmoid(b)`, `g = fused_gdn_gating(A_log, a, dt_bias)`
   - Run `chunk_gated_delta_rule` with initial recurrent state
   - Update final recurrent state in cache

**Key Parameters**:
- `A_log`, `dt_bias`: Temporal dynamics parameters
- `a`, `b`: Gating parameters
- `conv_weights`, `bias`: 1D convolution parameters
- `head_k_dim`, `head_v_dim`: Per-head dimensions

#### Mamba2AttnBackend

Specialized backend for Mamba2 architecture with chunked processing support.

**Unique Features**:

1. **Chunked Metadata** (`Mamba2Metadata`):
   - Splits sequences into physical chunks of size `mamba_chunk_size`
   - Computes logical chunk boundaries respecting sequence boundaries
   - Tracks chunk indices and offsets for state passing

2. **Mixed Batch Handling**:
   - Distinguishes prefill vs decode requests
   - Computes `has_initial_states` flag from context lengths
   - Pre-computes chunk metadata for efficient kernel execution

3. **Direct Forward Call**:
   - Calls `mixer.forward()` directly (not through `forward_decode`/`forward_extend`)
   - Passes complete `Mamba2Metadata` to mixer layer
   - Supports mixed prefill+decode in single batch

**Metadata Preparation**:
- `prepare_decode()`: Decode-only path for CUDA graph
- `prepare_mixed()`: Handles mixed prefill/decode batches
- `_query_start_loc_to_chunk_indices_offsets()`: Computes chunk boundaries

### 4. Unified Hybrid Backend

#### HybridLinearAttnBackend

Orchestrates full attention and linear attention backends for hybrid models.

**Architecture**:

```
Layer 0:  Full Attention  ←─── full_attn_backend
Layer 1:  Linear Attention ←─── linear_attn_backend
Layer 2:  Full Attention  ←─── full_attn_backend
Layer 3:  Linear Attention ←─── linear_attn_backend
...
```

**Design Pattern**: Composite pattern with layer-based routing

**Key Features**:

1. **Layer-Based Routing**:
   - `full_attn_layers`: List of layer IDs using traditional attention
   - Routes operations based on `layer_id`
   - Transparent delegation to appropriate backend

2. **Unified Interface**:
   - Implements complete `AttentionBackend` interface
   - Coordinates metadata initialization across both backends
   - Manages CUDA graph state for both paths

3. **Forward Methods**:
   - `forward()`: Main entry point, dispatches by forward mode
   - `forward_decode()`: Routes decode operations
   - `forward_extend()`: Routes extend operations

4. **Speculative Decoding Support**:
   - `update_mamba_state_after_mtp_verify()`: Updates Mamba states after verification
   - Handles accepted token updates for both SSM and conv states
   - Efficient chunked updates to reduce memory pressure

**State Update Process** (Speculative):
```python
valid_mask = accepted_length > 0
last_steps = accepted_length - 1
# Update SSM states from intermediate cache
ssm_states[:, valid_indices, :] = intermediate_cache[:, valid_indices, last_steps]
# Update conv states from intermediate cache
conv_states[:, valid_indices, :, :] = intermediate_conv[:, valid_indices, last_steps]
```

## Forward Modes and Processing

### Forward Mode Types

1. **DECODE**: Single token generation per request
   - Uses decode backend/path
   - Efficient recurrent state update
   - CUDA graph compatible

2. **EXTEND/PREFILL**: Multi-token prompt processing
   - Uses extend/prefill backend/path
   - Chunked processing for long sequences
   - Handles initial state loading

3. **TARGET_VERIFY**: Speculative decoding verification
   - Configurable backend selection
   - Processes draft tokens in batch
   - Maintains intermediate states

4. **DRAFT_EXTEND**: Speculative decoding draft generation
   - Similar routing to target_verify
   - Generates multiple candidate tokens

5. **IDLE**: No computation phase
   - Returns empty tensors
   - Used during batch transitions

### Metadata Flow

```
ForwardBatch
    ↓
init_forward_metadata()
    ↓
ForwardMetadata
    - query_start_loc: Cumulative sequence positions
    - mamba_cache_indices: Cache line mappings
    ↓
[GDN] → Raw metadata
[Mamba2] → Mamba2Metadata (with chunk info)
    ↓
forward() operations
```

## Memory Management

### Request-to-Token Pool

The `HybridReqToTokenPool` manages:

1. **Traditional KV Cache**: For full attention layers
   - Token-level granularity
   - Radix tree structure for prefix sharing

2. **Mamba State Cache** (`MambaPool`):
   - **Convolution States**: Sliding window for causal conv
   - **SSM/Temporal States**: Recurrent hidden states
   - **Intermediate Caches** (Speculative):
     - `intermediate_ssm`: Step-by-step SSM states
     - `intermediate_conv_window`: Step-by-step conv windows

3. **Cache Indexing**:
   - `get_mamba_indices(req_pool_indices)`: Maps requests to cache lines
   - Per-layer cache: `mamba2_layer_cache(layer_id)`

### State Shapes

- **Conv State**: `[num_layers, pool_size, conv_width, hidden_dim]`
- **SSM State**: `[num_layers, pool_size, num_heads, head_dim, state_dim]`
- **Intermediate SSM**: `[num_layers, pool_size, max_draft_len, ...]`
- **Intermediate Conv**: `[num_layers, pool_size, max_draft_len, ...]`

## CUDA Graph Support

### Initialization Phase

```python
init_cuda_graph_state(max_bs, max_num_tokens):
    # Pre-allocate buffers
    for i in range(max_bs):
        state_indices_list[i] = Tensor[i+1] filled with pad_slot_id
        query_start_loc_list[i] = Tensor[i+2]

    # Cache common patterns
    decode_query_start_loc = [0, 1, 2, ..., max_bs]
    verify_query_start_loc = [0, step, 2*step, ..., max_bs*step]
```

### Capture Phase

```python
init_forward_metadata_capture_cuda_graph():
    # Select pre-allocated buffers
    query_start_loc = query_start_loc_list[bs-1]
    state_indices = state_indices_list[bs-1]

    # Copy cached patterns
    if decode: copy from cached_decode_query_start_loc
    if verify: copy from cached_verify_query_start_loc

    # Populate state indices
    mamba_indices = get_mamba_indices(req_pool_indices)
    state_indices[:len(mamba_indices)].copy_(mamba_indices)
```

### Replay Phase

```python
init_forward_metadata_replay_cuda_graph():
    # Handle padding requests
    num_padding = count(seq_lens == fill_value)
    req_pool_indices[bs-num_padding:] = 0

    # Adjust for padding
    if num_padding > 0:
        query_start_loc[bs-num_padding:] = (bs-num_padding)
        state_indices[bs-num_padding:] = -1
```

## Integration with Models

### Model Registration

Models register their hybrid attention configuration through `ModelRunner`:

```python
# In model_runner.py
if model_config.hybrid_gdn_layers:
    runner.hybrid_gdn_config = HybridGDNConfig(...)
    runner.mambaish_config = MambaishConfig(
        full_attention_layer_ids=[...],
        ...
    )
```

### Backend Construction

```python
# attention_registry.py:attn_backend_wrapper
if cfg := runner.mambaish_config:
    if runner.hybrid_gdn_config:
        linear_attn_backend = GDNAttnBackend(runner)
    elif runner.mamba2_config:
        linear_attn_backend = Mamba2AttnBackend(runner)

    full_attn_layers = cfg.full_attention_layer_ids
    return HybridLinearAttnBackend(
        full_attn_backend, linear_attn_backend, full_attn_layers
    )
```

### Layer-Level Usage

#### Hybrid GDN Model (e.g., Falcon-H1)

```python
class FalconH1Block:
    def forward(self, hidden_states, forward_batch):
        # Standard attention
        attn_out = self.self_attention(hidden_states, forward_batch)

        # Mamba block via backend
        attn_backend = forward_batch.attn_backend
        mamba_out = torch.empty_like(hidden_states)
        attn_backend.linear_attn_backend.forward(
            self.mamba,
            hidden_states,
            mamba_out,
            layer_id=self.layer_id,
        )

        return attn_out + mamba_out
```

#### GDN Attention Layer

```python
class GatedDeltaNetAttention(RadixAttention):
    def forward(self, hidden_states, forward_batch):
        mixed_qkv = self.qkv_proj(hidden_states)

        # Backend handles all compute
        return forward_batch.attn_backend.forward(
            q=None, k=None, v=None,  # Not pre-computed
            layer=self,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            conv_weights=self.conv_weights,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            a=self.a, b=self.b,
            # ... other parameters
        )
```

## Performance Optimizations

### Kernel Fusion

1. **Conv + Gating**: `causal_conv1d_fn` combines convolution with activation
2. **Attention + Update**: `fused_sigmoid_gating_delta_rule_update` fuses gating, attention, state update
3. **QK Normalization**: `use_qk_l2norm_in_kernel=True` eliminates separate normalization pass

### Memory Efficiency

1. **In-Place State Updates**: Direct writes to cache tensors
2. **Chunked Processing**: Limits peak memory for long sequences
3. **Shared Buffers**: Reuses pre-allocated tensors across CUDA graph invocations

### Platform-Specific Optimizations

```python
if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
```

Automatically selects optimized kernels for:
- CUDA (NVIDIA GPUs)
- NPU (Ascend accelerators)

## Design Patterns and Principles

### Strategy Pattern
`HybridAttnBackend` uses strategy pattern to select prefill/decode backends based on forward mode, enabling different optimizations per workload.

### Composite Pattern
`HybridLinearAttnBackend` composes full and linear attention backends, routing by layer ID for hybrid models.

### Template Method
`MambaAttnBackendBase` defines metadata construction template, with GDN and Mamba2 specializing specific steps.

### Dependency Injection
Backends receive `ModelRunner` as dependency, accessing pools, configs, and device information through unified interface.

## Key Differences from Standard Attention

| Aspect | Standard Attention | Mamba/Linear Attention |
|--------|-------------------|----------------------|
| State Type | KV cache (per token) | Recurrent state (per sequence) |
| Complexity | O(n²) or O(n) with sparsity | O(n) always |
| Cache Indexing | By sequence length | By request ID |
| Prefill Strategy | Parallel across tokens | Chunked sequential |
| CUDA Graph Fill | seq_len = 0 | seq_len = 1 |
| State Shape | `[num_tokens, num_heads, head_dim]` | `[num_heads, head_dim, state_dim]` |

## Configuration and Tuning

### Key Parameters

- `mamba_chunk_size`: Physical chunk size for Mamba2 (default: 256)
- `speculative_attention_mode`: "decode" or "prefill" for spec decoding
- `full_attention_layer_ids`: Which layers use traditional attention

### Environment Detection

The system automatically detects:
- GPU architecture (Blackwell, Ampere, etc.)
- Platform (CUDA, NPU)
- Available kernels (Triton, CUTLASS, native)

### Assertions and Validation

```python
# Enforces correct backend pairing
assert isinstance(attn_backend, HybridLinearAttnBackend)
assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)

# Validates CUDA graph constraints
assert max_num_tokens % max_bs == 0, "must be divisible for verify step"
```

## Code References

### Core Files

- `hybrid_attn_backend.py:13-148`: HybridAttnBackend implementation
- `hybrid_linear_attn_backend.py:54-228`: MambaAttnBackendBase
- `hybrid_linear_attn_backend.py:230-443`: GDNAttnBackend
- `hybrid_linear_attn_backend.py:446-525`: Mamba2AttnBackend
- `hybrid_linear_attn_backend.py:528-706`: HybridLinearAttnBackend
- `base_attn_backend.py:15-146`: AttentionBackend interface
- `attention_registry.py:178-219`: Backend wrapper and registration

### Related Components

- `mamba2_metadata.py`: Metadata and chunking logic
- `mem_cache/memory_pool.py`: HybridReqToTokenPool and MambaPool
- `models/falcon_h1.py`: Example hybrid model usage
- `models/nemotron_h.py`: Another hybrid model implementation

## Future Directions

### Potential Enhancements

1. **Multi-Backend Routing**: Support 3+ backends routed by layer type
2. **Dynamic Chunk Sizing**: Adapt `mamba_chunk_size` based on batch composition
3. **Kernel Autotune**: Profile and select optimal kernel per GPU/workload
4. **State Compression**: Quantize or compress recurrent states
5. **Cross-Layer State Sharing**: Exploit patterns across Mamba layers

### Research Opportunities

- Hybrid sparsity patterns combining sparse attention + linear attention
- Learned routing decisions (which layers use which mechanism)
- Mixed-precision state management for memory efficiency
- Unified KV cache + recurrent state pool abstraction

## Summary

SGLang's hybrid attention architecture provides a flexible, efficient framework for serving models with mixed attention mechanisms. Key strengths:

- **Modularity**: Clean separation between attention backend types
- **Performance**: Fused kernels, CUDA graphs, memory pooling
- **Flexibility**: Easy to add new attention variants
- **Production-Ready**: Handles speculative decoding, batching, multi-GPU

The design successfully abstracts complexity while maintaining high performance, enabling SGLang to support cutting-edge hybrid architectures like Falcon-H1 and NemotronH.
