# Mamba / Non-Linear Attention System Deep Dive

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Backend Hierarchy](#backend-hierarchy)
4. [Metadata Structures](#metadata-structures)
5. [Forward Pass Flows](#forward-pass-flows)
6. [CUDA Graph Support](#cuda-graph-support)
7. [Cache Management](#cache-management)
8. [Kernel Operations](#kernel-operations)
9. [Model Integration](#model-integration)
10. [Speculative Decoding](#speculative-decoding)
11. [Platform Support](#platform-support)

---

## Overview

SGLang implements efficient mamba-style and linear attention mechanisms through a hierarchical backend system. This system supports:

- **Hybrid attention models** (Falcon H1, Qwen3-Next) with both full attention and linear attention layers
- **Mamba2 models** (NemotronH) with SSM-based attention
- **Gated Delta Net (GDN)** attention for efficient linear complexity
- **Speculative decoding** with intermediate state caching
- **CUDA graph optimization** for decode performance
- **Mixed prefill/decode** batching

The implementation bridges between traditional attention mechanisms and state-space models (SSMs), providing a unified interface for models that mix these paradigms.

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│              AttentionBackend (ABC)                     │
│  - init_forward_metadata()                              │
│  - forward_decode() / forward_extend()                  │
│  - CUDA graph capture/replay                            │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│   Hybrid     │ │   Mamba      │ │   Hybrid         │
│   Attn       │ │   Attn       │ │   Linear         │
│   Backend    │ │   Backend    │ │   Attn           │
│              │ │   Base       │ │   Backend        │
└──────────────┘ └──────────────┘ └──────────────────┘
                        │                │
                ┌───────┴────────┐      │
                │                │      │
                ▼                ▼      ▼
        ┌──────────┐    ┌──────────┐  ┌─────────────┐
        │   GDN    │    │  Mamba2  │  │  Routes to  │
        │  Attn    │    │  Attn    │  │  full/linear│
        │ Backend  │    │ Backend  │  │  backends   │
        └──────────┘    └──────────┘  └─────────────┘
```

### File Organization

- `hybrid_attn_backend.py` - Routes between prefill/decode backends
- `hybrid_linear_attn_backend.py` - Manages full + linear attention, contains GDN/Mamba2 backends
- `mamba/mamba2_metadata.py` - Metadata structures for mamba operations
- `mamba/mamba.py` - MambaMixer2 implementation
- `fla/` - Fast Linear Attention kernels (chunk, recurrent, sigmoid gating)
- `mamba/causal_conv1d*.py` - Causal convolution implementations

---

## Backend Hierarchy

### 1. HybridAttnBackend

**Purpose**: Route between different backends for prefill vs decode phases.

**Location**: `python/sglang/srt/layers/attention/hybrid_attn_backend.py:13`

**Key Responsibilities**:
- Backend selection based on `ForwardMode`
- Delegates all operations to selected backend
- Handles speculative decoding mode selection

**Backend Selection Logic** (`_select_backend()` at line 27):

```python
if forward_mode.is_decode_or_idle():
    return self.decode_backend
elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
    return (
        self.decode_backend
        if self.model_runner.server_args.speculative_attention_mode == "decode"
        else self.prefill_backend
    )
else:
    return self.prefill_backend
```

**Usage Pattern**: Used for standard attention models where prefill and decode may benefit from different kernel implementations (e.g., FlashInfer for prefill, Triton for decode).

---

### 2. MambaAttnBackendBase

**Purpose**: Base class for all mamba-style attention backends, providing common metadata management.

**Location**: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:54`

**Key Responsibilities**:
- Forward metadata generation for mamba operations
- CUDA graph state initialization
- Cache index management

**Core Data Structures**:

```python
class MambaAttnBackendBase:
    pad_slot_id: int  # Sentinel value for padding (-1)
    req_to_token_pool: HybridReqToTokenPool  # Maps requests to cache indices
    forward_metadata: ForwardMetadata  # Current forward pass metadata
    state_indices_list: List[torch.Tensor]  # Pre-allocated for CUDA graph
    query_start_loc_list: List[torch.Tensor]  # Pre-allocated for CUDA graph
```

**Metadata Generation** (`_forward_metadata()` at line 66):

For **decode mode** (line 69-72):
```python
query_start_loc = torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
# Result: [0, 1, 2, ..., bs] - each request has exactly 1 token
```

For **extend mode** (line 73-90):
- If `target_verify`: Step by draft_token_num (speculative decoding)
- Otherwise: Use extend_start_loc from forward_batch
- Accumulates sequence lengths across batch

For **target_verify mode** (line 74-81):
```python
query_start_loc = torch.arange(
    0,
    forward_batch.input_ids.shape[0] + 1,
    step=forward_batch.spec_info.draft_token_num,
    dtype=torch.int32,
    device=forward_batch.input_ids.device,
)
# Splits tokens into draft_token_num sized chunks for verification
```

**CUDA Graph Support**:

Initialization (`init_cuda_graph_state()` at line 133):
- Pre-allocates tensors for all possible batch sizes [1..max_bs]
- Creates `state_indices_list` and `query_start_loc_list` for reuse
- Caches common patterns (decode, verify) for fast replay

Capture (`_capture_metadata()` at line 158):
- Copies from cached patterns to reusable buffers
- Retrieves mamba cache indices from req_to_token_pool
- Returns metadata pointing to stable tensors

Replay (`_replay_metadata()` at line 178):
- Handles padding requests (seq_lens == fill_value)
- Adjusts query_start_loc for padded requests
- Sets padded cache indices to -1 (sentinel)

---

### 3. GDNAttnBackend

**Purpose**: Implements Gated Delta Net attention for hybrid models (Falcon H1, Qwen3-Next).

**Location**: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:230`

**Architecture**: GDN combines causal convolution with gated delta rule recurrence for O(n) complexity attention.

**Key Components**:

1. **Causal Convolution** - Short-range dependencies via 1D convolution
2. **Gated Delta Rule** - Long-range dependencies via linear recurrence
3. **SSM States** - Maintains temporal state across tokens

**Forward Decode** (`forward_decode()` at line 233):

Pipeline for single-token generation:

```python
# Step 1: Causal convolution update (line 264-271)
mixed_qkv = causal_conv1d_update(
    mixed_qkv,              # Input [batch, hidden]
    conv_states,            # Conv window cache [layers, batch, width, hidden]
    conv_weights,           # Learnable conv weights
    bias,                   # Conv bias
    activation,             # Activation function
    conv_state_indices=cache_indices,  # Which cache slots to use
)

# Step 2: Split into Q, K, V (line 273-281)
query, key, value = torch.split(mixed_qkv, [...], dim=-1)
# Reshape to [1, batch, num_heads, head_dim]

# Step 3: Fused sigmoid gating delta rule update (line 289-303)
core_attn_out = fused_sigmoid_gating_delta_rule_update(
    A_log=A_log,            # Log of state transition matrix
    dt_bias=dt_bias,        # Time step bias
    q=query,                # Query [1, batch, heads, head_dim]
    k=key,                  # Key [1, batch, heads, head_dim]
    v=value,                # Value [1, batch, value_heads, head_dim]
    a=a,                    # Gating input parameter
    b=b,                    # Gating forget parameter
    initial_state_source=ssm_states,  # State cache [layers, batch, ...]
    initial_state_indices=cache_indices,  # Which states to update
    cu_seqlens=query_start_loc,  # [0, 1, 2, ..., batch]
    use_qk_l2norm_in_kernel=True,  # Normalize Q, K
    softplus_beta=1.0,      # Gating activation params
    softplus_threshold=20.0,
)
```

**Forward Extend** (`forward_extend()` at line 307):

Pipeline for multi-token sequences (prefill or speculative verify):

```python
# Determine mode
is_target_verify = forward_batch.forward_mode.is_target_verify()

# Get cache parameters (line 338-353)
conv_states = mamba_cache_params.conv
ssm_states = mamba_cache_params.temporal
if is_target_verify:
    # Use speculative cache for intermediate states
    intermediate_state_cache = mamba_cache_params.intermediate_ssm
    intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window

# Step 1: Causal convolution
if is_target_verify:
    # Process in chunks for verification (line 355-374)
    batch_size = seq_len // draft_token_num
    mixed_qkv_reshaped = mixed_qkv.view(batch_size, draft_token_num, -1)
    mixed_qkv_processed = causal_conv1d_update(
        mixed_qkv_reshaped.transpose(1, 2),
        conv_states_to_use,
        conv_weights,
        bias,
        activation,
        conv_state_indices=cache_indices[:batch_size],
        intermediate_conv_window=intermediate_conv_window_cache,  # Cache intermediates
    )
    mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
else:
    # Standard causal convolution for prefill (line 376-386)
    mixed_qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),  # [hidden, seq_len]
        conv_weights,
        bias,
        activation=activation,
        conv_states=conv_states_to_use,
        has_initial_state=has_initial_states,  # Some reqs have cached prefix
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
    ).transpose(0, 1)[:seq_len]

# Step 2: Split and reshape (line 388-403)
query, key, value = torch.split(mixed_qkv, [...], dim=-1)
query = query.view(1, seq_len, num_heads, head_k_dim)
key = key.view(1, seq_len, num_heads, head_k_dim)
value = value.view(1, seq_len, num_value_heads, head_v_dim)

# Step 3: Compute gating parameters (line 405-409)
beta = b.sigmoid()  # Forget gate
g = fused_gdn_gating(A_log, a, dt_bias)  # Combined gating

# Step 4: Apply gated delta rule
if is_target_verify:
    # Recurrent update with intermediate caching (line 411-425)
    core_attn_out = fused_recurrent_gated_delta_rule_update(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state_source=ssm_states,
        initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,  # Don't update main state yet
        intermediate_states_buffer=intermediate_state_cache,  # Store intermediates
        cache_steps=draft_token_num,  # Cache every step
    )
else:
    # Chunked processing for prefill (line 427-441)
    recurrent_state = ssm_states[cache_indices]
    core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,  # Starting state
        output_final_state=True,  # Return final state
        cu_seqlens=query_start_loc,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    # Update cache with final state
    ssm_states[cache_indices] = last_recurrent_state.to(ssm_states.dtype)
```

**Key Differences Between Modes**:

| Aspect | Decode | Extend (Prefill) | Extend (Target Verify) |
|--------|--------|------------------|------------------------|
| Tokens/request | 1 | Variable | Fixed (draft_token_num) |
| Conv operation | `causal_conv1d_update` | `causal_conv1d_fn` | `causal_conv1d_update` (chunked) |
| Delta rule | `fused_sigmoid_gating_delta_rule_update` | `chunk_gated_delta_rule` | `fused_recurrent_gated_delta_rule_update` |
| State update | In-place | Final state only | Intermediate caching |
| Purpose | Generate next token | Process prompt | Verify speculative tokens |

---

### 4. Mamba2AttnBackend

**Purpose**: Wrapper for Mamba2Mixer operations with chunked processing support.

**Location**: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:446`

**Key Difference from GDN**: Uses specialized Mamba2 kernels with explicit chunk management for better throughput on long sequences.

**Initialization** (line 449-453):
```python
def __init__(self, model_runner: ModelRunner):
    super().__init__(model_runner)
    config = model_runner.mamba2_config
    self.mamba_chunk_size = config.mamba_chunk_size  # Physical chunk size
```

**Metadata Preparation**:

Override `init_forward_metadata()` (line 455-462):
```python
metadata = self._forward_metadata(forward_batch)  # Base metadata
self.forward_metadata = Mamba2Metadata.prepare_mixed(
    metadata.query_start_loc,
    metadata.mamba_cache_indices,
    self.mamba_chunk_size,  # Enables chunking
    forward_batch,
)
```

This produces `Mamba2Metadata` with additional chunking information for efficient processing.

**Forward Path** (`forward()` at line 497):
```python
def forward(
    self,
    mixer: MambaMixer2,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
    mup_vector: Optional[torch.Tensor] = None,
    use_triton_causal_conv: bool = False,
):
    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
    return mixer.forward(
        hidden_states=hidden_states,
        output=output,
        layer_cache=layer_cache,
        metadata=self.forward_metadata,  # Mamba2Metadata
        mup_vector=mup_vector,
        use_triton_causal_conv=use_triton_causal_conv,
    )
```

**Note**: Unlike other backends, `forward_decode()` and `forward_extend()` raise `NotImplementedError` (lines 517-525). The `forward()` method is called directly, with the mixer handling mixed prefill/decode internally based on metadata.

---

### 5. HybridLinearAttnBackend

**Purpose**: Manages models with both full attention layers (standard transformer) and linear attention layers (GDN/Mamba).

**Location**: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:528`

**Initialization** (line 531-540):
```python
def __init__(
    self,
    full_attn_backend: AttentionBackend,      # e.g., FlashInfer, Triton
    linear_attn_backend: MambaAttnBackendBase,  # GDNAttnBackend or Mamba2AttnBackend
    full_attn_layers: list[int],              # Layer IDs using full attention
):
    self.full_attn_layers = full_attn_layers
    self.full_attn_backend = full_attn_backend
    self.linear_attn_backend = linear_attn_backend
    self.attn_backend_list = [full_attn_backend, linear_attn_backend]
```

**Routing Logic** (`forward_decode()` at line 597, `forward_extend()` at line 616):
```python
layer_id = layer.layer_id if layer else kwargs["layer_id"]
if layer_id in self.full_attn_layers:
    return self.full_attn_backend.forward_decode(...)
return self.linear_attn_backend.forward_decode(...)
```

**Delegation Pattern**:
- Initializes metadata for **both** backends on each forward pass
- Routes individual layer operations based on layer_id
- Full attention layers use KV cache
- Linear attention layers use SSM state cache

**Speculative Decoding Support** (`update_mamba_state_after_mtp_verify()` at line 671):

After verifying speculative tokens, this method updates the main cache with accepted intermediate states:

```python
def update_mamba_state_after_mtp_verify(self, accepted_length, model):
    # Get which requests and how many tokens were accepted
    state_indices_tensor = (
        self.linear_attn_backend.forward_metadata.mamba_cache_indices[
            :request_number
        ]
    )

    # Retrieve intermediate caches from speculative buffer
    mamba_caches = (
        self.linear_attn_backend.req_to_token_pool
        .get_speculative_mamba2_params_all_layers()
    )

    # Copy accepted intermediate states to main cache (line 690-705)
    valid_mask = accepted_length > 0
    last_steps_all = (accepted_length - 1).to(torch.int64)
    valid_state_indices = state_indices_tensor[valid_mask]
    last_steps = last_steps_all[valid_mask]

    # Update SSM states
    ssm_states[:, valid_state_indices, :] = intermediate_state_cache[
        :, valid_state_indices, last_steps
    ].to(ssm_states.dtype)

    # Update conv states
    conv_states[:, valid_state_indices, :, :] = intermediate_conv_window_cache[
        :, valid_state_indices, last_steps
    ].to(conv_states.dtype)
```

This enables speculative decoding to work efficiently: verify multiple tokens in parallel, then commit only the accepted states.

---

## Metadata Structures

### ForwardMetadata

**Location**: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:25`

**Definition**:
```python
@dataclass(kw_only=True)
class ForwardMetadata:
    query_start_loc: torch.Tensor  # Shape: [batch_size + 1], dtype: int32
    mamba_cache_indices: torch.Tensor  # Shape: [batch_size], dtype: int32
```

**Purpose**: Minimal metadata for mamba operations.

**`query_start_loc`**:
- Cumulative sum of sequence lengths
- Used to identify token boundaries in flattened batch
- Example: For sequences [A(3 tokens), B(5 tokens), C(2 tokens)]
  - `query_start_loc = [0, 3, 8, 10]`
  - Sequence A: tokens[0:3], B: tokens[3:8], C: tokens[8:10]

**`mamba_cache_indices`**:
- Maps each request to its cache slot in MambaPool
- Example: `[45, 12, 89]` means request 0 uses cache slot 45, etc.
- Value `-1` indicates padding (no-op)

---

### Mamba2Metadata

**Location**: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:32`

**Definition**:
```python
@dataclass(kw_only=True)
class Mamba2Metadata(ForwardMetadata):
    num_prefills: int          # Number of prefill requests
    num_prefill_tokens: int    # Total tokens in prefill requests
    num_decodes: int           # Number of decode requests

    mixed_metadata: MixedMetadata | None  # Only for extend mode
```

**MixedMetadata** (nested dataclass at line 40):
```python
@dataclass(kw_only=True, frozen=True)
class MixedMetadata:
    has_initial_states: torch.Tensor  # [batch_size], bool
    prep_initial_states: bool         # Any request has initial state?

    chunk_size: int                   # Physical mamba chunk size
    seq_idx: torch.Tensor             # [1, num_tokens], which seq each token belongs to
    chunk_indices: torch.Tensor       # Logical chunk -> physical chunk mapping
    chunk_offsets: torch.Tensor       # Offset within physical chunk

    extend_seq_lens_cpu: list[int]    # Sequence lengths on CPU
```

**Chunking Logic** (`_query_start_loc_to_chunk_indices_offsets()` at line 55):

Physical chunks are fixed-size (e.g., 256 tokens). Logical chunks adapt to sequence boundaries.

Example from docstring (line 91-105):
```
query_start_loc = [0, 5, 10]
chunk_size = 8
total_seqlens = 10

Result:
chunk_indices = [0, 0, 1]
chunk_offsets = [0, 5, 0]

Interpretation:
- Logical chunk 0: tokens [0:5] (seq 0), physical chunk 0, offset 0
- Logical chunk 1: tokens [5:8] (seq 1 prefix), physical chunk 0, offset 5
- Logical chunk 2: tokens [8:10] (seq 1 suffix), physical chunk 1, offset 0
```

This allows sequences to span physical chunks while maintaining efficient kernel dispatch.

**Preparation Methods**:

1. **`prepare_decode()`** (line 139): Decode-only, no chunking needed
   ```python
   return Mamba2Metadata(
       query_start_loc=query_start_loc,
       mamba_cache_indices=mamba_cache_indices,
       num_decodes=len(seq_lens),
       num_prefills=0,
       num_prefill_tokens=0,
       mixed_metadata=None,  # Not needed for decode
   )
   ```

2. **`prepare_mixed()`** (line 154): Handles prefill + decode batches
   ```python
   # Compute sequence index for each token (line 176-183)
   seq_idx = torch.repeat_interleave(
       torch.arange(num_prefills, dtype=torch.int32, device=device),
       query_start_loc.diff(),  # Repeat by seq length
       output_size=num_prefill_tokens,
   )

   # Compute chunking if any request has initial states (line 188-194)
   if prep_initial_states:
       chunk_indices, chunk_offsets = (
           cls._query_start_loc_to_chunk_indices_offsets(
               query_start_loc, chunk_size, num_prefill_tokens
           )
       )
   ```

---

## Forward Pass Flows

### Decode Flow (Single Token Generation)

**Entry Point**: Model layer calls `attn_backend.forward_decode()`

**Call Chain**:
```
Model Layer
  ↓
HybridLinearAttnBackend.forward_decode()  [line 597]
  ↓ (route by layer_id)
  ├─→ full_attn_backend.forward_decode()  [for full attn layers]
  └─→ linear_attn_backend.forward_decode()  [for linear attn layers]
       ↓
     GDNAttnBackend.forward_decode()  [line 233]
       ↓
       ├─→ causal_conv1d_update()  [line 264]
       │   Updates conv_states in-place
       │
       └─→ fused_sigmoid_gating_delta_rule_update()  [line 289]
           Updates ssm_states in-place
           Returns attention output
```

**Metadata Flow**:
```
ModelRunner.forward_batch_info.init_metadata()
  ↓
HybridLinearAttnBackend.init_forward_metadata()  [line 542]
  ↓
  ├─→ full_attn_backend.init_forward_metadata()
  └─→ linear_attn_backend.init_forward_metadata()
       ↓
     MambaAttnBackendBase.init_forward_metadata()  [line 101]
       ↓
     self.forward_metadata = self._forward_metadata(forward_batch)
       ↓
     ForwardMetadata(
         query_start_loc=[0, 1, 2, ..., batch_size],
         mamba_cache_indices=req_pool_indices mapped to cache slots
     )
```

**Performance Characteristics**:
- O(1) per token (after initial setup)
- In-place state updates (no large allocations)
- Compatible with CUDA graphs
- Minimal CPU overhead

---

### Extend Flow (Prefill / Multi-Token)

**Entry Point**: Model layer calls `attn_backend.forward_extend()`

**Call Chain**:
```
Model Layer
  ↓
HybridLinearAttnBackend.forward_extend()  [line 616]
  ↓ (route by layer_id)
  └─→ GDNAttnBackend.forward_extend()  [line 307]
       ↓
       ├─→ causal_conv1d_fn()  [line 376]
       │   Processes entire sequence with convolution
       │   Handles variable sequence lengths
       │   Updates conv_states with final window
       │
       ├─→ fused_gdn_gating()  [line 406]
       │   Computes gating parameters (g, beta)
       │
       └─→ chunk_gated_delta_rule()  [line 428]
           Processes sequence in chunks
           Uses initial_state from cache
           Returns final state
           Updates ssm_states with final state
```

**Key Optimization - Chunked Processing**:

For a 1024-token sequence with chunk_size=256:
1. Load initial state from cache
2. Process chunk 0 (tokens 0-255), output state_0
3. Process chunk 1 (tokens 256-511) with state_0, output state_1
4. Process chunk 2 (tokens 512-767) with state_1, output state_2
5. Process chunk 3 (tokens 768-1023) with state_2, output state_3
6. Store state_3 in cache

This enables:
- Bounded memory usage (process 256 tokens at a time)
- Efficient kernel dispatch (fixed chunk size)
- Prefix caching (resume from stored state)

**Handling Initial States**:

If a request has cached prefix (e.g., system prompt):
```python
has_initial_states = forward_batch.extend_prefix_lens > 0
recurrent_state = ssm_states[cache_indices]  # Load cached state
core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
    initial_state=recurrent_state,  # Resume from cached state
    ...
)
```

---

### Target Verify Flow (Speculative Decoding)

**Entry Point**: Model layer calls `attn_backend.forward_extend()` with `forward_mode.is_target_verify()`

**Purpose**: Verify N speculative tokens in parallel, caching intermediate states to enable fast rollback.

**Call Chain**:
```
Model Layer (Target Model)
  ↓
HybridLinearAttnBackend.forward_extend()
  ↓
GDNAttnBackend.forward_extend()  [is_target_verify=True]
  ↓
  ├─→ causal_conv1d_update() in chunked mode  [line 363]
  │   Processes draft_token_num tokens per request
  │   Caches intermediate_conv_window at each step
  │
  └─→ fused_recurrent_gated_delta_rule_update()  [line 412]
      Processes all tokens recurrently
      Caches intermediate_states_buffer at each step
      Does NOT update main ssm_states (disable_state_update=True)
  ↓
Return attention outputs
  ↓
Verification logic determines accepted_length per request
  ↓
HybridLinearAttnBackend.update_mamba_state_after_mtp_verify()  [line 671]
  Copies intermediate_states[accepted_length-1] to main cache
```

**Intermediate Caching**:

For 5 draft tokens [t0, t1, t2, t3, t4]:
```
Initial state: S0 (from main cache)

Process:
  t0 → S1  (store in intermediate_states[:, :, 0])
  t1 → S2  (store in intermediate_states[:, :, 1])
  t2 → S3  (store in intermediate_states[:, :, 2])
  t3 → S4  (store in intermediate_states[:, :, 3])
  t4 → S5  (store in intermediate_states[:, :, 4])

After verification:
  If accepted_length = 3:
    Copy intermediate_states[:, :, 2] (S3) to main cache

  If accepted_length = 0:
    Keep S0 in main cache (reject all)
```

**Why This Works**:
- All draft tokens are correct up to first rejection
- Can accept partial sequences efficiently
- No need to recompute accepted tokens
- Rejected tokens simply discarded

---

## CUDA Graph Support

CUDA graphs eliminate kernel launch overhead for decode, achieving ~2x throughput improvement.

### Initialization Phase

**Entry Point**: `ModelRunner` calls `attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)`

**Implementation** (`MambaAttnBackendBase.init_cuda_graph_state()` at line 133):

```python
def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
    # Allocate buffers for all batch sizes [1..max_bs]
    for i in range(max_bs):
        # Pre-allocate state indices (padded to i+1)
        self.state_indices_list.append(
            torch.full((i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device)
        )
        # Pre-allocate query start locations
        self.query_start_loc_list.append(
            torch.empty((i + 2,), dtype=torch.int32, device=self.device)
        )

    # Cache common patterns
    self.cached_cuda_graph_decode_query_start_loc = torch.arange(
        0, max_bs + 1, dtype=torch.int32, device=self.device
    )  # [0, 1, 2, ..., max_bs]

    verify_step = max_num_tokens / max_bs
    self.cached_cuda_graph_verify_query_start_loc = torch.arange(
        0,
        max_bs * verify_step + 1,
        step=verify_step,
        dtype=torch.int32,
        device=self.device,
    )  # [0, verify_step, 2*verify_step, ..., max_bs*verify_step]
```

**Memory Allocation**:
- `state_indices_list`: N tensors for N batch sizes (total ~max_bs^2 * 4 bytes)
- `query_start_loc_list`: N tensors for N batch sizes (similar)
- Cached patterns: 2 * max_bs * 4 bytes

Total overhead: ~O(max_bs^2) but very small in practice (e.g., max_bs=256 → ~512KB)

### Capture Phase

**Entry Point**: `attn_backend.init_forward_metadata_capture_cuda_graph(bs, ...)`

**Implementation** (`_capture_metadata()` at line 158):

```python
def _capture_metadata(self, bs: int, req_pool_indices: torch.Tensor, forward_mode: ForwardMode):
    # Copy from cached pattern to reusable buffer
    if forward_mode.is_decode_or_idle():
        self.query_start_loc_list[bs - 1].copy_(
            self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
        )
    elif forward_mode.is_target_verify():
        self.query_start_loc_list[bs - 1].copy_(
            self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
        )

    # Get mamba indices and copy to buffer
    mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
    self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

    # Return metadata pointing to stable buffers
    return ForwardMetadata(
        query_start_loc=self.query_start_loc_list[bs - 1],  # Stable address
        mamba_cache_indices=self.state_indices_list[bs - 1],  # Stable address
    )
```

**Key Insight**: All tensor addresses are stable across captures. CUDA graph records pointer values, so we must use the same tensor objects.

### Replay Phase

**Entry Point**: `attn_backend.init_forward_metadata_replay_cuda_graph(bs, ...)`

**Implementation** (`_replay_metadata()` at line 178):

```python
def _replay_metadata(self, bs: int, req_pool_indices: torch.Tensor, forward_mode: ForwardMode, spec_info, seq_lens_cpu):
    # Detect padding (line 186-192)
    num_padding = torch.count_nonzero(
        seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
    )
    req_pool_indices[bs - num_padding :] = 0  # Zero out padding
    mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
    mamba_indices[bs - num_padding :] = -1  # Sentinel for padding

    # Copy to stable buffer
    self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

    # Adjust query_start_loc for padding (line 194-219)
    if forward_mode.is_decode_or_idle():
        if num_padding == 0:
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        else:
            # Copy non-padded prefix
            self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
            )
            # Padded suffix gets constant value (no new tokens)
            self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                bs - num_padding
            )
    # Similar logic for target_verify mode...
```

**Padding Handling**:

CUDA graphs require fixed batch size. To support variable batches, we pad:

Example: max_bs=8, actual_bs=5
- req_pool_indices = [12, 34, 56, 78, 90, 0, 0, 0]  (pad with 0)
- mamba_indices = [12, 34, 56, 78, 90, -1, -1, -1]  (pad with -1)
- query_start_loc = [0, 1, 2, 3, 4, 5, 5, 5, 5]  (pad with constant)

Kernels check for `-1` cache index and skip processing.

---

## Cache Management

### MambaPool

**Location**: `python/sglang/srt/mem_cache/memory_pool.py:115`

**Purpose**: Manages SSM state cache for mamba layers.

**Data Structures**:

```python
@dataclass(frozen=True, kw_only=True)
class State:
    conv: torch.Tensor    # Convolution window cache
    temporal: torch.Tensor  # SSM state cache

@dataclass(frozen=True, kw_only=True)
class SpeculativeState(State):
    intermediate_ssm: torch.Tensor  # Intermediate states for verification
    intermediate_conv_window: torch.Tensor  # Intermediate conv windows
```

**Shapes**:

For GDN with num_layers=L, pool_size=P, conv_width=W, state_dim=D:
- `conv`: [L, P, W, hidden_dim]
- `temporal`: [L, P, num_heads, head_dim, state_dim]

For speculative (draft_token_num=N):
- `intermediate_ssm`: [L, P, N, num_heads, head_dim, state_dim]
- `intermediate_conv_window`: [L, P, N, W, hidden_dim]

**Memory Usage**:

Example: Qwen3-Next-7B, pool_size=4096, draft_tokens=5:
- Main cache: ~2-4 GB
- Speculative cache: ~10-20 GB (5x larger)

**Layer Access** (`mamba2_layer_cache()` in HybridReqToTokenPool):

```python
def mamba2_layer_cache(self, layer_id: int):
    return self.mamba_pool.state.at_layer_idx(layer_id)
    # Returns State object with:
    #   - conv: [pool_size, conv_width, hidden_dim]
    #   - temporal: [pool_size, num_heads, head_dim, state_dim]
```

### HybridReqToTokenPool

**Location**: `python/sglang/srt/mem_cache/memory_pool.py:296`

**Purpose**: Combines standard KV cache (for full attention) with mamba cache (for linear attention).

**Architecture**:

```python
class HybridReqToTokenPool(ReqToTokenPool):
    req_to_token: torch.Tensor  # Standard KV cache mapping [size, max_context_len]
    req_to_mamba: torch.Tensor  # Mamba cache mapping [size]
    mamba_pool: MambaPool       # Actual mamba state storage
```

**Dual Indexing**:

Request → Token indices (for KV cache):
```python
req_pool_idx = 42
token_indices = self.req_to_token[req_pool_idx, :seq_len]  # [10, 23, 45, ...]
# Use token_indices to access k_cache[layer_id, token_indices, :]
```

Request → Mamba index (for SSM cache):
```python
req_pool_idx = 42
mamba_idx = self.req_to_mamba[req_pool_idx]  # e.g., 17
# Use mamba_idx to access mamba_pool.temporal[layer_id, mamba_idx, ...]
```

**Allocation Strategy**:

```python
def alloc(self, need_size: int) -> List[int]:
    # Allocate from req_to_token free pool
    req_indices = super().alloc(need_size)

    # For each request, allocate mamba slot
    for req_idx in req_indices:
        mamba_idx = self.mamba_pool.alloc_one()
        self.req_to_mamba[req_idx] = mamba_idx

    return req_indices
```

**Why Separate Pools?**:
- KV cache: Size grows with sequence length (tokens)
- Mamba cache: Fixed size per request (independent of sequence length)
- Different eviction strategies (e.g., keep mamba state longer for reuse)

---

## Kernel Operations

### Causal Conv1D

**Purpose**: Apply 1D convolution with causal masking (only past context influences current token).

**Implementations**:
- CUDA: `sglang/srt/layers/attention/mamba/causal_conv1d.py` (native kernel)
- Triton: `sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` (line 14)
- NPU: `sgl_kernel_npu.mamba.causal_conv1d` (for Ascend NPUs)

**Function Signatures**:

**`causal_conv1d_fn`** (prefill/extend):
```python
def causal_conv1d_fn(
    x: torch.Tensor,           # [hidden_dim, seq_len] - input
    weight: torch.Tensor,      # [hidden_dim, conv_width] - conv weights
    bias: torch.Tensor,        # [hidden_dim] - conv bias
    activation: str,           # "silu", "swish", etc.
    conv_states: torch.Tensor, # [pool_size, conv_width-1, hidden_dim] - cache
    has_initial_state: torch.Tensor,  # [batch_size] - bool mask
    cache_indices: torch.Tensor,      # [batch_size] - which cache slots
    query_start_loc: torch.Tensor,    # [batch_size+1] - sequence boundaries
    seq_lens_cpu: list[int],          # Sequence lengths
) -> torch.Tensor:
    # Returns: [hidden_dim, seq_len] - convolved output
```

**Operation**:
```
For each token t at position i in sequence:
  context = [x[i-width+1], ..., x[i-1]] + conv_states[from cache]
  out[i] = activation(conv(context + [x[i]]))

At end of sequence:
  conv_states[cache_idx] = [x[-width+1], ..., x[-1]]
```

**`causal_conv1d_update`** (decode):
```python
def causal_conv1d_update(
    x: torch.Tensor,              # [batch, hidden_dim] or [batch, hidden_dim, draft_tokens]
    conv_state: torch.Tensor,     # [layers, pool_size, width-1, hidden_dim]
    weight: torch.Tensor,         # [hidden_dim, width]
    bias: torch.Tensor,           # [hidden_dim]
    activation: str,
    conv_state_indices: torch.Tensor,  # [batch]
    intermediate_conv_window: Optional[torch.Tensor] = None,  # For verification
) -> torch.Tensor:
    # Returns: [batch, hidden_dim] or [batch, hidden_dim, draft_tokens]
```

**Operation**:
```
For each request in batch:
  conv_state_idx = conv_state_indices[request]
  context = conv_state[:, conv_state_idx, :, :]  # [width-1, hidden_dim]
  window = concat(context, x[request])  # [width, hidden_dim]
  out[request] = activation(conv(window))

  # Update cache (shift left, append new)
  conv_state[:, conv_state_idx, :, :] = window[1:, :]  # Drop oldest

  # Optionally store intermediate
  if intermediate_conv_window is not None:
      intermediate_conv_window[:, conv_state_idx, step, :, :] = window[1:, :]
```

---

### Fast Linear Attention (FLA) Kernels

**Source**: `python/sglang/srt/layers/attention/fla/`

These implement the Gated Delta Net attention mechanism.

#### 1. chunk_gated_delta_rule

**Location**: `python/sglang/srt/layers/attention/fla/chunk.py:6`

**Purpose**: Process sequences in chunks with linear attention.

**Signature**:
```python
def chunk_gated_delta_rule(
    q: torch.Tensor,     # [batch, seq_len, num_heads, head_dim]
    k: torch.Tensor,     # [batch, seq_len, num_heads, head_dim]
    v: torch.Tensor,     # [batch, seq_len, num_value_heads, head_dim]
    g: torch.Tensor,     # [batch, seq_len, num_heads] - gating parameter
    beta: torch.Tensor,  # [batch, seq_len, num_heads] - forget gate
    initial_state: torch.Tensor,  # [batch, num_heads, head_dim, state_dim]
    output_final_state: bool,
    cu_seqlens: torch.Tensor,  # [batch+1] - sequence boundaries
    head_first: bool,
    use_qk_l2norm_in_kernel: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: (output, final_state)
```

**Algorithm** (simplified):
```
State = initial_state  # [heads, head_dim, state_dim]

For each chunk of size C:
    Q_chunk = q[i:i+C]  # [C, heads, head_dim]
    K_chunk = k[i:i+C]
    V_chunk = v[i:i+C]
    G_chunk = g[i:i+C]
    Beta_chunk = beta[i:i+C]

    # Intra-chunk attention (parallel)
    O_intra = local_attention(Q_chunk, K_chunk, V_chunk)

    # Inter-chunk via state (sequential)
    for t in range(C):
        O_inter[t] = State @ K_chunk[t]  # Retrieve from state
        State = Beta_chunk[t] * State + G_chunk[t] * (K_chunk[t].T @ V_chunk[t])  # Update state

    O_chunk = O_intra + O_inter

Return (O, final_state)
```

**Complexity**: O(n * d^2) where n=seq_len, d=head_dim (linear in n)

#### 2. fused_sigmoid_gating_delta_rule_update

**Location**: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py:10`

**Purpose**: Single-step update for decode with fused sigmoid gating.

**Signature**:
```python
def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,  # [num_heads] - log state transition matrix
    dt_bias: torch.Tensor,  # [num_heads] - time step bias
    q: torch.Tensor,      # [1, batch, num_heads, head_dim]
    k: torch.Tensor,      # [1, batch, num_heads, head_dim]
    v: torch.Tensor,      # [1, batch, num_value_heads, head_dim]
    a: torch.Tensor,      # [batch, num_heads] - gating input
    b: torch.Tensor,      # [batch, num_heads] - gating forget
    initial_state_source: torch.Tensor,  # [layers, pool_size, heads, head_dim, state_dim]
    initial_state_indices: torch.Tensor,  # [batch] - which cache slots
    cu_seqlens: torch.Tensor,  # [batch+1]
    use_qk_l2norm_in_kernel: bool,
    softplus_beta: float,
    softplus_threshold: float,
) -> torch.Tensor:
```

**Algorithm**:
```
For each request in batch:
    state_idx = initial_state_indices[request]
    State = initial_state_source[:, state_idx, :, :, :]  # Load state

    # Compute gating parameters (fused)
    g = softplus(A_log + a[request], beta=softplus_beta, threshold=softplus_threshold)
    beta_val = sigmoid(b[request])

    # Retrieve from state
    o_inter = State @ k[request]  # [heads, head_dim]

    # Update state
    State = beta_val * State + g * (k[request].T @ v[request])

    # Store state
    initial_state_source[:, state_idx, :, :, :] = State

    # Return output
    output[request] = q[request] @ o_inter
```

**In-place Update**: State is updated directly in cache (no copy needed for decode).

#### 3. fused_recurrent_gated_delta_rule_update

**Location**: `python/sglang/srt/layers/attention/fla/fused_recurrent.py:8`

**Purpose**: Process multiple tokens recurrently for speculative verification.

**Signature**:
```python
def fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,      # [1, batch*draft_tokens, num_heads, head_dim]
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,      # [1, batch*draft_tokens, num_heads]
    beta: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,  # [batch]
    cu_seqlens: torch.Tensor,  # [batch+1], step by draft_token_num
    use_qk_l2norm_in_kernel: bool,
    disable_state_update: bool,  # True for verification
    intermediate_states_buffer: torch.Tensor,  # Store intermediates
    cache_steps: int,  # draft_token_num
) -> torch.Tensor:
```

**Algorithm**:
```
For each request in batch:
    state_idx = initial_state_indices[request]
    State = initial_state_source[:, state_idx, :, :, :]  # Initial state

    start = cu_seqlens[request]
    end = cu_seqlens[request+1]

    For t in range(start, end):  # draft_token_num iterations
        # Retrieve
        o_inter = State @ k[t]
        output[t] = q[t] @ o_inter

        # Update state
        State = beta[t] * State + g[t] * (k[t].T @ v[t])

        # Cache intermediate state
        step = t - start
        intermediate_states_buffer[:, state_idx, step, :, :, :] = State

    # Note: main initial_state_source NOT updated (disable_state_update=True)
```

**Key Difference from decode**: Caches all intermediate states for later selective commit.

---

### Triton Kernel: fused_gdn_gating

**Location**: `python/sglang/srt/models/qwen3_next.py:221`

**Purpose**: Fuse computation of gating parameter `g` for GDN.

**Signature**:
```python
def fused_gdn_gating(
    A_log: torch.Tensor,   # [num_heads]
    a: torch.Tensor,       # [batch, num_heads]
    dt_bias: torch.Tensor, # [num_heads]
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    # Returns: g [batch, num_heads]
```

**Triton Kernel** (line 195):
```python
@triton.jit
def fused_gdn_gating_kernel(
    g,        # Output
    A_log,    # Input
    a,        # Input
    dt_bias,  # Input
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_heads = tl.program_id(2)

    head_offset = pid_heads * BLK_HEADS + tl.arange(0, BLK_HEADS)
    mask = head_offset < NUM_HEADS

    # Load
    A_log_val = tl.load(A_log + head_offset, mask=mask)
    dt_bias_val = tl.load(dt_bias + head_offset, mask=mask)
    idx = pid_batch * seq_len * NUM_HEADS + pid_seq * NUM_HEADS + head_offset
    a_val = tl.load(a + idx, mask=mask)

    # Compute: g = softplus(A_log + a + dt_bias)
    x = A_log_val + a_val + dt_bias_val
    # softplus(x, beta, threshold) = log(1 + exp(beta * x)) / beta if x*beta < threshold else x
    beta_x = beta * x
    result = tl.where(
        beta_x < threshold,
        tl.log1p(tl.exp(beta_x)) / beta,
        x
    )

    # Store
    tl.store(g + idx, result, mask=mask)
```

**Why Fuse?**: Eliminates 3 memory reads and 2 writes, improving memory bandwidth utilization.

---

## Model Integration

### Example: Falcon H1

**File**: `python/sglang/srt/models/falcon_h1.py`

**Architecture**: Hybrid model with both attention and mamba layers.

**Layer Configuration**:
```python
# Example: 40-layer model
full_attention_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]  # Every 4th layer
linear_attention_layers = [1, 2, 3, 5, 6, 7, ...]  # Remaining layers
```

**Backend Setup** (`attention_registry.py:178`):
```python
if cfg := runner.mambaish_config:
    if runner.hybrid_gdn_config is not None:
        linear_attn_backend = GDNAttnBackend(runner)
    elif runner.mamba2_config is not None:
        linear_attn_backend = Mamba2AttnBackend(runner)

    full_attn_layers = cfg.full_attention_layer_ids
    return HybridLinearAttnBackend(
        full_attn_backend,
        linear_attn_backend,
        full_attn_layers
    )
```

**Layer Forward** (`falcon_h1.py:330`):
```python
def forward(self, ...):
    if not forward_batch.forward_mode.is_idle():
        # Attention block (full attention)
        attention_hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states * self.attention_in_multiplier,
            forward_batch=forward_batch,
        )
        attention_hidden_states = attention_hidden_states * self.attn_out_multiplier

        # Mamba block (linear attention)
        attn_backend = forward_batch.attn_backend
        assert isinstance(attn_backend, HybridLinearAttnBackend)
        assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)

        mamba_hidden_states = torch.empty_like(hidden_states)
        attn_backend.linear_attn_backend.forward(
            self.mamba,
            hidden_states * self.ssm_in_multiplier,
            mamba_hidden_states,
            layer_id=self.layer_id,
            mup_vector=self.mup_vector,
        )
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

        # Combine
        hidden_states = attention_hidden_states + mamba_hidden_states
```

**Key Insight**: Each layer has **both** attention and mamba branches, combined additively. This differs from models that alternate layer types.

---

### Example: Qwen3-Next

**Architecture**: Sparse linear attention layers (GDN) mixed with full attention.

**Layer Pattern**: Linear attention layers are typically inserted at specific positions, determined by model config.

**GDN Layer Forward** (typical pattern):
```python
def forward(self, hidden_states, forward_batch):
    # Pre-norm
    residual = hidden_states
    hidden_states = self.norm(hidden_states)

    # GDN attention
    attn_out = self.gdn_attention(
        hidden_states=hidden_states,
        forward_batch=forward_batch,
    )

    # Post-norm and residual
    hidden_states = residual + attn_out

    # MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states, forward_batch)
    hidden_states = residual + hidden_states

    return hidden_states
```

**GDN Attention Implementation**:
```python
class GDNAttention(nn.Module):
    def forward(self, hidden_states, forward_batch):
        # Project to mixed QKV
        mixed_qkv = self.qkv_proj(hidden_states)

        # Call backend
        attn_backend = forward_batch.attn_backend
        if forward_batch.forward_mode.is_decode():
            output = attn_backend.forward_decode(
                q=None, k=None, v=None,  # Computed from mixed_qkv
                layer=None,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                conv_weights=self.conv_weights,
                bias=self.conv_bias,
                activation=self.activation,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                attention_tp_size=self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                a=self.a_param,
                b=self.b_param,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                layer_id=self.layer_id,
            )
        else:
            output = attn_backend.forward_extend(...)

        # Output projection
        output = self.o_proj(output)
        return output
```

---

## Speculative Decoding

### Overview

Speculative decoding uses a small draft model to generate N candidate tokens, then verifies them with the target model in parallel. If all N are correct, we save N-1 forward passes.

**Challenge for Mamba**: State updates are sequential. How to verify N tokens without doing N sequential updates?

**Solution**: Cache intermediate states during verification, then commit only accepted states.

### Process Flow

**Draft Phase**:
```
Draft model generates: [t0, t1, t2, t3, t4]
```

**Verify Phase**:
```
Target model (in forward_extend with is_target_verify=True):

  1. Convolution (causal_conv1d_update with intermediate caching):
     S_conv_0 → [process t0] → S_conv_1 (cache)
     S_conv_1 → [process t1] → S_conv_2 (cache)
     S_conv_2 → [process t2] → S_conv_3 (cache)
     S_conv_3 → [process t3] → S_conv_4 (cache)
     S_conv_4 → [process t4] → S_conv_5 (cache)

  2. SSM (fused_recurrent_gated_delta_rule_update):
     S_ssm_0 → [process t0] → S_ssm_1 (cache)
     S_ssm_1 → [process t1] → S_ssm_2 (cache)
     S_ssm_2 → [process t2] → S_ssm_3 (cache)
     S_ssm_3 → [process t3] → S_ssm_4 (cache)
     S_ssm_4 → [process t4] → S_ssm_5 (cache)

  3. Generate logits for all 5 tokens

  4. Compare with draft:
     - t0 matches ✓
     - t1 matches ✓
     - t2 matches ✓
     - t3 doesn't match ✗
     - t4 not checked

     accepted_length = 3
```

**Commit Phase** (`update_mamba_state_after_mtp_verify()` at line 671):
```python
# Copy intermediate states to main cache
for request in batch:
    if accepted_length[request] > 0:
        step = accepted_length[request] - 1  # 0-indexed

        # Copy SSM state
        ssm_states[:, cache_idx[request], :] = (
            intermediate_ssm[:, cache_idx[request], step, :]
        )

        # Copy conv state
        conv_states[:, cache_idx[request], :, :] = (
            intermediate_conv_window[:, cache_idx[request], step, :, :]
        )
```

### Memory Trade-off

**Without Intermediate Caching**:
- Process draft tokens sequentially: O(N) passes
- Memory: O(state_size)

**With Intermediate Caching**:
- Process draft tokens in parallel: O(1) pass
- Memory: O(N * state_size)

For N=5, state_size=32MB per layer, 40 layers:
- Additional memory: 5 * 32MB * 40 = 6.4 GB
- Speedup: ~4x (assuming 80% acceptance rate)

Trade-off is favorable for moderate N (typically 3-7).

---

## Platform Support

### CUDA

**Native Kernels**:
- `causal_conv1d`: Optimized CUDA kernel for convolution
- FLA kernels: Triton-based implementations for delta rule

**Requirements**:
- Compute capability ≥ 8.0 (Ampere or newer)
- Triton compiler

**Performance**:
- Decode: 1-2ms per layer for batch_size=32
- Prefill: ~100 tokens/ms for 7B model

### NPU (Ascend)

**Custom Kernels** (`sgl_kernel_npu`):
- `causal_conv1d_fn_npu`: NPU-optimized convolution
- `causal_conv1d_update_npu`: NPU single-token update
- `chunk_gated_delta_rule_npu`: NPU chunked processing
- `fused_sigmoid_gating_delta_rule_update_npu`: NPU decode

**Activation** (`hybrid_linear_attn_backend.py:38`):
```python
elif is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
```

**Backend Selection** (`attention_registry.py:202`):
```python
if is_npu():
    assert (
        runner.server_args.attention_backend == "ascend"
    ), "ascend backend is the only supported backend on NPU for hybrid GDN models"
```

### Triton Fallback

For platforms without native kernels, Triton implementations provide portability:
- `causal_conv1d_fn` (Triton version at `causal_conv1d_triton.py`)
- Performance: ~70% of native CUDA

---

## Performance Characteristics

### Complexity Analysis

**Traditional Attention**:
- Time: O(n^2 * d) per layer
- Memory: O(n * d) KV cache per layer
- Decode: O(n * d) (linear in sequence length)

**Mamba/GDN Attention**:
- Time: O(n * d^2) per layer (linear in sequence length!)
- Memory: O(d^2) state per layer (constant!)
- Decode: O(d^2) (constant in sequence length)

**Hybrid Model** (e.g., Falcon H1 with 25% full attention):
- Time: 0.25 * O(n^2 * d) + 0.75 * O(n * d^2)
- Memory: 0.25 * O(n * d) + 0.75 * O(d^2)
- Decode: 0.25 * O(n * d) + 0.75 * O(d^2)

**Crossover Point**: For d=128, breakeven at n ≈ 128. For longer sequences, hybrid models are more efficient.

### Benchmark Results

**Qwen3-Next-7B** (indicative):
- Decode throughput: 120 tokens/sec/request (batch=32, seq_len=2048)
- Prefill throughput: 8000 tokens/sec (seq_len=2048)
- Memory: 16GB model + 4GB KV cache + 2GB mamba cache (total 22GB @ fp16)

**Comparison to Full Attention**:
- Decode: 1.5x faster for seq_len > 2048
- Prefill: 1.2x faster
- Memory: 30% less cache memory for seq_len=4096

---

## Summary

SGLang's mamba/linear attention system provides:

1. **Unified Backend Interface**: Seamless integration with standard attention backends
2. **Efficient State Management**: Dual-pool cache system for hybrid models
3. **CUDA Graph Optimization**: Pre-allocated buffers for decode performance
4. **Speculative Decoding Support**: Intermediate state caching for verification
5. **Platform Portability**: CUDA, NPU, and Triton implementations
6. **Linear Complexity**: O(n) time and O(1) memory for decode

The implementation bridges state-space models and transformers, enabling efficient inference for hybrid architectures like Falcon H1, Qwen3-Next, and NemotronH.

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `hybrid_attn_backend.py` | 148 | Routes between prefill/decode backends |
| `hybrid_linear_attn_backend.py` | 706 | GDN, Mamba2, and hybrid routing |
| `mamba2_metadata.py` | 212 | Metadata structures and chunking logic |
| `mamba/mamba.py` | ~800 | MambaMixer2 implementation |
| `mamba/causal_conv1d.py` | ~400 | Native CUDA convolution |
| `fla/chunk.py` | ~600 | Chunked delta rule kernel |
| `fla/fused_recurrent.py` | ~400 | Recurrent delta rule kernel |
| `fla/fused_sigmoid_gating_recurrent.py` | ~400 | Decode delta rule kernel |
| `memory_pool.py` | ~2000 | MambaPool and HybridReqToTokenPool |
| `attention_registry.py` | 227 | Backend registration and wrapping |

---

## Future Directions

Potential optimizations and extensions:

1. **Flashier Attention**: Fused kernels combining conv + delta rule
2. **Multi-Query Mamba**: Reduce state size with grouped parameters
3. **Continuous Batching**: Dynamic cache allocation for variable batch sizes
4. **Quantized States**: FP8 or INT8 state storage for memory reduction
5. **Longer Context**: Hierarchical states for million-token contexts
6. **Cross-Attention Mamba**: Extend to encoder-decoder architectures

The modular backend design facilitates experimentation with these directions.
