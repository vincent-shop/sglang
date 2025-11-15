# SGLang Triton Attention Backend System Documentation

## 1. Introduction

The Triton attention backend provides GPU-optimized attention implementations using OpenAI's Triton programming language. This backend serves as an alternative to FlashInfer for attention computation in SGLang's LLM serving system, supporting page size = 1 KV cache management.

### Major Subsystems

**Attention Operation Kernels (`triton_ops/`):**
- **Prefill Attention (`prefill_attention.py`)**: Memory-efficient attention for initial context processing
- **Decode Attention (`decode_attention.py`)**: Two-stage flash decoding for autoregressive generation
- **Extend Attention (`extend_attention.py`)**: Hybrid attention combining cached prefix with new tokens
- **Merge State (`merge_state.py`)**: Combines attention states from multiple computation stages

**Backend Integration (`triton_backend.py`):**
- ForwardMetadata management
- Buffer allocation and lifecycle
- CUDA graph support
- Sliding window attention coordination
- Deterministic inference mode

### Architecture Overview

The system operates in three distinct phases:

- `prefill_attention.py:170-217` - Prefill processes initial prompts using blocked matrix operations with online softmax
- `decode_attention.py:633-777` - Decode uses split-K parallelization for autoregressive token generation
- `extend_attention.py:542-644` - Extend handles prefix KV cache + new token computation

### Complexity Notes

The extend attention implementation has two operational modes:

1. **Standard 2-stage mode:** Separate prefix and extend token processing with state merging
2. **Unified 1-stage mode (deterministic):** Single-pass computation through merged KV indices for batch-invariant execution

The decode attention employs adaptive KV splitting based on sequence length distribution and hardware core count, which can be challenging to tune for optimal performance.

---

## 2. Overview

### System Architecture

The Triton backend implements a staged attention pipeline where computation flows through different kernels based on the inference phase (prefill/decode/extend). Each stage is optimized for its specific access patterns.

### Key Components

**Attention Kernels** - Triton JIT-compiled GPU kernels implementing flashattention algorithms
- Use tiled computation with online softmax to minimize memory bandwidth
- Support grouped-query attention (GQA), multi-query attention (MQA), and multi-head attention (MHA)
- Handle variable sequence lengths through indirection buffers

**Memory Management:**
- KV cache accessed via indirection pointers (`kv_indptr`, `kv_indices`)
- Page-based memory pooling with size-1 pages
- Supports sliding window attention via separate window buffers

**Backend Coordinator (`TritonAttnBackend`):**
- Manages metadata initialization per forward batch
- Allocates intermediate buffers (logits, LSE)
- Handles CUDA graph capture/replay
- Coordinates between different attention types per layer

### Inter-Component Communication

`triton_backend.py:226-426` - `init_forward_metadata()` prepares indirection buffers based on ForwardMode:

- **Decode mode:** Creates `kv_indptr`/`kv_indices` from `req_to_token` mapping
- **Extend mode:** Creates separate `qo_indptr` for query offsets, `kv_indptr` for prefix
- **Target verify:** Builds custom mask for speculative decoding tree attention

**Data Flow:**
1. ModelRunner calls `init_forward_metadata(forward_batch)` before each layer
2. Backend extracts sequence metadata from `forward_batch`
3. Triton kernels launched with metadata tensors as arguments
4. Kernels read/write KV cache through indirection buffers
5. Output accumulated in provided output tensor

### Key Data Structures

**ForwardMetadata (`triton_backend.py:37-53`):**
```python
@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor      # [bs, heads, splits, v_dim] - partial outputs
    attn_lse: torch.Tensor         # [bs, heads, splits] - log-sum-exp values
    max_extend_len: int            # Maximum extend sequence length
    num_kv_splits: torch.Tensor    # [bs] - KV split counts per sequence
    kv_indptr: torch.Tensor        # [bs+1] - CSR-style sequence offsets
    kv_indices: torch.Tensor       # [total_kv] - Physical KV cache locations
    qo_indptr: torch.Tensor        # [bs+1] - Query/output offsets (extend mode)
    custom_mask: torch.Tensor      # Custom attention mask (speculative)
    mask_indptr: torch.Tensor      # [bs+1] - Mask offsets
    # Sliding window specific buffers
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
```

**Indirection Buffer Layout:**
- `kv_indptr[i]` to `kv_indptr[i+1]` spans sequence i's KV cache locations
- `kv_indices[kv_indptr[i]:kv_indptr[i+1]]` contains physical page indices
- Similar CSR structure for query offsets in extend mode

### Important Algorithms

**Online Softmax (used across all kernels):**
- Computes softmax incrementally as tiles are processed
- Maintains running max (`m_i`) and sum (`l_i`)
- Updates via rescaling: `acc = acc * exp(m_old - m_new) + new_contribution`
- See `prefill_attention.py:130-157` for reference implementation

**Split-K Decode Attention (`decode_attention.py`):**
- Stage 1 (`_fwd_kernel_stage1`): Splits KV sequence into chunks, computes partial attention per chunk
- Stage 2 (`_fwd_kernel_stage2`): Reduces partial results using softmax combining
- Adaptive splitting via `get_num_kv_splits()` based on sequence length distribution

**Unified Extend Attention (`extend_attention.py:935-1046`):**
- Single-pass kernel processing merged prefix+extend KV indices
- Deterministic execution for batch-invariant operations
- Properly handles causal masking boundaries between prefix and extend regions

### Synchronization and Concurrency

**Global Synchronization:**
- All Triton kernels execute asynchronously on default CUDA stream
- Synchronization implicit through PyTorch's execution model
- CUDA graphs capture entire attention operation for replay

**Within-Kernel Parallelism:**
- Each thread block processes independent (batch, head, tile_m) combinations
- No inter-block communication required
- Atomic operations not used (accumulation done via separate output buffers per split)

### Design Analysis

**Why Split-K Decoding Works:**
- Long sequences (>8K tokens) would otherwise underutilize GPU cores
- Splitting KV dimension allows more parallelism than just (batch × heads)
- Trade-off: Extra reduction overhead vs. improved occupancy

**Limitation - Page Size Restriction:**

Both documentation and code comments state "supports page size = 1" (`prefill_attention.py:16`, `decode_attention.py:16`). This is because:
- Triton implementation uses direct token→physical location mapping
- Larger pages would require different addressing logic
- FlashInfer backend supports arbitrary page sizes through more complex indexing

**Sliding Window Attention:**
Implemented via separate indirection buffers (`window_kv_*`) that track only the recent window of KV cache. The kernel applies additional masking to enforce window boundaries (`extend_attention.py:337-343, extend_attention.py:825-840`).

---

## 3. Topics

### 3.1 Prefill Attention

- **Files:** `triton_ops/prefill_attention.py`
- **Purpose:** Processes initial prompt tokens in a single pass, computing full bidirectional or causal attention over the input sequence.
- **Key Function:** `context_attention_fwd()` (`prefill_attention.py:170-217`)

**Algorithm:**
1. Launch 3D grid: (batch, heads, num_tiles_m) where each block processes BLOCK_M query tokens
2. For each query block, iterate over all KV blocks (or up to diagonal for causal)
3. Compute `qk = q @ k.T * sm_scale` with proper masking
4. Apply online softmax update: update `m_i`, `l_i`, rescale accumulator
5. Compute `acc += softmax(qk) @ v`
6. Store final `acc / l_i` to output

**Triton Kernel:** `_fwd_kernel()` (`prefill_attention.py:34-168`)

**Blocking Strategy:**
- BLOCK_M, BLOCK_N: Query/KV tile sizes (64 or 128 depending on GPU architecture)
- BLOCK_DMODEL: Padded to next power-of-2 for vectorization
- Handles non-power-of-2 head dimensions via masking (`mask_d`)

**Causal Masking:**
```python
# prefill_attention.py:118-128
if IS_CAUSAL:
    qk += tl.where(
        (start_n + offs_n[None, :] < cur_batch_seq_len)
        & (offs_m[:, None] >= (start_n + offs_n[None, :])),
        0,
        float("-inf"),
    )
```
Only loads KV up to `(start_m + 1) * BLOCK_M` tokens for causal attention.

**Usage:** Called during initial prompt processing (PREFILL forward mode).

---

### 3.2 Decode Attention

- **Files:** `triton_ops/decode_attention.py`
- **Purpose:** Efficiently computes attention for single-token generation using a two-stage flash decoding algorithm.
- **Entry Points:**
  - `decode_attention_fwd()` - Main dispatcher (`decode_attention.py:719-777`)
  - `decode_attention_fwd_normal()` - MHA path (`decode_attention.py:633-674`)
  - `decode_attention_fwd_grouped()` - GQA/MQA/MLA path (`decode_attention.py:676-717`)

**Two-Stage Algorithm:**

**Stage 1** (`_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1`):
- Grid: (batch, heads/head_groups, max_kv_splits)
- Each block processes a chunk of KV cache: `[split_start:split_end]`
- Computes partial attention output and log-sum-exp per chunk
- Stores to intermediate buffers: `Att_Out[b,h,split,:]`, `Att_Lse[b,h,split]`

```python
# decode_attention.py:97-101
kv_len_per_split = (
    tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
)
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
```

**Stage 2** (`_fwd_kernel_stage2`, `_decode_softmax_reducev_fwd`):
- Grid: (batch, heads)
- Reduces partial outputs using softmax combining formula
- Handles up to `MAX_KV_SPLITS` partial results

```python
# decode_attention.py:564-572
n_e_max = tl.maximum(tlogic, e_max)
old_scale = tl.exp(e_max - n_e_max)
acc *= old_scale
exp_logic = tl.exp(tlogic - n_e_max)
acc += exp_logic * tv
e_sum = e_sum * old_scale + exp_logic
e_max = n_e_max
```

**Grouped Attention Optimization** (`_fwd_grouped_kernel_stage1`):
- Processes multiple query heads per block (`BLOCK_H=16`) that share same KV head
- Vectorizes QK computation across head dimension
- Critical for high GQA ratios (e.g., 8:1, 16:1)

**Adaptive KV Splitting:**

The `get_num_kv_splits()` method (`triton_backend.py:174-224`) dynamically determines split count based on:
1. Sequence length variance: More splits if sequences have similar lengths
2. Hardware utilization: Target `device_core_count` total thread blocks
3. Head grouping: Accounts for BLOCK_H fusion in grouped kernel

**Deterministic Mode:**
When `enable_deterministic=True`, uses fixed split tile size (`split_tile_size`, default 256):

```python
num_kv_splits[:] = (seq_lens + split_tile_size - 1) // split_tile_size
```
This ensures same computation graph regardless of batch composition.

**Special Features:**
- **Logit capping (`logit_cap`)**: Applies tanh capping to attention scores
- **XAI temperature scaling (`xai_temperature_len`)**: Position-dependent temperature
- **Sink tokens (`sinks`)**: Adds fixed attention mass to specific tokens

---

### 3.3 Extend Attention

- **Files:** `triton_ops/extend_attention.py`
- **Purpose:** Handles "prefill with KV cache" scenario where new tokens attend to both cached prefix and themselves.
- **Two Operational Modes:**

**Standard 2-Stage Mode** (`extend_attention_fwd`, `triton_backend.py:787-856`):

1. Compute attention to prefix KV cache (loaded via indirection)
2. Compute attention to new extend tokens (contiguous)
3. Combine states using online softmax updates within same kernel

**Unified 1-Stage Mode** (`extend_attention_fwd_unified`, `triton_backend.py:858-977`):
- Used when `enable_deterministic=True`
- Merges prefix and extend KV indices into unified buffer
- Single kernel pass processes all KV through indirection
- Simplifies control flow for batch-invariant execution

**Standard Mode Kernel** (`_fwd_kernel`, `extend_attention.py:211-540`):

```python
# Stage 1: Prefix attention (lines 319-413)
for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
    # Load K,V from k_buffer/v_buffer via kv_indices
    # Apply custom mask and sliding window mask
    # Update accumulator with online softmax

# Stage 2: Extend attention (lines 421-517)
for start_n in range(0, cur_block_m_end, BLOCK_N):
    # Load K,V from k_extend/v_extend (contiguous)
    # Apply causal mask for triangular portion
    # Update accumulator with online softmax
```

**Unified Mode Kernel** (`_fwd_kernel_unified`, `extend_attention.py:683-933`):

```python
# Unified loop over all KV (prefix + extend)
for start_n in range(0, cur_seq_kv_len, BLOCK_N):
    # Load K,V from k_buffer via unified_kv_indices
    # Causal mask only for extend×extend portion:
    if IS_CAUSAL and not USE_CUSTOM_MASK:
        k_is_extend = k_idx_in_total >= cur_seq_prefix_len
        causal_mask = tl.where(k_is_extend, q_idx >= k_idx_in_extend, True)
```

**Unified Index Building** (`build_unified_kv_indices`, `extend_attention.py:163-208`):
- Fuses prefix and extend indices into single contiguous buffer
- Uses Triton kernel `_copy_unified_indices_kernel` for parallel copying
- Maintains `prefix_lens` tensor to distinguish boundaries

**Block Size Selection:**

`_get_block_sizes_for_extend_attention()` (`extend_attention.py:35-96`) chooses tile sizes based on:
- Head dimension (special handling for 576, 288, 192)
- GPU architecture (sm_80: A100, sm_86: A10/L4, sm_89: H100)
- Shared memory constraints (sm_86/89 have 100KB vs. 160KB on sm_80)

**Sliding Window Attention:**
- Applies mask: `q_abs_pos <= k_abs_pos + SLIDING_WINDOW_SIZE`
- `window_kv_offsets` tracks the trimmed prefix start position
- See `extend_attention.py:337-343` (standard mode), `extend_attention.py:825-840` (unified mode)

**Custom Masks:**
Used for speculative decoding tree attention (DRAFT_EXTEND, TARGET_VERIFY modes):
- `mask_ptr` points to per-query-token mask array
- `mask_indptr` provides CSR-style indexing per sequence
- `SKIP_PREFIX_CUSTOM_MASK` flag can skip mask checks for prefix region

---

### 3.4 Merge State

- **Files:** `triton_ops/merge_state.py`
- **Purpose:** Combines two attention states (with separate log-sum-exp values) into a single output using the flashattention state merging formula.
- **Function:** `merge_state_triton()` (`merge_state.py:66-96`)

**Use Case:** Used when combining outputs from multiple attention computation stages, though the current codebase doesn't show active usage (likely vestigial or for future extensions).

**Kernel:** `merge_state_kernel()` (`merge_state.py:8-64`)

**Algorithm:**
```python
# merge_state.py:30-58
max_lse = tl.maximum(p_lse, s_lse)
p_lse = p_lse - max_lse
s_lse = s_lse - max_lse
out_se = tl.exp(p_lse) + tl.exp(s_lse)

p_scale = tl.exp(p_lse) / out_se
s_scale = tl.exp(s_lse) / out_se
out = p_out * p_scale + s_out * s_scale
```
This implements the standard formula for merging softmax-normalized states:
- Given states v_a with log-sum-exp s_a, and v_b with s_b
- Merged state: v = (exp(s_a)*v_a + exp(s_b)*v_b) / (exp(s_a) + exp(s_b))
- Numerically stable via max subtraction

**Grid:** (num_tokens, num_heads) - each thread block processes one (token, head) pair.

---

### 3.5 Backend Integration

- **Files:** `triton_backend.py`
- **Main Class:** `TritonAttnBackend` (`triton_backend.py:55-1030`)

**Initialization** (`__init__`, `triton_backend.py:56-172`):
- Imports kernel functions with `torch.compiler.disable` to prevent `torch.compile` interference
- Allocates reusable buffers: `kv_indptr`, `qo_indptr`, `mask_indptr`
- Configures KV splitting strategy based on `static_kv_splits`, `split_tile_size`
- Sets up sliding window buffers if `sliding_window_size > 0`
- Initializes deterministic mode if `enable_deterministic_inference=True`

**Metadata Initialization** (`init_forward_metadata`, `triton_backend.py:226-426`):

Prepares per-batch metadata based on ForwardMode:

- **DECODE/IDLE:**
  ```python
  # triton_backend.py:237-289
  kv_indptr[1:bs+1] = torch.cumsum(forward_batch.seq_lens, dim=0)
  kv_indices = torch.empty(forward_batch.seq_lens_sum, ...)
  create_flashinfer_kv_indices_triton[(bs,)](...)  # Fills kv_indices
  num_kv_splits = torch.empty((bs,), ...)
  self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)
  ```
- **EXTEND:**
  ```python
  # triton_backend.py:368-410
  kv_indptr: from extend_prefix_lens (prefix only)
  qo_indptr: from extend_seq_lens (new tokens)
  max_extend_len = max(extend_seq_lens_cpu)
  ```
- **TARGET_VERIFY** (speculative decoding verify phase):
  ```python
  # triton_backend.py:295-348
  qo_indptr: uniform draft_token_num per sequence
  custom_mask: from spec_info
  mask_indptr: CSR indexing for mask
  ```

**Forward Dispatch:**
- **Extend:** `forward_extend()` (`triton_backend.py:787-856`)
  - Saves KV cache first (must precede unified kernel in deterministic mode)
  - Dispatches to `extend_attention_fwd()` or `_forward_extend_unified()`
  - Handles sliding window by swapping metadata buffers

- **Decode:** `forward_decode()` (`triton_backend.py:979-1029`)
  - Saves KV cache
  - Calls `decode_attention_fwd()` with prepared metadata
  - Handles sliding window via metadata buffer selection

**CUDA Graph Support:**

- **Capture:** `init_forward_metadata_capture_cuda_graph()` (`triton_backend.py:495-649`)
  - Pre-allocates fixed-size CUDA graph buffers
  - Fills metadata using capture-time values

- **Replay:** `init_forward_metadata_replay_cuda_graph()` (`triton_backend.py:651-769`)
  - Updates variable metadata (seq_lens, req_pool_indices)
  - Recomputes num_kv_splits for current batch
  - Reuses pre-allocated kv_indices buffers

**Multi-Step Draft Backend** (`TritonMultiStepDraftBackend`, `triton_backend.py:1032-1186`):
- Wraps multiple TritonAttnBackend instances for speculative decoding
- Generates KV indices for all draft steps in parallel
- Each step uses a slice of the kv_indptr and kv_indices buffers

---

### 3.6 Sliding Window Attention Support

**Implementation:** `update_sliding_window_buffer()` (`triton_backend.py:1240-1277`)

**Algorithm:**
```python
window_kv_lens = min(seq_lens, sliding_window_size)  # Trim to window
window_kv_start_idx = seq_lens - window_kv_lens      # Start offset
# Build indirection for only the window region
create_flashinfer_kv_indices_triton(
    ..., window_kv_lens, ..., window_kv_start_idx, window_kv_indices, ...
)
```

**Usage:**
- Called during metadata initialization for DECODE and EXTEND modes
- Populates `forward_metadata.window_kv_*` fields
- Kernels apply additional masking to enforce window boundaries
- See `extend_attention.py:337-343`, `extend_attention.py:825-840`

**Translation for SWA Allocator:**
If the KV pool uses separate sliding window allocation (`translate_loc_from_full_to_swa`), translates physical indices after building the indirection buffer.

---

## 4. Code Organization

**Directory Structure:**
```
triton_ops/
  prefill_attention.py    # Context attention for prefill
  decode_attention.py     # Two-stage flash decoding
  extend_attention.py     # Extend + unified extend kernels
  merge_state.py          # Attention state merging
  double_sparsity_attention.py  # Advanced sparsity (not covered)
  rocm_mla_decode_rope.py       # ROCm-specific MLA kernels

triton_backend.py         # Backend integration and orchestration
```

**Key External Dependencies:**
- `sglang.srt.layers.attention.base_attn_backend.AttentionBackend` - Base class
- `sglang.srt.model_executor.forward_batch_info.ForwardBatch` - Batch metadata
- `sglang.srt.layers.radix_attention.RadixAttention` - Layer-specific configs
- `sglang.srt.layers.attention.utils.create_flashinfer_kv_indices_triton` - Index builder

**Calling Relationships:**
```
ModelRunner.forward_batch_generation()
  → TritonAttnBackend.init_forward_metadata(forward_batch)
  → attention_layer.forward(...)
    → TritonAttnBackend.forward_{extend|decode}(q, k, v, layer, forward_batch)
      → {extend|decode}_attention_fwd(q, k, v, o, kv_indptr, kv_indices, ...)
        → Triton kernel launch: _fwd_kernel[grid](...)
```

### Hardware-Specific Code Paths

**CUDA Capability Detection:**
```python
# prefill_attention.py:179-182, extend_attention.py:67-92
if CUDA_CAPABILITY[0] >= 9:      # Hopper (H100)
    BLOCK_M, BLOCK_N = (128, 64)
elif CUDA_CAPABILITY[0] >= 8:   # Ampere (A100)
    if CUDA_CAPABILITY[1] == 6:  # A10/L4 (100KB shmem)
        BLOCK_M, BLOCK_N = (64, 64)
```

**ROCm (AMD GPU) Support:**
```python
# decode_attention.py:472-476, extend_attention.py:594-596
if _is_hip:
    extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
    num_stages = 1  # Reduce stages for shmem constraints
```

---

## 5. Edge Cases and Design Decisions

### Edge Cases Handled

**Empty Sequences:**
- Kernels check `cur_batch_seq_len > 0` / `split_kv_end > split_kv_start`
- Empty splits skipped in decode stage 1, stage 2 handles gracefully

**Non-Power-of-2 Head Dimensions:**
- Pad to `triton.next_power_of_2(head_dim)` for vectorization
- Use `mask_d` to disable out-of-bounds elements
- See `prefill_attention.py:81-87`, `extend_attention.py:280-281`

**Sequence Length Variation in Batch:**
- Adaptive KV splitting based on length distribution
- Per-sequence `num_kv_splits[i]` accounts for individual lengths

**Speculative Decoding Tree Attention:**
- Custom masks provided via `custom_mask` tensor (uint8)
- Tree structure encoded in mask, applied per query-key pair
- See TARGET_VERIFY and DRAFT_EXTEND modes

### Design Rationale

**Why Split-K Decoding?**
- Long sequences (>8K tokens) underutilize GPU without splitting
- Splitting KV creates more thread blocks than (batch × heads) alone
- Trade-off: Reduction overhead vs. improved parallelism

**Why Two Extend Modes?**
- Standard mode: Optimized for dynamic batches, minimizes kernel complexity
- Unified mode: Batch-invariant execution for deterministic inference, enables CUDA graph sharing across batch sizes

**Why Triton Over CUDA?**
- Triton provides higher-level abstractions (tile programming, auto-tuning)
- Easier to maintain and extend than raw CUDA
- Performance competitive with hand-tuned CUDA for attention kernels

**Limitation - Page Size = 1:**
- Simplifies addressing: `kv_indices[i]` directly maps to physical location
- Larger pages require more complex offset calculations within page
- Trade-off: Simpler code vs. slightly higher metadata overhead

---

## 6. Integration Points

**Model Runner Integration:**
```python
# ModelRunner.__init__ creates backend
self.attn_backend = TritonAttnBackend(self)

# Per-batch initialization
self.attn_backend.init_forward_metadata(forward_batch)

# Layer forward pass
output = self.attn_backend.forward_{extend|decode}(q, k, v, layer, forward_batch)
```

**Attention Layer Integration:**
See `sglang.srt.layers.radix_attention.RadixAttention.forward()` which calls backend methods.

**Speculative Decoding Integration:**
- SpecInput provides `kv_indptr`, `kv_indices`, `custom_mask`
- Multi-step draft uses `TritonMultiStepDraftBackend` for parallel step processing

**CUDA Graph Integration:**
- Capture: `init_cuda_graph_state()`, `init_forward_metadata_capture_cuda_graph()`
- Replay: `init_forward_metadata_replay_cuda_graph()`
- Buffers pre-allocated and reused across replays

---

## 7. Performance Characteristics

**Prefill:**
- Memory-bound due to KV cache writes
- Performance scales with sequence length and batch size
- Block sizes tuned per GPU architecture

**Decode:**
- Compute-bound for split-K parallelization
- Stage 1: Parallelism = batch × heads × num_splits
- Stage 2: Parallelism = batch × heads (reduction bottleneck)
- Performance sensitive to split count tuning

**Extend:**
- Hybrid characteristics: prefix reads (memory-bound), extend compute (compute-bound)
- Unified mode has simpler control flow, potentially better for large batches

**Tuning Knobs:**
- `triton_attention_num_kv_splits`: Maximum KV splits for decode
- `triton_attention_split_tile_size`: Deterministic split size
- `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS`: Force static splits
- Block sizes hardcoded per architecture (could be auto-tuned)

---

This documentation provides a comprehensive view of the Triton attention backend implementation, covering architecture, algorithms, code organization, and design decisions. The system demonstrates sophisticated memory management and kernel fusion techniques optimized for LLM serving workloads.
