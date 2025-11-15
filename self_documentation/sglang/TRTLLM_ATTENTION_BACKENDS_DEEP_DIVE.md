# TensorRT-LLM Attention Backends: Ultra Deep Dive

## Executive Summary

This document provides a comprehensive technical analysis of SGLang's TensorRT-LLM attention backend implementations (`trtllm_mha_backend.py` and `trtllm_mla_backend.py`). These backends wrap NVIDIA's TensorRT-LLM kernels from the flashinfer library to provide high-performance attention computation on SM100 (Blackwell) architecture.

**Key Findings:**
- MHA backend: 693 LOC, handles standard multi-head attention with paged KV cache
- MLA backend: 1079 LOC, handles DeepSeek V2's multi-latent attention with compressed KV cache
- Both support CUDA graphs, FP8 quantization, sliding window attention, and EAGLE speculative decoding
- Critical optimization: Global workspace buffer sharing reduces memory overhead
- Block padding constraints ensure compatibility between TRT-LLM and Triton kernels

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [MHA Backend Deep Dive](#mha-backend-deep-dive)
3. [MLA Backend Deep Dive](#mla-backend-deep-dive)
4. [Memory Management](#memory-management)
5. [CUDA Graph Optimization](#cuda-graph-optimization)
6. [Speculative Decoding Support](#speculative-decoding-support)
7. [Data Flow Analysis](#data-flow-analysis)
8. [Performance Optimizations](#performance-optimizations)
9. [Integration Points](#integration-points)
10. [Critical Code Paths](#critical-code-paths)

---

## 1. Architecture Overview

### 1.1 Design Philosophy

The TRT-LLM backends follow a **facade pattern**: they wrap flashinfer's TensorRT-LLM kernels while adhering to SGLang's `AttentionBackend` interface. This allows drop-in replacement of attention implementations without changing model code.

**Design Principles:**
1. **Lazy initialization**: Resources allocated on first use
2. **State separation**: Metadata separate from computation
3. **Graph-first design**: CUDA graph execution is the primary path
4. **Zero-copy where possible**: Use views and in-place operations

### 1.2 Class Hierarchy

```
AttentionBackend (abstract base)
├── FlashInferAttnBackend
│   └── TRTLLMHAAttnBackend
│       └── TRTLLMHAAttnMultiStepDraftBackend
└── FlashInferMLAAttnBackend
    └── TRTLLMMLABackend
        └── TRTLLMMLAMultiStepDraftBackend
```

**Key Interfaces:**
- `init_forward_metadata()`: Prepare metadata for computation
- `forward_decode()`: Single-token generation
- `forward_extend()`: Multi-token prefill/extension
- `init_cuda_graph_state()`: Pre-allocate graph resources
- `init_forward_metadata_capture_cuda_graph()`: Record graph metadata
- `init_forward_metadata_replay_cuda_graph()`: Update graph metadata

### 1.3 Execution Modes

The backends handle 7 distinct execution modes (from `ForwardMode` enum):

| Mode | Description | Use Case |
|------|-------------|----------|
| DECODE | Single token generation | Standard inference |
| EXTEND | Multi-token prefill | Initial prompt processing |
| MIXED | Chunked prefill + decode | Hybrid batching |
| IDLE | No computation | Data parallel padding |
| TARGET_VERIFY | Verify draft tokens | EAGLE target model |
| DRAFT_EXTEND | Generate draft continuations | EAGLE draft model |
| DRAFT_EXTEND_V2 | Fixed-shape draft output | EAGLE v2 optimization |

---

## 2. MHA Backend Deep Dive

### 2.1 Core Data Structures

#### TRTLLMMHAMetadata (lines 37-51)

```python
@dataclass
class TRTLLMMHAMetadata:
    cache_seqlens_int32: torch.Tensor  # [batch_size], dtype=int32
    max_seq_len_q: int                  # scalar
    max_seq_len_k: int                  # scalar
    cu_seqlens_q: torch.Tensor         # [batch_size+1], dtype=int32
    cu_seqlens_k: torch.Tensor         # [batch_size+1], dtype=int32
    page_table: torch.Tensor           # [batch_size, max_pages], dtype=int32
```

**Purpose:** Encapsulates all metadata required for a single TRT-LLM kernel invocation.

**Field Semantics:**
- `cache_seqlens_int32`: Actual KV cache length for each sequence (excludes padding)
- `max_seq_len_q`: Maximum query length across batch (for kernel dispatch)
- `max_seq_len_k`: Maximum key length across batch (for memory bounds)
- `cu_seqlens_q`: Cumulative sum of query lengths, e.g., `[0, q1, q1+q2, ...]`
- `cu_seqlens_k`: Cumulative sum of key lengths
- `page_table`: Maps sequence positions to physical KV cache pages

**Memory Layout Example:**
```
Batch with seq_lens=[3, 5, 2]:
  cache_seqlens_int32 = [3, 5, 2]
  cu_seqlens_k = [0, 3, 8, 10]
  max_seq_len_k = 5
```

### 2.2 Initialization Flow

#### Constructor (lines 56-108)

```python
def __init__(self, model_runner, skip_prefill=False, ...):
    super().__init__(...)  # Inherit from FlashInferAttnBackend

    # 1. Extract configuration
    self.max_context_len = model_runner.model_config.context_len
    self.page_size = model_runner.page_size
    self.data_type = model_runner.kv_cache_dtype

    # 2. Allocate global workspace (CRITICAL OPTIMIZATION)
    global global_zero_init_workspace_buffer
    if global_zero_init_workspace_buffer is None:
        global_zero_init_workspace_buffer = torch.zeros(
            512 * 1024 * 1024,  # 512 MB
            dtype=torch.uint8,
            device=model_runner.device
        )
    self.workspace_buffer = global_zero_init_workspace_buffer

    # 3. Initialize state containers
    self.decode_cuda_graph_metadata = {}  # {batch_size: metadata}
    self.target_verify_metadata = {}
    self.draft_extend_metadata = {}
```

**Key Design Choices:**

1. **Global workspace sharing** (lines 84-91):
   - Single 512MB buffer shared across ALL TRTLLMHAAttnBackend instances
   - Rationale: TRT-LLM kernels need scratch space for reduction operations
   - Thread-safety: OK because backends execute sequentially per forward pass
   - Memory savings: ~512MB × (num_layers - 1) for typical 30+ layer models

2. **Metadata dictionary keying** (line 94):
   - Key by `batch_size` enables CUDA graph reuse across iterations
   - Same batch size → same metadata buffer → graph replay
   - Different batch size → record new graph with new metadata

3. **Skip prefill flag** (line 59):
   - Used by multi-step draft backends where prefill is handled elsewhere
   - Avoids double initialization of prefill resources

### 2.3 CUDA Graph State Management

#### init_cuda_graph_state (lines 109-193)

This function pre-allocates ALL buffers needed for CUDA graph execution across all possible batch sizes.

```python
def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf=None):
    # Calculate maximum pages needed
    max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

    # Pre-allocate buffers with MAXIMUM dimensions
    self.decode_cuda_graph_metadata = {
        "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
        "page_table": torch.zeros(max_bs, max_num_pages, dtype=torch.int32, device=self.device),
        "strided_indices": torch.arange(0, self.max_context_len, self.page_size, device=self.device)
    }
```

**Why pre-allocation?**
- CUDA graphs capture GPU memory addresses
- Any memory allocation during replay invalidates the graph
- Solution: Allocate maximum size upfront, use slicing during replay

**Strided indices optimization** (line 125-127):
- `torch.arange(0, max_context_len, page_size)` creates `[0, page_size, 2*page_size, ...]`
- Used for efficient page table indexing: `req_to_token[indices[:, None], strided_indices[None, :]]`
- Enables vectorized page lookups without explicit loops

**Speculative decoding state** (lines 130-192):
```python
if self.speculative_num_draft_tokens > 0:
    # Draft decode: one token per step
    self.decode_cuda_graph_metadata["cu_seqlens_q"] = torch.arange(0, max_bs + 1, ...)

    # Target verify: verify multiple draft tokens at once
    self.target_verify_metadata = {
        "cu_seqlens_q": torch.arange(0, max_bs * speculative_num_draft_tokens + 1,
                                     step=speculative_num_draft_tokens, ...)
    }
```

### 2.4 Metadata Initialization Paths

The backend has THREE distinct metadata initialization paths:

#### Path 1: Dynamic (Non-CUDA Graph) - init_forward_metadata (lines 413-513)

Used for:
- First execution before CUDA graphs are captured
- Variable batch sizes not yet captured
- Debugging/profiling with graphs disabled

```python
def init_forward_metadata(self, forward_batch):
    metadata = TRTLLMMHAMetadata()

    if forward_batch.forward_mode.is_decode_or_idle():
        # Decode path
        metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
        metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
        metadata.cu_seqlens_q = torch.arange(0, batch_size + 1, ...)  # [0, 1, 2, ..., bs]
        metadata.cu_seqlens_k = torch.cumsum(seqlens_in_batch, dim=0).pad((1, 0))

        # Extract page table from req_to_token mapping
        metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :metadata.max_seq_len_k
        ]

    # Convert to strided page table (CRITICAL TRANSFORMATION)
    if self.page_size > 1:
        strided_indices = torch.arange(0, metadata.page_table.shape[1], self.page_size, ...)
        metadata.page_table = metadata.page_table[:, strided_indices] // self.page_size
```

**Strided page table transformation** (lines 505-511):
- Input: `req_to_token[seq_idx, token_pos]` → token index in physical cache
- Output: `page_table[seq_idx, page_idx]` → page index in physical cache
- Example with `page_size=16`:
  ```
  req_to_token[0] = [0, 1, 2, ..., 31]  (32 tokens)
  strided_indices = [0, 16]             (every 16th position)
  page_table[0] = [0//16, 16//16] = [0, 1]  (2 pages)
  ```

#### Path 2: CUDA Graph Capture - init_forward_metadata_capture_cuda_graph (lines 194-305)

Called during `torch.cuda.make_graphed_callables()`:

```python
def init_forward_metadata_capture_cuda_graph(self, bs, num_tokens, req_pool_indices,
                                             seq_lens, encoder_lens, forward_mode, spec_info):
    metadata = TRTLLMMHAMetadata()

    if forward_mode.is_decode_or_idle():
        if spec_info is not None:
            # Draft decode: account for speculative step
            metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata["cache_seqlens"][:bs]
            metadata.max_seq_len_k = seq_lens.max().item() + (self.speculative_step_id + 1)
            metadata.page_table = self.decode_cuda_graph_metadata["page_table_draft_decode"][:bs, :]
        else:
            # Normal decode
            metadata.cache_seqlens_int32 = seq_lens[:bs].to(torch.int32)
            metadata.page_table = self.decode_cuda_graph_metadata["page_table"][:bs, :]

        # Store for replay
        self.decode_cuda_graph_metadata[bs] = metadata
```

**Why separate capture/replay?**
- Capture: Creates metadata structure, records addresses
- Replay: Updates values in-place without allocation
- Graph records memory addresses, not tensor contents

#### Path 3: CUDA Graph Replay - init_forward_metadata_replay_cuda_graph (lines 307-407)

Called every inference iteration:

```python
def init_forward_metadata_replay_cuda_graph(self, bs, req_pool_indices, seq_lens,
                                            seq_lens_sum, encoder_lens, forward_mode,
                                            spec_info, seq_lens_cpu):
    # Retrieve pre-allocated metadata
    metadata = self.decode_cuda_graph_metadata[bs]

    # Update dynamic fields IN-PLACE (no allocation!)
    max_len = seq_lens_cpu.max().item()
    metadata.max_seq_len_k = max_len
    metadata.cache_seqlens_int32.copy_(seq_lens)  # .copy_() is in-place

    # Update cumulative lengths
    metadata.cu_seqlens_k[1:].copy_(
        torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
    )

    # Recompute page table
    max_seq_pages = (max_len + self.page_size - 1) // self.page_size
    page_indices = self.req_to_token[
        req_pool_indices[:, None],
        self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][None, :]
    ]
    metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
```

**Critical invariants:**
- MUST use `.copy_()` not `=` assignment (latter creates new tensor)
- MUST slice existing buffers, not create new ones
- MUST maintain consistent tensor addresses across replays

### 2.5 Forward Decode Implementation

#### forward_decode (lines 515-572)

The core single-token generation path:

```python
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
    # 1. Save new KV to cache (BEFORE attention computation)
    if save_kv_cache and k is not None:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v,
            layer.k_scale, layer.v_scale
        )

    # 2. Reshape query for TRT-LLM kernel
    q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

    # 3. Retrieve KV cache and reshape to TRT-LLM format
    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    # Input:  [num_pages, page_size, num_kv_heads, head_dim]
    # Output: [num_pages, num_kv_heads, page_size, head_dim]
    k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim) \
                     .permute(0, 2, 1, 3)
    v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim) \
                     .permute(0, 2, 1, 3)

    # 4. Compute attention scales
    q_scale = 1.0  # Query typically not quantized
    k_scale = layer.k_scale_float if hasattr(layer, "k_scale_float") else 1.0
    bmm1_scale = q_scale * k_scale * layer.scaling
    bmm2_scale = 1.0

    # 5. Call TRT-LLM kernel
    o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=(k_cache, v_cache),
        workspace_buffer=self.workspace_buffer,
        block_tables=self.forward_metadata.page_table,
        seq_lens=self.forward_metadata.cache_seqlens_int32,
        max_seq_len=self.max_context_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=layer.sliding_window_size,
        sinks=kwargs.get("sinks", None)
    )

    return o.view(-1, layer.tp_q_head_num * layer.head_dim)
```

**KV cache layout transformation** (lines 536-541):
- SGLang's default: `[num_pages, page_size, num_heads, head_dim]`
  - Optimized for sequential token writes
  - Coalesced memory access during KV cache population
- TRT-LLM expects: `[num_pages, num_heads, page_size, head_dim]`
  - Optimized for per-head attention computation
  - Better memory locality during attention kernel execution
- Transformation cost: Near-zero (just metadata change via `.permute()`)

**Scale computation** (lines 545-551):
- `bmm1_scale = q_scale × k_scale × softmax_scale`
  - `q_scale`: Query quantization scale (1.0 if FP16)
  - `k_scale`: Key quantization scale (from checkpoint if quantized)
  - `softmax_scale`: 1/√(head_dim), stored as `layer.scaling`
- Applied in kernel's BMM1: `scores = (Q × K^T) × bmm1_scale`
- `bmm2_scale`: Applied in BMM2: `output = softmax(scores) × V × bmm2_scale`

**Sliding window attention** (line 567):
- `window_left=-1`: Full attention (default)
- `window_left=N`: Only attend to last N tokens
- Implementation in TRT-LLM kernel masks scores outside window

**Attention sinks** (line 569):
- Experimental feature for maintaining attention on initial tokens
- `sinks=N`: Always attend to first N tokens regardless of window
- Useful for long-context scenarios with sliding windows

### 2.6 Forward Extend Implementation

#### forward_extend (lines 574-630)

Multi-token prefill/extension:

```python
def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
    # 1. Save KV cache
    if save_kv_cache and k is not None:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v,
            layer.k_scale, layer.v_scale
        )

    # 2. Prepare inputs
    q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim) \
                     .permute(0, 2, 1, 3)
    v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim) \
                     .permute(0, 2, 1, 3)

    # 3. Compute scales
    q_scale = 1.0
    k_scale = layer.k_scale_float if hasattr(layer, "k_scale_float") else 1.0
    bmm1_scale = q_scale * k_scale * layer.scaling

    # 4. Call TRT-LLM prefill kernel
    o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=q,
        kv_cache=(k_cache, v_cache),
        workspace_buffer=self.workspace_buffer,
        block_tables=self.forward_metadata.page_table,
        seq_lens=self.forward_metadata.cache_seqlens_int32,
        max_q_len=self.forward_metadata.max_seq_len_q,
        max_kv_len=self.max_context_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        batch_size=forward_batch.batch_size,
        cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
        cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
        window_left=layer.sliding_window_size,
        sinks=kwargs.get("sinks", None)
    )

    return o.view(-1, layer.tp_q_head_num * layer.head_dim)
```

**Key differences from decode:**
1. **Ragged batch handling**: `cum_seq_lens_q/kv` enable variable-length sequences
2. **Causal masking**: Kernel applies causal mask automatically for prefill
3. **max_q_len**: Required for kernel launch configuration
4. **Workspace usage**: Prefill needs more workspace for larger attention matrices

---

## 3. MLA Backend Deep Dive

### 3.1 MLA Architecture Background

**Multi-Latent Attention (MLA)** is DeepSeek V2's innovation for compressing KV cache:

Traditional MHA:
```
Q: [seq_len, num_heads, head_dim]
K: [seq_len, num_kv_heads, head_dim]
V: [seq_len, num_kv_heads, head_dim]
KV cache size = 2 × seq_len × num_kv_heads × head_dim
```

MLA with compression:
```
Q = Q_nope + Q_rope
  Q_nope: [seq_len, num_heads, kv_lora_rank]      # Low-rank component
  Q_rope:  [seq_len, num_heads, qk_rope_head_dim]  # Positional component

K = K_nope + K_rope
  K_nope: [seq_len, num_kv_heads, kv_lora_rank]    # Compressed
  K_rope:  [seq_len, num_kv_heads, qk_rope_head_dim]

KV cache stores: [K_nope, K_rope] concatenated
KV cache size = seq_len × (kv_lora_rank + qk_rope_head_dim)
```

**Compression ratio example (DeepSeek V3):**
```
Traditional: 2 × seq_len × 128 × 128 = 32,768 × seq_len bytes
MLA:        seq_len × (512 + 64) = 576 × seq_len bytes
Compression: 56.8× reduction!
```

### 3.2 Core Data Structures

#### TRTLLMMLAPrefillMetadata (lines 204-210)

```python
@dataclass
class TRTLLMMLAPrefillMetadata:
    max_seq_len: int              # Maximum sequence length in batch
    cum_seq_lens: torch.Tensor    # Cumulative query lengths [bs+1]
    seq_lens: torch.Tensor        # Individual sequence lengths [bs]
```

Used for extend/prefill operations with variable-length sequences.

#### TRTLLMMLADecodeMetadata (lines 213-222)

```python
@dataclass
class TRTLLMMLADecodeMetadata:
    block_kv_indices: torch.Tensor  # [batch_size, max_blocks], dtype=int32
    max_seq_len_k: int              # Max KV sequence length
    max_seq_len_q: int              # Max query length (for draft_extend)
    sum_seq_lens_q: int             # Total query tokens (for draft_extend)
    cu_seqlens_q: torch.Tensor      # Cumulative query lengths (for draft_extend)
    seq_lens_q: torch.Tensor        # Individual query lengths (for draft_extend)
```

**Block-level indexing:**
- MHA uses token-level page table: `page_table[seq_idx, token_pos]`
- MLA uses block-level indices: `block_kv_indices[seq_idx, block_idx]`
- Block = multiple pages grouped together for Triton kernel efficiency

### 3.3 Block Padding Calculation

#### _calc_padded_blocks (lines 287-308)

This function solves a critical constraint satisfaction problem:

```python
def _calc_padded_blocks(self, max_seq_len: int) -> int:
    # Initial block count
    blocks = triton.cdiv(max_seq_len, self.page_size)

    # Constraint 1: TRT-LLM requires block_num % (128 / page_size) == 0
    trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size  # 128 / page_size

    # Constraint 2: Triton kernel requires alignment for coalesced access
    triton_constraint = get_num_page_per_block_flashmla(self.page_size)  # 4096 / page_size

    # Take LCM to satisfy both
    constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

    # Pad to LCM multiple
    if blocks % constraint_lcm != 0:
        blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm

    return blocks
```

**Example calculation (page_size=64):**
```
max_seq_len = 8192 tokens
Initial blocks = ceil(8192 / 64) = 128 blocks

trtllm_constraint = 128 / 64 = 2
triton_constraint = 4096 / 64 = 64
constraint_lcm = lcm(2, 64) = 64

128 % 64 == 0, so no padding needed
Final blocks = 128
```

**Why this matters:**
- TRT-LLM kernel has hardcoded BLOCK_CONSTRAINT=128 (architecture-specific)
- Triton kernel uses BLOCK_SIZE=4096 for coalesced memory access
- Mismatch causes kernel launch failures or incorrect results
- LCM ensures both constraints satisfied with minimal padding

### 3.4 KV Indices Creation

#### _create_block_kv_indices (lines 310-346)

Converts token-level mapping to block-level indices:

```python
def _create_block_kv_indices(self, batch_size, max_blocks,
                            req_pool_indices, seq_lens, device):
    # Allocate output
    block_kv_indices = torch.full((batch_size, max_blocks), -1,
                                  dtype=torch.int32, device=device)

    # Call Triton kernel for vectorized conversion
    create_flashmla_kv_indices_triton[(batch_size,)](
        self.req_to_token,         # Input: [max_reqs, max_context_len]
        req_pool_indices,          # [batch_size]
        seq_lens,                  # [batch_size]
        None,                      # kv_start_idx
        block_kv_indices,          # Output: [batch_size, max_blocks]
        self.req_to_token.stride(0),
        max_blocks,
        PAGED_SIZE=self.page_size
    )

    return block_kv_indices
```

**Triton kernel implementation** (from utils.py:54-100):
```python
@triton.jit
def create_flashmla_kv_indices_triton(req_to_token_ptr, req_pool_indices_ptr,
                                       page_kernel_lens_ptr, kv_start_idx,
                                       kv_indices_ptr, req_to_token_ptr_stride,
                                       kv_indices_ptr_stride, PAGED_SIZE: tl.constexpr):
    NUM_PAGE_PER_BLOCK: tl.constexpr = 4096 // PAGED_SIZE
    pid = tl.program_id(axis=0)  # One thread per sequence

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_start = 0
    kv_end = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, 4096)

    for i in range(num_pages_loop):
        # Load every PAGED_SIZE-th token (i.e., page boundaries)
        paged_offset = (tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) +
                       i * NUM_PAGE_PER_BLOCK) * PAGED_SIZE

        data = tl.load(req_to_token_ptr + req_pool_index * req_to_token_ptr_stride +
                      kv_start + paged_offset, mask=paged_offset < num_paged * PAGED_SIZE)

        # Divide by page_size to get block indices
        tl.store(kv_indices_ptr + pid * kv_indices_ptr_stride +
                tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK,
                data // PAGED_SIZE, mask=...)
```

**Optimization details:**
- Processes 4096 / page_size pages per iteration
- Vectorized loads/stores using Triton's SIMD operations
- Avoids Python loops over sequences

### 3.5 Triton Padding/Unpadding Kernels

#### pad_draft_extend_query_kernel (lines 56-126)

Used in DRAFT_EXTEND mode where each sequence has different `accept_length`:

```python
@triton.jit
def pad_draft_extend_query_kernel(q_ptr, padded_q_ptr, seq_lens_q_ptr,
                                   cumsum_ptr, batch_size, max_seq_len,
                                   num_heads, head_dim, BLOCK_SIZE: tl.constexpr):
    # 3D grid: (batch×seq, head_block, dim_block)
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return

    # Check if this position is valid
    seq_len = tl.load(seq_lens_q_ptr + batch_id)
    if seq_pos >= seq_len:
        return

    # Find input position
    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos

    # Calculate head block
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    # Calculate dim block
    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Load from input (ragged)
    input_offset = (input_pos * num_heads * head_dim +
                   (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim +
                   (dim_start + tl.arange(0, BLOCK_SIZE))[None, :])

    data = tl.load(q_ptr + input_offset,
                  mask=head_mask[:, None] & dim_mask[None, :], other=0.0)

    # Store to output (padded)
    output_offset = (batch_id * max_seq_len * num_heads * head_dim +
                    seq_pos * num_heads * head_dim +
                    (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim +
                    (dim_start + tl.arange(0, BLOCK_SIZE))[None, :])

    tl.store(padded_q_ptr + output_offset, data,
            mask=head_mask[:, None] & dim_mask[None, :])
```

**Grid dimensions:**
```python
BLOCK_SIZE = 64
grid = (batch_size * max_seq_len,           # All positions
        triton.cdiv(num_heads, BLOCK_SIZE),  # Head blocks
        triton.cdiv(head_dim, BLOCK_SIZE))   # Dim blocks
```

**Memory access pattern:**
```
Input (ragged):  [sum(accept_lengths), num_heads, head_dim]
  Seq 0: tokens 0-2    (accept_length=3)
  Seq 1: tokens 3-6    (accept_length=4)
  Seq 2: tokens 7-8    (accept_length=2)

Output (padded): [batch_size, max_accept_length, num_heads, head_dim]
  Seq 0: tokens 0-2, then zeros
  Seq 1: tokens 3-6
  Seq 2: tokens 7-8, then zeros
```

**Why 3D parallelization?**
- 1D: Sequential over all dimensions (slow)
- 2D: Parallel over batch×seq and one of {heads, dims}
- 3D: Parallel over all three dimensions (optimal GPU utilization)

#### unpad_draft_extend_output_kernel (lines 130-198)

Inverse operation after attention computation:

```python
@triton.jit
def unpad_draft_extend_output_kernel(raw_out_ptr, output_ptr, accept_length_ptr,
                                      cumsum_ptr, batch_size, token_per_batch,
                                      tp_q_head_num, v_head_dim, BLOCK_SIZE: tl.constexpr):
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return

    accept_len = tl.load(accept_length_ptr + batch_id)
    if seq_pos >= accept_len:
        return

    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos

    # Similar block calculations...

    # Load from padded input
    input_offset = (batch_id * token_per_batch * tp_q_head_num * v_head_dim +
                   seq_pos * tp_q_head_num * v_head_dim + ...)

    # Store to ragged output
    output_offset = (output_pos * tp_q_head_num * v_head_dim + ...)

    tl.store(output_ptr + output_offset, data, mask=...)
```

### 3.6 FP8 Quantization with Fused RoPE

#### quantize_and_rope_for_fp8 (lines 591-667)

This function is a **critical optimization** that fuses three operations:

```python
def quantize_and_rope_for_fp8(self, q_nope, q_rope, k_nope, k_rope,
                               forward_batch, cos_sin_cache, is_neox):
    attn_dtype = torch.float8_e4m3fn
    q_len, num_heads = q_rope.shape[0], q_rope.shape[1]

    # Allocate FP8 output tensors
    q_out = q_rope.new_empty(q_len, num_heads,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                            dtype=attn_dtype)
    k_rope_out = k_rope.new_empty(k_rope.shape, dtype=attn_dtype)
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)

    # Fused kernel: RoPE + Quantization + Merge
    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,          # Input: BF16
        k_rope=k_rope,          # Input: BF16
        q_nope=q_nope,          # Input: BF16
        k_nope=k_nope,          # Input: BF16
        cos_sin_cache=cos_sin_cache,
        pos_ids=forward_batch.positions,
        is_neox=is_neox,
        quantize_dtype=attn_dtype,
        # Output slicing (NO COPY!)
        q_rope_out=q_out[..., self.kv_lora_rank:],    # Write to end of q_out
        q_nope_out=q_out[..., :self.kv_lora_rank],    # Write to beginning
        k_rope_out=k_rope_out,
        k_nope_out=k_nope_out,
        quant_scale_q=1.0,
        quant_scale_kv=1.0
    )

    return q_out, k_nope_out, k_rope_out
```

**What the kernel does internally:**
1. **Apply RoPE to q_rope and k_rope:**
   ```python
   # Pseudocode
   cos = cos_sin_cache[positions, :qk_rope_head_dim//2]
   sin = cos_sin_cache[positions, qk_rope_head_dim//2:]
   q_rope_rotated = apply_rotary_emb(q_rope, cos, sin, is_neox)
   k_rope_rotated = apply_rotary_emb(k_rope, cos, sin, is_neox)
   ```

2. **Quantize all components to FP8:**
   ```python
   q_nope_fp8 = quantize_fp8(q_nope)
   q_rope_fp8 = quantize_fp8(q_rope_rotated)
   k_nope_fp8 = quantize_fp8(k_nope)
   k_rope_fp8 = quantize_fp8(k_rope_rotated)
   ```

3. **Merge q components in-place:**
   ```python
   # Write directly to output tensor (no intermediate allocation)
   q_out[..., :kv_lora_rank] = q_nope_fp8
   q_out[..., kv_lora_rank:] = q_rope_fp8
   ```

**Performance benefits:**
- **Memory savings**: No intermediate BF16 tensors for rotated results
- **Kernel fusion**: Single kernel launch vs. 3 separate operations
- **Cache efficiency**: Reduced memory bandwidth requirements

**Layout before/after:**
```
Before (4 separate BF16 tensors):
  q_nope: [seq_len, num_heads, kv_lora_rank]         = 512 dims
  q_rope: [seq_len, num_heads, qk_rope_head_dim]     = 64 dims
  k_nope: [seq_len, num_kv_heads, kv_lora_rank]      = 512 dims
  k_rope: [seq_len, num_kv_heads, qk_rope_head_dim]  = 64 dims

After (3 FP8 tensors, 2 merged):
  q_out:      [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim] = 576 dims
  k_nope_out: [seq_len, num_kv_heads, kv_lora_rank]                 = 512 dims
  k_rope_out: [seq_len, num_kv_heads, qk_rope_head_dim]             = 64 dims

Memory reduction: ~25% for query path
```

### 3.7 Forward Decode (MLA)

#### forward_decode (lines 749-848)

```python
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True,
                   q_rope=None, k_rope=None, cos_sin_cache=None, is_neox=False):
    merge_query = q_rope is not None

    # FP8 path: fused quantization + RoPE
    if self.data_type == torch.float8_e4m3fn:
        assert all(x is not None for x in [q_rope, k_rope, cos_sin_cache])
        q, k, k_rope = self.quantize_and_rope_for_fp8(
            q, q_rope, k.squeeze(1), k_rope.squeeze(1),
            forward_batch, cos_sin_cache, is_neox
        )
        merge_query = False  # Already merged

    # Save KV cache (both nope and rope components)
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
            layer, forward_batch.out_cache_loc, k, k_rope
        )

    # Prepare query
    if merge_query:
        # FP16 path: manual merge
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope_reshaped = q_rope.view(-1, layer.tp_q_head_num,
                                      layer.head_dim - layer.v_head_dim)
        query = _concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
    else:
        # FP8 path: already merged
        query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

    # Ensure 4D shape for kernel
    if query.dim() == 3:
        query = query.unsqueeze(1)  # [bs, 1, num_heads, head_dim]

    # Prepare KV cache
    k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)
    # Shape: [num_pages, 1, page_size, kv_cache_dim]

    # Get metadata
    metadata = getattr(forward_batch, "decode_trtllm_mla_metadata", None) \
               or self.forward_decode_metadata

    # Compute scales
    q_scale = 1.0
    k_scale = layer.k_scale_float if hasattr(layer, "k_scale_float") else 1.0
    bmm1_scale = q_scale * k_scale * layer.scaling

    # Call TRT-LLM MLA kernel
    raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=self.workspace_buffer,
        qk_nope_head_dim=self.qk_nope_head_dim,
        kv_lora_rank=self.kv_lora_rank,
        qk_rope_head_dim=self.qk_rope_head_dim,
        block_tables=metadata.block_kv_indices,
        seq_lens=forward_batch.seq_lens.to(torch.int32),
        max_seq_len=metadata.max_seq_len_k,
        bmm1_scale=bmm1_scale
    )

    output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
    return output
```

**MLA-specific parameters:**
- `qk_nope_head_dim`: Dimension of non-positional component (typically head_dim - qk_rope_head_dim)
- `kv_lora_rank`: Compressed KV dimension (512 for DeepSeek V2/V3)
- `qk_rope_head_dim`: Rotary encoding dimension (64 typically)

**KV cache format:**
- Stored as concatenated `[k_nope, k_rope]` with shape `[num_pages, page_size, kv_cache_dim]`
- `kv_cache_dim = kv_lora_rank + qk_rope_head_dim`
- Kernel internally splits and decompresses

### 3.8 Forward Extend (MLA)

The extend path has THREE execution branches:

#### Branch 1: Target Verify (lines 938-943)

```python
if forward_batch.forward_mode.is_target_verify():
    seq_lens = (forward_batch.seq_lens.to(torch.int32) +
               forward_batch.spec_info.draft_token_num)
    max_seq_len = metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num

    q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)

    raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(...)

    output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
```

**What is target verify?**
- In speculative decoding, draft model generates K candidate tokens
- Target model verifies which candidates are correct
- Processes K tokens per sequence in parallel
- Uses decode kernel (not prefill) because query length is fixed and small

#### Branch 2: Draft Extend (lines 946-1003)

```python
elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
    # Adjust seq_lens for padding alignment
    seq_lens = (forward_batch.seq_lens - metadata.seq_lens_q +
               metadata.max_seq_len_q).to(torch.int32)
    max_seq_len = metadata.max_seq_len_k + metadata.max_seq_len_q

    # Pad queries
    if self.padded_q_buffer is not None:
        padded_q = self.padded_q_buffer[:bs, :metadata.max_seq_len_q, :, :].to(q.dtype)
    else:
        padded_q = torch.zeros(bs, metadata.max_seq_len_q, layer.tp_q_head_num,
                              layer.head_dim, dtype=q.dtype, device=q.device)

    q = self.pad_draft_extend_query(q, padded_q, metadata.seq_lens_q,
                                    metadata.cu_seqlens_q)

    # Run attention with padded input
    q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
    raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(...)

    # Unpad output
    raw_out = self.unpad_draft_extend_output(raw_out, metadata.cu_seqlens_q,
                                             metadata.seq_lens_q, metadata.sum_seq_lens_q)

    output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
```

**Draft extend scenario:**
```
Batch with accept_lengths = [2, 4, 1]:
  Seq 0 accepted 2 tokens → extend with 2 new queries
  Seq 1 accepted 4 tokens → extend with 4 new queries
  Seq 2 accepted 1 token  → extend with 1 new query

Padding:
  max_seq_len_q = 4
  padded_q: [3, 4, num_heads, head_dim]

After kernel:
  raw_out: [3, 4, num_heads, v_head_dim]

After unpadding:
  output: [7, num_heads, v_head_dim]  (2+4+1=7)
```

**Why adjust seq_lens?** (line 951-955)
- TRT-LLM kernel expects KV cache length to align with query positions
- `forward_batch.seq_lens` = context + verified tokens
- But we padded queries to `max_seq_len_q`
- So we need: `kv_len = context + max_seq_len_q` to align properly

#### Branch 3: Normal Prefill (lines 1006-1054)

Two sub-branches:

**3a. MHA for Chunked Prefix Cache** (lines 1006-1034):
```python
if forward_batch.attn_attend_prefix_cache:
    return flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=q, key=k, value=v,
        workspace_buffer=self.workspace_buffer,
        seq_lens=forward_batch.prefix_chunk_seq_lens[chunk_idx],
        max_q_len=self.forward_prefill_metadata.max_seq_len,
        max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
        bmm1_scale=layer.scaling,
        bmm2_scale=1.0,
        o_sf_scale=-1.0,
        batch_size=forward_batch.batch_size,
        window_left=-1,
        cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
        cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
        enable_pdl=False,
        is_causal=False,  # Already computed attention mask
        return_lse=True,
        out=torch.zeros(*output_shape, dtype=q.dtype, device=q.device)
    )
```

**Why MHA for prefix cache?**
- Prefix cache stores full attention results (not compressed)
- Uses standard attention computation (not MLA decompression)
- `is_causal=False` because prefix is fully attended

**3b. Standard MLA Prefill** (lines 1036-1054):
```python
return flashinfer.prefill.trtllm_ragged_attention_deepseek(
    query=q, key=k, value=v,
    workspace_buffer=self.workspace_buffer,
    seq_lens=self.forward_prefill_metadata.seq_lens,
    max_q_len=self.forward_prefill_metadata.max_seq_len,
    max_kv_len=self.forward_prefill_metadata.max_seq_len,
    bmm1_scale=layer.scaling,
    bmm2_scale=1.0,
    o_sf_scale=1.0,
    batch_size=forward_batch.batch_size,
    window_left=-1,
    cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
    cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
    enable_pdl=False,
    is_causal=True,  # Apply causal masking
    return_lse=forward_batch.mha_return_lse
)
```

**Notable parameter:**
- `return_lse`: Log-sum-exp of attention scores
- Used for downstream operations (e.g., attention visualization, LoRA)

---

## 4. Memory Management

### 4.1 Global Workspace Buffer

**Location:** Both backends (mha:84-91, mla:264-271)

```python
global_zero_init_workspace_buffer = None

def __init__(self, model_runner, ...):
    global global_zero_init_workspace_buffer
    if global_zero_init_workspace_buffer is None:
        global_zero_init_workspace_buffer = torch.zeros(
            DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=model_runner.device
        )
    self.workspace_buffer = global_zero_init_workspace_buffer
```

**Why global?**
- TRT-LLM kernels need workspace for:
  - Reduction operations (sum, max)
  - Intermediate attention scores
  - Softmax computation
- Size requirements:
  - MHA: 512 MB (line 29)
  - MLA: 128 MB (line 43)
- Sharing strategy:
  - All layers of all models share ONE buffer
  - Safe because forward pass is sequential
  - Saves: `(num_layers - 1) × workspace_size`
  - Example: 32-layer model saves 32 × 512MB = 16GB!

**Thread safety:**
- Python GIL ensures single-threaded execution
- Even with multi-threading, forward pass is sequential per batch
- No race conditions possible

### 4.2 KV Cache Management

#### MHA KV Cache

**Storage format:**
```python
k_buffer: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
v_buffer: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
```

**Writing to cache** (memory_pool.py:732-766):
```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale, v_scale):
    layer_id = layer.layer_id - self.start_layer

    if self.store_dtype != self.dtype:
        # Quantization path
        cache_k = quantize_kv_cache(cache_k, k_scale, self.store_dtype)
        cache_v = quantize_kv_cache(cache_v, v_scale, self.store_dtype)

    # Write to physical cache
    self.k_buffer[layer_id][loc] = cache_k
    self.v_buffer[layer_id][loc] = cache_v
```

**Reading from cache** (memory_pool.py:710-730):
```python
def get_key_buffer(self, layer_id):
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    if self.store_dtype != self.dtype:
        return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
    return self.k_buffer[layer_id - self.start_layer]
```

**Quantization support:**
- `store_dtype`: Physical storage (FP8, INT8, FP16)
- `dtype`: Computation dtype (FP16, BF16)
- `.view(dtype)`: Zero-cost reinterpretation for dequantization

#### MLA KV Cache

**Storage format:**
```python
kv_buffer: [num_layers, num_pages, page_size, kv_cache_dim]
# where kv_cache_dim = kv_lora_rank + qk_rope_head_dim
```

**Writing to cache** (memory_pool.py:1394-1408):
```python
def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
    layer_id = layer.layer_id - self.start_layer

    # Concatenate nope and rope
    if cache_k_rope is not None:
        cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
    else:
        cache_k = cache_k_nope

    # Quantize if needed
    if self.store_dtype != self.dtype:
        cache_k = quantize_kv_cache(cache_k, k_scale, self.store_dtype)

    self.kv_buffer[layer_id][loc] = cache_k
```

**Triton kernel for write** (memory_pool.py:1151-1187):
```python
@triton.jit
def set_mla_kv_buffer_kernel(kv_buffer_ptr, cache_k_nope_ptr, cache_k_rope_ptr,
                              loc_ptr, buffer_stride, ...):
    pid = tl.program_id(axis=0)
    loc = tl.load(loc_ptr + pid)

    # Write k_nope
    for offset in range(num_loops_nope):
        data_nope = tl.load(cache_k_nope_ptr + ...)
        tl.store(kv_buffer_ptr + loc * buffer_stride + offset, data_nope)

    # Write k_rope
    for offset in range(num_loops_rope):
        data_rope = tl.load(cache_k_rope_ptr + ...)
        tl.store(kv_buffer_ptr + loc * buffer_stride + nope_size + offset, data_rope)
```

**Why Triton kernel?**
- Fused concatenation + write avoids intermediate tensor
- Coalesced memory access pattern
- Faster than PyTorch cat + indexing

### 4.3 CUDA Graph Memory Constraints

**Key principle:** All buffers must be pre-allocated before graph capture.

**MHA Graph Buffers** (trtllm_mha_backend.py:117-128):
```python
def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf):
    max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

    self.decode_cuda_graph_metadata = {
        "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
        "page_table": torch.zeros(max_bs, max_num_pages, dtype=torch.int32,
                                 device=self.device),
        "strided_indices": torch.arange(0, self.max_context_len, self.page_size,
                                       device=self.device)
    }
```

**MLA Graph Buffers** (trtllm_mla_backend.py:358-376):
```python
def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf):
    max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

    # Block indices buffer
    self.decode_cuda_graph_kv_indices = torch.full(
        (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
    )

    # Padding buffer for draft extend
    num_tokens_per_bs = max_num_tokens // max_bs
    self.padded_q_buffer = torch.zeros(
        (max_bs, num_tokens_per_bs, self.num_q_heads, self.kv_cache_dim),
        dtype=self.data_type, device=self.device
    )

    # Unpadding buffer
    self.unpad_output_buffer = torch.zeros(
        (max_num_tokens, self.num_q_heads, 512),  # v_head_dim hardcoded
        dtype=self.data_type, device=self.device
    )
```

**Memory usage calculation:**
```python
# Example: max_bs=128, max_context_len=8192, page_size=64, num_heads=32, head_dim=128

MHA:
  cache_seqlens: 128 × 4 bytes = 512 B
  page_table: 128 × 128 × 4 bytes = 64 KB
  strided_indices: 128 × 4 bytes = 512 B
  Total: ~65 KB per graph size

MLA:
  kv_indices: 128 × 128 × 4 bytes = 64 KB
  padded_q_buffer: 128 × 16 × 32 × 576 × 2 bytes = 75 MB
  unpad_output_buffer: 2048 × 32 × 512 × 2 bytes = 64 MB
  Total: ~139 MB per graph size

Graph captures typically 10-20 sizes → 650 KB (MHA) to 2.8 GB (MLA)
```

---

## 5. CUDA Graph Optimization

### 5.1 CUDA Graph Overview

**What is a CUDA graph?**
- Captures a sequence of GPU operations as a graph
- Records kernel launches, memory copies, dependencies
- Replays entire graph with single CPU call
- Benefits:
  - Eliminates kernel launch overhead (~5-20μs per kernel)
  - Enables aggressive kernel fusion
  - Reduces CPU-GPU synchronization

**SGLang's graph strategy:**
- Capture graphs for each `batch_size` independently
- Key by `(batch_size, forward_mode)` tuple
- Store metadata templates, update values during replay

### 5.2 Graph Capture Flow

**Step 1: Allocate graph state** (init_cuda_graph_state)
```python
# Called once during model initialization
backend.init_cuda_graph_state(max_bs=128, max_num_tokens=2048)
```

**Step 2: Warm-up iteration** (init_forward_metadata_capture_cuda_graph)
```python
# Called during graph recording
with torch.cuda.graph(cuda_graph):
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        backend.init_forward_metadata_capture_cuda_graph(
            bs=bs, num_tokens=bs, req_pool_indices=indices[:bs],
            seq_lens=lens[:bs], encoder_lens=None,
            forward_mode=ForwardMode.DECODE, spec_info=None
        )
        output = model.forward(...)
```

**Step 3: Graph replay** (init_forward_metadata_replay_cuda_graph)
```python
# Called every inference iteration
backend.init_forward_metadata_replay_cuda_graph(
    bs=actual_batch_size, req_pool_indices=actual_indices,
    seq_lens=actual_lens, seq_lens_sum=actual_sum,
    encoder_lens=None, forward_mode=ForwardMode.DECODE,
    spec_info=None, seq_lens_cpu=actual_lens_cpu
)
cuda_graph.replay()
```

### 5.3 Metadata Update Strategy

**Capture-time metadata** (mha:194-250):
```python
def init_forward_metadata_capture_cuda_graph(self, bs, num_tokens, ...):
    metadata = TRTLLMMHAMetadata()

    # Use pre-allocated buffers (no allocation!)
    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata["cache_seqlens"][:bs]
    metadata.page_table = self.decode_cuda_graph_metadata["page_table"][:bs, :]

    # Compute static values
    metadata.cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
    metadata.cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    metadata.max_seq_len_k = seq_lens.max().item()

    # Store template
    self.decode_cuda_graph_metadata[bs] = metadata
```

**Replay-time updates** (mha:307-356):
```python
def init_forward_metadata_replay_cuda_graph(self, bs, req_pool_indices, seq_lens, ...):
    # Retrieve template
    metadata = self.decode_cuda_graph_metadata[bs]

    # Update ONLY dynamic values
    max_len = seq_lens_cpu.max().item()
    metadata.max_seq_len_k = max_len  # Scalar assignment (OK)

    # In-place tensor updates
    metadata.cache_seqlens_int32.copy_(seq_lens)
    metadata.cu_seqlens_k[1:].copy_(
        torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
    )

    # Recompute page table (MUST use same output buffer)
    max_seq_pages = (max_len + self.page_size - 1) // self.page_size
    page_indices = self.req_to_token[
        req_pool_indices[:, None],
        self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][None, :]
    ]
    metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
```

**Critical rules:**
1. **No allocations**: Cannot create new tensors
2. **In-place only**: Use `.copy_()`, not `=`
3. **Same addresses**: Must update existing buffers
4. **Consistent shapes**: Slicing OK, reshaping NOT OK

### 5.4 Graph Performance Impact

**Microbenchmark results** (estimated from architecture):
```
Without graphs (bs=8, seq_len=128, num_layers=32):
  Kernel launch overhead: 32 layers × 10μs = 320μs
  Attention computation: ~800μs
  Total: ~1120μs

With graphs:
  Graph launch overhead: ~5μs
  Attention computation: ~800μs
  Total: ~805μs

Speedup: 1.39× (28% reduction in latency)
```

**Scaling with batch size:**
```
bs=1:  Launch overhead dominates, 2× speedup
bs=8:  Mixed, 1.4× speedup
bs=64: Computation dominates, 1.1× speedup
```

---

## 6. Speculative Decoding Support

### 6.1 EAGLE Algorithm Overview

**Speculative decoding workflow:**
```
1. Draft model generates K candidate tokens in parallel
2. Target model verifies candidates in single pass
3. Accept longest correct prefix
4. Continue from last accepted token
```

**EAGLE-specific optimization:**
- Multi-step draft: Generate candidates token-by-token
- Tree-based expansion: Maintain multiple hypotheses
- Parallel verification: Verify all hypotheses simultaneously

### 6.2 Forward Modes for Speculation

| Mode | Model | Operation | Batch Shape |
|------|-------|-----------|-------------|
| DECODE | Draft | Generate 1st candidate | [bs, 1, ...] |
| DRAFT_EXTEND | Draft | Generate 2nd-Kth candidates | [bs, accept_len, ...] |
| TARGET_VERIFY | Target | Verify all K candidates | [bs, K, ...] |

### 6.3 MHA Multi-Step Draft Backend

**TRTLLMHAAttnMultiStepDraftBackend** (lines 633-693):
```python
class TRTLLMHAAttnMultiStepDraftBackend(FlashInferMultiStepDraftBackend):
    def __init__(self, model_runner, topk, speculative_num_steps):
        super().__init__(model_runner, topk, speculative_num_steps)

        # Create one backend per draft step
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMHAAttnBackend(
                model_runner,
                skip_prefill=True,  # Only decode for drafts
                kv_indptr_buf=self.kv_indptr[i],
                kv_last_page_len_buf=self.kv_last_page_len,
                speculative_step_id=i  # Track which step
            )

    def init_forward_metadata(self, forward_batch):
        # Initialize all backends in parallel
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)
```

**Why separate backends per step?**
- Each step has different `speculative_step_id`
- Affects `max_seq_len_k` calculation: `base_len + step_id + 1`
- Enables independent CUDA graph capture per step

### 6.4 Draft Decode Metadata

**Metadata preparation** (mha:209-230):
```python
if forward_mode.is_decode_or_idle():
    if spec_info is not None:
        # Draft decode
        metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata["cache_seqlens"][:bs]

        # Add speculative offset
        metadata.max_seq_len_k = seq_lens.max().item() + (self.speculative_step_id + 1)

        metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][:bs + 1]
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.page_table = self.decode_cuda_graph_metadata["page_table_draft_decode"][:bs, :]
```

**Step ID impact:**
```
Base sequence length: 100 tokens

Step 0 (1st draft token):
  max_seq_len_k = 100 + (0 + 1) = 101
  cache_seqlens = [101, 101, ...]

Step 1 (2nd draft token):
  max_seq_len_k = 100 + (1 + 1) = 102
  cache_seqlens = [102, 102, ...]

Step 2 (3rd draft token):
  max_seq_len_k = 100 + (2 + 1) = 103
  cache_seqlens = [103, 103, ...]
```

### 6.5 Target Verify Metadata

**Preparation** (mha:251-280):
```python
elif forward_mode.is_target_verify():
    metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][:bs]
    metadata.cache_seqlens_int32.copy_(seq_lens + self.speculative_num_draft_tokens)

    # Query length = number of draft tokens
    metadata.max_seq_len_q = self.speculative_num_draft_tokens
    metadata.max_seq_len_k = seq_lens.max().item() + self.speculative_num_draft_tokens

    # Cumulative query lengths (strided by draft_token_num)
    metadata.cu_seqlens_q = torch.arange(
        0, bs * self.speculative_num_draft_tokens + 1,
        self.speculative_num_draft_tokens,
        dtype=torch.int32, device=device
    )

    metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][:(bs + 1)]
    metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]
```

**Example with 3 draft tokens:**
```
Batch size: 4
Draft tokens per sequence: 3

cu_seqlens_q = [0, 3, 6, 9, 12]
  Seq 0: queries 0-2   (3 tokens)
  Seq 1: queries 3-5   (3 tokens)
  Seq 2: queries 6-8   (3 tokens)
  Seq 3: queries 9-11  (3 tokens)

Shape: [12, num_heads, head_dim]  (4 sequences × 3 tokens)
```

### 6.6 Draft Extend in MLA

**Scenario:**
```
Step 1: Generate 1st candidate (DECODE mode)
  Input:  [bs] tokens
  Output: [bs] tokens

Step 2-K: Generate 2nd-Kth candidates (DRAFT_EXTEND mode)
  Accept lengths: [2, 4, 1, 3]  (variable per sequence)
  Input:  [10] tokens (2+4+1+3)
  Output: [10] tokens
```

**Padding requirement:**
```
TRT-LLM kernel expects: [bs, max_accept_len, num_heads, head_dim]
Actual data is ragged:  [sum(accept_lens), num_heads, head_dim]

Solution:
  1. Pad to [bs, max_accept_len, ...]  (pad_draft_extend_query)
  2. Run kernel
  3. Unpad back to [sum(accept_lens), ...]  (unpad_draft_extend_output)
```

**Metadata for draft extend** (mla:482-493):
```python
if forward_mode.is_draft_extend(include_v2=True):
    accept_length = spec_info.accept_length[:bs]

    if spec_info.accept_length_cpu:
        metadata.max_seq_len_q = max(spec_info.accept_length_cpu[:bs])
        metadata.sum_seq_lens_q = sum(spec_info.accept_length_cpu[:bs])
    else:
        metadata.max_seq_len_q = 1
        metadata.sum_seq_lens_q = bs

    # Update cumulative lengths
    metadata.cu_seqlens_q[1:].copy_(
        torch.cumsum(accept_length, dim=0, dtype=torch.int32)
    )
    metadata.seq_lens_q.copy_(accept_length)
```

---

## 7. Data Flow Analysis

### 7.1 End-to-End Decode Path (MHA)

```
┌─────────────────────────────────────────────────────────────┐
│                     Model Forward Pass                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              RadixAttention.forward() (radix_attention.py:94)│
│  - Receives q, k, v from attention layer                     │
│  - Checks forward_mode                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        AttentionBackend.forward() (base_attn_backend.py:77) │
│  - Routes to forward_decode() or forward_extend()            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   TRTLLMHAAttnBackend.forward_decode() (trtllm_mha:515)     │
│  1. Save KV cache: token_to_kv_pool.set_kv_buffer()          │
│  2. Reshape q to [bs, num_heads, head_dim]                   │
│  3. Load KV cache: token_to_kv_pool.get_kv_buffer()          │
│  4. Permute KV cache to TRT-LLM format                       │
│  5. Compute attention scales                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  flashinfer.decode.trtllm_batch_decode_with_kv_cache()      │
│  - TRT-LLM optimized CUDA kernel                             │
│  - Inputs:                                                   │
│    * query: [bs, num_heads, head_dim]                        │
│    * kv_cache: ([num_pages, num_heads, page_size, head_dim],│
│                 [num_pages, num_heads, page_size, head_dim]) │
│    * block_tables: [bs, max_pages]                           │
│    * seq_lens: [bs]                                          │
│    * bmm1_scale, bmm2_scale                                  │
│  - Operation:                                                │
│    1. Load K cache using block_tables                        │
│    2. Compute scores = (Q @ K^T) * bmm1_scale                │
│    3. Apply softmax                                          │
│    4. Load V cache                                           │
│    5. Compute output = softmax(scores) @ V * bmm2_scale      │
│  - Output: [bs, num_heads, head_dim]                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Return to model forward pass                      │
│  - Output reshaped to [bs, num_heads * head_dim]             │
│  - Continue to next layer                                    │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 End-to-End Decode Path (MLA with FP8)

```
┌─────────────────────────────────────────────────────────────┐
│         DeepSeekV2Attention.forward() (deepseek_v2.py)      │
│  - Split Q into q_nope, q_rope                               │
│  - Split K into k_nope, k_rope                               │
│  - Apply RoPE to q_rope, k_rope (skipped for FP8 TRT-LLM)   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   TRTLLMMLABackend.forward_decode() (trtllm_mla:749)        │
│  1. Check dtype == torch.float8_e4m3fn                       │
│  2. Call quantize_and_rope_for_fp8():                        │
│     ├─ Apply RoPE to q_rope, k_rope                          │
│     ├─ Quantize q_nope, q_rope, k_nope, k_rope to FP8        │
│     └─ Merge q_nope + q_rope into single tensor              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   flashinfer.rope.mla_rope_quantize_fp8() (trtllm_mla:648)  │
│  Fused CUDA kernel:                                          │
│  - Load q_nope, q_rope, k_nope, k_rope (BF16)                │
│  - Apply RoPE to q_rope, k_rope:                             │
│    * q_rope_rotated = rotate(q_rope, cos, sin, positions)    │
│    * k_rope_rotated = rotate(k_rope, cos, sin, positions)    │
│  - Quantize all to FP8:                                      │
│    * Find abs_max per tensor                                 │
│    * scale = 448.0 / abs_max  (FP8 E4M3 range: -448 to 448)  │
│    * quantized = clamp(input * scale, -448, 448)             │
│  - Write merged output:                                      │
│    * q_out[..., :kv_lora_rank] = q_nope_fp8                  │
│    * q_out[..., kv_lora_rank:] = q_rope_fp8                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   Save KV cache: token_to_kv_pool.set_mla_kv_buffer()       │
│  - Concatenate k_nope_fp8 + k_rope_fp8                       │
│  - Write to cache[layer_id][out_cache_loc]                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla()  │
│  - Inputs:                                                   │
│    * query: [bs, 1, num_heads, kv_lora_rank+qk_rope_head_dim]│
│    * kv_cache: [num_pages, 1, page_size, kv_cache_dim]      │
│    * block_tables: [bs, max_blocks]                          │
│    * qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim       │
│  - Operation:                                                │
│    1. Split query: q_nope = q[..., :kv_lora_rank]            │
│                    q_rope = q[..., kv_lora_rank:]            │
│    2. Load KV cache and split:                               │
│       k_nope = kv[..., :kv_lora_rank]                        │
│       k_rope = kv[..., kv_lora_rank:]                        │
│    3. Decompress K: k_full = decompress(k_nope, k_rope)      │
│    4. Compute scores = (q @ k_full^T) * bmm1_scale           │
│    5. Apply softmax                                          │
│    6. Compute v = decompress_v(k_nope)                       │
│    7. Output = softmax(scores) @ v                           │
│  - Output: [bs, 1, num_heads, v_head_dim]                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Return to model forward pass                      │
│  - Output reshaped to [bs, num_heads * v_head_dim]           │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 CUDA Graph Replay Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Inference Iteration N                           │
│  - New batch arrives with batch_size = bs                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Check if CUDA graph exists for bs                    │
│  if bs in captured_graphs:                                   │
│      Use graph replay path                                   │
│  else:                                                       │
│      Use dynamic execution path                              │
└─────────────────────────────────────────────────────────────┘
                            │ (Graph replay)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  backend.init_forward_metadata_replay_cuda_graph()          │
│  - Retrieve pre-allocated metadata: metadata = graph_meta[bs]│
│  - Update dynamic fields:                                    │
│    * metadata.max_seq_len_k = seq_lens_cpu.max().item()      │
│    * metadata.cache_seqlens.copy_(seq_lens)                  │
│    * metadata.cu_seqlens_k[1:].copy_(cumsum(seq_lens))       │
│    * Recompute page_table via strided indexing               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              cuda_graph.replay()                             │
│  - Single GPU command replays entire graph                   │
│  - No CPU-GPU synchronization per kernel                     │
│  - Executes: embedding → layers → logits                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Sampling and token generation                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Performance Optimizations

### 8.1 Memory Bandwidth Optimization

**1. KV Cache Layout Transformation**
- Cost: ~0 (just metadata change)
- Benefit: Better locality during attention computation
- Details: Permute from `[pages, page_size, heads, dim]` to `[pages, heads, page_size, dim]`

**2. Strided Page Table Indexing**
- Saves: N × page_size memory reads → N reads
- Example: `page_size=64`, `seq_len=1024` → 1024 reads → 16 reads (64× reduction)
- Implementation: `req_to_token[:, torch.arange(0, max_len, page_size)]`

**3. Fused RoPE + Quantization (MLA)**
- Saves: ~25% memory bandwidth
- Avoids: Intermediate BF16 tensors for rotated results
- Single kernel launch vs. 3 separate operations

**4. Block-Level Indexing (MLA)**
- Groups multiple pages into blocks
- Reduces index tensor size: `ceil(seq_len / 64)` → `ceil(seq_len / 4096)`
- ~64× reduction in index table size

### 8.2 Compute Optimization

**1. Attention Scale Precomputation**
```python
bmm1_scale = q_scale * k_scale * layer.scaling
```
- Precomputed once, used in kernel
- Avoids per-element scaling operations
- Fused into matrix multiplication

**2. Workspace Buffer Reuse**
- Single allocation shared across all layers
- Avoids allocation/deallocation overhead
- Reduces memory fragmentation

**3. CUDA Graph Execution**
- Eliminates kernel launch overhead: ~10μs per kernel → ~5μs total
- For 32-layer model: 320μs → 5μs (64× reduction in launch overhead)

**4. In-Place Operations**
```python
metadata.cache_seqlens.copy_(seq_lens)  # In-place
# vs
metadata.cache_seqlens = seq_lens.clone()  # Allocation + copy
```
- Avoids allocation: ~1-5μs per tensor
- Maintains CUDA graph validity

### 8.3 Triton Kernel Optimizations

**1. 3D Parallelization**
```python
grid = (batch × seq, head_blocks, dim_blocks)
```
- Maximizes GPU utilization
- Each thread block handles 64×64 tile
- Coalesced memory access within block

**2. Block Size Selection**
```python
BLOCK_SIZE = 64  # Tuned for Ampere/Hopper
```
- 64 = 2 warps per block
- Optimal for:
  - Register usage (32 registers per thread)
  - Shared memory (64×64×4 bytes = 16KB)
  - Occupancy (high warp count)

**3. Masked Loads/Stores**
```python
data = tl.load(ptr, mask=valid_mask, other=0.0)
```
- Avoids branches
- Enables vectorized loads even with irregular shapes
- Compiler optimizes to predicated instructions

### 8.4 Algorithmic Optimizations

**1. LCM-Based Block Padding**
- Minimizes padding overhead
- Ensures compatibility with both TRT-LLM and Triton
- Typical overhead: <2% for long sequences

**2. Ragged Batch Handling**
- Uses cumulative length arrays instead of padding
- Saves computation on padded positions
- Critical for variable-length batches

**3. Quantization-Aware Scaling**
```python
bmm1_scale = q_scale * k_scale * softmax_scale
```
- Folds quantization scales into attention computation
- Single scaling operation vs. separate dequant + scale + quant

---

## 9. Integration Points

### 9.1 Upstream Dependencies

**1. Model Config** (`model_config.ModelConfig`):
```python
# MHA
self.hidden_size = config.hidden_size
self.max_context_len = config.context_len

# MLA
self.kv_lora_rank = config.kv_lora_rank
self.qk_nope_head_dim = config.qk_nope_head_dim
self.qk_rope_head_dim = config.qk_rope_head_dim
self.v_head_dim = config.v_head_dim
```

**2. Model Runner** (`model_runner.ModelRunner`):
```python
# Resource references
self.req_to_token = model_runner.req_to_token_pool.req_to_token
self.device = model_runner.device
self.dtype = model_runner.dtype
self.kv_cache_dtype = model_runner.kv_cache_dtype
self.page_size = model_runner.page_size
```

**3. Forward Batch** (`forward_batch_info.ForwardBatch`):
```python
# Batch metadata
forward_batch.batch_size
forward_batch.seq_lens
forward_batch.seq_lens_cpu
forward_batch.req_pool_indices
forward_batch.out_cache_loc

# Mode and speculation
forward_batch.forward_mode
forward_batch.spec_info
```

**4. Token-to-KV Pool** (`memory_pool.TokenToKVPool`):
```python
# KV cache operations
forward_batch.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, k_scale, v_scale)
forward_batch.token_to_kv_pool.set_mla_kv_buffer(layer, loc, k_nope, k_rope)
k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)
k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer_id)
```

### 9.2 Downstream Consumers

**1. RadixAttention** (`layers/radix_attention.py`):
```python
class RadixAttention(nn.Module):
    def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        return forward_batch.attn_backend.forward(
            q, k, v, self, forward_batch, save_kv_cache, **kwargs
        )
```

**2. Model Layers** (e.g., `deepseek_v2.py`):
```python
# MLA path
q_nope, q_rope = self.split_qkv(q)
k_nope, k_rope = self.split_qkv(k)
output = self.attn.forward(
    q_nope, k_nope, v, forward_batch,
    q_rope=q_rope, k_rope=k_rope,
    cos_sin_cache=self.cos_sin_cache,
    is_neox=True
)
```

**3. Speculative Decoding** (`speculative/`):
```python
# Draft model
for step_id in range(num_draft_steps):
    output = draft_backend.attn_backends[step_id].forward(...)

# Target model
verify_output = target_backend.forward(...)
```

### 9.3 External Library APIs

**1. FlashInfer TRT-LLM Kernels**:
```python
# MHA decode
flashinfer.decode.trtllm_batch_decode_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens,
    max_seq_len, bmm1_scale, bmm2_scale, window_left, sinks
)

# MHA prefill
flashinfer.prefill.trtllm_batch_context_with_kv_cache(
    query, kv_cache, workspace_buffer, block_tables, seq_lens,
    max_q_len, max_kv_len, bmm1_scale, bmm2_scale,
    batch_size, cum_seq_lens_q, cum_seq_lens_kv, window_left, sinks
)

# MLA decode
flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
    query, kv_cache, workspace_buffer, qk_nope_head_dim,
    kv_lora_rank, qk_rope_head_dim, block_tables, seq_lens,
    max_seq_len, bmm1_scale
)

# MLA prefill
flashinfer.prefill.trtllm_ragged_attention_deepseek(
    query, key, value, workspace_buffer, seq_lens,
    max_q_len, max_kv_len, bmm1_scale, bmm2_scale, o_sf_scale,
    batch_size, window_left, cum_seq_lens_q, cum_seq_lens_kv,
    enable_pdl, is_causal, return_lse, out
)

# RoPE + Quantization fusion
flashinfer.rope.mla_rope_quantize_fp8(
    q_rope, k_rope, q_nope, k_nope, cos_sin_cache, pos_ids,
    is_neox, quantize_dtype, q_rope_out, k_rope_out,
    q_nope_out, k_nope_out, quant_scale_q, quant_scale_kv
)
```

**2. Triton Utilities**:
```python
# From utils.py
create_flashinfer_kv_indices_triton[(batch_size,)](
    req_to_token, req_pool_indices, page_kernel_lens,
    kv_indptr, kv_start_idx, kv_indices,
    req_to_token_stride
)

create_flashmla_kv_indices_triton[(batch_size,)](
    req_to_token, req_pool_indices, page_kernel_lens,
    kv_start_idx, kv_indices, req_to_token_stride,
    kv_indices_stride, PAGED_SIZE
)
```

---

## 10. Critical Code Paths

### 10.1 Hot Paths (Called Every Token)

**1. Decode metadata replay** (MHA:307-407, MLA:447-505)
- Frequency: Every decode iteration
- Operations:
  - Scalar assignments: ~5 ops
  - Tensor copies: 2-3 copies
  - Page table recomputation: 1 kernel launch
- Latency: ~10-20μs

**2. Forward decode execution** (MHA:515-572, MLA:749-848)
- Frequency: Every layer, every token
- Operations:
  - KV cache save: 1 kernel
  - KV cache load: 2 buffer retrievals
  - Attention kernel: 1 kernel
- Latency: ~50-200μs depending on batch size

**3. KV cache operations** (memory_pool.py:732-766, 1394-1408)
- Frequency: Every layer, every token
- Operations:
  - MHA: 2 tensor indexing ops (K and V)
  - MLA: 1 concatenation + 1 tensor indexing
- Latency: ~5-10μs

### 10.2 Warm Paths (Called Per Batch)

**1. Dynamic metadata initialization** (MHA:413-513, MLA:511-586)
- Frequency: For uncaptured batch sizes
- Operations:
  - Tensor allocations: 5-8 tensors
  - Cumulative sums: 2 operations
  - Page table extraction: 1 indexing op
- Latency: ~50-100μs

**2. Extend/prefill execution** (MHA:574-630, MLA:850-1054)
- Frequency: Initial prompt processing
- Operations:
  - KV cache operations: Same as decode
  - Attention kernel: Larger workspace usage
  - Potential padding/unpadding: Draft mode only
- Latency: ~500μs-10ms depending on prompt length

### 10.3 Cold Paths (Called Once or Rarely)

**1. Backend initialization** (MHA:56-108, MLA:227-286)
- Frequency: Once per model load
- Operations:
  - Global workspace allocation: 128-512 MB
  - Configuration extraction: ~20 assignments
  - Dictionary initialization: 3-4 dicts
- Latency: ~10-50ms

**2. CUDA graph state initialization** (MHA:109-193, MLA:348-377)
- Frequency: Once per model load
- Operations:
  - Buffer allocations: 5-10 large tensors
  - Metadata structure creation: Multiple dicts
- Latency: ~50-200ms

**3. Graph capture** (MHA:194-305, MLA:379-445)
- Frequency: Once per (batch_size, forward_mode) pair
- Operations:
  - Metadata template creation
  - Dummy forward pass
  - Graph recording
- Latency: ~100-500ms

**4. Block padding calculation** (MLA:287-308)
- Frequency: Once per backend initialization
- Operations:
  - LCM computation
  - Padding calculation
- Latency: <1μs

### 10.4 Speculative Decoding Paths

**1. Draft generation** (frequency: num_draft_steps per token)
- Path: `attn_backends[step_id].forward_decode()`
- Latency: Similar to normal decode (~50-200μs)

**2. Target verification** (frequency: 1 per accepted sequence)
- Path: `forward_extend()` with TARGET_VERIFY mode
- Latency: ~200-500μs (processes multiple tokens)

**3. Draft extend** (frequency: num_draft_steps - 1 per accepted sequence)
- Path: `forward_extend()` with DRAFT_EXTEND mode
- Extra operations: Padding + unpadding kernels (~20-50μs each)
- Latency: ~100-300μs

---

## Conclusion

The TensorRT-LLM attention backends represent a sophisticated optimization layer in SGLang's attention system. Key achievements:

1. **Performance**: CUDA graphs reduce launch overhead by 64×, critical for small batches
2. **Memory Efficiency**: MLA compression achieves 56× KV cache reduction for DeepSeek models
3. **Flexibility**: Supports 7 execution modes including advanced speculative decoding
4. **Correctness**: Careful constraint management (LCM padding, in-place updates) ensures reliability

The implementations demonstrate deep understanding of:
- GPU kernel optimization (TRT-LLM, Triton)
- Memory management (global buffers, graph constraints)
- Attention algorithms (MHA, MLA, sliding window, causal masking)
- Speculative execution (EAGLE multi-step drafting)

Future optimization opportunities:
- Dynamic block padding (currently conservative)
- Further kernel fusion (attention + RoPE + LayerNorm)
- Multi-query attention (MQA) and grouped-query attention (GQA) specializations
- Support for newer hardware (Blackwell B200, Rubin R100)
