# SGLang Memory Pool System

## Overview

The SGLang memory pool system manages GPU memory allocation for KV (Key-Value) cache during LLM inference. It implements a sophisticated two-level memory management architecture that efficiently handles different attention mechanisms and model architectures.

**Location**: `python/sglang/srt/mem_cache/memory_pool.py`

## Architecture

### Two-Level Memory Management

SGLang uses a two-level memory pool architecture:

1. **Request-to-Token Pool** (`ReqToTokenPool`): Maps requests to their token locations
2. **Token-to-KV Pool** (`KVCache` + allocators): Manages the actual physical KV cache data

```
Request → ReqToTokenPool → Token Indices → TokenToKVPoolAllocator → Physical KV Cache
```

### Component Hierarchy

```
KVCache (Abstract Base)
├── MHATokenToKVPool              # Multi-Head Attention
├── MLATokenToKVPool              # Multi-Latent Attention (DeepSeek)
│   ├── NSATokenToKVPool          # Native Structured Attention with FP8 quantization
│   └── AscendMLAPagedTokenToKVPool  # Paged MLA for Ascend NPU
├── HybridLinearKVPool            # Hybrid attention + Mamba layers
├── SWAKVPool                     # Sliding Window Attention
├── AscendTokenToKVPool           # Ascend NPU optimized
└── DoubleSparseTokenToKVPool     # Double sparse attention

ReqToTokenPool
├── HybridReqToTokenPool          # Request pool with Mamba support
└── MambaPool                     # Mamba state cache management
```

## Core Components

### 1. ReqToTokenPool

**Purpose**: Maps request indices to token locations in the KV cache.

**Key Features**:
- Maintains a 2D tensor: `[num_requests, max_context_len]`
- Each row stores token indices for a request
- Manages request slot allocation/deallocation
- Supports memory saver mode for efficient GPU memory usage

**Implementation**:
```python
class ReqToTokenPool:
    req_to_token: torch.Tensor  # [size, max_context_len]
    free_slots: List[int]       # Available request slots
```

**Usage Pattern**:
```python
# Allocate slots for 4 requests
indices = pool.alloc(4)  # Returns [0, 1, 2, 3]

# Write token locations
pool.write(indices, token_locations)

# Free slots when done
pool.free(indices)
```

### 2. KVCache (Abstract Base Class)

**Purpose**: Defines the interface for physical KV cache storage.

**Core Methods**:
- `get_key_buffer(layer_id)`: Retrieve K cache for a layer
- `get_value_buffer(layer_id)`: Retrieve V cache for a layer
- `set_kv_buffer(layer, loc, cache_k, cache_v)`: Write KV cache
- `get_cpu_copy(indices)`: CPU offloading support
- `load_cpu_copy(kv_cache_cpu, indices)`: Load from CPU

**Common Properties**:
- `size`: Total cache size in tokens
- `page_size`: Granularity of allocation (1 for token-level, >1 for paged)
- `dtype`: Storage data type (fp16, bf16, fp8, etc.)
- `layer_num`: Number of transformer layers
- `device`: Device placement (cuda, npu, etc.)
- `mem_usage`: Memory usage in GB

### 3. MHATokenToKVPool

**Purpose**: Standard multi-head attention KV cache.

**Architecture**:
```
K buffer: [num_layers, size, num_heads, head_dim]
V buffer: [num_layers, size, num_heads, head_dim]
```

**Key Features**:
- Separate K and V buffers per layer
- Padded slot 0 for dummy outputs from padded tokens
- Optional FP8 storage (stores as uint8, views as fp8)
- Dual-stream optimization for K/V overlap during CUDA graph capture
- Optional tiled KV copy for efficient cache migration

**Performance Optimizations**:
1. **Alternate Stream**: Overlaps K and V cache writes during CUDA graph mode
2. **Tiled Copy Kernel**: Efficient multi-layer KV cache movement using Triton
3. **CPU Offloading**: Chunked transfer (8192 tokens per chunk) for memory pressure

**Example - Tiled Copy Configuration**:
```python
# Heuristics based on stride
if stride_bytes >= 8192:
    bytes_per_tile = 512
elif stride_bytes >= 4096:
    bytes_per_tile = 256
else:
    bytes_per_tile = 128
```

### 4. MLATokenToKVPool

**Purpose**: Multi-Latent Attention cache for DeepSeek models.

**Architecture**:
```
KV buffer: [num_layers, size, 1, kv_lora_rank + qk_rope_head_dim]
```

**Key Innovation**:
- Combines K nope (non-positional) and K rope (rotary positional) components
- Single unified buffer instead of separate K/V
- Reduces memory footprint by ~2x compared to standard MHA

**Memory Layout**:
```
[token_0][token_1]...[token_N]
   ↓
[nope_dim: 512 | rope_dim: 64]
```

**Triton Kernels**:
1. `set_mla_kv_buffer_kernel`: Writes nope + rope to unified buffer
2. `get_mla_kv_buffer_kernel`: Extracts nope and rope components

### 5. NSATokenToKVPool

**Purpose**: Native Structured Attention with FP8 quantization for MLA models.

**Key Features**:
- FP8 quantized index_k storage for memory efficiency
- Separate scale factors per quantization block (128 elements)
- Combined buffer layout for index_k data and scales

**Buffer Layout**:
```
Page i:
  [FP8 data: page_size * head_dim bytes]
  [Scales: page_size * (head_dim / 128) * 4 bytes as float32]
```

**Quantization**:
- Block size: 128 elements
- Store dtype: uint8 (viewed as float8_e4m3fn)
- Paged allocation: page_size = 64 tokens

### 6. MambaPool

**Purpose**: Manages state cache for Mamba/SSM (State Space Model) layers.

**State Tensors**:
```python
conv_state: [num_layers, size+1, conv_dim, conv_kernel_size]
temporal_state: [num_layers, size+1, heads, head_dim_1, head_dim_2]
```

**Speculative Decoding Support**:
```python
# Additional buffers for draft token speculation
intermediate_ssm: [num_layers, size+1, draft_tokens, H, K, V]
intermediate_conv_window: [num_layers, size+1, draft_tokens, dim, K-1]
```

**Custom Memory Pool**:
- Optional NVLink-backed memory pool for disaggregated serving
- Environment variable: `SGLANG_MOONCAKE_CUSTOM_MEM_POOL`

### 7. HybridReqToTokenPool

**Purpose**: Unified request pool for hybrid models (attention + Mamba layers).

**Architecture**:
```
ReqToTokenPool (base)
  └── MambaPool
      └── req_index_to_mamba_index_mapping
```

**Key Features**:
- Manages both attention token indices and Mamba state indices
- Supports radix cache for Mamba states
- Handles chunk prefill without premature state allocation

**Allocation Flow**:
```python
# Allocates both token and Mamba slots
select_index = hybrid_pool.alloc(need_size, reqs)
# Internally:
#   1. Allocate token slots via parent class
#   2. Allocate Mamba slots for each request
#   3. Update mapping: req_index -> mamba_index
```

### 8. SWAKVPool

**Purpose**: Sliding Window Attention with separate pools for full and windowed layers.

**Architecture**:
```
SWAKVPool
├── full_kv_pool: MHATokenToKVPool (size: full)
├── swa_kv_pool: MHATokenToKVPool (size: swa, smaller)
└── layers_mapping: {layer_id → (pool_layer_id, is_swa)}
```

**Key Insight**:
- SWA layers only need to cache recent tokens within the window
- Full attention layers need complete history
- Reduces memory for hybrid architectures (e.g., some layers with sliding window)

**Index Translation**:
```python
full_to_swa_index_mapping: torch.Tensor
# Maps full attention indices to SWA pool indices
swa_loc = pool.translate_loc_from_full_to_swa(full_loc)
```

### 9. AscendTokenToKVPool

**Purpose**: Optimized KV cache for Ascend NPU hardware.

**Key Difference**:
```python
# Continuous memory layout for better NPU performance
kv_buffer: [2, num_layers, num_pages, page_size, num_heads, head_dim]
#           ↑
#           0=K, 1=V
k_buffer = kv_buffer[0]
v_buffer = kv_buffer[1]
```

**Native Operations**:
- Uses `torch_npu._npu_reshape_and_cache` for efficient cache writing
- Optimized for Ascend's transfer backend

### 10. DoubleSparseTokenToKVPool

**Purpose**: Implements double sparse attention (sparse in both tokens and channels).

**Buffers**:
```python
k_buffer: [num_layers, size, num_heads, head_dim]
v_buffer: [num_layers, size, num_heads, head_dim]
label_buffer: [num_layers, size, num_heads, heavy_channel_num]
```

**Label Buffer**: Tracks which channels are "heavy" (important) for each token.

## Allocator System

The allocator layer (`python/sglang/srt/mem_cache/allocator.py`) manages token-level or page-level allocation on top of the physical KV cache.

### TokenToKVPoolAllocator

**Purpose**: Token-level allocation for non-paged KV cache.

**Allocation Strategy**:
```python
free_pages: torch.Tensor  # Immediately available indices
release_pages: torch.Tensor  # Freed but not yet sorted
```

**Sort-on-Demand**:
- If `need_sort=True`, freed pages go to `release_pages`
- Merged and sorted only when allocation fails
- Reduces overhead during normal operation

### PagedTokenToKVPoolAllocator

**Purpose**: Page-aligned allocation for paged attention.

**Key Features**:
- Allocates in page_size multiples
- Optimized Triton kernels for extend and decode phases
- Handles partial page filling

**Extend Phase Allocation**:
```python
alloc_extend_kernel:
  # Part 1: Fill old partial page
  # Part 2: Allocate new full pages
  # Part 3: Allocate new partial page
```

**Decode Phase Allocation**:
```python
alloc_decode_kernel:
  # Per request: allocate next token location
  # Reuse partial page if available, else allocate new page
```

### SWATokenToKVPoolAllocator

**Purpose**: Manages allocation for hybrid SWA architectures.

**Dual Allocation**:
```python
alloc(need_size):
    full_indices = full_allocator.alloc(need_size)
    swa_indices = swa_allocator.alloc(need_size)
    mapping[full_indices] = swa_indices
    return full_indices
```

**Memory Tracking**:
```python
full_available_size()  # Full attention tokens available
swa_available_size()   # SWA tokens available
```

## Memory Management Utilities

### Helper Functions (`common.py`)

**alloc_token_slots**:
```python
def alloc_token_slots(tree_cache, num_tokens, backup_state=False):
    # 1. Evict from tree cache if needed
    # 2. Backup state if requested
    # 3. Allocate tokens
    # 4. Handle OOM with detailed error message
```

**alloc_for_extend**:
```python
def alloc_for_extend(batch):
    # 1. Allocate request slots
    # 2. Allocate KV cache (paged or non-paged)
    # 3. Write token indices to req_to_token_pool
    return out_cache_loc, req_pool_indices_device, req_pool_indices
```

**alloc_for_decode**:
```python
def alloc_for_decode(batch, token_per_req):
    # 1. Allocate KV cache for new tokens
    # 2. Write to req_to_token_pool
    return out_cache_loc
```

### Triton Optimization

**write_req_to_token_pool_triton**:
- Fused kernel for writing cache indices
- Handles both prefix (from radix cache) and new tokens
- Block size: 512 elements

**get_last_loc_triton**:
- Extracts last token location for each request
- Used in paged allocation for partial page continuation

## Advanced Features

### 1. CPU Offloading

**Chunked Transfer**:
```python
chunk_size = 8192  # Tokens per transfer
for i in range(0, len(indices), chunk_size):
    chunk = kv_cache[indices[i:i+chunk_size]]
    cpu_copy.append(chunk.to('cpu', non_blocking=True))
```

**Synchronization**:
- `torch.cuda.synchronize()` before and after transfers
- Ensures data integrity

### 2. Memory Saver Mode

**TorchMemorySaverAdapter Integration**:
```python
with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
    # Allocate KV cache
    # Tagged for memory profiling and management
```

### 3. Custom Memory Pools

**Mooncake Integration** (Disaggregated Serving):
```python
from mooncake.allocator import NVLinkAllocator
allocator = NVLinkAllocator.get_allocator(device)
custom_mem_pool = torch.cuda.MemPool(allocator.allocator())

with torch.cuda.use_mem_pool(custom_mem_pool):
    # Allocate from NVLink-backed memory
```

### 4. Layer-wise Transfer Control

**Synchronization for Disaggregated Prefill**:
```python
kv_cache.register_layer_transfer_counter(layer_done_counter)

# During forward pass
def get_key_buffer(layer_id):
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id)
    return self.k_buffer[layer_id]
```

### 5. FP8 Storage

**Transparent Conversion**:
```python
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
    store_dtype = torch.uint8  # index_put not implemented for fp8
else:
    store_dtype = dtype

# On read
if self.store_dtype != self.dtype:
    return buffer.view(self.dtype)
```

### 6. CUDA Graph Optimization

**Dual Stream K/V Writing**:
```python
if get_is_capture_mode() and self.alt_stream is not None:
    current_stream = torch.cuda.current_stream()
    self.alt_stream.wait_stream(current_stream)

    # Write K on main stream
    k_buffer[loc] = cache_k

    # Write V on alternate stream
    with torch.cuda.stream(self.alt_stream):
        v_buffer[loc] = cache_v

    current_stream.wait_stream(self.alt_stream)
```

## Usage Examples

### Basic MHA Model

```python
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

# Create KV cache
kv_cache = MHATokenToKVPool(
    size=32768,           # Total tokens
    page_size=1,          # Token-level allocation
    dtype=torch.float16,
    head_num=32,
    head_dim=128,
    layer_num=32,
    device="cuda",
    enable_memory_saver=False,
)

# Create allocator
allocator = TokenToKVPoolAllocator(
    size=32768,
    dtype=torch.float16,
    device="cuda",
    kvcache=kv_cache,
    need_sort=True,
)

# Create request pool
req_pool = ReqToTokenPool(
    size=512,             # Max concurrent requests
    max_context_len=4096,
    device="cuda",
    enable_memory_saver=False,
)

# Allocate
token_indices = allocator.alloc(128)  # Allocate 128 tokens
req_indices = req_pool.alloc(4)       # Allocate 4 request slots

# Use during forward pass
k_buffer, v_buffer = kv_cache.get_kv_buffer(layer_id=0)
kv_cache.set_kv_buffer(layer, loc=token_indices, cache_k=k, cache_v=v)

# Free when done
allocator.free(token_indices)
req_pool.free(req_indices)
```

### Hybrid Mamba + Attention Model

```python
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, HybridLinearKVPool

# Create Mamba pool
mamba_pool = MambaPool(
    size=4096,
    cache_params=mamba2_cache_params,
    device="cuda",
)

# Create hybrid request pool
hybrid_req_pool = HybridReqToTokenPool(
    size=32768,
    mamba_size=4096,
    max_context_len=4096,
    device="cuda",
    enable_memory_saver=False,
    cache_params=mamba2_cache_params,
)

# Create hybrid KV pool
hybrid_kv_pool = HybridLinearKVPool(
    size=32768,
    dtype=torch.float16,
    page_size=1,
    head_num=32,
    head_dim=128,
    full_attention_layer_ids=[0, 2, 4, ...],  # Which layers use attention
    enable_kvcache_transpose=False,
    device="cuda",
    mamba_pool=mamba_pool,
)

# Allocation includes both KV and Mamba state
req_indices = hybrid_req_pool.alloc(need_size=4, reqs=request_list)
```

### Paged Attention with SWA

```python
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

# Create paged allocator
paged_allocator = PagedTokenToKVPoolAllocator(
    size=32768,
    page_size=16,         # 16 tokens per page
    dtype=torch.float16,
    device="cuda",
    kvcache=kv_cache,
    need_sort=True,
)

# Extend phase allocation (prefill)
out_indices = paged_allocator.alloc_extend(
    prefix_lens=torch.tensor([0, 10, 20]),
    prefix_lens_cpu=torch.tensor([0, 10, 20]),
    seq_lens=torch.tensor([32, 42, 52]),
    seq_lens_cpu=torch.tensor([32, 42, 52]),
    last_loc=torch.tensor([-1, 159, 335]),
    extend_num_tokens=96,
)

# Decode phase allocation (one token per request)
decode_indices = paged_allocator.alloc_decode(
    seq_lens=torch.tensor([33, 43, 53]),
    seq_lens_cpu=torch.tensor([33, 43, 53]),
    last_loc=torch.tensor([31, 47, 63]),
)
```

## Memory Layout Examples

### Token-level (page_size=1)

```
Token indices: [1, 2, 3, 4, 5, ...]
K buffer: [layer_0: tokens, layer_1: tokens, ...]
V buffer: [layer_0: tokens, layer_1: tokens, ...]

Request allocation returns: [1, 2, 3, 4]  (contiguous)
```

### Paged (page_size=16)

```
Pages: [0, 1, 2, 3, ...]
Each page: 16 tokens

Request with 50 tokens:
  Page 0: tokens [0-15]
  Page 1: tokens [16-31]
  Page 2: tokens [32-47]
  Page 3: tokens [48-49] (partial)

Allocation returns: [0, 16, 32, 48, ...]
```

### MLA Layout

```
Unified buffer per token:
  [nope: 512 dims | rope: 64 dims] = 576 total

Layer 0: [token_0: 576 dims | token_1: 576 dims | ...]
Layer 1: [token_0: 576 dims | token_1: 576 dims | ...]
...
```

## Performance Considerations

### Memory Efficiency

1. **FP8 Storage**: ~50% memory reduction with minimal accuracy loss
2. **MLA**: ~50% reduction vs standard MHA (single buffer vs K+V)
3. **SWA**: Proportional to window size vs full history
4. **Paging**: Reduces fragmentation, enables oversubscription

### Compute Efficiency

1. **Triton Kernels**: Fused operations reduce kernel launch overhead
2. **Dual Streams**: Overlaps K and V writes during CUDA graph
3. **Tiled Copy**: Efficient multi-layer cache migration
4. **Sort-on-Demand**: Reduces allocation overhead

### Memory vs Compute Tradeoffs

| Feature | Memory Savings | Compute Overhead |
|---------|----------------|------------------|
| FP8 Storage | 50% | Minimal (hw accelerated) |
| MLA | 50% | None |
| Paging | Variable | ~5% (page management) |
| CPU Offload | High | High (PCIe bandwidth) |

## Integration Points

### Radix Cache (Prefix Caching)

```python
# Radix cache provides prefix_indices
prefix_indices = radix_cache.match_prefix(req)
req_pool.write((req_idx, slice(0, len(prefix_indices))), prefix_indices)
```

### Scheduler

```python
# Scheduler calls allocation during batching
from sglang.srt.mem_cache.common import alloc_for_extend, alloc_for_decode

# Prefill phase
out_cache_loc, req_indices, _ = alloc_for_extend(batch)

# Decode phase
out_cache_loc = alloc_for_decode(batch, token_per_req=1)
```

### Model Executor

```python
# Model runner accesses KV cache during forward pass
k_buffer, v_buffer = kv_cache.get_kv_buffer(layer_id)

# Attention layer writes to cache
kv_cache.set_kv_buffer(self, loc=cache_loc, cache_k=k, cache_v=v)
```

## Error Handling

### Out of Memory

```python
if out_cache_loc is None:
    error_msg = (
        f"Out of memory. Try to lower your batch size.\n"
        f"Try to allocate {num_tokens} tokens.\n"
        f"Available tokens: {available_size}\n"
        f"Evictable tokens: {evictable_size}\n"
    )
    logger.error(error_msg)
    tree_cache.pretty_print()  # Debug output
    raise RuntimeError(error_msg)
```

### Assertions (Debug Mode)

```python
SGLANG_DEBUG_MEMORY_POOL=1  # Enable debug assertions

# Checks:
# - Page alignment
# - No duplicate allocations
# - Last location correctness
# - Sufficient free pages
```

## Future Directions

1. **Dynamic Paging**: Variable page sizes per request
2. **Hierarchical Caching**: Multi-tier (GPU → CPU → Disk)
3. **Compression**: Online compression of old tokens
4. **Prefetch**: Predictive loading for CPU-offloaded cache
5. **Distributed KV**: Cross-node KV cache sharing

## References

- Main implementation: `python/sglang/srt/mem_cache/memory_pool.py`
- Allocators: `python/sglang/srt/mem_cache/allocator.py`
- Utilities: `python/sglang/srt/mem_cache/common.py`
- Scheduler integration: `python/sglang/srt/managers/scheduler.py`
- Model runner: `python/sglang/srt/model_executor/model_runner.py`

## Summary

The SGLang memory pool system provides:

- **Flexible architecture** supporting MHA, MLA, Mamba, and hybrid models
- **Efficient allocation** with token-level and paged strategies
- **Memory optimization** via FP8, compression, and CPU offloading
- **Hardware acceleration** with Triton kernels and CUDA graphs
- **Production ready** with error handling and debugging support

The two-level design (request → tokens → physical cache) provides clean abstraction while enabling advanced features like radix caching, paging, and disaggregated serving.
