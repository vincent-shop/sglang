# SGLang Cache Mechanisms: Deep Technical Documentation

## Table of Contents

1. [Overview and Architecture](#overview-and-architecture)
2. [Memory Hierarchy](#memory-hierarchy)
3. [Core Cache Implementations](#core-cache-implementations)
4. [Radix Tree Data Structure](#radix-tree-data-structure)
5. [Eviction Policies](#eviction-policies)
6. [Memory Allocators](#memory-allocators)
7. [Advanced Features](#advanced-features)
8. [Integration and Usage](#integration-and-usage)
9. [Performance Considerations](#performance-considerations)

---

## Overview and Architecture

SGLang implements a sophisticated multi-tier caching system for Key-Value (KV) cache management in large language model inference. The system is designed to maximize cache hit rates through intelligent prefix matching and reuse, reducing redundant computation and improving throughput.

### Design Philosophy

The cache system is built around three core principles:

1. **Prefix Reuse**: Share computation across requests with common token prefixes
2. **Memory Efficiency**: Manage GPU memory through eviction and hierarchical storage
3. **Flexibility**: Support multiple cache strategies and architectural patterns

### Cache Hierarchy

```
┌─────────────────────────────────────────────────────┐
│                 BasePrefixCache                     │
│            (Abstract Base Interface)                │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴──────────┬──────────────────┐
        │                    │                  │
┌───────▼────────┐  ┌────────▼─────────┐  ┌────▼──────────┐
│  ChunkCache    │  │   RadixCache     │  │ SWARadixCache │
│  (No prefix    │  │  (Full radix     │  │  (Sliding     │
│   matching)    │  │   tree cache)    │  │   window)     │
└────────────────┘  └──────┬───────────┘  └───────────────┘
                           │
                    ┌──────▼───────┐
                    │ HiRadixCache │
                    │ (Hierarchical│
                    │  + Storage)  │
                    └──────────────┘
```

---

## Memory Hierarchy

### Three-Tier Memory Architecture

SGLang organizes memory across three distinct tiers for optimal performance and capacity:

#### 1. Device Memory (GPU VRAM)
- **Purpose**: Active KV cache for immediate inference
- **Speed**: Fastest access (10-100 GB/s)
- **Capacity**: Limited (typically 16-80 GB)
- **Management**: Token-to-KV pool allocators

#### 2. Host Memory (System RAM)
- **Purpose**: Intermediate storage for evicted cache
- **Speed**: Medium access via PCIe (10-30 GB/s)
- **Capacity**: Larger (typically 128-512 GB)
- **Management**: Host memory pool allocators
- **Feature**: Used in HiRadixCache (HiCache)

#### 3. Storage Backend (Disk/Network Storage)
- **Purpose**: Long-term persistent cache
- **Speed**: Slower (1-10 GB/s for NVMe, varies for network)
- **Capacity**: Very large (TBs)
- **Backends**: File, S3, Mooncake, nixl, etc.
- **Feature**: Optional persistent storage in HiRadixCache

### Memory Pool Structure

```
ReqToTokenPool                    TokenToKVPoolAllocator               KVCache
┌─────────────┐                  ┌──────────────────┐            ┌──────────────┐
│             │                  │                  │            │              │
│ Request ID  ├─────────────────►│  Token Indices  ├───────────►│  Physical    │
│ Mappings    │                  │  Management     │            │  KV Storage  │
│             │                  │                  │            │              │
│ [Req → Tok] │                  │  Free Pages     │            │  K Buffers   │
│             │                  │  Allocation     │            │  V Buffers   │
│             │                  │  Deallocation   │            │              │
└─────────────┘                  └──────────────────┘            └──────────────┘
```

#### ReqToTokenPool

**Location**: `python/sglang/srt/mem_cache/memory_pool.py:66-114`

A mapping pool that tracks which token positions belong to which request.

**Key Attributes**:
```python
self.size: int                    # Maximum number of requests
self.max_context_len: int         # Maximum sequence length per request
self.req_to_token: torch.Tensor   # Shape: (size, max_context_len)
self.free_slots: List[int]        # Available request slots
```

**Operations**:
- `alloc(need_size: int)`: Allocates request slots
- `free(free_index: Union[int, List[int]])`: Frees request slots
- `write(indices, values)`: Updates token mappings

**Example**: Request 5 with tokens at indices [100, 101, 102, 103]
```python
req_to_token_pool.req_to_token[5, 0:4] = torch.tensor([100, 101, 102, 103])
```

#### TokenToKVPoolAllocator

**Location**: `python/sglang/srt/mem_cache/allocator.py:118-173`

Manages the mapping from token indices to physical KV cache locations.

**Variants**:

1. **TokenToKVPoolAllocator**: Standard allocator (page_size=1)
   ```python
   def alloc(self, need_size: int) -> torch.Tensor:
       # Returns contiguous indices for tokens
       select_index = self.free_pages[:need_size]
       self.free_pages = self.free_pages[need_size:]
       return select_index
   ```

2. **PagedTokenToKVPoolAllocator**: Page-aligned allocator
   - **Page Size**: Configurable (typically 16-64 tokens)
   - **Optimization**: Better memory locality, reduced fragmentation
   - **Location**: `allocator.py:411-580`

3. **SWATokenToKVPoolAllocator**: Dual-pool allocator for SWA
   - **Full Pool**: For all tokens
   - **SWA Pool**: For sliding window tokens
   - **Mapping**: `full_to_swa_index_mapping` tensor
   - **Location**: `allocator.py:175-293`

**Key Methods**:
```python
def alloc(self, need_size: int) -> Optional[torch.Tensor]:
    """Allocate indices for tokens"""

def free(self, free_index: torch.Tensor):
    """Free token indices back to pool"""

def merge_and_sort_free(self):
    """Consolidate free pages for defragmentation"""
```

#### KVCache

**Location**: `python/sglang/srt/mem_cache/memory_pool.py:394-496`

Abstract base class for physical KV cache storage.

**Implementations**:

1. **MHATokenToKVPool** (`memory_pool.py:498-796`)
   - Multi-Head Attention cache
   - Separate K and V buffers per layer
   - Shape: `[size + page_size, head_num, head_dim]` per layer

2. **MLATokenToKVPool** (`memory_pool.py:1275-1474`)
   - Multi-Latent Attention (DeepSeek models)
   - Combined KV buffer with compression
   - Shape: `[size + page_size, 1, kv_lora_rank + qk_rope_head_dim]`

3. **NSATokenToKVPool** (`memory_pool.py:1476-1598`)
   - Next State Attention with FP8 quantization
   - Additional index_k buffer for efficient attention
   - Quantization-aware storage

4. **HybridLinearKVPool** (`memory_pool.py:798-901`)
   - Mixed MHA + Mamba layers
   - Separate pools for attention and state-space layers

5. **SWAKVPool** (`memory_pool.py:903-1059`)
   - Dual KV pools for sliding window attention
   - Full attention pool + SWA pool
   - Index mapping between pools

6. **DoubleSparseTokenToKVPool** (`memory_pool.py:1796-1884`)
   - Support for double sparse attention patterns
   - Additional label buffer for channel sparsity

**Key Interface**:
```python
class KVCache(abc.ABC):
    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Retrieve K buffer for a layer"""

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Retrieve V buffer for a layer"""

    @abc.abstractmethod
    def set_kv_buffer(self, layer: RadixAttention, loc: torch.Tensor,
                      cache_k: torch.Tensor, cache_v: torch.Tensor):
        """Write KV cache at specified locations"""
```

**Memory Layout Example (MHA)**:
```
Layer 0:  K[0:size, 0:head_num, 0:head_dim]  V[0:size, 0:head_num, 0:head_dim]
Layer 1:  K[0:size, 0:head_num, 0:head_dim]  V[0:size, 0:head_num, 0:head_dim]
...
Layer N:  K[0:size, 0:head_num, 0:head_dim]  V[0:size, 0:head_num, 0:head_dim]
```

---

## Core Cache Implementations

### BasePrefixCache

**Location**: `python/sglang/srt/mem_cache/base_prefix_cache.py`

Abstract interface defining the contract for all cache implementations.

**Core Methods**:

```python
class BasePrefixCache(ABC):
    @abstractmethod
    def match_prefix(self, key: Any, **kwargs) -> MatchResult:
        """Find longest cached prefix matching the given key"""

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        """Cache a completed request"""

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        """Cache an incomplete request (chunked prefill)"""

    @abstractmethod
    def evict(self, num_tokens: int):
        """Evict tokens to free memory"""

    @abstractmethod
    def inc_lock_ref(self, node: Any):
        """Lock cache entries to prevent eviction"""

    @abstractmethod
    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        """Unlock cache entries"""
```

**MatchResult Structure**:
```python
class MatchResult(NamedTuple):
    device_indices: torch.Tensor    # GPU cache indices matched
    last_device_node: Any           # Last matched node on GPU
    last_host_node: Any             # Last matched node on host
    host_hit_length: int = 0        # Length of host cache hit
```

### ChunkCache

**Location**: `python/sglang/srt/mem_cache/chunk_cache.py`

Simplest cache implementation with **no prefix matching**. Used when RadixCache is disabled.

**Design Rationale**:
- Minimal overhead for scenarios where prefix reuse is not expected
- Direct memory management without tree structure
- Suitable for diverse, non-repetitive workloads

**Implementation Details**:

```python
class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.protected_size_ = 0  # Tracks locked tokens
```

**Key Characteristics**:

1. **No Prefix Matching**:
   ```python
   def match_prefix(self, **unused_kwargs) -> MatchResult:
       return MatchResult(
           device_indices=torch.empty((0,), dtype=torch.int64),
           last_device_node=None,
           last_host_node=None,
       )
   ```
   Always returns empty match - no reuse.

2. **Direct Memory Management**:
   ```python
   def cache_finished_req(self, req: Req, is_insert: bool = True):
       kv_indices = self.req_to_token_pool.req_to_token[
           req.req_pool_idx,
           : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
       ]
       self.req_to_token_pool.free(req.req_pool_idx)
       self.token_to_kv_pool_allocator.free(kv_indices)
       self.protected_size_ -= len(req.prefix_indices)
   ```
   Directly frees all KV cache for completed requests.

3. **Chunked Prefill Support**:
   ```python
   def cache_unfinished_req(self, req: Req, chunked=False):
       kv_indices = self.req_to_token_pool.req_to_token[
           req.req_pool_idx, : len(req.fill_ids)
       ]
       self.protected_size_ += len(kv_indices) - len(req.prefix_indices)
       req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
   ```
   Stores KV indices for later chunked prefill stages.

**SWAChunkCache Variant**:

```python
class SWAChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def evict_swa(self, req: Req, prelen: int, attention_chunk_size: int):
        if prelen >= req.evicted_seqlen_local + attention_chunk_size:
            new_evicted_seqlen_local = attention_chunk_size * (
                prelen // attention_chunk_size
            )
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx,
                req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local
```

Extends ChunkCache for sliding window attention with selective eviction.

---

## Radix Tree Data Structure

### Core Concepts

The radix tree (also called prefix tree or compact trie) is the foundation of SGLang's cache reuse mechanism. Unlike a standard trie where each node represents a single token, a radix tree node can represent multiple tokens, making it more space-efficient.

**Location**: `python/sglang/srt/mem_cache/radix_cache.py`

### RadixKey

**Definition** (`radix_cache.py:51-73`):
```python
class RadixKey:
    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        self.token_ids = token_ids  # Sequence of token IDs
        self.extra_key = extra_key  # Additional namespace (e.g., lora_id, cache_salt)
```

**Purpose of extra_key**:
- Isolates cache namespaces (different LoRA adapters, sampling salts, etc.)
- Entries with different `extra_key` values never share nodes
- Enables multi-tenancy and request isolation

**Key Operations**:
```python
def __len__(self) -> int:
    return len(self.token_ids)

def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
    if isinstance(idx, slice):
        return RadixKey(self.token_ids[idx], self.extra_key)
    return RadixKey([self.token_ids[idx]], self.extra_key)
```

### TreeNode

**Definition** (`radix_cache.py:75-134`):
```python
class TreeNode:
    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)  # Child nodes
        self.parent: TreeNode = None           # Parent node
        self.key: RadixKey = None              # Token sequence
        self.value: Optional[torch.Tensor] = None  # KV cache indices
        self.lock_ref = 0                      # Lock counter
        self.last_access_time = time.monotonic()  # For LRU
        self.creation_time = time.monotonic()     # For FIFO
        self.hit_count = 0                     # For LFU
        self.host_ref_counter = 0              # Host protection counter
        self.host_value: Optional[torch.Tensor] = None  # Host KV indices
        self.hash_value: Optional[List[str]] = None     # Page hashes
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
```

**Node Properties**:

1. **evicted**: Node's GPU cache has been evicted
   ```python
   @property
   def evicted(self):
       return self.value is None
   ```

2. **backuped**: Node has host backup
   ```python
   @property
   def backuped(self):
       return self.host_value is not None
   ```

3. **protect_host()**: Prevent host eviction
   ```python
   def protect_host(self):
       self.host_ref_counter += 1
   ```

### Key Matching Functions

**Token-by-Token Matching** (`radix_cache.py:143-150`):
```python
def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i
```

**Paged Matching** (`radix_cache.py:153-163`):
```python
def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i
```

**Benefits of Paged Matching**:
- Reduces comparison operations (page_size comparisons → 1 comparison)
- Aligns with GPU memory page boundaries
- Better cache locality

### RadixCache Implementation

**Initialization** (`radix_cache.py:188-237`):
```python
class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
        eviction_policy: str = "lru",
        is_eagle: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.is_eagle = is_eagle

        # Select matching function based on page size
        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=page_size)

        # Select key conversion for EAGLE (bigram keys)
        if is_eagle:
            self.key_convert_fn = _convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        # Initialize eviction strategy
        if eviction_policy.lower() == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        # ... other strategies ...

        self.reset()
```

### Prefix Matching Algorithm

**match_prefix Method** (`radix_cache.py:251-321`):

```python
def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
    """
    Find the longest cached prefix of `key` in the radix tree.

    Process:
    1. Convert key (e.g., to bigram for EAGLE)
    2. Align to page boundaries if paged
    3. Traverse tree matching prefixes
    4. Split nodes if match ends mid-node
    5. Return matched indices and last node
    """
    key.token_ids = self.key_convert_fn(key.token_ids)

    if self.disable or len(key) == 0:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
        )

    # Page alignment
    if self.page_size != 1:
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]

    if len(key) == 0:
        return empty_match_result()

    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = torch.empty((0,), dtype=torch.int64, device=self.device)
    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
    )
```

**Helper Method** (`radix_cache.py:563-586`):
```python
def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    node.last_access_time = time.monotonic()
    child_key = self.get_child_key_fn(key)

    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = time.monotonic()
        prefix_len = self.key_match_fn(child.key, key)

        if prefix_len < len(child.key):
            # Partial match: split node
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            # Full match: continue traversal
            value.append(child.value)
            node = child
            key = key[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

    return value, node
```

**Example Traversal**:

```
Tree state:
         root
          / \
       [1,2,3]  [4,5]
         /        \
      [4,5,6]   [6,7,8]

Query: [1,2,3,4,5,7,8]

Step 1: Match root → child [1,2,3]
        prefix_len = 3, matched tokens: [1,2,3]

Step 2: Match [1,2,3] → child [4,5,6]
        prefix_len = 2 (partial match with [4,5])
        Split [4,5,6] into [4,5] and [6]
        matched tokens: [1,2,3,4,5]

Step 3: No child for [7,8], stop
        Return: matched=[1,2,3,4,5], last_node=[4,5]
```

### Node Splitting

**Purpose**: When a match ends in the middle of a node's key, split the node to expose the exact boundary.

**_split_node Method** (`radix_cache.py:588-605`):
```python
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    # Create: parent.children[key] = new_node → child

    new_node = TreeNode()
    new_node.children = {self.get_child_key_fn(key[split_len:]): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]  # First part
    new_node.value = child.value[:split_len]

    child.parent = new_node
    child.key = child.key[split_len:]  # Second part
    child.value = child.value[split_len:]

    new_node.parent.children[self.get_child_key_fn(key)] = new_node

    self._record_store_event(new_node)
    self._record_store_event(child)

    return new_node
```

**Visualization**:
```
Before split (split_len=2):
    parent
      |
    child: [A,B,C,D] → value=[10,11,12,13]

After split:
    parent
      |
    new_node: [A,B] → value=[10,11]
      |
    child: [C,D] → value=[12,13]
```

### Insertion Algorithm

**insert Method** (`radix_cache.py:323-336`):
```python
def insert(self, key: RadixKey, value=None, chunked=False):
    if self.disable:
        return 0

    key.token_ids = self.key_convert_fn(key.token_ids)

    if value is None:
        value = torch.tensor(key.token_ids, dtype=torch.int64)

    if self.is_eagle:
        value = value[: len(key)]

    return self._insert_helper(self.root_node, key, value)
```

**_insert_helper Method** (`radix_cache.py:607-638`):
```python
def _insert_helper(self, node: TreeNode, key: RadixKey, value):
    node.last_access_time = time.monotonic()
    if len(key) == 0:
        return 0

    child_key = self.get_child_key_fn(key)
    total_prefix_length = 0

    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        node.last_access_time = time.monotonic()
        prefix_len = self.key_match_fn(node.key, key)
        total_prefix_length += prefix_len
        key = key[prefix_len:]
        value = value[prefix_len:]

        if prefix_len < len(node.key):
            # Split node for partial match
            new_node = self._split_node(node.key, node, prefix_len)
            node = new_node

        if len(key):
            child_key = self.get_child_key_fn(key)

    # Create new leaf for remaining key
    if len(key):
        new_node = TreeNode()
        new_node.parent = node
        new_node.key = key
        new_node.value = value
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
        self._record_store_event(new_node)

    return total_prefix_length
```

**Insertion Example**:

```
Initial tree:
    root
     |
   [1,2,3,4] → kv=[100,101,102,103]

Insert: key=[1,2,3,4,5,6], value=[100,101,102,103,104,105]

Step 1: Match [1,2,3,4] completely
        total_prefix_length = 4
        Remaining: key=[5,6], value=[104,105]

Step 2: No child for [5,6], create new node

Result:
    root
     |
   [1,2,3,4] → kv=[100,101,102,103]
     |
   [5,6] → kv=[104,105]

Returns: total_prefix_length = 4 (tokens reused)
```

### Cache Lifecycle Management

**cache_finished_req** (`radix_cache.py:338-397`):

Handles request completion and KV cache insertion/cleanup.

```python
def cache_finished_req(self, req: Req, is_insert: bool = True):
    all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

    if self.disable:
        # Simple cleanup without caching
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :all_token_len
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)
        self.req_to_token_pool.free(req.req_pool_idx)
        return

    token_ids = (req.origin_input_ids + req.output_ids)[:all_token_len]
    actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, :all_token_len
    ]

    # Page alignment
    if self.page_size != 1:
        page_aligned_len = actual_kv_len // self.page_size * self.page_size
        page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
            dtype=torch.int64, copy=True
        )
    else:
        page_aligned_len = actual_kv_len
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

    page_aligned_token_len = (
        page_aligned_len + 1 if self.is_eagle else page_aligned_len
    )

    old_prefix_len = len(req.prefix_indices)
    if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
        old_prefix_len -= 1

    # Insert into radix tree
    if is_insert:
        new_prefix_len = self.insert(
            RadixKey(token_ids[:page_aligned_token_len], req.extra_key),
            page_aligned_kv_indices,
        )
        # Free duplicates that were already in tree
        self.token_to_kv_pool_allocator.free(
            kv_indices[old_prefix_len:new_prefix_len]
        )
    else:
        self.token_to_kv_pool_allocator.free(
            kv_indices[old_prefix_len:page_aligned_len]
        )

    # Free unaligned tail
    self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

    # Release locks
    self.req_to_token_pool.free(req.req_pool_idx)
    self.dec_lock_ref(req.last_node)
```

**cache_unfinished_req** (`radix_cache.py:398-473`):

Handles chunked prefill where a request is paused mid-computation.

```python
def cache_unfinished_req(self, req: Req, chunked=False):
    if self.disable:
        return

    token_ids = req.fill_ids
    all_token_len = len(token_ids)
    actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, :all_token_len
    ]

    # Page alignment
    if self.page_size != 1:
        page_aligned_len = actual_kv_len // self.page_size * self.page_size
        page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
            dtype=torch.int64, copy=True
        )
    else:
        page_aligned_len = actual_kv_len
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

    page_aligned_token_len = (
        page_aligned_len + 1 if self.is_eagle else page_aligned_len
    )
    page_aligned_token_ids = token_ids[:page_aligned_token_len]

    old_prefix_len = len(req.prefix_indices)
    if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
        old_prefix_len -= 1

    # Insert partial results into cache
    new_prefix_len = self.insert(
        RadixKey(page_aligned_token_ids, req.extra_key),
        page_aligned_kv_indices,
        chunked=chunked,
    )
    self.token_to_kv_pool_allocator.free(kv_indices[old_prefix_len:new_prefix_len])

    # Re-match to get updated indices
    new_indices, new_last_node, _, _ = self.match_prefix(
        RadixKey(token_ids=page_aligned_token_ids, extra_key=req.extra_key)
    )
    self.req_to_token_pool.write(
        (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
        new_indices[old_prefix_len:],
    )

    req.last_matched_prefix_len = len(new_indices)

    self.dec_lock_ref(req.last_node)
    self.inc_lock_ref(new_last_node)

    # Update prefix indices for next chunk
    if self.page_size != 1:
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices[len(new_indices) :]]
        )
    else:
        if self.is_eagle:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[actual_kv_len:]]
            )
        else:
            req.prefix_indices = new_indices
    req.last_node = new_last_node
```

### Lock Reference Management

**Purpose**: Prevent eviction of actively-used cache entries.

**inc_lock_ref** (`radix_cache.py:511-523`):
```python
def inc_lock_ref(self, node: TreeNode):
    if self.disable:
        return 0

    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            # Move from evictable to protected
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        node = node.parent
    return delta
```

Locks a node and all ancestors up to root. Locked nodes cannot be evicted.

**dec_lock_ref** (`radix_cache.py:525-541`):
```python
def dec_lock_ref(self, node: TreeNode):
    if self.disable:
        return 0

    delta = 0
    while node != self.root_node:
        if node.lock_ref == 1:
            # Move from protected to evictable
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
            delta += len(node.key)
        node.lock_ref -= 1
        if node.parent is None:
            assert (
                node is self.root_node
            ), f"This request holds the node from another tree"
        node = node.parent
    return delta
```

**Lock Invariant**: For any node to be locked, all ancestors must also be locked. This ensures the path from root to node is always valid.

---

## Eviction Policies

**Location**: `python/sglang/srt/mem_cache/evict_policy.py`

When GPU memory is full, the cache must evict entries to make room for new requests. Different eviction strategies optimize for different access patterns.

### Base Strategy

```python
class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        """
        Return priority for eviction.
        Lower priority = evicted first.
        """
        pass
```

### LRU (Least Recently Used)

```python
class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time
```

**Behavior**: Evicts entries that haven't been accessed for the longest time.

**Best For**: General-purpose workloads with temporal locality.

**Example**:
```
Access pattern: A, B, C, A, D
Last access times: A=4, B=2, C=3, D=5
Eviction order: B (oldest) → C → A → D (newest)
```

### LFU (Least Frequently Used)

```python
class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)
```

**Behavior**: Evicts entries with lowest hit counts. Ties broken by last access time.

**Best For**: Workloads with popular "hot" prefixes that should stay cached.

**Example**:
```
Access counts: A=10, B=3, C=5, D=2
Eviction order: D (least frequent) → B → C → A (most frequent)
```

### FIFO (First In, First Out)

```python
class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time
```

**Behavior**: Evicts oldest entries regardless of access patterns.

**Best For**: Streaming workloads where entries are used once.

**Example**:
```
Creation times: A=1, B=2, C=3, D=4
Eviction order: A (oldest) → B → C → D (newest)
```

### MRU (Most Recently Used)

```python
class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time
```

**Behavior**: Evicts most recently used entries first (inverted LRU).

**Best For**: Sequential scan workloads where recently accessed data won't be needed again soon.

### FILO (First In, Last Out)

```python
class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time
```

**Behavior**: Evicts newest entries first (stack-like).

**Best For**: Specialized scenarios with LIFO access patterns.

### Eviction Algorithm

**evict Method** (`radix_cache.py:482-509`):

```python
def evict(self, num_tokens: int):
    if self.disable:
        return

    # Collect all leaf nodes
    leaves = self._collect_leaves()

    # Build min-heap based on eviction priority
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)

        # Skip locked nodes
        if x == self.root_node:
            break
        if x.lock_ref > 0:
            continue

        # Evict node
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)

        # If parent becomes childless, add to heap
        if len(x.parent.children) == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))

        self._record_remove_event(x)
```

**Process**:
1. Collect all leaf nodes (only leaves can be evicted)
2. Build min-heap prioritized by eviction strategy
3. Pop nodes from heap until `num_tokens` freed
4. Skip locked nodes
5. When parent becomes childless leaf, add to heap
6. Record eviction events (for disaggregation)

**Why Leaves Only?**
- Internal nodes have children that depend on them
- Evicting internal nodes would break tree structure
- Once all children evicted, parent becomes leaf

---

## SWARadixCache (Sliding Window Attention)

**Location**: `python/sglang/srt/mem_cache/swa_radix_cache.py`

Advanced cache for models with sliding window attention (e.g., Mistral, Yi). Maintains separate full and SWA caches with sophisticated tombstone mechanism.

### Architecture

```
┌─────────────────────────────────────────────┐
│           SWARadixCache                     │
├─────────────────────────────────────────────┤
│                                             │
│  Full KV Cache          SWA KV Cache        │
│  ┌────────────┐         ┌────────────┐     │
│  │ All tokens │         │ Window     │     │
│  │ (Large)    │         │ tokens only│     │
│  └────────────┘         └────────────┘     │
│                                             │
│  Full Lock Refs         SWA Lock Refs       │
│  SWA Tombstones (marks evicted SWA tokens) │
│                                             │
│  Full LRU List          SWA LRU List        │
│  (all nodes)            (non-tombstone)     │
└─────────────────────────────────────────────┘
```

### TreeNode Extensions

**Location**: `swa_radix_cache.py:49-95`

```python
class TreeNode:
    def __init__(self, id: Optional[int] = None):
        # ... standard fields ...

        # SWA-specific fields
        self.swa_tombstone = False      # SWA tokens evicted
        self.full_lock_ref = 0          # Full cache locks
        self.swa_lock_ref = 0           # SWA cache locks
        self.swa_uuid = None            # UUID for lock management

        # LRU list pointers (doubled for two lists)
        self.prev = None                # Full LRU list
        self.next = None
        self.swa_prev = None            # SWA LRU list
        self.swa_next = None
```

**Tombstone Concept**:
- Node's full KV cache exists, but SWA cache evicted
- Happens for tokens outside sliding window
- Internal nodes can be tombstones; leaves cannot

**Lock Invariant**:
```
swa_lock_ref > 0  ⟹  full_lock_ref > 0
```
SWA locks always imply full locks, but not vice versa.

### LRU Lists

**Location**: `swa_radix_cache.py:102-320`

SWARadixCache maintains **two** separate LRU lists:

1. **Full LRU List**: All nodes (including tombstones)
2. **SWA LRU List**: Only non-tombstone nodes

```python
class LRUList:
    def __init__(self, swa: bool = False):
        self.swa = swa
        if self.swa:
            self.prv = "swa_prev"
            self.nxt = "swa_next"
            self.lock_ref = "swa_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"

        # Dummy head and tail for simpler operations
        self.head = TreeNode()  # Most recently used
        self.tail = TreeNode()  # Least recently used
        setattr(self.head, self.nxt, self.tail)
        setattr(self.tail, self.prv, self.head)
        self.cache = {}  # node.id → node mapping
```

**Operations**:

```python
def insert_mru(self, node):
    """Insert as most recently used"""
    assert node.id not in self.cache
    self.cache[node.id] = node
    self._add_node(node)  # After head

def reset_node_mru(self, node):
    """Move existing node to MRU position"""
    assert node.id in self.cache
    self._remove_node(node)
    self._add_node(node)

def get_lru_no_lock(self) -> Optional[TreeNode]:
    """Get least recently used unlocked node"""
    return self.get_prev_no_lock(self.tail, check_id=False)

def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
    """Get least recently used unlocked leaf"""
    return self.get_prev_leaf_no_lock(self.tail, check_id=False)
```

### Hybrid Eviction

**Location**: `swa_radix_cache.py:582-663`

SWARadixCache can evict from two pools:

```python
def evict(self, full_num_tokens: int, swa_num_tokens: int = 0) -> None:
    if self.disable:
        return

    full_num_evicted = 0
    swa_num_evicted = 0

    # Phase 1: Evict full tokens (also evicts SWA)
    if full_num_tokens > 0:
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert x != self.root_node
            assert x.full_lock_ref == 0

            # Free both full and SWA tokens
            self.token_to_kv_pool_allocator.free(x.value)
            full_num_evicted += len(x.value)
            swa_num_evicted += len(x.value)

            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
            self.full_lru_list.remove_node(x)
            self.swa_lru_list.remove_node(x)

            self._delete_leaf(x)
            x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
            full_num_evicted += leaf_full_num_evicted

            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()
            x = x_next

    # Phase 2: Evict SWA tokens only (create tombstones)
    if swa_num_evicted < swa_num_tokens:
        x = self.swa_lru_list.get_lru_no_lock()

        while swa_num_evicted < swa_num_tokens and self.swa_lru_list.in_list(x):
            assert not x.swa_tombstone
            assert x != self.root_node
            assert x.swa_lock_ref == 0

            if len(x.children) > 0:
                # Internal node: tombstone it
                self.token_to_kv_pool_allocator.free_swa(x.value)
                swa_num_evicted += len(x.value)

                x_next = self.swa_lru_list.get_prev_no_lock(x)
                self.swa_lru_list.remove_node(x)
                self._tombstone_internal_node(x)
            else:
                # Leaf node: evict full cache too
                assert x.full_lock_ref == 0
                self.token_to_kv_pool_allocator.free(x.value)
                full_num_evicted += len(x.value)
                swa_num_evicted += len(x.value)

                x_next = self.swa_lru_list.get_prev_no_lock(x)
                self.full_lru_list.remove_node(x)
                self.swa_lru_list.remove_node(x)
                self._delete_leaf(x)
                self._iteratively_delete_tombstone_leaf(x)

            x = x_next
```

**Two-Phase Eviction**:

1. **Phase 1 (Full Eviction)**: Evict leaf nodes completely
   - Frees both full and SWA cache
   - Only targets leaf nodes
   - Cleans up tombstone parent leaves

2. **Phase 2 (SWA-Only Eviction)**: Evict SWA cache only
   - Can target internal nodes (creates tombstones)
   - Leaf nodes still evicted completely
   - Creates tombstones for internal nodes

### Lock Management with Sliding Window

**Location**: `swa_radix_cache.py:665-741`

```python
def inc_lock_ref(self, node: TreeNode) -> Optional[int]:
    """
    Lock nodes from root to node.
    - Full locks: entire path
    - SWA locks: only first `sliding_window_size` tokens
    Returns: swa_uuid_for_lock (boundary marker)
    """
    if self.disable:
        return None

    swa_lock_size = 0
    swa_uuid_for_lock = None

    while node != self.root_node:
        # Lock full cache
        if node.full_lock_ref == 0:
            self.full_evictable_size_ -= len(node.value)
            self.full_protected_size_ += len(node.value)
        node.full_lock_ref += 1

        # Lock SWA cache (only up to window size)
        if swa_lock_size < self.sliding_window_size:
            assert not node.swa_tombstone
            if node.swa_lock_ref == 0:
                self.swa_evictable_size_ -= len(node.value)
                self.swa_protected_size_ += len(node.value)
            node.swa_lock_ref += 1
            swa_lock_size += len(node.value)

            if swa_lock_size >= self.sliding_window_size:
                # Mark boundary with UUID
                if node.swa_uuid is None:
                    node.swa_uuid = gen_swa_uuid()
                swa_uuid_for_lock = node.swa_uuid

        node = node.parent

    return swa_uuid_for_lock

def dec_lock_ref(self, node: TreeNode, swa_uuid_for_lock: Optional[int] = None):
    """
    Unlock nodes.
    - Full locks: entire path to root
    - SWA locks: only up to node with swa_uuid_for_lock
    """
    if self.disable:
        return

    dec_lock_swa = True
    while node != self.root_node:
        # Unlock full cache
        if node.full_lock_ref == 1:
            self.full_evictable_size_ += len(node.value)
            self.full_protected_size_ -= len(node.value)
        node.full_lock_ref -= 1

        # Unlock SWA cache (only if within window)
        if dec_lock_swa:
            assert not node.swa_tombstone
            if node.swa_lock_ref == 1:
                self.swa_evictable_size_ += len(node.value)
                self.swa_protected_size_ -= len(node.value)
            node.swa_lock_ref -= 1

            if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:
                dec_lock_swa = False  # Stop unlocking SWA

        node = node.parent
```

**Example**:

```
Sliding window size: 5 tokens
Request tokens: [A,B,C,D,E,F,G,H] (8 tokens)

Tree:
root → [A,B,C] (3 tokens) → [D,E,F] (3 tokens) → [G,H] (2 tokens)

inc_lock_ref on [G,H]:
- Full locks: [G,H], [D,E,F], [A,B,C]
- SWA locks: [G,H] (2 tokens), [D,E,F] (3 tokens, total 5)
- swa_uuid_for_lock = [D,E,F].swa_uuid (boundary at 5 tokens)
- [A,B,C] not SWA locked (outside window)

dec_lock_ref with swa_uuid_for_lock:
- Full unlocks: [G,H], [D,E,F], [A,B,C]
- SWA unlocks: [G,H], [D,E,F] (stops at uuid)
- [A,B,C] not SWA unlocked
```

### Prefix Matching with Tombstones

**Location**: `swa_radix_cache.py:790-854`

```python
def _match_prefix_helper(self, key: RadixKey) -> Tuple[List[torch.Tensor], TreeNode]:
    """
    Match prefix considering sliding window and tombstones.

    Strategy:
    - Track match length since last tombstone
    - Only return prefix if >= sliding_window_size from tombstone
    - Ensures valid SWA cache exists for matched prefix
    """
    node = self.root_node
    child_key = self.get_child_key_fn(key)

    value = []
    match_len_since_tombstone = float("inf")  # Inf for no tombstone
    best_value_len = 0
    best_last_node = node

    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]

        # Check if enough tokens since last tombstone
        if (
            child.swa_tombstone
            and match_len_since_tombstone >= self.sliding_window_size
        ):
            # Save this as valid match point
            best_value_len = len(value)
            best_last_node = node
            match_len_since_tombstone = 0  # Reset counter

        prefix_len = self.key_match_fn(child.key, key)

        if prefix_len < len(child.key):
            # Partial match, split and update
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            if not new_node.swa_tombstone:
                match_len_since_tombstone += len(new_node.value)
            node = new_node
            break
        else:
            # Full match, continue
            value.append(child.value)
            if not child.swa_tombstone:
                match_len_since_tombstone += len(child.value)
            node = child
            key = key[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

    # Final check
    if match_len_since_tombstone >= self.sliding_window_size:
        best_value_len = len(value)
        best_last_node = node

    # Update LRU lists (child to parent order)
    self.full_lru_list.reset_node_and_parents_mru(best_last_node, self.root_node)
    self.swa_lru_list.reset_node_and_parents_mru(best_last_node, self.root_node)

    return value[:best_value_len], best_last_node
```

**Tombstone Handling**:

```
Tree with tombstones (window_size=5):
root → [A,B,C] → [D,E,F] (tombstone) → [G,H,I] → [J,K]

Query: [A,B,C,D,E,F,G,H,I,J,K]

Step 1: Match [A,B,C]
        match_len = 3, no tombstone yet

Step 2: Match [D,E,F] (tombstone)
        match_len = 6 >= 5
        Save: best_value=[A,B,C], best_node=[A,B,C]
        Reset: match_len_since_tombstone = 0

Step 3: Match [G,H,I]
        match_len_since_tombstone = 3

Step 4: Match [J,K]
        match_len_since_tombstone = 5 >= 5
        Save: best_value=[A,B,C,G,H,I,J,K], best_node=[J,K]

Return: value=[A,B,C,G,H,I,J,K] (skips tombstone [D,E,F])
        last_node=[J,K]
```

### Tombstone Cleanup

**_iteratively_delete_tombstone_leaf** (`swa_radix_cache.py:960-981`):

```python
def _iteratively_delete_tombstone_leaf(self, node: TreeNode) -> Tuple[TreeNode, int]:
    """
    Iteratively delete tombstone parent leaves.
    Maintains invariant: leaf nodes are never tombstones.
    """
    full_num_evicted = 0
    while node.parent.swa_tombstone and len(node.parent.children) == 0:
        if node.parent == self.root_node:
            break  # Root never evicted
        if node.parent.full_lock_ref > 0:
            break  # In use
        assert node.parent.swa_lock_ref == 0  # Tombstones never SWA locked

        # Evict tombstone parent leaf
        self.token_to_kv_pool_allocator.free(node.parent.value)
        full_num_evicted += len(node.parent.value)
        self.full_lru_list.remove_node(node.parent)
        self._delete_tombstone_leaf(node.parent)
        node = node.parent

    return node, full_num_evicted
```

**Purpose**: After deleting a regular leaf, check if parent became a tombstone leaf. If so, delete it too (recursive).

**Example**:
```
Before:
parent (tombstone, no SWA) → child (leaf)

Delete child:
parent (tombstone, no children) → (becomes leaf)

Cleanup:
Delete parent (tombstone leaf)
```

---

## HiRadixCache (Hierarchical Cache)

**Location**: `python/sglang/srt/mem_cache/hiradix_cache.py`

Most advanced cache implementation with three-tier memory hierarchy: GPU (device) → CPU (host) → Storage (disk/network).

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HiRadixCache                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Device Cache (GPU)                                     │
│  ┌──────────────────────────────────────┐              │
│  │ node.value: torch.Tensor (indices)   │              │
│  │ Fastest access, limited capacity     │              │
│  └──────────────────────────────────────┘              │
│                    ↕ Write-through/Write-back          │
│  Host Cache (CPU RAM)                                   │
│  ┌──────────────────────────────────────┐              │
│  │ node.host_value: torch.Tensor        │              │
│  │ Medium speed, larger capacity        │              │
│  │ token_to_kv_pool_host                │              │
│  └──────────────────────────────────────┘              │
│                    ↕ Backup/Prefetch                    │
│  Storage Backend (Disk/Network)                         │
│  ┌──────────────────────────────────────┐              │
│  │ Persistent storage (nixl, S3, etc.)  │              │
│  │ node.hash_value: page hashes         │              │
│  │ Slowest access, unlimited capacity   │              │
│  └──────────────────────────────────────┘              │
│                                                         │
│  Async Operations:                                      │
│  - ongoing_write_through (Device → Host)                │
│  - ongoing_load_back (Host → Device)                    │
│  - ongoing_backup (Host → Storage)                      │
│  - ongoing_prefetch (Storage → Host)                    │
└─────────────────────────────────────────────────────────┘
```

### Initialization

**Location**: `hiradix_cache.py:28-142`

```python
class HiRadixCache(RadixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,           # Host/device memory ratio
        hicache_size: int,               # Host memory size (tokens)
        hicache_write_policy: str,       # "write_through" or "write_back"
        hicache_io_backend: str,         # "page_first" or "direct"
        hicache_mem_layout: str,         # Memory layout for host cache
        enable_metrics: bool,
        eviction_policy: str = "lru",
        hicache_storage_backend: Optional[str] = None,  # "file", "nixl", etc.
        hicache_storage_prefetch_policy: Optional[str] = "best_effort",
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[str] = None,
        is_eagle: bool = False,
    ):
        # Get KV cache from allocator
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()

        # Create host memory pool
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size, hicache_mem_layout
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size, hicache_mem_layout
            )

        self.tp_group = tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.enable_storage = hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and enable_metrics

        # Parse storage configuration
        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(storage_backend_extra_config)

        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            page_size / 1024 * prefetch_timeout_per_ki_token
        )

        # Create cache controller (manages async I/O)
        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
            io_backend=hicache_io_backend,
            storage_backend=hicache_storage_backend,
            prefetch_threshold=self.prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=extra_config,
        )

        # Track ongoing async operations
        self.ongoing_write_through = {}  # node_id → node
        self.ongoing_load_back = {}      # node_id → node
        self.ongoing_prefetch = {}       # req_id → (node, tokens, indices, op)
        self.ongoing_backup = {}         # operation_id → node

        # Thresholds for write-through and load-back
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        # Initialize parent RadixCache
        super().__init__(
            req_to_token_pool,
            token_to_kv_pool_allocator,
            page_size,
            disable=False,
            eviction_policy=eviction_policy,
            is_eagle=is_eagle,
        )
```

### Write Policies

#### Write-Through

```python
def _inc_hit_count(self, node: TreeNode, chunked=False):
    """Increment hit count and potentially write to host."""
    if self.cache_controller.write_policy == "write_back" or chunked:
        return
    node.hit_count += 1

    if not node.backuped:
        if node.hit_count >= self.write_through_threshold:
            self.write_backup(node)
```

**Behavior**:
- Writes to host immediately when node hit count reaches threshold
- Ensures host cache is up-to-date
- Higher overhead but better resilience

**Process**:
1. Node accessed → increment hit_count
2. hit_count ≥ threshold → initiate write to host
3. Lock node during write (prevent eviction)
4. Unlock after write completes

#### Write-Back

```python
def evict(self, num_tokens: int):
    # ... eviction logic ...

    write_back_nodes = []
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)

        if not x.backuped:
            if self.cache_controller.write_policy == "write_back":
                # Write to host on eviction
                num_evicted += self.write_backup(x, write_back=True)
                write_back_nodes.append(x)
            else:
                num_evicted += self._evict_regular(x)
        else:
            num_evicted += self._evict_backuped(x)

    if self.cache_controller.write_policy == "write_back":
        self.writing_check(write_back=True)  # Block until writes complete
        for node in write_back_nodes:
            self._evict_backuped(node)
```

**Behavior**:
- Writes to host only on eviction
- Lower overhead during normal operation
- Blocking write on eviction (ensures data safety)

### Host Memory Management

**write_backup** (`hiradix_cache.py:230-251`):

```python
def write_backup(self, node: TreeNode, write_back=False):
    """Write node's device cache to host memory."""
    host_indices = self.cache_controller.write(
        device_indices=node.value,
        node_id=node.id,
    )

    if host_indices is None:
        # Host memory full, evict some
        self.evict_host(len(node.value))
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )

    if host_indices is not None:
        node.host_value = host_indices
        assert len(node.host_value) > 0
        self.ongoing_write_through[node.id] = node
        if not write_back:
            self.inc_lock_ref(node)  # Lock during async write
    else:
        return 0

    return len(host_indices)
```

**writing_check** (`hiradix_cache.py:277-316`):

```python
def writing_check(self, write_back=False):
    """Check completion of async writes to host."""
    if write_back:
        # Blocking: wait for all writes
        while len(self.ongoing_write_through) > 0:
            for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                finish_event.synchronize()
                for ack_id in ack_list:
                    del self.ongoing_write_through[ack_id]
            self.cache_controller.ack_write_queue.clear()
            assert len(self.ongoing_write_through) == 0
        return

    # Non-blocking: check what's ready
    if len(self.ongoing_write_through) == 0:
        return

    finish_count = 0
    for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
        if not finish_event.query():
            break
        finish_count += 1

    # Sync across TP workers
    queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
    if self.tp_world_size > 1:
        torch.distributed.all_reduce(
            queue_size,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group,
        )

    finish_count = int(queue_size.item())
    while finish_count > 0:
        _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
        finish_event.synchronize()
        for ack_id in ack_list:
            backuped_node = self.ongoing_write_through.pop(ack_id)
            self.dec_lock_ref(backuped_node)
            if self.enable_storage:
                self.write_backup_storage(backuped_node)  # Next tier
        finish_count -= 1
```

**evict_host** (`hiradix_cache.py:392-421`):

```python
def evict_host(self, num_tokens: int):
    """Evict tokens from host memory."""
    leaves = self._collect_leaves()
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)

        if x == self.root_node:
            break

        # Only evict host value of already-evicted device nodes
        if not x.evicted:
            continue

        # Protected from eviction (ongoing operations)
        if x.host_ref_counter > 0:
            continue

        num_evicted += self.cache_controller.evict_host(x.host_value)

        # Delete node from tree
        for k, v in x.parent.children.items():
            if v == x:
                break
        del x.parent.children[k]

        if len(x.parent.children) == 0 and x.parent.evicted:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))
```

### Host-to-Device Loading

**load_back** (`hiradix_cache.py:423-472`):

```python
def load_back(
    self, node: TreeNode, mem_quota: Optional[int] = None
) -> Optional[torch.Tensor]:
    """Load KV cache from host back to device."""
    last_hit_node = node
    nodes_to_load = []

    # Find all evicted ancestors
    while node.evicted:
        assert node.backuped
        nodes_to_load.insert(0, node)
        node = node.parent
    else:
        ancester_node = node

    # Lock ancestor to prevent eviction
    delta = self.inc_lock_ref(ancester_node)

    # Check if within memory quota
    host_indices = torch.cat([n.host_value for n in nodes_to_load])
    if len(host_indices) < self.load_back_threshold or (
        len(host_indices) > mem_quota + delta if mem_quota is not None else False
    ):
        self.dec_lock_ref(ancester_node)
        return None

    # Allocate device memory
    device_indices = self.cache_controller.load(
        host_indices=host_indices, node_id=last_hit_node.id
    )
    if device_indices is None:
        # Try evicting and retry
        self.evict(len(host_indices))
        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )

    self.dec_lock_ref(ancester_node)
    if device_indices is None:
        return None

    # Track ongoing load
    self.ongoing_load_back[last_hit_node.id] = last_hit_node

    # Assign device indices to nodes
    offset = 0
    for node in nodes_to_load:
        node.value = device_indices[offset : offset + len(node.host_value)]
        offset += len(node.host_value)

    self.evictable_size_ += len(device_indices)
    self.inc_lock_ref(last_hit_node)

    return device_indices
```

**loading_check** (`hiradix_cache.py:318-331`):

```python
def loading_check(self):
    """Check completion of async loads from host."""
    finish_count = 0
    for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
        if not finish_event.query():
            break
        finish_count += 1

        for ack_id in ack_list:
            end_node = self.ongoing_load_back.pop(ack_id)
            self.dec_lock_ref(end_node)

    # ACK until all events processed
    del self.cache_controller.ack_load_queue[:finish_count]
```

### Storage Backend Operations

#### Backup to Storage

**write_backup_storage** (`hiradix_cache.py:253-265`):

```python
def write_backup_storage(self, node: TreeNode):
    """Write node's host cache to persistent storage."""
    prefix_keys = (
        node.get_prefix_hash_values(node.parent)
        if self.hicache_storage_pass_prefix_keys
        else None
    )

    operation_id = self.cache_controller.write_storage(
        node.host_value, node.key, node.hash_value, prefix_keys
    )
    self.ongoing_backup[operation_id] = node
    node.protect_host()  # Prevent host eviction during backup
```

#### Prefetch from Storage

**prefetch_from_storage** (`hiradix_cache.py:709-747`):

```python
def prefetch_from_storage(
    self,
    req_id: str,
    last_host_node: TreeNode,
    new_input_tokens: List[int],
    last_hash: Optional[str] = None,
    prefix_keys: Optional[List[str]] = None,
):
    """Prefetch KV cache from storage to host."""
    # Align to page size
    prefetch_length = len(new_input_tokens) - (
        len(new_input_tokens) % self.page_size
    )
    new_input_tokens = new_input_tokens[:prefetch_length]

    # Check if prefetch should be done
    if (
        not self.enable_storage
        or prefetch_length < self.prefetch_threshold
        or self.cache_controller.prefetch_rate_limited()
    ):
        return

    # Protect host node and allocate host memory
    last_host_node.protect_host()
    host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
    if host_indices is None:
        self.evict_host(prefetch_length)
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
    if host_indices is None:
        last_host_node.release_host()
        return

    # Start prefetch operation
    operation = self.cache_controller.prefetch(
        req_id, host_indices, new_input_tokens, last_hash, prefix_keys
    )
    self.ongoing_prefetch[req_id] = (
        last_host_node,
        new_input_tokens,
        host_indices,
        operation,
    )
    self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)
```

**check_prefetch_progress** (`hiradix_cache.py:612-671`):

```python
def check_prefetch_progress(self, req_id: str) -> bool:
    """Check if prefetch can be terminated and insert results."""
    if req_id not in self.ongoing_prefetch:
        return True

    last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
        req_id
    ]

    if operation.host_indices is None:
        return True  # Prefetch not issued

    if not self.can_terminate_prefetch(operation):
        return False  # Still fetching

    # Terminate and get results
    completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
        operation
    )

    # Sync across TP workers
    min_completed_tokens = completed_tokens
    if self.tp_world_size > 1:
        completed_tokens_tensor = torch.tensor(
            min_completed_tokens, dtype=torch.int
        )
        torch.distributed.all_reduce(
            completed_tokens_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group,
        )
        min_completed_tokens = completed_tokens_tensor.item()

    # Insert fetched data into tree
    fetched_token_ids = token_ids[:min_completed_tokens]
    written_indices = host_indices[:min_completed_tokens]
    matched_length = self._insert_helper_host(
        last_host_node,
        RadixKey(
            token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
        ),
        written_indices,
        hash_value[: min_completed_tokens // self.page_size],
    )

    # Free overlapped and extra host memory
    self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
    self.cache_controller.append_host_mem_release(
        host_indices[min_completed_tokens:completed_tokens]
    )
    last_host_node.release_host()
    del self.ongoing_prefetch[req_id]
    self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

    return True
```

### Hierarchical Eviction

**evict Method** (`hiradix_cache.py:336-375`):

```python
def evict(self, num_tokens: int):
    """Evict tokens from device memory with hierarchical backup."""
    leaves = self._collect_leaves_device()
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    write_back_nodes = []

    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)

        if x.lock_ref > 0:
            continue

        if not x.backuped:
            if self.cache_controller.write_policy == "write_back":
                # Write-back: backup before eviction
                num_evicted += self.write_backup(x, write_back=True)
                write_back_nodes.append(x)
            else:
                # Write-through: no backup, just evict
                num_evicted += self._evict_regular(x)
        else:
            # Already backed up, just evict device
            num_evicted += self._evict_backuped(x)

        # Add parent to heap if all children evicted
        for child in x.parent.children.values():
            if child in write_back_nodes:
                continue
            if not child.evicted:
                break
        else:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))

    # For write-back, wait for writes then evict
    if self.cache_controller.write_policy == "write_back":
        self.writing_check(write_back=True)
        for node in write_back_nodes:
            assert node.backuped
            self._evict_backuped(node)
```

**_evict_backuped vs _evict_regular**:

```python
def _evict_backuped(self, node: TreeNode):
    """Evict node with host backup (keep tree structure)."""
    num_evicted = self.cache_controller.evict_device(node.value)
    assert num_evicted > 0
    self.evictable_size_ -= num_evicted
    node.value = None  # Mark as evicted
    return num_evicted

def _evict_regular(self, node: TreeNode):
    """Evict node without backup (delete from tree)."""
    self.cache_controller.mem_pool_device_allocator.free(node.value)
    num_evicted = len(node.value)
    self._delete_leaf(node)  # Remove from tree
    return num_evicted
```

### Page Hashing for Storage

**Purpose**: Content-addressable storage using hash values.

**Hash Computation** (`hiradix_cache.py:892-904`):

```python
def insert(self, key: RadixKey, value=None, chunked=False):
    # ... insertion logic ...

    if len(key):
        new_node = TreeNode()
        new_node.parent = node
        new_node.key = key
        new_node.value = value
        node.children[child_key] = new_node
        self.evictable_size_ += len(value)

        if self.enable_storage:
            last_hash = node.get_last_hash_value()
            new_node.hash_value = []
            for idx in range(0, len(key), self.page_size):
                new_node.hash_value.append(
                    self.cache_controller.get_hash_str(
                        key.token_ids[idx : idx + self.page_size],
                        prior_hash=last_hash,
                    )
                )
                last_hash = new_node.hash_value[-1]
```

**Hash Chaining**: Each page hash depends on prior hash, creating a chain from root to leaf.

```
root
 |
[A,B,C] → hash_0 = hash([A,B,C])
 |
[D,E,F] → hash_1 = hash([D,E,F] + hash_0)
 |
[G,H,I] → hash_2 = hash([G,H,I] + hash_1)
```

This enables:
- Deduplication across requests
- Integrity verification
- Prefix-based lookup in storage

---

## Advanced Features

### EAGLE Support (Speculative Decoding)

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) uses bigram keys instead of token keys.

**Key Conversion** (`radix_cache.py:177-184`):

```python
def _convert_to_bigram_key(tokens: List[int]) -> List[Tuple[int, int]]:
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
    if len(tokens) < 2:
        return []
    if isinstance(tokens[0], tuple):
        return tokens
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
```

**Implication**: KV cache length = token length - 1

**Example**:
```
Tokens: [1, 2, 3, 4] (4 tokens)
Bigram keys: [(1,2), (2,3), (3,4)] (3 keys)
KV cache: 3 entries (for bigrams)

Tree structure:
root → (1,2) → (2,3) → (3,4)
       kv_0    kv_1    kv_2
```

### Disaggregation Support

**KV Cache Events** (`radix_cache.py:690-747`):

For disaggregated inference (separate prefill and decode clusters), cache modifications are recorded as events.

```python
def _record_store_event(self, node: TreeNode):
    """Record KV cache insertion."""
    if self.enable_kv_cache_events:
        for start in range(0, len(node.key), self.page_size):
            page_tokens = node.key.token_ids[start : start + self.page_size]
            if not page_tokens:
                continue

            block_hash = hash(tuple(page_tokens))

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=page_tokens,
                    block_size=len(page_tokens),
                    lora_id=None,
                )
            )

            parent_block_hash = block_hash

def _record_remove_event(self, node: TreeNode):
    """Record KV cache eviction."""
    if self.enable_kv_cache_events:
        for start in range(0, len(node.key), self.page_size):
            page_tokens = node.key.token_ids[start : start + self.page_size]
            if not page_tokens:
                continue
            block_hash = hash(tuple(page_tokens))
            self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

def take_events(self):
    """Atomically retrieve and clear events."""
    if not self.enable_kv_cache_events:
        return []
    events = self.kv_event_queue
    self.kv_event_queue = []
    return events
```

**Usage**: Prefill cluster records events, decode cluster replays events to maintain synchronized cache state.

### Mamba Support (State-Space Models)

**Location**: `python/sglang/srt/mem_cache/mamba_radix_cache.py`

For models with Mamba (state-space) layers mixed with attention layers.

**Key Difference**: Mamba uses **convolution state** and **temporal state** instead of KV cache.

```python
class MambaPool:
    def __init__(
        self,
        size: int,
        cache_params: "Mamba2CacheParams",
        device: str,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal

        # Allocate state tensors
        conv_state = torch.zeros(
            size=(num_mamba_layers, size + 1) + conv_state_shape,
            dtype=conv_dtype,
            device=device,
        )
        temporal_state = torch.zeros(
            size=(num_mamba_layers, size + 1) + temporal_state_shape,
            dtype=ssm_dtype,
            device=device,
        )
```

Mamba layers don't benefit from radix cache prefix reuse, so they use direct state management.

### Multi-Modal Cache

**Location**: `python/sglang/srt/mem_cache/multimodal_cache.py`

Extends RadixCache for vision-language models where images can be reused across requests.

**Key Insight**: Image embeddings are large and expensive to compute. Caching them improves performance for repeated images.

**Implementation**: Uses image hashes as keys in radix tree, stores image KV cache separately.

---

## Integration and Usage

### Integration with Scheduler

**Location**: `python/sglang/srt/managers/scheduler.py`

The scheduler integrates cache mechanisms into request processing:

```python
class Scheduler:
    def __init__(self, ...):
        # ... other init ...

        # Create cache based on configuration
        if server_args.disable_radix_cache:
            self.tree_cache = ChunkCache(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.model_config.page_size,
            )
        elif enable_hierarchical_cache:
            self.tree_cache = HiRadixCache(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                # ... HiCache parameters ...
            )
        elif enable_swa:
            self.tree_cache = SWARadixCache(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                sliding_window_size,
                self.model_config.page_size,
            )
        else:
            self.tree_cache = RadixCache(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.model_config.page_size,
                eviction_policy=server_args.eviction_policy,
            )
```

### Request Lifecycle

1. **Arrival**: New request arrives with token IDs

2. **Prefix Matching**:
   ```python
   match_result = self.tree_cache.match_prefix(
       RadixKey(token_ids=req.input_ids, extra_key=req.extra_key)
   )
   req.prefix_indices = match_result.device_indices
   req.last_node = match_result.last_device_node
   ```

3. **Lock Acquired**:
   ```python
   self.tree_cache.inc_lock_ref(req.last_node)
   ```

4. **Inference**:
   - Reuse cached KV for prefix tokens
   - Compute KV only for new tokens
   - Write new KV to allocated indices

5. **Chunked Prefill** (if applicable):
   ```python
   self.tree_cache.cache_unfinished_req(req)
   # Request paused, can be resumed later
   ```

6. **Completion**:
   ```python
   self.tree_cache.cache_finished_req(req)
   # Inserts new tokens into cache, frees req slot, releases locks
   ```

7. **Eviction** (when memory pressure):
   ```python
   if self.available_memory() < threshold:
       self.tree_cache.evict(num_tokens_to_free)
   ```

### Memory Quota Management

```python
def get_available_gpu_memory(self):
    """Calculate available GPU memory for new requests."""
    available = self.token_to_kv_pool_allocator.available_size()
    available -= self.tree_cache.evictable_size()
    return available

def try_allocate_tokens(self, num_tokens):
    """Attempt to allocate tokens, evicting if needed."""
    if self.get_available_gpu_memory() < num_tokens:
        evict_size = num_tokens - self.get_available_gpu_memory()
        self.tree_cache.evict(evict_size)

    return self.token_to_kv_pool_allocator.alloc(num_tokens)
```

---

## Performance Considerations

### Cache Hit Rate

**Definition**: Proportion of tokens served from cache vs. recomputed.

**Formula**:
```
hit_rate = matched_tokens / total_tokens
```

**Factors Affecting Hit Rate**:

1. **Workload Pattern**:
   - High similarity → high hit rate (e.g., same system prompt)
   - Diverse inputs → low hit rate

2. **Cache Size**:
   - Larger cache → more entries → higher hit rate
   - Trade-off with GPU memory for active inference

3. **Eviction Policy**:
   - LRU works well for temporal locality
   - LFU preserves popular prefixes
   - FIFO for streaming workloads

4. **Page Size**:
   - Larger pages → fewer nodes, faster lookup
   - Smaller pages → more granular reuse
   - Typical: 16-64 tokens per page

### Memory Efficiency

**Memory Overhead**:

1. **TreeNode Objects**:
   - Python object overhead: ~500 bytes per node
   - 1M tokens with page_size=16 → 62.5K nodes → 31 MB

2. **LRU Lists** (SWA):
   - Double pointer overhead per node
   - Minimal compared to KV cache size

3. **Host Memory** (HiCache):
   - Configurable ratio (e.g., 2x device memory)
   - Enables larger effective cache capacity

4. **Storage** (HiCache):
   - Minimal memory overhead (hash values only)
   - Unbounded capacity

**Memory vs. Hit Rate Trade-off**:

```
┌────────────────────────────────────┐
│                                    │
│   High Hit Rate                    │
│   ↑                                │
│   │     ╱────────────              │
│   │   ╱                            │
│   │  ╱                             │
│   │ ╱                              │
│   │╱                               │
│   └─────────────────────→          │
│     Low          High               │
│       Cache Memory                 │
│                                    │
└────────────────────────────────────┘

Diminishing returns: doubling cache size
doesn't double hit rate
```

### Latency Characteristics

**Cache Lookup**: O(L) where L = matched prefix length
- Traverse radix tree from root to last matched node
- Compare token sequences at each level
- Typical: microseconds for reasonable prefix lengths

**Insertion**: O(L) where L = inserted sequence length
- Traverse existing path, split nodes if needed
- Create new nodes for unmatched suffix
- Typical: microseconds

**Eviction**: O(N log N) where N = number of leaves
- Build min-heap of all leaves
- Pop and evict until quota met
- Typical: milliseconds for large caches
- Amortized: infrequent operation

**Host Write** (HiCache): Async, ~10-50 ms
- PCIe transfer latency
- Non-blocking with write-through
- Blocking with write-back on eviction

**Host Load** (HiCache): ~10-50 ms
- PCIe transfer latency
- Triggered on miss + host hit
- Can overlap with queuing

**Storage Prefetch** (HiCache): Hundreds of ms to seconds
- Network/disk latency dominant
- Async background operation
- Best-effort termination policies

### Scalability

**Single-GPU**:
- Cache shared by all requests on GPU
- Lock contention minimal (fine-grained locking)
- Scales to millions of cached tokens

**Tensor Parallel (TP)**:
- Each TP rank maintains **identical** radix cache
- Synchronization on critical operations (eviction, prefetch)
- Memory overhead: O(tp_size)
- Benefit: Distributed cache lookup

**Data Parallel (DP)**:
- Each DP rank maintains **independent** cache
- No synchronization needed
- Benefit: Linear scalability

**Disaggregation**:
- Prefill cluster: maintains cache, emits events
- Decode cluster: replays events, maintains sync'd cache
- Communication: Event streaming

### Tuning Guidelines

**Choosing Cache Type**:

| Workload | Recommended Cache |
|----------|-------------------|
| Diverse, no reuse | ChunkCache |
| Common prefixes | RadixCache (LRU) |
| Popular prompts | RadixCache (LFU) |
| Sliding window models | SWARadixCache |
| Memory constrained | HiRadixCache (host) |
| Multi-tenant | HiRadixCache (storage) |

**Page Size Selection**:

| Page Size | Pros | Cons |
|-----------|------|------|
| 1 | Finest granularity | Slowest lookup, most nodes |
| 16 | Good balance | Standard choice |
| 64 | Fast lookup | Coarser reuse |
| 256 | Fastest | Very coarse reuse |

**HiCache Configuration**:

- **Write Policy**:
  - Write-through: Better resilience, higher overhead
  - Write-back: Lower overhead, risk of data loss on crashes

- **Host Memory Ratio**:
  - 1x: Minimal host usage
  - 2x: Recommended default
  - 4x+: For high-reuse workloads

- **Storage Backend**:
  - File: Simple, local SSD
  - S3/Object store: Shared, scalable
  - nixl: High-performance, multi-tenant

- **Prefetch Threshold**:
  - Low (e.g., 128 tokens): Aggressive prefetching
  - High (e.g., 1024 tokens): Conservative, less overhead

---

## Summary

SGLang's cache system is a comprehensive, multi-tier solution for KV cache management in LLM inference:

1. **BasePrefixCache**: Abstract interface for all cache types
2. **ChunkCache**: Simple no-prefix-matching cache
3. **RadixCache**: Full radix tree with eviction strategies
4. **SWARadixCache**: Sliding window support with tombstones
5. **HiRadixCache**: Hierarchical cache with host and storage tiers

**Key Features**:
- Prefix matching and reuse via radix tree
- Multiple eviction policies (LRU, LFU, FIFO, MRU, FILO)
- Page-aligned memory management
- Chunked prefill support
- EAGLE speculative decoding
- Sliding window attention
- Hierarchical storage (GPU → CPU → Disk)
- Disaggregation support
- Multi-modal extensions

**Design Strengths**:
- Flexibility: Multiple cache types for different scenarios
- Efficiency: Intelligent prefix reuse reduces computation
- Scalability: Supports TP/DP parallelism and disaggregation
- Extensibility: Clean abstractions for new cache types

This cache system is central to SGLang's high performance, enabling efficient serving of real-world workloads with substantial prefix overlap.

---

**Document Version**: 1.0
**Generated**: 2025-10-31
**Lines of Code Analyzed**: ~15,000+
**Files Covered**: 10+ core cache implementation files
