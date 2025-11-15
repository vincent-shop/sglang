# Base Attention Backend: Ultra-Deep Technical Analysis

## Executive Architecture Summary

The `AttentionBackend` at `base_attn_backend.py:15` is the **central abstraction** for all attention computation in SGLang. It's a strategy pattern implementation enabling hot-swappable attention kernels (FlashInfer, Triton, PyTorch native, TensorRT-LLM, etc.) without model code changes. The system handles 20+ backend implementations supporting diverse hardware (CUDA, ROCm, NPU, XPU), attention variants (MHA, GQA, MLA, sparse), and execution modes (eager, CUDA graph, torch.compile).

---

## 1. Core Interface Contract

### 1.1 Mandatory Abstract Methods

**`init_forward_metadata(forward_batch: ForwardBatch)`** - The lifeblood of every forward pass
- **Called**: Before EVERY model forward at `model_runner.py:1976, 1995, 2036`
- **Purpose**: Transform high-level `ForwardBatch` data into backend-specific metadata (indices, pointers, split counts)
- **Contract**: Must prepare all internal state for subsequent `forward()` calls
- **Side effects**: Updates backend's `forward_metadata` field, constructs KV indices, computes attention metadata
- **Critical**: This is where backends compute **KV indirection** - mapping from logical sequence positions to physical KV cache locations

**Example from FlashInferBackend** (`flashinfer_backend.py:411-496`):
```python
def init_forward_metadata(self, forward_batch: ForwardBatch):
    if forward_batch.forward_mode.is_decode_or_idle():
        # Decode: create KV indptr and indices for paged attention
        self.indices_updater_decode.update(
            forward_batch.req_pool_indices,  # Request indices in pool
            forward_batch.seq_lens,           # Current sequence lengths
            forward_batch.seq_lens_cpu,       # CPU copy for zero-copy planning
            forward_batch.seq_lens_sum,       # Total tokens to attend
            decode_wrappers=self.decode_wrappers,
            ...
        )
        self.forward_metadata = DecodeMetadata(self.decode_wrappers)
    else:
        # Prefill: handle prefix lens, ragged vs paged, multi-item scoring
        prefix_lens = forward_batch.extend_prefix_lens
        use_ragged = not self.is_multimodal and not self.multi_item_scoring_delimiter

        self.indices_updater_prefill.update(...)
        self.forward_metadata = PrefillMetadata(
            self.prefill_wrappers_paged,
            use_ragged,
            extend_no_prefix,
        )
```

**`forward_decode(q, k, v, layer, forward_batch, save_kv_cache)`** - Single-token generation
- **Input shapes**: `q`: `(bs, num_heads*head_dim)`, `k/v`: `(bs, num_kv_heads, head_dim)`
- **Returns**: `(bs, num_heads*head_dim)` attention output
- **KV cache**: Must call `forward_batch.token_to_kv_pool.set_kv_buffer()` if `save_kv_cache=True`
- **Optimization**: This is the **hot path** - optimized with CUDA graphs

**`forward_extend(q, k, v, layer, forward_batch, save_kv_cache)`** - Prefill with cached prefix
- **Input shapes**: `q/k/v`: `(num_tokens, ...)` where `num_tokens = sum(extend_seq_lens)`
- **Complexity**: Handles variable-length sequences, prefix reuse, ragged/paged attention
- **Ragged attention**: New tokens attend to themselves + nothing in cache
- **Paged attention**: New tokens attend to cached prefix + themselves

### 1.2 CUDA Graph Interface (Optional)

CUDA graphs **freeze** the entire execution flow including kernel launches, memory accesses, and control flow into a single replayable unit. Performance gains: 2-5x for small batches.

**`init_cuda_graph_state(max_bs, max_num_tokens)`**
- **Called once**: During `CudaGraphRunner.__init__` at `cuda_graph_runner.py:286`
- **Purpose**: Pre-allocate **all buffers** that will be mutated during graph capture
- **Memory**: Allocated from `global_graph_memory_pool` shared across all batch sizes

**FlashInfer example** (`flashinfer_backend.py:498-531`):
```python
def init_cuda_graph_state(self, max_bs, max_num_tokens):
    # Pre-allocate KV indices buffer (largest allocation)
    self.cuda_graph_kv_indices = [
        torch.zeros(
            (max_num_tokens * self.max_context_len,),  # Worst case: every token attends to full context
            dtype=torch.int32,
            device="cuda",
        )
        for _ in range(self.num_wrappers)  # Separate buffers for sliding window
    ]

    # Custom mask for speculative decoding
    self.cuda_graph_custom_mask = torch.zeros(
        (max_num_tokens * self.max_context_len),
        dtype=torch.uint8,
        device="cuda",
    )
```

**`init_forward_metadata_capture_cuda_graph(...)`** - Called during **capture** phase
- **Purpose**: Initialize metadata using **pre-allocated** CUDA graph buffers
- **Constraint**: NO dynamic memory allocation, NO device-to-host copies, NO control flow based on tensor values
- **Key**: Creates FlashInfer `Wrapper` objects with `use_cuda_graph=True` and pre-allocated buffers

**`init_forward_metadata_replay_cuda_graph(...)`** - Called during **replay** phase
- **Purpose**: Update metadata tensors in-place for new batch data
- **Optimization**: Can skip some CPU-GPU synchronization using `seq_lens_cpu`

**Triton backend KV splits logic** (`triton_backend.py:174-224`):
```python
def get_num_kv_splits(self, num_kv_splits: torch.Tensor, seq_lens: torch.Tensor):
    # Deterministic mode (for reproducibility): splits determined by sequence length only
    if self.enable_deterministic:
        num_kv_splits[:] = (seq_lens + self.split_tile_size - 1) // self.split_tile_size
        return

    # Dynamic mode: adjust splits based on batch size and GPU occupancy
    # This Triton kernel balances load across SMs
    get_num_kv_splits_triton[(1,)](
        num_kv_splits,
        seq_lens,
        num_seq,
        num_group,
        self.num_head,
        self.num_kv_head,
        self.max_kv_splits,
        self.device_core_count,  # Adjust parallelism to GPU
    )
```

**`get_cuda_graph_seq_len_fill_value()`** - Padding value for unused batch slots
- **FlashInfer returns 1**: Empty sequences get seq_len=1 to avoid division by zero
- **Used in**: `cuda_graph_runner.py:289, 755` for padding `seq_lens` tensor

### 1.3 Speculative Decoding Interface (Optional)

**`get_verify_buffers_to_fill_after_draft()`** - Returns `[mask_buffer, positions_buffer]`
- **Purpose**: In speculative decoding, **draft model** generates a tree of candidates. **Target model** needs to verify them with custom attention masks.
- **Workflow**:
  1. Draft runs → generates tree mask
  2. Tree mask copied to `mask_buffer`
  3. Target model replays CUDA graph with updated mask
- **Triton returns**: `[self.cuda_graph_custom_mask, None]` at line 780

**`update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)`**
- **Purpose**: Recompute backend metadata that depends on the tree structure
- **Most backends**: No-op (mask update is sufficient)

### 1.4 Optional Feature Methods

**`support_triton()`** - Returns True if backend supports `torch.compile`
- **Default**: True (most backends compatible)
- **Used**: To decide whether to call Triton-based helper kernels

**`get_indexer_metadata(layer_id, forward_batch)`** - For Native Sparse Attention (NSA)
- **Returns**: `BaseIndexerMetadata` with `get_seqlens_int32()`, `get_page_table_64()`, `topk_transform()`
- **Purpose**: NSA performs top-k token selection; indexer tracks which tokens to attend to
- **Implementation**: Only `NativeSparseAttnBackend` at `nsa_backend.py`

---

## 2. Backend Registration & Selection

### 2.1 Registration System (`attention_registry.py`)

**Decorator-based registry** at lines 15-20:
```python
ATTENTION_BACKENDS = {}  # Global dict mapping name -> factory function

def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn  # fn: (ModelRunner) -> AttentionBackend
        return fn
    return decorator
```

**Factory pattern**: Each backend provides a factory that:
1. Takes `ModelRunner` as input (contains all config/state)
2. Returns initialized `AttentionBackend` instance
3. Handles conditional logic (e.g., MLA vs non-MLA)

**Example**: FlashInfer factory (`attention_registry.py:23-45`):
```python
@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    if not runner.use_mla_backend:
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        # EAGLE speculative needs special stream for planning
        if runner.server_args.speculative_algorithm == "EAGLE":
            if not hasattr(runner, "plan_stream_for_flashinfer"):
                runner.plan_stream_for_flashinfer = torch.cuda.Stream()

        return FlashInferAttnBackend(runner, init_new_workspace=runner.init_new_workspace)
    else:
        from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
        return FlashInferMLAAttnBackend(runner)
```

### 2.2 Backend Selection Logic (`model_runner.py:1829-1873`)

**Primary selection** at `_get_attention_backend_from_str()`:
```python
backend_str = server_args.attention_backend  # e.g., "flashinfer", "triton", "fa3"
full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
return attn_backend_wrapper(self, full_attention_backend)
```

**Wrapper layer** (`attention_registry.py:178-219`) handles special architectures:
- **Hybrid models** (e.g., Mamba + Transformer layers): Wraps with `HybridLinearAttnBackend`
  - Linear layers → `GDNAttnBackend` or `Mamba2AttnBackend`
  - Transformer layers → Original backend (FlashInfer/Triton)
- **Two-batch overlap**: Wraps with `TboAttnBackend` (splits batches for parallelism)

**Backend compatibility checks**:
```python
# MLA models require special backends
if runner.use_mla_backend:
    assert server_args.attention_backend in MLA_ATTENTION_BACKENDS  # flashinfer, fa3, fa4, trtllm_mla, cutlass_mla, aiter, nsa

# Chunked prefix cache only supported by specific backends
if not server_args.disable_chunked_prefix_cache:
    assert server_args.attention_backend in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
```

### 2.3 Available Backends (20+ implementations)

| Backend | Hardware | Attention Type | Key Feature |
|---------|----------|----------------|-------------|
| **flashinfer** | CUDA | MHA/MLA | Default, fastest, paged attention |
| **triton** | CUDA/ROCm | MHA | Customizable, double sparsity |
| **fa3** / **fa4** | CUDA (SM80-90) | MHA/MLA | FlashAttention v3/v4 |
| **torch_native** | Any | MHA | PyTorch SDPA, fallback |
| **flex_attention** | CUDA | MHA | PyTorch 2.5+ flex attention |
| **nsa** | CUDA | MLA | Native Sparse Attention (top-k) |
| **aiter** | ROCm | MLA | AMD-optimized |
| **wave** | CUDA | MHA | Wavelinker backend |
| **trtllm_mha/mla** | CUDA | MHA/MLA | TensorRT-LLM kernels |
| **ascend** | NPU | MHA/MLA | Huawei Ascend NPU |
| **intel_amx** | CPU | MHA | Intel AMX instructions |
| **intel_xpu** | XPU | MHA | Intel Data Center GPU |
| **cutlass_mla** | CUDA | MLA | CUTLASS-based MLA |
| **flashmla** | CUDA | MLA | Custom MLA kernel |
| **double_sparsity** | CUDA | MHA | H2O + channel sparsity |

---

## 3. Complete Data Flow & State Management

### 3.1 Batch Lifecycle

**Scheduler → Worker → ModelRunner → Backend** (`forward_batch_info.py:17-28`):
```
ScheduleBatch (CPU, scheduler)
  ↓ [schedule decision, memory allocation]
ModelWorkerBatch (CPU→GPU transfer)
  ↓ [ForwardBatch.init_new() at line 325]
ForwardBatch (GPU, contains attn_backend reference at line 281)
  ↓ [attn_backend.init_forward_metadata()]
Backend metadata (KV indices, indptr, attention masks)
  ↓ [model.forward() → RadixAttention.forward()]
Backend forward (decode/extend)
```

### 3.2 ForwardBatch: The Data Contract

**Critical fields** (`forward_batch_info.py:172-323`):
```python
@dataclass
class ForwardBatch:
    # Core batch data
    forward_mode: ForwardMode  # EXTEND, DECODE, MIXED, TARGET_VERIFY, DRAFT_EXTEND, etc.
    batch_size: int
    input_ids: torch.Tensor  # (num_tokens,)
    req_pool_indices: torch.Tensor  # (batch_size,) - indices into req_to_token_pool
    seq_lens: torch.Tensor  # (batch_size,)
    out_cache_loc: torch.Tensor  # (num_tokens,) - where to write new KV
    seq_lens_sum: int

    # Prefill-specific
    extend_seq_lens: Optional[torch.Tensor]  # (batch_size,) - tokens being added
    extend_prefix_lens: Optional[torch.Tensor]  # (batch_size,) - cached prefix length
    extend_start_loc: Optional[torch.Tensor]  # (batch_size,) - cumsum of extend_seq_lens

    # Backend references (THE KEY)
    req_to_token_pool: ReqToTokenPool  # Maps request → token locations
    token_to_kv_pool: KVCache  # Physical KV cache
    attn_backend: AttentionBackend  # THE BACKEND

    # Speculative decoding
    spec_info: Optional[SpecInput]  # Tree mask, positions, etc.

    # Data parallel attention
    global_num_tokens_gpu: Optional[torch.Tensor]  # For all-gather coordination
    dp_padding_mode: Optional[DpPaddingMode]
```

**Initialization** at `ForwardBatch.init_new()` (line 325):
- Transfers data from CPU (`ModelWorkerBatch`) to GPU
- Computes `positions` tensor (line 419-443)
- Handles Mamba/hybrid models
- **Critical**: Sets `forward_batch.attn_backend = model_runner.attn_backend`

### 3.3 KV Cache Integration: The Memory Contract

**Three-level indirection**:
1. **ReqToTokenPool** (`memory_pool.py:66-114`): `req_to_token[req_idx, pos] = kv_cache_idx`
2. **TokenToKVPoolAllocator**: Manages free/allocated indices
3. **KVCache** (`memory_pool.py:394-496`): Physical buffers `k_buffer[layer_id][kv_cache_idx]`

**Backend access pattern** (from `flashinfer_backend.py:820-828`):
```python
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
    # 1. Write new KV to cache
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            forward_batch.out_cache_loc,  # Where to write
            k, v,
            layer.k_scale, layer.v_scale  # For FP8 quantization
        )

    # 2. Read KV from cache (includes new KV + all history)
    o = decode_wrapper.forward(
        q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
        forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),  # Physical buffer
        sm_scale=layer.scaling,
    )
    return o
```

**KV index construction** (Triton kernel at `forward_batch_info.py:1098-1132`):
```python
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # (max_batch, max_context_len)
    req_pool_indices_ptr,  # (batch_size,)
    seq_lens_ptr,  # (batch_size,)
    kv_indptr_ptr,  # (batch_size+1,) output
    kv_start_idx_ptr,  # Optional offset for sliding window
    kv_indices_ptr,  # (sum(seq_lens),) output
    req_to_token_stride: tl.constexpr,
):
    pid = tl.program_id(0)  # One thread per sequence
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    seq_len = tl.load(seq_lens_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr_ptr + pid)

    # Gather token positions for this request
    for i in range(seq_len):
        kv_idx = tl.load(req_to_token_ptr + req_pool_index * req_to_token_stride + i)
        tl.store(kv_indices_ptr + kv_indices_offset + i, kv_idx)
```

---

## 4. Concrete Backend Implementation: FlashInfer Deep Dive

### 4.1 Architecture

FlashInfer provides **two execution paths**:
1. **Ragged attention**: For pure prefill (no cached prefix). Single kernel, `O(N^2)` complexity.
2. **Paged attention**: For decode + prefill with prefix. Requires indirection through page table.

**Wrapper pattern** (`flashinfer_backend.py:111-288`):
```python
class FlashInferAttnBackend:
    def __init__(self, model_runner):
        # Decode wrappers (one per layer type: full/sliding window)
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,  # Shared temporary buffer
                "NHD",  # Layout: (num_tokens, num_heads, head_dim)
                use_tensor_cores=self.decode_use_tensor_cores,
            )
            for _ in range(self.num_wrappers)
        ]

        # Prefill wrappers
        self.prefill_wrappers_paged = [
            BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
            for _ in range(self.num_wrappers)
        ]
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

        # Indices updaters (complex state machines)
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)
        self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(model_runner, self)
```

### 4.2 Metadata Initialization: Decode Path

**`FlashInferIndicesUpdaterDecode.call_begin_forward()`** (lines 999-1101):
```python
def call_begin_forward(self, wrapper, req_pool_indices, paged_kernel_lens, ...):
    # 1. Compute indptr (cumulative sum of sequence lengths)
    kv_indptr[1:bs+1] = torch.cumsum(paged_kernel_lens, dim=0)

    # 2. Build KV indices using Triton kernel (parallel gather)
    if wrapper.is_cuda_graph_enabled:
        kv_indices = wrapper._paged_kv_indices_buf  # Pre-allocated
    else:
        kv_indices = torch.empty(paged_kernel_lens_sum, dtype=torch.int32)

    create_flashinfer_kv_indices_triton[(bs,)](
        self.req_to_token,  # 2D array: [req_idx, pos] -> kv_idx
        req_pool_indices,
        paged_kernel_lens,
        kv_indptr,
        kv_start_idx,  # For sliding window: skip first N tokens
        kv_indices,
        self.req_to_token.shape[1],
    )

    # 3. Plan attention kernel launch
    #    FlashInfer analyzes sequence lengths to determine:
    #    - Grid dimensions
    #    - Shared memory allocation
    #    - Whether to split KV across multiple threadblocks
    wrapper.begin_forward(
        kv_indptr,  # (bs+1,)
        kv_indices,  # (sum(seq_lens),)
        self.kv_last_page_len[:bs],  # Tokens in last page of each sequence
        self.num_qo_heads,
        self.num_kv_heads,
        self.head_dim,
    )
```

### 4.3 Metadata Initialization: Prefill Path

**Decision tree** (`flashinfer_backend.py:455-496`):
```python
def init_forward_metadata(self, forward_batch):
    if forward_batch.forward_mode.is_extend():
        prefix_lens = forward_batch.extend_prefix_lens

        # Multimodal DISABLES ragged wrapper (needs paged for proper masking)
        if self.is_multimodal or self.multi_item_scoring_delimiter:
            use_ragged = False
            extend_no_prefix = False
        else:
            use_ragged = True
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

        self.forward_metadata = PrefillMetadata(
            self.prefill_wrappers_paged,
            use_ragged,
            extend_no_prefix,
        )
```

**Ragged vs Paged tradeoff**:
- **Ragged** (faster): Single kernel, contiguous memory access
- **Paged** (flexible): Supports prefix caching, sliding window, multi-item scoring

### 4.4 Forward Extend: Two-Stage Attention

**With prefix** (`flashinfer_backend.py:768-792`):
```python
# Stage 1: Ragged attention for new tokens (O(extend_len^2))
o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k.view(-1, layer.tp_k_head_num, layer.head_dim),
    v.view(-1, layer.tp_v_head_num, layer.head_dim),
    causal=True,  # Causal mask for new tokens
    sm_scale=layer.scaling,
)

# Stage 2: Paged attention for cached prefix (O(extend_len * prefix_len))
o2, s2 = prefill_wrapper_paged.forward_return_lse(
    q,
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
    causal=False,  # No causal mask needed (prefix is before new tokens)
    sm_scale=layer.scaling,
)

# Stage 3: Merge attention states using log-sum-exp trick
o, _ = merge_state(o1, s1, o2, s2)
```

**`merge_state` mathematics**:
```
softmax(concat([A_prefix, A_new]))
  = exp(A_prefix) / (exp(A_prefix).sum() + exp(A_new).sum())

Using LSE: log(sum(exp(A))) = LSE(A)
  o_merged = (o1 * exp(s1) + o2 * exp(s2)) / (exp(s1) + exp(s2))
```

### 4.5 Multi-Item Scoring: Advanced Feature

**Use case**: Multiple-choice questions where each choice must only attend to prompt + itself, NOT other choices.

**Tensor format** (`flashinfer_backend.py:289-409`):
```python
# Example: "What is capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
# token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0]
#   - Position 0 at each delimiter
#   - Increments within each item

multi_item_params = MultiItemScoringParams(
    prefix_len_ptr=torch.tensor([7]),  # Prompt length
    token_pos_in_items_ptr=torch.tensor([0, 1, 0, 1, 0, 1, 0]),
    token_pos_in_items_len=7,
    max_item_len_ptr=torch.tensor([1]),  # Longest item
)

# FlashInfer uses these to construct custom attention mask
# Each item token only attends to: prompt + its own item
```

---

## 5. CUDA Graph Integration: The Performance Multiplier

### 5.1 Capture Phase

**Goal**: Record ALL CUDA operations for a specific batch size into a replayable graph.

**Workflow** (`cuda_graph_runner.py:452-506`):
```python
def capture(self):
    for bs in reversed(self.capture_bs):  # Reverse for memory reuse
        with patch_model(
            self.model_runner.model,
            bs in self.compile_bs,  # torch.compile for small batches
            num_tokens=bs * self.num_tokens_per_bs,
            tp_group=self.model_runner.tp_group,
        ) as forward:
            graph, output_buffers = self.capture_one_batch_size(bs, forward)
            self.graphs[bs] = graph
            self.output_buffers[bs] = output_buffers
```

**Per-batch-size capture** (`cuda_graph_runner.py:540-698`):
```python
def capture_one_batch_size(self, bs, forward):
    # 1. Create inputs (sliced from pre-allocated max-size tensors)
    input_ids = self.input_ids[:num_tokens]
    req_pool_indices = self.req_pool_indices[:bs]

    # 2. Build ForwardBatch
    forward_batch = ForwardBatch(
        forward_mode=self.capture_forward_mode,  # DECODE or TARGET_VERIFY
        batch_size=bs,
        input_ids=input_ids,
        req_pool_indices=req_pool_indices,
        ...
        attn_backend=self.model_runner.attn_backend,
    )

    # 3. Initialize attention backend with CUDA graph buffers
    self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_batch.forward_mode,
        forward_batch.spec_info,
    )

    # 4. Warmup (run twice to initialize autotuning caches)
    for _ in range(2):
        self.device_module.synchronize()
        self.model_runner.tp_group.barrier()
        run_once()

    # 5. CAPTURE!
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=graph, pool=memory_pool, stream=stream):
        output = run_once()

    return graph, output
```

**Critical constraints**:
- **No CPU synchronization**: All control flow must be deterministic
- **No dynamic allocation**: All tensors pre-allocated
- **No device-to-host copies**: Can't read tensor values to make decisions
- **No tensor size queries**: `len(tensor)` forbidden inside graph

### 5.2 Replay Phase

**Workflow** (`cuda_graph_runner.py:732-852`):
```python
def replay(self, forward_batch, skip_attn_backend_init=False):
    # 1. Prepare: Copy new batch data into CUDA graph input buffers
    if not skip_attn_backend_init:
        self.replay_prepare(forward_batch)

    # 2. Replay: Execute pre-recorded graph (microseconds)
    self.graphs[self.bs].replay()

    # 3. Extract outputs (sliced from pre-allocated buffers)
    return LogitsProcessorOutput(
        next_token_logits=output.next_token_logits[:self.raw_num_token],
        hidden_states=output.hidden_states[:self.raw_num_token] if output.hidden_states else None,
    )
```

**`replay_prepare` details** (lines 732-820):
```python
def replay_prepare(self, forward_batch):
    # 1. Select captured batch size (with padding)
    index = bisect.bisect_left(self.capture_bs, raw_bs)
    bs = self.capture_bs[index]  # Next largest captured size

    # 2. Copy input data
    self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
    self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
    self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)

    # 3. Pad unused slots
    if bs != raw_bs:
        self.seq_lens[raw_bs:bs].fill_(self.seq_len_fill_value)

    # 4. Update attention metadata (CRITICAL)
    self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
        bs,
        self.req_pool_indices[:bs],
        self.seq_lens[:bs],
        forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
        self.encoder_lens[:bs] if self.is_encoder_decoder else None,
        self.capture_forward_mode,
        forward_batch.spec_info,
        seq_lens_cpu=self.seq_lens_cpu[:bs],
    )
```

### 5.3 FlashInfer CUDA Graph Optimization: Fast Decode Plan

**Problem**: FlashInfer's `begin_forward()` includes expensive device-to-host copy of `kv_indptr` to plan kernel launch.

**Solution**: Pre-copy `kv_indptr` to CPU, use `fast_decode_plan` bypass (`flashinfer_backend.py:1047-1080`):
```python
# Global state to avoid repeated CPU allocations
global global_override_indptr_cpu

# At begin_forward time:
if seq_lens_cpu is not None and global_override_indptr_cpu is None:
    global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
    global_override_indptr_cpu[0] = 0
    global_override_indptr_cpu[1:bs+1] = torch.cumsum(seq_lens_cpu, dim=0)

# Replace wrapper.begin_forward with optimized version
if wrapper_uses_fast_decode_plan:
    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        ...,
        global_override_indptr_cpu=global_override_indptr_cpu,  # Skip D2H copy!
    )
```

---

## 6. Speculative Decoding Integration

### 6.1 EAGLE Architecture

**EAGLE** (Extrapolation Algorithm for Greater Language-model Efficiency) generates a **tree** of draft tokens, verified in parallel by the target model.

**Tree structure** (`cuda_graph_runner.py:854-894`):
```python
spec_info = EagleVerifyInput(
    draft_token=None,  # Filled at runtime
    custom_mask=self.custom_mask,  # Tree attention mask
    positions=None,  # Position IDs for each tree node
    retrive_index=None,  # Which draft tokens to accept
    draft_token_num=self.num_tokens_per_bs,  # E.g., 63 for EAGLE
)
```

**Attention mask construction**:
- Each draft token can attend to:
  1. All prefix tokens
  2. Its ancestors in the tree
  3. NOT its siblings or descendants

**CUDA graph impact**:
- `capture_forward_mode = ForwardMode.TARGET_VERIFY`
- `num_tokens_per_bs = speculative_num_draft_tokens` (e.g., 63)
- Captured graphs verify **fixed-size** trees

### 6.2 Buffer Update Workflow

**Draft phase** (Draft model generates tree):
```python
# 1. Draft model runs, produces tree structure
draft_tokens, tree_mask = draft_model.generate_tree(...)

# 2. Get target model's verify buffers
mask_buffer, pos_buffer = target_backend.get_verify_buffers_to_fill_after_draft()

# 3. Copy tree data into buffers (outside CUDA graph)
mask_buffer[:tree_mask.numel()].copy_(tree_mask)

# 4. Replay target model CUDA graph (uses updated mask)
logits = cuda_graph_runner.replay(forward_batch)

# 5. Determine which draft tokens were correct
accept_length = verify_draft(logits, draft_tokens)
```

**Triton backend example** (`triton_backend.py:774-785`):
```python
def get_verify_buffers_to_fill_after_draft(self):
    return [self.cuda_graph_custom_mask, None]

def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
    # Triton: mask update is sufficient, no metadata recomputation needed
    pass
```

---

## 7. Advanced Features & Edge Cases

### 7.1 Sliding Window Attention

**Use case**: Models with limited attention window (e.g., Mistral with 4096 window).

**Implementation** (FlashInfer at `flashinfer_backend.py:147-155`):
```python
if model_runner.sliding_window_size is not None:
    self.num_wrappers = 2  # Separate wrappers for full vs windowed attention
    self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
```

**Layer dispatch** (`flashinfer_backend.py:832-841`):
```python
def _get_wrapper_idx(self, layer: RadixAttention):
    if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
        return layer.sliding_window_size == -1  # 0 for windowed, 1 for full
    # Layer config: sliding_window_size=-1 means full attention
```

**KV indices construction** (Triton at `triton_backend.py:1240-1277`):
```python
def update_sliding_window_buffer(...):
    # Clamp sequence lengths to window size
    window_kv_lens = torch.minimum(seq_lens, torch.tensor(sliding_window_size))

    # Start index: last window_size tokens
    window_kv_start_idx = seq_lens - window_kv_lens

    # Build indices for windowed attention
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,  # Skip early tokens
        window_kv_indices,
        req_to_token.stride(0),
    )
```

### 7.2 Encoder-Decoder Models

**Cross-attention requirements**:
- Decoder queries attend to encoder keys/values
- Encoder KV written to separate cache location

**FlashInfer support** (`flashinfer_backend.py:150-155`):
```python
if model_runner.model_config.is_encoder_decoder:
    self.num_wrappers = 2
    self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
```

**Forward routing** (`base_attn_backend.py:88-109`):
```python
def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True):
    if forward_batch.forward_mode.is_idle():
        return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
    elif forward_batch.forward_mode.is_decode():
        return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache)
    else:
        return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
```

### 7.3 Data Parallel Attention (DP Attention)

**Problem**: Large batches don't fit on single GPU. Solution: Shard batch across multiple GPUs, gather attention outputs.

**Coordination** (`forward_batch_info.py:676-714`):
```python
def prepare_mlp_sync_batch(self, model_runner):
    # Pad sequences to same length for all-gather
    max_num_tokens = max(global_num_tokens)
    if dp_padding_mode.is_max_len():
        global_num_tokens = [max_num_tokens] * sync_group_size
        buffer_len = max_num_tokens * sync_group_size
    else:
        buffer_len = sum(global_num_tokens)

    # Broadcast buffer length to all ranks
    set_dp_buffer_len(buffer_len, num_tokens, global_num_tokens)

    # Pad tensors to DP size
    self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
    self.seq_lens = self._pad_tensor_to_size(self.seq_lens, bs, value=seq_len_fill_value)
```

### 7.4 MLA (Multi-Head Latent Attention)

**Architecture** (DeepSeek-V2/V3): Compress KV into low-rank representation.
- Standard MHA: `kv_dim = num_kv_heads * head_dim` (e.g., 8 * 128 = 1024)
- MLA: `kv_dim = kv_lora_rank + qk_rope_head_dim` (e.g., 512 + 64 = 576)
- **Memory savings**: 40-50% KV cache reduction

**KV cache format** (`memory_pool.py:1275-1424`):
```python
class MLATokenToKVPool(KVCache):
    def __init__(self, kv_lora_rank, qk_rope_head_dim, ...):
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # 576
        self.kv_buffer = [
            torch.zeros((size, 1, self.kv_cache_dim), dtype=dtype)
            for _ in range(layer_num)
        ]

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        # Concatenate nope and rope components into single buffer
        set_mla_kv_buffer_triton(
            self.kv_buffer[layer.layer_id],
            loc,
            cache_k_nope,  # (bs, 1, 512)
            cache_k_rope,  # (bs, 1, 64)
        )
```

**Triton kernel** (lines 1150-1213):
```python
@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    ...,
):
    pid_loc = tl.program_id(0)
    loc = tl.load(loc_ptr + pid_loc)

    # Interleave nope and rope into contiguous buffer
    base_ptr = kv_buffer_ptr + loc * buffer_stride
    nope_data = tl.load(cache_k_nope_ptr + pid_loc * nope_stride + offs)
    tl.store(base_ptr + offs, nope_data)

    rope_data = tl.load(cache_k_rope_ptr + pid_loc * rope_stride + offs_rope)
    tl.store(base_ptr + nope_dim + offs_rope, rope_data)
```

---

## 8. Performance Optimizations

### 8.1 Tensor Core Utilization

**Decision logic** (`flashinfer_backend.py:1549-1595`):
```python
def should_use_tensor_core(kv_cache_dtype, num_attention_heads, num_kv_heads):
    # Environment override
    if env_override := os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE"):
        return env_override.lower() == "true"

    # FlashInfer heuristic: Tensor cores efficient for GQA group size >= 4
    gqa_group_size = num_attention_heads // num_kv_heads

    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True  # Always use for FP8
    elif kv_cache_dtype in (torch.float16, torch.bfloat16):
        return gqa_group_size >= 4  # Requires sufficient work per KV head
    else:
        return False
```

### 8.2 Workspace Buffer Sharing

**Global buffer** (`flashinfer_backend.py:103-202`):
```python
global_workspace_buffer = None  # Shared across all backend instances

class FlashInferAttnBackend:
    def __init__(self, model_runner):
        global global_workspace_buffer
        if global_workspace_buffer is None:
            workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()  # Default 256MB
            global_workspace_buffer = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")

        self.workspace_buffer = global_workspace_buffer
```

**Purpose**: FlashInfer uses workspace for:
- Intermediate attention outputs during KV splitting
- Page table transformations
- Temporary buffers for merge operations

### 8.3 Deterministic Inference Mode

**Requirement**: Same input → same output (bit-exact reproducibility).

**Challenge**: Dynamic KV splitting is non-deterministic (depends on GPU load, batch composition).

**Solution** (Triton at `triton_backend.py:116-127`):
```python
if self.enable_deterministic:
    # Use fixed tile size (deterministic split points)
    self.split_tile_size = get_int_env_var("SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256)
    self.static_kv_splits = False  # Disable dynamic optimization

# In get_num_kv_splits():
if self.enable_deterministic:
    # Deterministic: splits determined by sequence length only
    num_kv_splits[:] = (expanded_seq_lens + self.split_tile_size - 1) // self.split_tile_size
```

**FlashInfer** (`flashinfer_backend.py:168-183`):
```python
if self.enable_deterministic:
    self.decode_use_tensor_cores = True  # Disable optimized decode path
    self.prefill_split_tile_size = 4096
    self.decode_split_tile_size = 2048
    self.disable_cuda_graph_kv_split = True  # Force single-stage attention
```

### 8.4 KV Cache Copy Optimization

**Use case**: Prefix sharing via copy (faster than recomputation).

**Tiled implementation** (`memory_pool.py:1886-1919`):
```python
@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,  # Base pointers for all layers
    strides,    # Bytes per cache entry
    tgt_loc_ptr, src_loc_ptr,
    num_locs,
    BYTES_PER_TILE: tl.constexpr = 512,
):
    bid = tl.program_id(0)  # Layer ID
    tid = tl.program_id(1)  # Tile ID within entry

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)

    # Load source/target indices for all locs
    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc)

    # Compute byte offsets
    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    # Tile copy (safe for in-place: read entire tile before writing)
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)
```

---

## 9. Debugging & Introspection

### 9.1 Key Debug Points

**Backend selection**:
```python
# model_runner.py:1870
logger.info(f"Using attention backend: {backend_str}")
```

**Metadata initialization**:
```python
# Add to base_attn_backend.py:19
def init_forward_metadata(self, forward_batch: ForwardBatch):
    print(f"Backend: {self.__class__.__name__}")
    print(f"Mode: {forward_batch.forward_mode}")
    print(f"Batch size: {forward_batch.batch_size}")
    print(f"Seq lens: {forward_batch.seq_lens_cpu}")
```

**CUDA graph replay**:
```python
# cuda_graph_runner.py:838
print(f"Replaying CUDA graph: bs={self.bs}, raw_bs={self.raw_bs}")
```

### 9.2 Common Failure Modes

**1. CUDA graph capture failure**:
- **Symptom**: `RuntimeError: CUDA error: invalid configuration argument`
- **Cause**: Not enough memory for graph capture
- **Solution**: Reduce `--cuda-graph-max-bs` or `--mem-fraction-static`

**2. Incorrect attention output**:
- **Symptom**: Garbled text, NaN losses
- **Cause**: KV indices mismatch, wrong metadata
- **Debug**: Print `kv_indptr`, `kv_indices` in `init_forward_metadata`

**3. Speculative decoding failures**:
- **Symptom**: Lower acceptance rate than expected
- **Cause**: Tree mask not updated correctly
- **Debug**: Verify `spec_info.custom_mask` before replay

---

## 10. Summary & Key Takeaways

### Core Principles

1. **Abstraction through interface**: `AttentionBackend` defines contract, implementations vary wildly
2. **Metadata-driven execution**: Backends pre-compute indices/pointers in `init_forward_metadata()`
3. **Three-level indirection**: Request → Token → KV Cache (enables prefix caching, paged attention)
4. **CUDA graph optimization**: 2-5x speedup by freezing execution flow
5. **Speculative decoding synergy**: Tree mask update without graph recapture

### Critical Paths

**Decode (hot path)**:
```
forward_batch.attn_backend.init_forward_metadata()
  → indices_updater_decode.update()
    → create_flashinfer_kv_indices_triton (build KV indices)
    → wrapper.begin_forward (plan attention kernel)
  → decode_wrapper.forward (execute FlashInfer kernel)
```

**Prefill with prefix**:
```
init_forward_metadata()
  → decide ragged vs paged
  → prefill_wrapper_ragged.forward_return_lse (new tokens)
  → prefill_wrapper_paged.forward_return_lse (cached prefix)
  → merge_state (combine attention outputs)
```

### Extension Points

**To add a new backend**:
1. Subclass `AttentionBackend`
2. Implement `init_forward_metadata`, `forward_decode`, `forward_extend`
3. Register with `@register_attention_backend("name")`
4. Optionally: CUDA graph support (`init_cuda_graph_state`, etc.)

**To add a new attention variant**:
1. Extend `ForwardBatch` with necessary metadata
2. Modify backend's `init_forward_metadata` to handle new mode
3. Implement kernel in `forward_decode`/`forward_extend`

This architecture has proven robust across 20+ backend implementations, 5+ attention types, and 4+ hardware platforms. Its power lies in the clean separation between **scheduling** (ModelRunner), **indirection** (ForwardBatch), and **computation** (AttentionBackend).
