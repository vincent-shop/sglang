# CUDA Graph Infrastructure: Deep Dive

SGLang Team
January 2025

---

## Purpose

This document provides complementary technical details to PIECEWISE_CUDA_GRAPH_ONBOARDING.md, focusing on shared infrastructure, data structures, and integration patterns used by both standard and piecewise CUDA graph runners. Read PIECEWISE_CUDA_GRAPH_ONBOARDING.md first for system overview.

---

## 1. Shared Infrastructure Components

### 1.1 Global Memory Pool Management

Both runners share a single global memory pool for symmetric memory allocation across CUDA graphs:

```python
# cuda_graph_runner.py:212-223 and piecewise_cuda_graph_runner.py:115-126
global_graph_memory_pool = None

def get_global_graph_memory_pool():
    return global_graph_memory_pool

def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val
```

**Initialization Pattern** (first runner to initialize sets the pool):

```python
# Standard runner (cuda_graph_runner.py:690-693)
if get_global_graph_memory_pool() is None:
    set_global_graph_memory_pool(self.device_module.graph_pool_handle())
set_graph_pool_id(get_global_graph_memory_pool())

# Piecewise runner (piecewise_cuda_graph_runner.py:183-186)
if get_global_graph_memory_pool() is None:
    set_global_graph_memory_pool(self.device_module.graph_pool_handle())
set_graph_pool_id(get_global_graph_memory_pool())
```

**Key Property**: Once set, the memory pool is reused across all subsequent graph captures, enabling better memory locality and reducing fragmentation.

### 1.2 Context Managers

#### `model_capture_mode()`
Sets global flag indicating graph capture is active:

```python
# cuda_graph_runner.py:93-100
is_capture_mode = False

@contextmanager
def model_capture_mode():
    global is_capture_mode
    is_capture_mode = True
    yield
    is_capture_mode = False
```

**Usage**: Wraps capture calls in both runners (cuda_graph_runner.py:380, piecewise_cuda_graph_runner.py:204)

**Purpose**: Allows other subsystems to detect capture mode and adjust behavior (e.g., skip validation checks, use dummy values)

#### `freeze_gc(enable_cudagraph_gc: bool)`
Optimizes garbage collection during capture:

```python
# cuda_graph_runner.py:104-119
@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    gc.collect()                    # Clean before capture
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()                 # Freeze objects from GC
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
```

**Rationale**:
- Prevents GC from running during capture, which could cause non-deterministic memory addresses
- Reduces overhead by avoiding unnecessary collections during intensive capture phase
- Controlled unfreeze after capture completes

**Usage Locations**:
- Standard runner: cuda_graph_runner.py:464
- Piecewise runner: piecewise_cuda_graph_runner.py:283

#### `patch_model(model, compiler, num_tokens, tp_group)` (Standard) vs `patch_model(model, compiler)` (Piecewise)

**Standard Runner Version** (cuda_graph_runner.py:134-164):
```python
@contextmanager
def patch_model(model, enable_compile, num_tokens, tp_group):
    backup_ca_comm = None
    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Backup custom allreduce communicator
            yield torch.compile(torch.no_grad()(model.forward), mode="...", ...)
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm
```

**Piecewise Runner Version** (piecewise_cuda_graph_runner.py:106-112):
```python
@contextmanager
def patch_model(model, compiler):
    try:
        if compiler != "eager":
            _to_torch(model, reverse=False, num_tokens=16)
        yield model
    finally:
        _to_torch(model, reverse=True, num_tokens=16)
```

**Key Differences**:
1. Standard wraps with `torch.compile()` in-place; piecewise uses separate `install_torch_compiled()` call
2. Standard backs up TP communicator; piecewise doesn't need to
3. Piecewise uses fixed `num_tokens=16` for patching (actual size handled during capture)

#### `_to_torch(model, reverse, num_tokens)`
Recursively converts CustomOp layers for compilation compatibility:

```python
# cuda_graph_runner.py:122-131 and piecewise_cuda_graph_runner.py:94-102
def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)
```

**Purpose**: CustomOp layers (e.g., custom kernels) need to expose torch-compatible interfaces during compilation, then restore original implementations for execution.

---

## 2. ForwardBatch Data Structure

### 2.1 Core Architecture

ForwardBatch is the primary data container for a forward pass, defined in forward_batch_info.py:172-324. It flows through the system as:

```
ScheduleBatch → ModelWorkerBatch → ForwardBatch
   (CPU)            (CPU→GPU)         (GPU tensors)
```

### 2.2 Essential Fields by Category

#### Batch Metadata
```python
forward_mode: ForwardMode          # EXTEND, DECODE, MIXED, IDLE, TARGET_VERIFY, etc.
batch_size: int                    # Number of sequences in batch
seq_lens: torch.Tensor             # Sequence lengths (batch_size,)
seq_lens_sum: int                  # Sum of all sequence lengths
seq_lens_cpu: Optional[torch.Tensor]  # CPU copy for async operations
```

#### Input Tensors
```python
input_ids: torch.Tensor            # Token IDs (num_tokens,)
positions: torch.Tensor            # Position indices (num_tokens,)
req_pool_indices: torch.Tensor     # Request pool indices (batch_size,)
out_cache_loc: torch.Tensor        # KV cache output locations (num_tokens,)
```

#### Memory Pool References
```python
req_to_token_pool: ReqToTokenPool  # Maps requests to token indices
token_to_kv_pool: KVCache          # Maps tokens to KV cache locations
attn_backend: AttentionBackend     # Attention implementation (FlashInfer, etc.)
```

#### Extend/Prefill Specific
```python
extend_num_tokens: Optional[int]           # Total tokens in extend batch
extend_seq_lens: Optional[torch.Tensor]    # Per-sequence extend lengths
extend_prefix_lens: Optional[torch.Tensor] # Cached prefix lengths
extend_start_loc: Optional[torch.Tensor]   # Start index for each sequence
```

#### CUDA Graph Specific
```python
capture_hidden_mode: CaptureHiddenMode     # NULL, LAST, or FULL
num_token_non_padded: Optional[torch.Tensor]  # Actual non-padded token count
padded_static_len: int                     # Padded length (-1 if not padded)
```

#### Data Parallel (DP) Attention
```python
global_num_tokens_cpu: Optional[List[int]]         # Token counts per DP rank
global_num_tokens_gpu: Optional[torch.Tensor]      # GPU copy
dp_padding_mode: Optional[DpPaddingMode]           # Padding strategy
global_dp_buffer_len: Optional[int]                # Total DP buffer size
is_extend_in_batch: bool                           # Has extend operations
can_run_dp_cuda_graph: bool                        # DP graph compatible
```

### 2.3 ForwardBatch Lifecycle

**Creation** (forward_batch_info.py:326-462):
```python
@classmethod
def init_new(cls, batch: ModelWorkerBatch, model_runner: ModelRunner):
    ret = cls(
        forward_mode=batch.forward_mode,
        batch_size=len(batch.seq_lens),
        input_ids=batch.input_ids,
        # ... copy all fields from ModelWorkerBatch
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=model_runner.attn_backend,
    )

    # Initialize positions for decode vs extend
    if ret.forward_mode.is_decode():
        ret.positions = clamp_position(batch.seq_lens)
    else:
        ret.positions, ret.extend_start_loc = compute_position(...)

    return ret
```

**Padding for DP** (forward_batch_info.py:676-791):
When using data parallel attention with multiple DP ranks, tensors are padded to consistent sizes:
```python
def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
    # Determine padding mode (max_len or per-rank)
    self.dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        self.is_extend_in_batch, global_num_tokens
    )

    if dp_padding_mode.is_max_len():
        max_num_tokens = max(global_num_tokens)
        global_num_tokens = [max_num_tokens] * sync_group_size

    # Pad all tensors
    self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
    self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)
    self.seq_lens = self._pad_tensor_to_size(self.seq_lens, bs, value=seq_len_fill_value)
```

**Post-Forward Cleanup** (forward_batch_info.py:792-840):
After forward pass with DP, remove padding from output:
```python
def post_forward_mlp_sync_batch(self, logits_output: LogitsProcessorOutput):
    # Restore original batch size/forward mode
    self.forward_mode = getattr(self, "_original_forward_mode", self.forward_mode)
    self.batch_size = getattr(self, "_original_batch_size", self.batch_size)

    # Slice outputs back to original size
    if self.forward_mode.is_decode():
        logits_output.next_token_logits = logits_output.next_token_logits[:bs]
    elif self.forward_mode.is_extend():
        logits_output.next_token_logits = logits_output.next_token_logits[:num_tokens]
```

---

## 3. ForwardMode Enum and Execution Routing

### 3.1 ForwardMode Definitions

```python
# forward_batch_info.py:64-147
class ForwardMode(IntEnum):
    EXTEND = auto()          # Prefill/extend with KV cache prefix
    DECODE = auto()          # Single token generation
    MIXED = auto()           # Chunked prefill (both extend + decode)
    IDLE = auto()            # No work for this DP rank
    TARGET_VERIFY = auto()   # Speculative decoding verification
    DRAFT_EXTEND = auto()    # Speculative draft model extend
    DRAFT_EXTEND_V2 = auto() # Alternative draft extend
    SPLIT_PREFILL = auto()   # PD multiplexing split prefill
```

### 3.2 Mode-Specific Properties

**CUDA Graph Compatibility** (forward_batch_info.py:135-143):
```python
def is_cuda_graph(self):
    return (
        self == ForwardMode.DECODE
        or self == ForwardMode.TARGET_VERIFY
        or self == ForwardMode.IDLE
    )
```
Only DECODE, TARGET_VERIFY, and IDLE modes use standard CUDA graph runner. EXTEND uses piecewise runner.

**Prefill Detection** (forward_batch_info.py:85-99):
```python
def is_extend(self, include_draft_extend_v2: bool = False):
    return (
        self == ForwardMode.EXTEND
        or self == ForwardMode.MIXED
        or self == ForwardMode.DRAFT_EXTEND
        or (self == ForwardMode.DRAFT_EXTEND_V2 if include_draft_extend_v2 else False)
        or self == ForwardMode.TARGET_VERIFY
    )
```

### 3.3 Runner Selection Logic

**Standard Runner** (from model_runner.py, as referenced in PIECEWISE_CUDA_GRAPH_ONBOARDING.md:213-223):
```python
def forward_decode(forward_batch, **kwargs):
    # Standard CUDA graph is used for DECODE mode
    if forward_mode.is_cuda_graph() and cuda_graph_runner.can_run(forward_batch):
        return cuda_graph_runner.replay(forward_batch, **kwargs)

    # Fallback to eager
    return model.forward(input_ids, positions, forward_batch, **kwargs)
```

**Piecewise Runner** (from model_runner.py:2147-2156):
```python
def forward_extend(forward_batch, **kwargs):
    # Piecewise graph is tried FIRST for extend mode
    if piecewise_cuda_graph_runner is not None:
        if piecewise_cuda_graph_runner.can_run(forward_batch):
            return piecewise_cuda_graph_runner.replay(forward_batch, **kwargs)

    # Fallback to eager execution
    return model.forward(input_ids, positions, forward_batch, **kwargs)
```

**Decision Tree**:
```
ForwardBatch.forward_mode
├── DECODE/TARGET_VERIFY/IDLE → Standard CudaGraphRunner
│   └── can_run? → Yes: replay(), No: eager
└── EXTEND/MIXED/DRAFT_EXTEND → PiecewiseCudaGraphRunner
    └── can_run? → Yes: replay(), No: eager
```

---

## 4. Runner Compatibility Checks

### 4.1 Standard CudaGraphRunner.can_run()

```python
# cuda_graph_runner.py:390-450
def can_run(self, forward_batch: ForwardBatch):
    # 1. Batch size check
    cuda_graph_bs = (
        forward_batch.batch_size if not require_mlp_tp_gather
        else max(forward_batch.global_num_tokens_cpu)
    )
    is_bs_supported = (
        cuda_graph_bs in self.graphs if self.disable_padding
        else cuda_graph_bs <= self.max_bs
    )

    # 2. DP CUDA graph compatibility
    if self.require_mlp_sync:
        is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

    # 3. Encoder-decoder mixed batch check
    is_encoder_lens_supported = (
        torch.all(forward_batch.encoder_lens > 0)
        if self.is_encoder_decoder else True
    )

    # 4. Capture hidden mode match
    requested_mode = max(
        forward_batch.capture_hidden_mode,
        getattr(forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL)
    )
    capture_hidden_mode_matches = (
        requested_mode == CaptureHiddenMode.NULL or
        requested_mode == self.capture_hidden_mode
    )

    # 5. Two-batch overlap support
    is_tbo_supported = (
        forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
    )

    # 6. N-gram speculative decoding check
    is_ngram_supported = (
        (forward_batch.batch_size * self.num_tokens_per_bs == forward_batch.input_ids.numel())
        if self.model_runner.spec_algorithm.is_ngram() else True
    )

    return (
        is_bs_supported and is_encoder_lens_supported and
        is_tbo_supported and capture_hidden_mode_matches and is_ngram_supported
    )
```

**Key Rejection Reasons**:
1. Batch size not captured or exceeds max
2. DP synchronization requirements not met
3. Mixed encoder-decoder batch (some sequences with encoder_len=0)
4. Requested capture_hidden_mode doesn't match captured graphs
5. Two-batch overlap requirements not satisfied
6. N-gram token count mismatch

### 4.2 PiecewiseCudaGraphRunner.can_run()

```python
# piecewise_cuda_graph_runner.py:265-277
def can_run(self, forward_batch: ForwardBatch):
    num_tokens = len(forward_batch.input_ids)

    # 1. Token count must not exceed max
    if num_tokens > self.max_num_tokens:
        return False

    # 2. Cannot compute input token logprobs during graph replay
    if forward_batch.return_logprob:
        for start_len, seq_len in zip(
            forward_batch.extend_logprob_start_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
        ):
            if start_len is not None and start_len < seq_len:
                return False  # Need logprobs for input tokens

    return True
```

**Key Rejection Reasons**:
1. Token count exceeds `--piecewise-cuda-graph-max-tokens`
2. Logprobs requested for input tokens (not just output tokens)

**Why Logprobs Fail**: During graph replay with padding, intermediate logits for padded positions are invalid. Computing logprobs requires access to all intermediate token logits, which would include these invalid values.

---

## 5. Memory Management Deep Dive

### 5.1 Capture Ordering Strategy

Both runners capture in **reverse order** (largest first):

**Standard Runner**:
```python
# cuda_graph_runner.py:474-479
capture_range = (
    tqdm.tqdm(list(reversed(self.capture_bs)))
    if get_tensor_model_parallel_rank() == 0
    else reversed(self.capture_bs)
)
for i, bs in enumerate(capture_range):
    # Capture batch size bs
```

**Piecewise Runner**:
```python
# piecewise_cuda_graph_runner.py:293-297
capture_range = (
    tqdm.tqdm(list(reversed(self.capture_num_tokens)))
    if get_tensor_model_parallel_rank() == 0
    else reversed(self.capture_num_tokens)
)
for i, num_tokens in enumerate(capture_range):
    # Capture num_tokens
```

**Rationale**: Capturing largest size first allocates the maximum required memory block. Subsequent smaller captures can reuse portions of this block, reducing fragmentation and total memory usage.

### 5.2 Memory Pool Lifecycle

```
1. First runner initialization
   ↓
2. set_global_graph_memory_pool(device_module.graph_pool_handle())
   ↓
3. set_graph_pool_id(pool) for symmetric memory
   ↓
4. Capture largest graphs (allocate peak memory)
   ↓
5. Capture smaller graphs (reuse memory)
   ↓
6. Both runners share same pool for replay
```

### 5.3 Buffer Pre-allocation

**Standard Runner** (cuda_graph_runner.py:305-376):
```python
with torch.device(self.device):
    # Pre-allocate max size buffers
    self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
    self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
    self.seq_lens = torch.full((self.max_bs,), self.seq_len_fill_value, dtype=torch.int32)
    self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
    self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
    self.mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
    self.next_token_logits_buffer = torch.zeros(
        (self.max_num_token, vocab_size), dtype=torch.float
    )
```

**Piecewise Runner** (piecewise_cuda_graph_runner.py:173-179):
```python
with torch.device(self.device):
    # Only allocate input buffers (no logits buffer)
    self.input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
    self.out_cache_loc = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
    self.positions = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
```

**Key Difference**: Standard runner pre-allocates logits buffer; piecewise doesn't need to because torch.compile handles output buffer management.

---

## 6. Edge Cases and Limitations

### 6.1 Logprob Computation Limitation (Piecewise)

**Code Location**: piecewise_cuda_graph_runner.py:268-274

**Problem**: Cannot compute logprobs for input tokens during piecewise graph replay.

**Example Scenario**:
```python
# User request: compute logprobs for prompt "Hello world"
extend_logprob_start_lens_cpu = [0]  # Start from beginning
extend_seq_lens_cpu = [2]            # "Hello" + "world"

# This will return False from can_run()
if start_len < seq_len:  # 0 < 2
    return False  # Fallback to eager execution
```

**Workaround**: System automatically falls back to eager execution for these cases.

### 6.2 Encoder-Decoder Mixed Batches (Standard)

**Code Location**: cuda_graph_runner.py:412-416

**Problem**: CUDA graph cannot handle batches with some sequences having `encoder_len=0` (decoder-only) and others with `encoder_len>0` (encoder-decoder).

**Reason**: The `full_text_row_masked_out_mask` tensor shape depends on encoder_lens, and mixed values create incompatible tensor shapes.

**Detection**:
```python
is_encoder_lens_supported = (
    torch.all(forward_batch.encoder_lens > 0)
    if self.is_encoder_decoder else True
)
```

### 6.3 Hidden State Capture Mode Mismatch

**Code Location**: cuda_graph_runner.py:418-430

**Problem**: If a request needs hidden states but graphs were captured without hidden state support, replay fails.

**Solution**: Recapture all graphs with new hidden mode (cuda_graph_runner.py:700-730):
```python
def recapture_if_needed(self, forward_batch: ForwardBatch):
    required_capture_hidden_mode = max(
        forward_batch.capture_hidden_mode,
        getattr(forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL),
        CaptureHiddenMode.FULL if enable_return_hidden_states else CaptureHiddenMode.NULL
    )

    if self.capture_hidden_mode != required_capture_hidden_mode:
        self.capture_hidden_mode = required_capture_hidden_mode
        self.capture()  # Recapture all graphs
```

**Performance Impact**: Recapture adds 30-60 seconds but only happens once per hidden mode change.

### 6.4 Pipeline Parallelism Incompatibility (Piecewise)

**Documented In**: PIECEWISE_CUDA_GRAPH_ONBOARDING.md:32

**Reason**: Proxy tensor handling complexity with torch.compile and PP communication patterns.

**Detection** (from model_runner.py:1591-1597):
```python
if server_args.pp_size > 1:
    logger.warning("Piecewise CUDA graph not supported with pipeline parallelism")
    piecewise_cuda_graph_runner = None
```

### 6.5 N-gram Speculative Decoding Alignment (Standard)

**Code Location**: cuda_graph_runner.py:435-442

**Problem**: N-gram speculative decoding requires exact token count alignment.

**Check**:
```python
is_ngram_supported = (
    (forward_batch.batch_size * self.num_tokens_per_bs == forward_batch.input_ids.numel())
    if self.model_runner.spec_algorithm.is_ngram() else True
)
```

**Example**:
- `batch_size=4`, `num_tokens_per_bs=5` (5 draft tokens per sequence)
- Expected `input_ids.numel()=20` tokens
- If actual is 19 or 21, graph cannot run → eager execution

---

## 7. Integration with ModelRunner

### 7.1 Initialization Order

```
1. ModelRunner.__init__()
   ├── Initialize memory pools (req_to_token_pool, token_to_kv_pool)
   ├── Initialize attention backend
   └── Check compatibility
       ├── enable_piecewise_cuda_graph?
       │   └── Collect attention_layers
       │       └── PiecewiseCudaGraphRunner.__init__()
       │           ├── Install torch.compile
       │           ├── Warmup
       │           └── Capture all token sizes
       └── !disable_cuda_graph?
           └── CudaGraphRunner.__init__()
               ├── Set torch.compile config
               └── Capture all batch sizes
```

### 7.2 Forward Pass Routing

**High-Level Flow** (from model_runner.py):
```python
def forward(self, forward_batch):
    if forward_batch.forward_mode.is_extend():
        return self.forward_extend(forward_batch)
    elif forward_batch.forward_mode.is_decode():
        return self.forward_decode(forward_batch)
    elif forward_batch.forward_mode.is_mixed():
        return self.forward_mixed(forward_batch)
    else:
        return self.forward_idle(forward_batch)

def forward_extend(self, forward_batch):
    # Try piecewise graph FIRST
    if self.piecewise_cuda_graph_runner is not None:
        if self.piecewise_cuda_graph_runner.can_run(forward_batch):
            return self.piecewise_cuda_graph_runner.replay(forward_batch)

    # Fallback to eager
    return self.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch)

def forward_decode(self, forward_batch):
    # Try standard graph FIRST
    if self.cuda_graph_runner is not None:
        if self.cuda_graph_runner.can_run(forward_batch):
            return self.cuda_graph_runner.replay(forward_batch)

    # Fallback to eager
    return self.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch)
```

### 7.3 Attention Backend Coordination

**Standard Runner Metadata Init** (cuda_graph_runner.py:649-657):
```python
# During capture
self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
    bs, num_tokens, req_pool_indices, seq_lens, encoder_lens,
    forward_batch.forward_mode, forward_batch.spec_info
)

# During replay
self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
    bs, self.req_pool_indices[:bs], self.seq_lens[:bs],
    forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
    self.encoder_lens[:bs], self.capture_forward_mode, forward_batch.spec_info
)
```

**Piecewise Runner Metadata Init** (implicit via set_forward_context):
```python
# piecewise_cuda_graph_runner.py:391
with set_forward_context(forward_batch, self.attention_layers):
    # This context manager handles attention backend initialization
    self.model_runner.model.forward(...)
```

**Key Difference**: Piecewise relies on context manager to set up attention metadata; standard calls backend methods directly.

---

## 8. PPProxyTensors for Pipeline Parallelism

### 8.1 Structure

```python
# forward_batch_info.py:970-998
class PPProxyTensors:
    tensors: Dict[str, torch.Tensor]

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})
```

### 8.2 Usage in Standard Runner

**Initialization** (cuda_graph_runner.py:322-332):
```python
if self.pp_size > 1:
    self.pp_proxy_tensors = {
        "hidden_states": torch.zeros(
            (self.max_bs, hidden_size), dtype=model_dtype
        ),
        "residual": torch.zeros(
            (self.max_bs, hidden_size), dtype=model_dtype
        ),
    }
```

**Capture** (cuda_graph_runner.py:671-673):
```python
if self.pp_size > 1:
    kwargs["pp_proxy_tensors"] = PPProxyTensors(
        {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
    )
```

**Replay** (cuda_graph_runner.py:772-775):
```python
if pp_proxy_tensors:
    for key in self.pp_proxy_tensors.keys():
        dim = pp_proxy_tensors[key].shape[0]
        self.pp_proxy_tensors[key][:dim].copy_(pp_proxy_tensors[key])
```

### 8.3 Why Piecewise Doesn't Support PP

**Code Reference**: piecewise_cuda_graph_runner.py:505-509
```python
if isinstance(output, PPProxyTensors):
    raise NotImplementedError(
        "PPProxyTensors is not supported in PiecewiseCudaGraphRunner yet."
    )
```

**Reason**: torch.compile's graph tracing has difficulty with dynamic PP tensor shapes and inter-stage communication patterns. Standard runner's explicit graph capture handles this better.

---

## 9. Debugging and Profiling

### 9.1 Capture Profiling

**Enable** via `--enable-profile-cuda-graph`:

```python
# cuda_graph_runner.py:453-521
if self.enable_profile_cuda_graph:
    profile_context = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    )
    torch.cuda.memory._record_memory_history()

# After capture
torch.cuda.memory._dump_snapshot(f"cuda_graph_runner_memory_usage.pickle")
logger.info(prof.key_averages(group_by_input_shape=True).table(...))
```

**Output**: Memory usage snapshot and sorted kernel timing tables.

### 9.2 Capture Progress Monitoring

Both runners show tqdm progress bars on rank 0:

```python
# Standard runner
capture_range.set_description(f"Capturing batches ({bs=} {avail_mem=:.2f} GB)")

# Piecewise runner
capture_range.set_description(f"Capturing num tokens ({num_tokens=} {avail_mem=:.2f} GB)")
```

### 9.3 Memory Tracking

**Available Memory Checks** (cuda_graph_runner.py:469-473, 482-486):
```python
avail_mem = get_available_gpu_memory(
    self.model_runner.device,
    self.model_runner.gpu_id,
    empty_cache=False,
)
```

Logged before and during capture to detect memory pressure.

---

## 10. Configuration Summary

### 10.1 Standard CUDA Graph Server Args

#### Core Configuration
- `--cuda-graph-max-bs <int>`: Maximum batch size to capture (default: dynamic based on GPU memory and TP size)
  - Auto-configured based on GPU memory: 8-512 depending on model size and available memory
  - Can be overridden explicitly for custom memory tuning
- `--cuda-graph-bs <int> [<int> ...]`: Explicit list of batch sizes to capture (default: auto-generated)
  - Default generation: `[1, 2, 4, 8, 12] + range(16, 257, 8) + range(272, 512, 16) + range(512, max_bs+1, 32)`
  - When `--disable-cuda-graph-padding` is set: `range(1, max_bs+1)` (captures every batch size)
- `--disable-cuda-graph`: Disable CUDA graph entirely (default: False)
  - Automatically disabled for: PD prefill mode, torch native attention backend, torch Flex Attention, deepeek_mode=normal, tensor dump mode
- `--disable-cuda-graph-padding`: Require exact batch size match, no padding (default: False)
  - Captures graphs for all batch sizes from 1 to max_bs
  - Uses more memory but avoids padding overhead
  - Useful when batch sizes are predictable

#### Torch Compile Integration
- `--enable-torch-compile`: Enable torch.compile for standard CUDA graphs (default: False)
  - Conflicts with `--enable-piecewise-cuda-graph` (mutually exclusive)
  - Uses torch.compile in "max-autotune-no-cudagraphs" mode by default
- `--torch-compile-max-bs <int>`: Maximum batch size for torch.compile (default: 32)
  - Should be a subset of `cuda_graph_bs`
  - Smaller values reduce compilation time

#### Performance Tuning
- `--enable-cudagraph-gc`: Allow garbage collection during CUDA graph capture (default: False)
  - When disabled (default), GC is frozen during capture to speed up the process
  - Enable if you encounter memory issues during capture
- `--enable-profile-cuda-graph`: Enable profiling of CUDA graph capture (default: False)
  - Generates memory snapshots and kernel timing tables
  - Outputs: `cuda_graph_runner_memory_usage.pickle` and profiling logs

#### Overlap Optimization
- `--enable-two-batch-overlap`: Enable two micro batches to overlap (default: False)
  - Requires compatible batch configuration
  - Checked via `forward_batch.can_run_tbo` during `can_run()`
- `--enable-single-batch-overlap`: Let computation and communication overlap within one micro batch (default: False)
- `--tbo-token-distribution-threshold <float>`: Threshold for token distribution between batches (default: 0.48)
  - Determines whether to use two-batch-overlap or two-chunk-overlap
  - Set to 0 to disable two-chunk-overlap

#### Hidden State Capture
- `--enable-return-hidden-states`: Enable capturing hidden states for all tokens (default: False)
  - Triggers automatic recapture with `CaptureHiddenMode.FULL`
  - Used for speculative decoding (EAGLE) and other features requiring intermediate states

**CaptureHiddenMode States**:
```python
NULL = 0   # No hidden state capture (default, minimal memory)
LAST = 1   # Capture only last token hidden state (used by some spec decode methods)
FULL = 2   # Capture all token hidden states (required by --enable-return-hidden-states)
```

### 10.2 Piecewise CUDA Graph Server Args

#### Core Configuration
- `--enable-piecewise-cuda-graph`: Enable piecewise CUDA graphs for extend/prefill (default: False)
  - Experimental feature for optimizing prefill/extend operations
  - Conflicts with `--enable-torch-compile` (mutually exclusive)
  - Not supported with pipeline parallelism (PP > 1)
- `--piecewise-cuda-graph-max-tokens <int>`: Maximum token count to capture (default: 4096)
  - Controls the upper bound of token sizes that will use CUDA graphs
  - Requests exceeding this fall back to eager execution
- `--piecewise-cuda-graph-tokens <int> [<int> ...]`: Explicit list of token sizes to capture (default: auto-generated)
  - Default generation: `[16, 32, 64, 128, 256] + range(512, 4097, 256) + range(4352, max_tokens+1, 256)`
  - Auto-generated based on `piecewise_cuda_graph_max_tokens`
- `--piecewise-cuda-graph-compiler <str>`: Compiler backend to use (default: "inductor")
  - Choices: `"eager"` or `"inductor"`
  - `"eager"`: Faster capture, no compilation overhead
  - `"inductor"`: Better runtime performance through torch.compile optimizations

### 10.3 Environment Variables

#### Standard CUDA Graph Runner
- `SGLANG_TORCH_COMPILE_MODE`: Torch compile mode (default: "max-autotune-no-cudagraphs")
  - Used when `--enable-torch-compile` is set
  - Other valid modes: "reduce-overhead", "max-autotune", "default"
- `SGLANG_TORCH_DYNAMIC_SHAPE`: Enable dynamic shapes in torch.compile (default: False)
  - Only used on AMD ROCm/HIP platforms (`_is_hip`)
  - Format: set to "1", "true", or "True" to enable
- `SGLANG_MEMORY_SAVER_CUDA_GRAPH`: Enable memory saver mode for CUDA graphs (default: False)
  - Must also set `--enable-memory-saver` server arg
  - Allows release/resume of CUDA graph memory for dynamic workloads

#### Piecewise CUDA Graph Runner
- No specific environment variables currently used

### 10.4 Automatic Disablement Conditions

CUDA graphs are automatically disabled when:
1. **Attention Backend Incompatibility**:
   - Torch native attention backend is used
   - Torch Flex Attention backend is used
2. **Mode Incompatibility**:
   - `deepeek_mode=normal` is set
   - Disaggregation mode is "prefill" (prefill-only server)
   - Debug tensor dump mode is enabled
3. **Piecewise Specific**:
   - Pipeline parallelism size > 1 (`--pp-size > 1`)

### 10.5 Runtime State Configuration

#### Batch Size Generation Logic
The system auto-generates batch sizes based on GPU memory capacity:

**Small GPU (<35GB HBM, TP<4)**:
- `cuda_graph_max_bs`: 8-32
- `chunked_prefill_size`: 2048-4096
- Example: L4/T4 GPUs

**Medium GPU (35-55GB HBM)**:
- `cuda_graph_max_bs`: 64-256
- `chunked_prefill_size`: 8192-32768
- Example: A100-40GB, A6000

**Large GPU (>55GB HBM or TP≥4)**:
- `cuda_graph_max_bs`: 128-512
- `chunked_prefill_size`: 16384-32768
- Example: A100-80GB, H100

**Memory Reservation Formula**:
```python
reserved_mem = chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2  # In GB
# With DP attention enabled:
reserved_mem += cuda_graph_max_bs * dp_size * 3
if cuda_graph_max_bs > 300:
    reserved_mem += cuda_graph_max_bs * dp_size * 1.5  # Extra overhead for large batch sizes
```

#### Token Size Generation Logic (Piecewise)
```python
# Default: [16, 32, 64, 128, 256, 512, 768, 1024, 1280, ..., 4096, 4352, ...]
capture_sizes = (
    [16, 32, 64, 128, 256]
    + list(range(512, 4097, 256))
    + list(range(4352, piecewise_cuda_graph_max_tokens + 1, 256))
)
```

### 10.6 Compatibility Matrix

| Feature | Standard CUDA Graph | Piecewise CUDA Graph |
|---------|-------------------|---------------------|
| Pipeline Parallelism (PP) | ✅ Supported | ❌ Not Supported |
| Tensor Parallelism (TP) | ✅ Supported | ✅ Supported |
| Data Parallelism (DP) | ✅ Supported | ✅ Supported |
| Encoder-Decoder Models | ✅ Supported (no mixed batches) | ✅ Supported |
| Two-Batch Overlap | ✅ Supported | ✅ Supported |
| N-gram Speculative Decoding | ✅ Supported | ✅ Supported |
| EAGLE Speculative Decoding | ✅ Supported | ✅ Supported |
| LoRA | ✅ Supported | ✅ Supported |
| Input Token Logprobs | ✅ Supported | ❌ Falls back to eager |
| torch.compile | ✅ Optional (--enable-torch-compile) | ✅ Built-in (inductor mode) |
| Custom AllReduce | ✅ Supported (backed up) | ✅ Supported (disabled during capture/replay) |
| FlashInfer Attention | ✅ Supported | ✅ Supported |
| Torch Native Attention | ❌ Disables CUDA graph | ❌ Disables CUDA graph |
| Torch Flex Attention | ❌ Disables CUDA graph | ❌ Disables CUDA graph |

### 10.7 Example Configurations

#### High Throughput (Large Batch, TP=4)
```bash
--cuda-graph-max-bs 512 \
--disable-cuda-graph-padding \
--enable-two-batch-overlap \
--enable-piecewise-cuda-graph \
--piecewise-cuda-graph-max-tokens 8192
```

#### Low Latency (Small Batch, TP=1)
```bash
--cuda-graph-max-bs 32 \
--cuda-graph-bs 1 2 4 8 16 32 \
--piecewise-cuda-graph-max-tokens 2048 \
--piecewise-cuda-graph-tokens 16 32 64 128 256 512 1024 2048
```

#### Memory Constrained (Small GPU)
```bash
--cuda-graph-max-bs 16 \
--enable-cudagraph-gc \
--piecewise-cuda-graph-max-tokens 2048 \
--piecewise-cuda-graph-compiler eager
```

#### Torch Compile Optimization
```bash
--enable-torch-compile \
--torch-compile-max-bs 64 \
--cuda-graph-max-bs 128
# Note: Cannot use with --enable-piecewise-cuda-graph
```

#### Speculative Decoding with Hidden States
```bash
--enable-return-hidden-states \
--cuda-graph-max-bs 128
# Will auto-recapture with CaptureHiddenMode.FULL
```

---

## 11. Common Patterns

### 11.1 Replay Padding Pattern

Both runners use identical padding logic:

1. Binary search for next larger captured size
2. Zero-fill padding regions
3. Copy actual data to pre-allocated buffers
4. Create static batch with padded size
5. After forward, slice outputs back to original size

**Standard**:
```python
# cuda_graph_runner.py:750-763
index = bisect.bisect_left(self.capture_bs, raw_bs)
bs = self.capture_bs[index]
if bs != raw_bs:
    self.seq_lens.fill_(self.seq_len_fill_value)
    self.out_cache_loc.zero_()
self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
```

**Piecewise**:
```python
# piecewise_cuda_graph_runner.py:413-426
index = bisect.bisect_left(self.capture_num_tokens, num_tokens)
static_num_tokens = self.capture_num_tokens[index]
if static_num_tokens != num_tokens:
    self.out_cache_loc.zero_()
self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
```

### 11.2 Custom Allreduce Handling

Standard runner backs up TP communicator during compilation:

```python
# cuda_graph_runner.py:146-163
backup_ca_comm = tp_group.ca_comm
# tp_group.ca_comm = None  # Optionally disable during compile
yield torch.compile(...)
tp_group.ca_comm = backup_ca_comm
```

Piecewise disables during capture/replay:

```python
# piecewise_cuda_graph_runner.py:284-286, 482-484
old_ca_disable = self.model_runner.tp_group.ca_comm.disabled
self.model_runner.tp_group.ca_comm.disabled = True
# ... capture/replay ...
self.model_runner.tp_group.ca_comm.disabled = old_ca_disable
```

---

## 12. File Reference Map

| Component | Standard Runner | Piecewise Runner | Shared |
|-----------|----------------|-----------------|--------|
| Main implementation | cuda_graph_runner.py | piecewise_cuda_graph_runner.py | - |
| Data structures | forward_batch_info.py:172-324 | forward_batch_info.py:172-324 | ✓ |
| Memory pool | cuda_graph_runner.py:212-223 | piecewise_cuda_graph_runner.py:115-126 | ✓ |
| Context managers | cuda_graph_runner.py:93-119 | piecewise_cuda_graph_runner.py:67-92 | ✓ |
| Model patching | cuda_graph_runner.py:134-164 | piecewise_cuda_graph_runner.py:106-112 | Δ |
| Capture logic | cuda_graph_runner.py:452-698 | piecewise_cuda_graph_runner.py:279-405 | - |
| Replay logic | cuda_graph_runner.py:822-853 | piecewise_cuda_graph_runner.py:477-512 | - |
| Compatibility checks | cuda_graph_runner.py:390-450 | piecewise_cuda_graph_runner.py:265-277 | - |

---

## 13. References

- **Main Onboarding**: PIECEWISE_CUDA_GRAPH_ONBOARDING.md
- **Source Files**:
  - python/sglang/srt/model_executor/cuda_graph_runner.py
  - python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
  - python/sglang/srt/model_executor/forward_batch_info.py
- **Related Systems**:
  - python/sglang/srt/compilation/compile.py
  - python/sglang/srt/compilation/piecewise_context_manager.py
  - python/sglang/srt/layers/attention/flashinfer_backend.py
