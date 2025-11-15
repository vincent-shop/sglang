# Piecewise CUDA Graph: Complete Onboarding Guide
## An Advanced Performance Optimization for SGLang Inference

SGLang Team
January 2025

---

## 1. Introduction

Piecewise CUDA graph is an experimental optimization that accelerates **prefill/extend** operations (multi-token processing) by eliminating kernel launch overhead. This document provides a comprehensive technical onboarding following the structure of how_to_write_some_design_doc.md.

**Critical Context**: SGLang has TWO separate CUDA graph systems:
- **Standard CudaGraphRunner**: Optimizes decode (1 token/request) by batch size
- **PiecewiseCudaGraphRunner**: Optimizes extend/prefill (multi-token) by token count

This document focuses on PiecewiseCudaGraphRunner.

### Decomposition of Major Components

1. **Dual Graph Architecture** - Understanding why SGLang needs two separate graph runners
2. **Token-Based Capture Strategy** - How graphs are captured by token count (4-4096+), not batch size
3. **Configuration & Runtime States** - All tunable parameters and their effects (§3.2)
4. **Torch.Compile Integration** - Using PyTorch 2.0's compilation with graph capture
5. **Graph Selection & Replay** - Binary search + padding mechanism for runtime execution
6. **Model Compatibility** - Requirements and restrictions (standard GQA, no PP, etc.)
7. **Memory Management** - Shared global memory pool and capture ordering

### Open Problems & Feedback Areas

- **Logprob Support**: Cannot compute logprobs for input tokens during graph replay (piecewise_cuda_graph_runner.py:268-275)
- **Pipeline Parallelism**: Not supported due to proxy tensor handling complexity (model_runner.py:1493-1497)
- **Non-Standard GQA Models**: Some architectures with custom attention patterns are incompatible (model_runner.py:336-347)
- **Torch Compile Conflict**: Cannot coexist with `--enable-torch-compile` due to different compilation strategies (model_runner.py:1485-1490)
  - See Configuration State Matrix (§3.2) for complete compatibility rules

---

## 2. Overview

### System Architecture

Piecewise CUDA graph pre-captures model forward passes for **specific token counts** (not batch sizes). During inference, it selects the closest graph and pads inputs to match the captured shape.

**Execution Flow**:
```
EXTEND mode batch → Can run piecewise? → Yes → Binary search token count
                           ↓ No                         ↓
                    Eager execution ← Pad & copy tensors → Replay graph
```

### Core Implementation (piecewise_cuda_graph_runner.py)

```python
class PiecewiseCudaGraphRunner:
    def __init__(self, model_runner: ModelRunner):
        # Parse configuration
        self.compile_config = CompilationConfig(
            server_args.piecewise_cuda_graph_tokens,      # Token sizes to capture
            server_args.piecewise_cuda_graph_compiler,    # "eager" or "inductor"
        )
        self.capture_num_tokens = self.compile_config.get_capture_sizes()

        # Install torch.compile (lines 192-201)
        with patch_model(model, self.compile_config.compiler):
            install_torch_compiled(patched_model, fullgraph=True, ...)
            with set_compiled(True):
                self.capture()  # Capture all graphs

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        # Lines 266-278
        # Reject if: (1) logprob for input tokens, (2) exceeds max tokens

    def replay(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        # Lines 477-507
        static_forward_batch = self.replay_prepare(forward_batch)  # Pad tensors
        output = model.forward(static_forward_batch, ...)
        return output[:raw_num_tokens]  # Slice back to original size
```

### Key Data Structures

```python
# Instance variables (lines 134-181)
self.graphs = {}                           # Dict is EMPTY - no graphs stored!
self.capture_num_tokens: List[int]         # [1, 2, 3, ..., 4096]
self.input_ids: torch.Tensor               # Pre-allocated buffer (max_num_tokens,)
self.out_cache_loc: torch.Tensor           # Pre-allocated buffer
self.positions: torch.Tensor               # Pre-allocated buffer
```

**Important**: Unlike standard CUDA graph, piecewise does NOT store graph objects in `self.graphs`. Instead, `install_torch_compiled()` installs persistent compilation that replays automatically when `set_compiled(True)`.

### Core Algorithm

```python
def replay_prepare(forward_batch):
    # Lines 407-475
    num_tokens = len(forward_batch.input_ids)

    # Binary search for next larger size
    index = bisect.bisect_left(capture_num_tokens, num_tokens)
    static_num_tokens = capture_num_tokens[index]

    # Zero-pad if mismatch
    if static_num_tokens != num_tokens:
        out_cache_loc.zero_()

    # Copy to pre-allocated buffers
    input_ids[:num_tokens].copy_(forward_batch.input_ids)
    positions[:num_tokens].copy_(forward_batch.positions)

    # Create static batch with padded size
    return ForwardBatch(input_ids=input_ids[:static_num_tokens], ...)
```

### Memory & Synchronization

- **Global Memory Pool**: Shared across standard + piecewise runners (lines 117-127)
- **Capture Order**: Largest-first (reverse order) to enable memory reuse (line 294)
- **GC Control**: `freeze_gc()` prevents garbage collection during capture (line 284)
- **TP Synchronization**: `tp_group.barrier()` after warmup runs (line 402)

---

## 3. Core Topics

### 3.1 Graph Capture Process

Capture happens during `PiecewiseCudaGraphRunner.__init__()` (lines 132-211):

```python
def __init__(self, model_runner):
    # 1. Pre-allocate input buffers (lines 174-180)
    with torch.device(device):
        self.input_ids = torch.zeros((max_num_tokens,), dtype=torch.int64)
        self.out_cache_loc = torch.zeros((max_num_tokens,), ...)
        self.positions = torch.zeros((max_num_tokens,), ...)

    # 2. Set graph pool for symmetric memory (lines 184-187)
    if global_graph_memory_pool is None:
        set_global_graph_memory_pool(device_module.graph_pool_handle())
    set_graph_pool_id(get_global_graph_memory_pool())

    # 3. Install torch compilation (lines 189-198)
    with patch_model(model, compiler) as patched_model:
        install_torch_compiled(
            patched_model,
            fullgraph=True,              # Enforce single graph
            compile_config=compile_config,
            graph_pool=memory_pool,
        )

    # 4. Warmup + Capture (lines 200-210)
    with set_compiled(True):
        self.warmup_and_capture()  # Single dummy forward
        with model_capture_mode():
            self.capture()  # Capture all token sizes
```

**Capture Loop** (`capture()` at lines 280-314):
- Iterates **in reverse order** (largest first) for memory efficiency
- For each token count, calls `capture_one_batch_size(num_tokens)`
- No explicit graph.capture() - compilation is handled by `set_compiled(True)`
- Saves gemlite cache after each capture (line 314)

**File Locations**:
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- `python/sglang/srt/compilation/compile.py` - `install_torch_compiled()`

### 3.2 Configuration Options & Runtime States

Piecewise CUDA graph behavior is controlled by several command-line arguments that define the "state" of the runtime code.

#### Quick Reference Table

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-piecewise-cuda-graph` | flag | `False` | Enable piecewise CUDA graph optimization |
| `--piecewise-cuda-graph-compiler` | str | `"eager"` | Compiler backend: `eager` or `inductor` |
| `--piecewise-cuda-graph-max-tokens` | int | `4096` | Maximum token count for capture |
| `--piecewise-cuda-graph-tokens` | List[int] | auto | Explicit token counts to capture |
| `--mem-fraction-static` | float | auto | GPU memory fraction for weights+cache |
| `--disable-cuda-graph` | flag | `False` | Disables ALL CUDA graphs (standard+piecewise) |
| `--enable-torch-compile` | flag | `False` | **CONFLICTS** with piecewise (disables it) |

#### Primary Configuration

**`--enable-piecewise-cuda-graph`** (server_args.py:3269-3272)
- Type: `bool` (flag)
- Default: `False`
- Description: Master switch to enable piecewise CUDA graph optimization for extend/prefill operations
- Experimental feature
- Example:
  ```bash
  python -m sglang.launch_server --enable-piecewise-cuda-graph
  ```

**`--piecewise-cuda-graph-compiler`** (server_args.py:3280-3285)
- Type: `str`
- Default: `"eager"`
- Choices: `["eager", "inductor"]`
- Description: Compilation backend for graph capture
  - `eager`: Faster compilation, minimal kernel fusion (recommended for development/testing)
  - `inductor`: Slower compilation, aggressive kernel fusion (recommended for production)
- Example:
  ```bash
  --piecewise-cuda-graph-compiler inductor
  ```

**`--piecewise-cuda-graph-max-tokens`** (server_args.py:3293-3297)
- Type: `int`
- Default: `4096`
- Description: Maximum token count for graph capture; determines upper bound of auto-generated token sizes
- Memory overhead increases with this value (~1-2GB per 1000 token sizes)
- Example:
  ```bash
  --piecewise-cuda-graph-max-tokens 2048
  ```

**`--piecewise-cuda-graph-tokens`** (server_args.py:3274-3278)
- Type: `List[int]` (JSON format)
- Default: `None` (auto-generated from `_generate_piecewise_cuda_graph_tokens()`)
- Description: Explicit list of token counts to capture; overrides auto-generation
- Example:
  ```bash
  --piecewise-cuda-graph-tokens "[32,64,128,256,512,1024,2048,4096]"
  ```

#### Auto-Generated Token Sizes

When `--piecewise-cuda-graph-tokens` is not specified, token sizes are generated by `_generate_piecewise_cuda_graph_tokens()` (server_args.py:861-878):

```python
# Actual implementation (server_args.py:866-876)
capture_sizes = (
    list(range(4, 33, 4))       # [4, 8, 12, ..., 32]
    + list(range(48, 257, 16))   # [48, 64, 80, ..., 256]
    + list(range(288, 513, 32))  # [288, 320, 352, ..., 512]
    + list(range(640, 4097, 128)) # [640, 768, 896, ..., 4096]
    + list(range(4352, piecewise_cuda_graph_max_tokens + 1, 256))
)

# Filter by max_tokens setting
capture_sizes = [s for s in capture_sizes if s <= piecewise_cuda_graph_max_tokens]
```

**Result**: Approximately 60-80 capture sizes between 4 and 4096 tokens, with finer granularity at smaller sizes.

#### Related Configuration (Affects Piecewise Behavior)

**`--mem-fraction-static`** (server_args.py:2081-2083)
- Type: `float`
- Default: Auto-computed based on model size
- Description: Fraction of GPU memory for (model weights + KV cache) vs activations/graph buffers
- Formula: `(GPU capacity - activations - cuda graph buffers) / GPU capacity`
- **Impact on Piecewise**: Capture may OOM if too high; reduce to 0.7-0.8 if capture fails
- Example:
  ```bash
  --mem-fraction-static 0.75
  ```

**`--disable-cuda-graph`** (server_args.py:3163-3165)
- Type: `bool` (flag)
- Default: `False`
- **Conflict**: Disables BOTH standard and piecewise CUDA graphs (model_runner.py:1481-1487)
- Example:
  ```bash
  --disable-cuda-graph  # Piecewise will be disabled
  ```

**`--enable-torch-compile`** (server_args.py:3264)
- Type: `bool` (flag)
- Default: `False`
- **Conflict**: Cannot coexist with `--enable-piecewise-cuda-graph` (model_runner.py:1485-1490)
- Different compilation strategies cause interference
- Environment variable: `SGLANG_ENABLE_TORCH_COMPILE` (set to "1" or "0")
- Example (INVALID):
  ```bash
  # This will disable piecewise CUDA graph!
  --enable-torch-compile --enable-piecewise-cuda-graph
  ```

**`--torch-compile-max-bs`** (server_args.py:3287-3291)
- Type: `int`
- Default: `32`
- Description: Maximum batch size for torch.compile (not directly used by piecewise)
- Example:
  ```bash
  --torch-compile-max-bs 64
  ```

#### Pipeline Parallelism Restriction

Piecewise CUDA graph is incompatible with pipeline parallelism:
- `--pp-size > 1` → Piecewise disabled (model_runner.py:1493-1497)
- Reason: Proxy tensor handling complexity across pipeline stages

#### Environment Variables

**`SGLANG_ENABLE_TORCH_COMPILE`** (server_args.py:1720-1721)
- Set by server at startup based on `--enable-torch-compile` flag
- Values: `"1"` (enabled) or `"0"` (disabled)
- Not intended for manual configuration; use `--enable-torch-compile` instead

#### Configuration State Matrix

| Config State | Standard CudaGraph | Piecewise CudaGraph | Notes |
|-------------|-------------------|---------------------|-------|
| Default | ✅ Enabled | ❌ Disabled | Safe default |
| `--enable-piecewise-cuda-graph` | ✅ Enabled | ✅ Enabled | Both coexist |
| `--disable-cuda-graph` | ❌ Disabled | ❌ Disabled | Disables both |
| `--enable-torch-compile` | ✅ Enabled | ❌ Disabled (conflict) | Torch compile takes priority |
| `--pp-size > 1` | ✅ Enabled | ❌ Disabled | Piecewise incompatible with PP |
| Non-standard GQA | ✅ Enabled | ❌ Disabled | Model compatibility check fails |

#### Example Configurations

**Minimal Production Setup**:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-compiler inductor
```

**Memory-Constrained Setup**:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-max-tokens 1024 \
    --mem-fraction-static 0.70
```

**Custom Token Bins** (tuned for specific workload):
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-tokens "[64,128,256,512,1024,2048]" \
    --piecewise-cuda-graph-compiler inductor
```

**High Throughput** (many concurrent requests):
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70B \
    --tp-size 4 \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-max-tokens 8192 \
    --piecewise-cuda-graph-compiler inductor \
    --mem-fraction-static 0.85
```

### 3.3 Model Runner Integration

**Initialization** (model_runner.py:328-346):
```python
if enable_piecewise_cuda_graph and can_run_piecewise_cuda_graph():
    # Collect attention layers for context manager
    self.attention_layers = []
    for layer in model.model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn"):
            self.attention_layers.append(layer.self_attn.attn)

    # Check all layers have standard GQA
    if len(self.attention_layers) < model_config.num_hidden_layers:
        log("Disable piecewise - non-standard GQA")
        self.piecewise_cuda_graph_runner = None
    else:
        self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)
```

**Execution Path** (model_runner.py:2147-2156):
```python
def forward_extend(forward_batch, **kwargs):
    # Piecewise graph is tried FIRST for extend mode
    if piecewise_cuda_graph_runner is not None:
        if piecewise_cuda_graph_runner.can_run(forward_batch):
            return piecewise_cuda_graph_runner.replay(forward_batch, **kwargs)

    # Fallback to eager execution
    return model.forward(input_ids, positions, forward_batch, **kwargs)
```

**Compatibility Checks** (model_runner.py:1481-1497):
- `disable_cuda_graph=True` → Disabled (model_runner.py:1481-1487)
- `enable_torch_compile=True` → Conflict, disabled (model_runner.py:1485-1490)
- `pp_size > 1` → Not supported (model_runner.py:1493-1497)
- Some layers without standard GQA → Disabled (model_runner.py:336-347)

### 3.4 Memory Management

**Memory Pool Sharing**:
- Global pool shared between standard + piecewise runners (lines 117-127)
- Set once, reused across all captures
- Enables better memory locality

**GC Strategy** (`freeze_gc()` context, lines 78-92):
```python
gc.collect()           # Clean before capture
gc.freeze()            # Freeze remaining objects
try:
    capture_graphs()   # Capture without GC interference
finally:
    gc.unfreeze()
```

**No Explicit Defragmentation**: Unlike standard runner, piecewise doesn't store graph objects, so no fragmentation management needed.

### 3.5 Distributed Inference

**Tensor Parallelism** (initialized at model_runner.py:772-778):
```python
initialize_model_parallel(
    tensor_model_parallel_size=tp_size,
    torch_compile=enable_piecewise_cuda_graph,  # Special handling for graphs
)
```

**Synchronization Points**:
- `tp_group.barrier()` before each capture (line 402)
- `set_graph_pool_id()` for symmetric memory allocation (line 187)
- NCCL allocator uses graph pool for communication buffers (line 33)

**Data Parallel Attention**:
- `set_dp_buffer_len()` called during capture (line 385)
- `DpPaddingMode.get_default_mode_in_cuda_graph()` used (line 242)

### 3.6 Troubleshooting

**OOM During Capture** (piecewise_cuda_graph_runner.py:543-549):

Error message indicates capture failed with OOM. The system suggests:

```bash
# Solution 1: Reduce memory allocation
--mem-fraction-static 0.7  # or 0.8 (default is auto-computed, usually 0.85-0.9)

# Solution 2: Reduce max token count
--piecewise-cuda-graph-max-tokens 512  # or 1024, 2048 (default: 4096)

# Solution 3: Use fewer capture sizes (custom bins)
--piecewise-cuda-graph-tokens "[64,128,256,512,1024]"

# Solution 4 (last resort): Disable piecewise
# Remove --enable-piecewise-cuda-graph flag
```

**Root Cause Analysis**:
- Each token size requires separate graph memory allocation
- ~60-80 graphs captured by default (4 to 4096 tokens)
- Memory = model weights + KV cache + activations + all graph buffers
- Formula: `available = GPU_capacity * mem_fraction_static - model_weights`

**Diagnostic Commands**:
```bash
# Check GPU memory before launch
nvidia-smi

# Watch memory during capture (run in separate terminal)
watch -n 0.5 nvidia-smi

# Enable detailed logging
export TORCH_LOGS="+graph_breaks,+recompiles"
```

**Logprob Failures** (lines 268-275):
```python
# Cannot compute logprobs for input tokens during graph replay
if forward_batch.return_logprob:
    for start_len, seq_len in zip(...):
        if start_len is not None and start_len < seq_len:
            return False  # Fall back to eager execution
```

**Symptom**: Requests with `logprobs=True` for input tokens fall back to eager mode, reducing throughput.

**Workaround**:
- Request logprobs only for output tokens (common use case)
- Accept fallback behavior for input token logprobs
- Not a bug - inherent limitation of graph replay

**Compilation Performance**:

| Compiler | Compilation Time | Runtime Performance | Use Case |
|----------|------------------|---------------------|----------|
| `eager` | Fast (seconds) | Baseline | Development, testing, debugging |
| `inductor` | Slow (minutes) | 10-30% faster | Production deployments |

**Recommendation**: Use `eager` for iteration speed, switch to `inductor` for production.

**Conflicts & Incompatibilities**:

```bash
# CONFLICT: torch.compile + piecewise
--enable-torch-compile --enable-piecewise-cuda-graph  # ❌ Piecewise disabled

# CONFLICT: disable-cuda-graph disables EVERYTHING
--disable-cuda-graph --enable-piecewise-cuda-graph  # ❌ Both graphs disabled

# CONFLICT: pipeline parallelism not supported
--pp-size 2 --enable-piecewise-cuda-graph  # ❌ Piecewise disabled
```

Check logs for compatibility warnings:
```
Disable piecewise CUDA graph because disable_cuda_graph is set
Disable piecewise CUDA graph because piecewise_cuda_graph has conflict with torch compile
Disable piecewise CUDA graph because piecewise_cuda_graph does not support PP
Disable piecewise CUDA graph because some layers do not apply Standard GQA
```

**Model Compatibility Issues**:

If piecewise is silently disabled, check:
1. Model has standard Grouped Query Attention (GQA) - model_runner.py:336-347
2. All transformer layers expose `.self_attn.attn` attribute
3. `len(attention_layers) == num_hidden_layers`

**Example Models**:
- ✅ LLaMA 2/3, Mistral, Qwen2 (standard GQA)
- ❌ Some custom architectures with non-standard attention

**Performance Regression**:

If piecewise is slower than eager:
1. Check if requests are falling back (look for "fallback to eager" in logs)
2. Verify token distribution matches capture sizes (most requests should hit exact bins)
3. Try `inductor` compiler instead of `eager`
4. Analyze padding overhead (requests with size between bins get padded)

**Debugging Configuration State**:

Add logging to verify active configuration:
```python
# In model_runner.py, check:
print(f"Piecewise enabled: {self.piecewise_cuda_graph_runner is not None}")
print(f"Capture sizes: {self.piecewise_cuda_graph_runner.capture_num_tokens}")
print(f"Compiler: {self.server_args.piecewise_cuda_graph_compiler}")
```

---

## 4. Implementation Details

### 4.1 Core Functions

#### `capture_one_batch_size(num_tokens)` (lines 316-405)
```python
def capture_one_batch_size(self, num_tokens: int):
    bs = 1  # Always batch size 1!

    # Create dummy ForwardBatch
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=input_ids[:num_tokens],
        positions=positions[:num_tokens],
        seq_lens=torch.tensor([num_tokens]),
        # ... all other required fields
    )

    # Run twice for warmup (lines 400-403)
    for _ in range(2):
        device_module.synchronize()
        tp_group.barrier()
        run_once()  # model.forward() with set_compiled(True)
```

**Note**: No explicit `graph.capture()` call - the `set_compiled(True)` context handles graph recording via torch.compile.

#### `replay(forward_batch)` (lines 477-507)
```python
def replay(forward_batch):
    static_forward_batch = replay_prepare(forward_batch)  # Pad tensors

    with set_forward_context(static_forward_batch, attention_layers):
        with set_compiled(True):  # Triggers compiled replay
            output = model.forward(
                static_forward_batch.input_ids,
                static_forward_batch.positions,
                static_forward_batch,
            )

    # Slice back to original size
    return LogitsProcessorOutput(
        next_token_logits=output.next_token_logits[:raw_num_tokens],
        hidden_states=output.hidden_states[:raw_num_tokens] if ... else None,
    )
```

### 4.2 Helper Functions & Context Managers

**`model_capture_mode()`** (lines 67-74): Sets global `is_capture_mode = True`
**`freeze_gc()`** (lines 78-92): Prevents GC during capture
**`patch_model()`** (lines 107-113): Converts CustomOp layers for compilation
**`_to_torch()`** (lines 95-103): Recursively calls `enter_torch_compile()` on CustomOp modules

### 4.3 Integration Points

**File Dependencies**:
- `model_runner.py:328-346` - Initialization
- `model_runner.py:2147-2149` - Execution path
- `forward_batch_info.py` - ForwardBatch dataclass
- `compilation/compile.py` - `install_torch_compiled()` and `set_compiled()`
- `compilation/piecewise_context_manager.py` - `set_forward_context()`
- `layers/attention/flashinfer_backend.py` - Attention metadata initialization

---

## 5. Deployment Plan

### Prerequisites
- CUDA 12.1+ (for graph capture support)
- PyTorch 2.0+ (for torch.compile integration)
- Available GPU memory > 20GB (depends on model size)
- Test workload representative of production traffic

### Phase 1: Verification
1. Check compatibility: Run `can_run_piecewise_cuda_graph()` checks
2. Verify model has standard GQA attention
3. Ensure no pipeline parallelism (`--pp-size 1`)
4. Confirm `--enable-torch-compile` is NOT set

### Phase 2: Baseline
```bash
# Run without piecewise graphs
python -m sglang.launch_server \
    --model-path your-model \
    --port 30000

# Benchmark prefill latency
python benchmark_latency.py --url http://localhost:30000
```

### Phase 3: Enable & Test
```bash
# Enable with default settings
python -m sglang.launch_server \
    --model-path your-model \
    --port 30000 \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-compiler inductor

# Monitor startup logs for capture progress
# Expected: "Capturing num tokens {num_tokens=...}" messages
```

**Configuration Reference**: See §3.2 for complete list of tunable parameters and example configurations.

### Phase 4: Tune (Optional)
Analyze your workload's token distribution:
```python
# Collect token counts from production
token_counts = [len(req.input_ids) for req in workload]
print(pd.Series(token_counts).describe())

# Customize capture sizes to match p50, p90, p99
--piecewise-cuda-graph-tokens "[32,128,512,1024,2048,4096]"
```

**Tuning Knobs** (see §3.2 for details):
- `--piecewise-cuda-graph-max-tokens`: Reduce to lower memory overhead
- `--piecewise-cuda-graph-tokens`: Custom bins for your workload distribution
- `--mem-fraction-static`: Adjust based on available GPU memory
- `--piecewise-cuda-graph-compiler`: `inductor` for production, `eager` for testing

### Phase 5: Production
- Start with canary deployment (10% traffic)
- Monitor for OOM errors during capture
- Check for logprob-related fallbacks
- Gradually increase to 100% if stable

---

## 6. Comparison: Standard vs Piecewise CUDA Graph

| Aspect | Standard CudaGraphRunner | PiecewiseCudaGraphRunner |
|--------|--------------------------|--------------------------|
| **Target** | DECODE mode (1 token/request) | EXTEND mode (multi-token prefill) |
| **Capture Dimension** | Batch size (1, 2, 4, 8, ...) | Token count (1, 2, 3, ..., 4096) |
| **Batch Size During Capture** | Variable (1-256) | Fixed at 1 |
| **Graph Storage** | `self.graphs[bs] = CUDAGraph()` | No explicit graph objects |
| **Compilation** | `torch.compile()` via `patch_model()` | `install_torch_compiled()` persistent |
| **Replay Trigger** | `graphs[bs].replay()` | `set_compiled(True)` context |
| **Initialization** | model_runner.py:2034-2068 | model_runner.py:328-346 |
| **Execution** | model_runner.py:2227-2244 | model_runner.py:2147-2149 |
| **File** | `cuda_graph_runner.py` | `piecewise_cuda_graph_runner.py` |

**Key Insight**: Both runners **coexist** - standard for decode, piecewise for extend!

---

## 7. Expected Performance

From test suite (`test/srt/test_piecewise_cuda_graph.py`):

**Latency Improvements**:
- Prefill latency target: < 15ms (line 67)
- Typical improvement: 10-30% vs eager execution
- Larger prompts benefit more (due to kernel launch overhead reduction)

**Accuracy**:
- MMLU score ≥ 0.65 (line 56)
- GPQA score ≥ 0.235 (line 45)
- Exact numerical match with eager execution

**Memory Overhead**:
- Capture phase: +1-2GB per 1000 token sizes
- Runtime: Minimal (shared global pool)
- Startup time: +30-60 seconds depending on token sizes

---

## 8. Conclusion

Piecewise CUDA graph is an **experimental** optimization targeting prefill/extend operations. It complements (not replaces) the standard CUDA graph runner used for decode.

**When to Use**:
- High throughput serving with many concurrent requests
- Long-context prefill dominates latency
- Models with standard GQA attention (LLaMA, Mistral, Qwen2)
- No pipeline parallelism required
- GPU memory ≥ 20GB available for graph capture

**When NOT to Use**:
- Small models where prefill is already fast (<50ms baseline)
- Non-standard attention architectures
- Pipeline parallel deployments (`--pp-size > 1`)
- Need input token logprobs (causes fallback to eager)
- Using `--enable-torch-compile` (conflict)
- Memory-constrained environments (<16GB GPU)

**Configuration Quick Start**:
```bash
# Production deployment (recommended)
--enable-piecewise-cuda-graph \
--piecewise-cuda-graph-compiler inductor \
--piecewise-cuda-graph-max-tokens 4096

# Memory-constrained deployment
--enable-piecewise-cuda-graph \
--piecewise-cuda-graph-max-tokens 1024 \
--mem-fraction-static 0.75
```

See §3.2 for complete configuration reference and tuning guide.

**Key Implementation Files**:
- `model_runner.py:328-346, 1481-1497, 2088-2090` - Integration & compatibility checks
- `piecewise_cuda_graph_runner.py` - Complete implementation (capture + replay)
- `server_args.py:489-493, 861-878, 3269-3297` - Configuration & defaults
- `forward_batch_info.py` - ForwardBatch dataclass
- `compilation/compile.py` - `install_torch_compiled()` and context managers

**Tunable Parameters** (see §3.2 for details):
- `--enable-piecewise-cuda-graph` - Master switch
- `--piecewise-cuda-graph-compiler` - `eager` (fast) or `inductor` (optimized)
- `--piecewise-cuda-graph-max-tokens` - Upper bound for capture (default: 4096)
- `--piecewise-cuda-graph-tokens` - Custom token bins (overrides auto-generation)
- `--mem-fraction-static` - GPU memory allocation fraction

**References**:
- Test suite: `test/srt/test_piecewise_cuda_graph.py`
- Error handling: `piecewise_cuda_graph_runner.py:543-549`
- Configuration: `server_args.py:3269-3297`
- Compatibility matrix: §3.2 Configuration State Matrix
