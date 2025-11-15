# SGLang Batch Invariant Operations
## Design Document

**SGLang Project**
January 2025

---

## 1. Introduction

Batch invariant operations ensure deterministic inference results in SGLang by replacing non-deterministic CUDA kernels with custom Triton implementations that guarantee identical outputs regardless of batch size. This system addresses a critical problem: standard PyTorch matrix operations can produce different numerical results when computing `mm(A[0:1], B)` versus `mm(A, B)[0:1]` due to floating-point accumulation order and GPU scheduling non-determinism.

The batch invariant ops module decomposes into three major components:

1. **Triton Kernel Implementations**: Custom GPU kernels for matrix multiply, log softmax, and mean operations using persistent execution patterns
2. **PyTorch Dispatch Hooking**: Runtime interception of standard PyTorch operations via `torch.library.Library` to redirect to batch-invariant implementations
3. **Mode Management API**: Context managers and global state for enabling/disabling batch-invariant behavior

This document outlines the design for operations that guarantee bit-exact reproducibility when `--enable-deterministic-inference` is activated in the model runner.

**Open Problems:**
- Performance optimization for very large matrices (>2^31 elements)
- Support for additional dtypes beyond float16/bfloat16/float32
- Multi-dimensional mean reduction optimization (currently falls back to torch.sum)

---

## 2. Overview

### Architecture

The batch invariant ops system intercepts PyTorch tensor operations at runtime and replaces them with deterministic Triton kernels. The interception happens through PyTorch's dispatch mechanism using `torch.library.Library("aten", "IMPL")`.

```python
# High-level flow
enable_batch_invariant_mode()
  -> Register implementations via torch.library.Library
  -> torch.mm calls redirect to mm_batch_invariant
  -> mm_batch_invariant calls matmul_persistent
  -> matmul_kernel_persistent executes with deterministic scheduling
```

### Key APIs

**Public API:**
- `enable_batch_invariant_mode()`: Globally activate batch-invariant kernels
- `disable_batch_invariant_mode()`: Deactivate and restore original PyTorch ops
- `set_batch_invariant_mode(enabled=True)`: Context manager for scoped activation
- `is_batch_invariant_mode_enabled()`: Query current mode state

**Internal Implementation Functions:**
- `matmul_persistent(a, b, bias=None)`: Persistent matmul kernel wrapper
- `log_softmax(input, dim=-1)`: Triton log softmax implementation
- `mean_dim(input, dim, keepdim, dtype)`: Single-dimension mean reduction
- `bmm_batch_invariant(a, b, out=None)`: Batched matrix multiply

### Data Structures

**Global State:**
```python
_batch_invariant_MODE: bool          # Current activation state
_batch_invariant_LIB: torch.library.Library  # PyTorch dispatch library
_original_torch_bmm: Callable        # Backup of original torch.bmm
```

**Configuration:**
```python
AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])
# Fixed at (16, 16) for attention operations
```

### Algorithms

**Persistent Matmul Strategy:**

The core algorithm uses persistent thread blocks that process multiple output tiles per SM (streaming multiprocessor):

```
for each SM (parallel):
    tile_id = start_pid
    while tile_id < total_tiles:
        (pid_m, pid_n) = compute_tile_position(tile_id)
        accumulator = zeros(BLOCK_M, BLOCK_N)

        for k_tile in range(K_tiles):
            a_block = load_with_mask(A[pid_m, k_tile])
            b_block = load_with_mask(B[k_tile, pid_n])
            accumulator += dot(a_block, b_block)

        if HAS_BIAS:
            accumulator += bias[pid_n]

        store_with_mask(C[pid_m, pid_n], accumulator)
        tile_id += NUM_SMS
```

This persistent approach ensures deterministic execution order by:
1. Launching exactly NUM_SMS blocks (one per SM)
2. Each block processes tiles in a fixed sequential order
3. Avoiding GPU scheduler variance through explicit tile assignment

**Log Softmax Numerical Stability:**

```
for each row (parallel):
    # Pass 1: Find max for stability
    max_val = -inf
    for col_block in range(0, n_cols, BLOCK_SIZE):
        vals = load(row[col_block:col_block+BLOCK_SIZE])
        max_val = max(max_val, max(vals))

    # Pass 2: Sum of exponentials
    sum_exp = 0
    for col_block in range(0, n_cols, BLOCK_SIZE):
        vals = load(row[col_block:col_block+BLOCK_SIZE])
        sum_exp += sum(exp(vals - max_val))

    log_sum_exp = log(sum_exp)

    # Pass 3: Compute final values
    for col_block in range(0, n_cols, BLOCK_SIZE):
        vals = load(row[col_block:col_block+BLOCK_SIZE])
        output = vals - max_val - log_sum_exp
        store(output_row[col_block:col_block+BLOCK_SIZE], output)
```

### Synchronization

**Global State Management:**

The module uses global variables for mode state, protected by Python's GIL. This design is acceptable because:
- Mode changes happen during model initialization, not during inference
- The context manager properly saves and restores state for nested usage
- No concurrent modification of `_batch_invariant_MODE` occurs during forward passes

**PyTorch Dispatch Registration:**

When enabling batch-invariant mode:
1. Create `torch.library.Library("aten", "IMPL")` instance
2. Register CUDA implementations for target ops: `mm`, `addmm`, `_log_softmax`, `mean.dim`, `bmm`
3. Monkeypatch `torch.bmm` directly as fallback for non-dispatch code paths
4. Store library instance in `_batch_invariant_LIB` to prevent premature destruction

Disabling reverses this process and calls `_destroy()` on the library.

### Why This Design Works

**Determinism Guarantee:**

Standard PyTorch matmul uses cuBLAS which:
- May split work differently based on input sizes
- Uses atomic operations for accumulation in some cases
- Schedules thread blocks non-deterministically

Batch-invariant ops guarantee determinism by:
- Fixed tile processing order per SM
- No atomic operations
- Explicit loop-based accumulation in predictable order
- Consistent block sizes based on dtype

**Performance Characteristics:**

Trade-offs made for determinism:
- Persistent kernels reduce kernel launch overhead (good for many small matrices)
- May underutilize GPU for very large single matrices (bad for throughput)
- Three-pass log_softmax is slower than fused implementations but numerically stable
- Mean fallback to torch.sum for multi-dim reduces to PyTorch's implementations

---

## 3. Detailed Design Topics

### 3.1 Persistent Matrix Multiplication

**File:** `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:49-217`

**Function:** `matmul_kernel_persistent`

This Triton kernel implements persistent thread block strategy for deterministic matmul.

**Key Parameters:**
- `NUM_SMS`: Number of streaming multiprocessors, determines parallelism level
- `BLOCK_SIZE_M/N/K`: Tile sizes, configured per dtype for optimal occupancy
- `GROUP_SIZE_M`: Tile grouping for cache locality (set to 8)
- `A_LARGE/B_LARGE/C_LARGE`: Flags for >2^31 element tensors requiring int64 indexing

**Algorithm Details:**

Program ID assignment:
```python
start_pid = tl.program_id(axis=0)  # Each program gets one SM
for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
    pid_m, pid_n = _compute_pid(tile_id, ...)  # Deterministic tile mapping
```

The `_compute_pid` function computes tile coordinates deterministically:
```python
group_id = tile_id // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (tile_id % group_size_m)
pid_n = (tile_id % num_pid_in_group) // group_size_m
```

This ensures tiles are processed in a fixed M-major, N-minor order within groups.

**Masking Strategy:**

Edge handling uses `tl.where` to clamp out-of-bounds indices to 0:
```python
offs_am = tl.where(offs_am < M, offs_am, 0)
```

Then loads use additional masks:
```python
a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
```

This double-masking ensures both address validity and zero-padding for partial tiles.

**Dtype Configuration:**

Wrapper function `matmul_persistent` selects block sizes based on dtype:

| dtype | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages |
|-------|---------|---------|---------|-----------|------------|
| bfloat16 | 128 | 128 | 64 | 8 | 3 |
| float16 | 128 | 256 | 64 | 8 | 3 |
| float32 | 128 | 128 | 32 | 8 | 3 |

These configurations balance occupancy, register usage, and shared memory.

**Bias Fusion:**

Optional bias addition is fused into the output kernel:
```python
if HAS_BIAS:
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    accumulator += bias
```

This saves a separate kernel launch for `addmm` operations.

**Helper Functions:**
- `mm_batch_invariant(a, b)`: Simple wrapper calling `matmul_persistent`
- `addmm_batch_invariant(bias, a, b)`: Wrapper with bias support

### 3.2 Log Softmax

**File:** `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:220-323`

**Function:** `_log_softmax_kernel`

Implements numerically stable log_softmax using three-pass reduction.

**Why Three Passes?**

Single-pass log_softmax is numerically unstable for large values. The max-subtraction trick requires:
1. First pass: compute max (for stability)
2. Second pass: compute sum(exp(x - max))
3. Third pass: compute x - max - log(sum_exp)

**Row-Parallel Processing:**

Each thread block processes one complete row:
```python
row_idx = tl.program_id(0).to(tl.int64)  # One block per row
```

This design:
- Simplifies synchronization (no cross-block communication)
- Works well for attention logits (many rows, moderate columns)
- Uses 1024-element blocks for column iteration

**Numerical Stability:**

Max computation handles infinities correctly:
```python
max_val = -float("inf")
vals = tl.load(..., mask=mask, other=-float("inf"))
max_val = tl.max(tl.maximum(vals, max_val))
```

Masked-out elements use `-inf` as sentinel, which correctly propagates through max.

**Wrapper Function:**

`log_softmax(input, dim=-1)` handles reshaping:
```python
input_2d = input.reshape(-1, input.shape[-1])  # Flatten to 2D
output = _log_softmax_kernel(input_2d, ...)
return output.reshape(original_shape)
```

Currently only supports last-dimension reduction (common for attention).

**Dispatch Registration:**

`_log_softmax_batch_invariant` wrapper matches PyTorch's internal signature:
```python
def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)
```

The `_half_to_float` parameter is part of PyTorch's internal API but not yet supported.

### 3.3 Mean Reduction

**File:** `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:326-495`

**Function:** `mean_kernel`

Implements mean along a single dimension via reduction.

**Three-Dimensional View:**

Input tensor is conceptually reshaped to `(M, N, K)` where:
- `M` = product of dimensions before reduction dim
- `N` = size of reduction dimension
- `K` = product of dimensions after reduction dim

Example: `mean(tensor[4, 8, 16, 32], dim=2)` becomes:
- `M = 4 * 8 = 32`
- `N = 16` (dimension being reduced)
- `K = 32`
- Output shape: `(32, 32)` then reshaped to `(4, 8, 32)`

**Program Organization:**

Each output element gets one thread block:
```python
pid = tl.program_id(0)  # Ranges from 0 to M*K-1
m_idx = pid // K
k_idx = pid % K
```

This allows independent parallel computation of each output element.

**Reduction Loop:**

Each block iterates over the reduction dimension:
```python
acc = 0.0
for n_start in range(0, N, BLOCK_SIZE):
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
    mask = n_offsets < N
    input_idx = m_idx * stride0 + n_offsets * stride1 + k_idx * stride2
    vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    acc += tl.sum(vals)
mean_val = acc / N
```

**Multi-Dimensional Mean Fallback:**

For multiple reduction dimensions, falls back to PyTorch:
```python
if len(dim) == 1:
    return mean_dim(input, dim[0], keepdim=keepdim)
else:
    n_elems = prod([input.shape[d] for d in dim])
    return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems
```

This preserves correctness while avoiding complex multi-pass reduction implementation.

**Dtype Handling:**

Automatically promotes integer types to float32:
```python
if dtype is None:
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        dtype = torch.float32
    else:
        dtype = input.dtype
```

### 3.4 Batched Matrix Multiply

**File:** `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:498-515`

**Function:** `bmm_batch_invariant`

Implements batched matmul by sequentially calling persistent kernel per batch element.

**Implementation:**

```python
def bmm_batch_invariant(a, b, *, out=None):
    if a.ndim == 3 and b.ndim == 3:
        results = []
        for i in range(a.shape[0]):
            results.append(matmul_persistent(a[i], b[i]))
        result = torch.stack(results, dim=0)
        if out is not None:
            out.copy_(result)
            return out
        return result
```

**Why Sequential?**

This naive approach ensures maximum determinism by:
- Processing batches in fixed order
- Avoiding cuBLAS batch scheduling variance
- Reusing well-tested persistent kernel

**Performance Note:**

This is slower than batched cuBLAS for large batch sizes. Future optimization could:
- Launch persistent kernel with batch dimension in tile_id space
- Use separate streams per batch element
- Implement fused batch-aware persistent kernel

Trade-off accepted because:
- Determinism is primary goal
- Most attention operations use small batch sizes in inference
- Per-batch persistent kernel still faster than standard non-persistent approach

**Fallback Monkeypatch:**

The system also monkeypatches `torch.bmm` directly:
```python
_original_torch_bmm = torch.bmm
torch.bmm = bmm_batch_invariant
```

This catches code paths that don't use PyTorch dispatch (e.g., JIT compiled code).

### 3.5 Mode Management and State Control

**File:** `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:518-569`

**Global State Variables:**

```python
_batch_invariant_MODE = False        # Current activation state
_batch_invariant_LIB = None          # torch.library.Library instance
_original_torch_bmm = None           # Backup of torch.bmm for restoration
```

**Enable Sequence:**

```python
def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    if _batch_invariant_MODE:
        return  # Idempotent

    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    # Register CUDA implementations
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "CUDA")

    # Monkeypatch torch.bmm as fallback
    _original_torch_bmm = torch.bmm
    torch.bmm = bmm_batch_invariant
```

**Disable Sequence:**

```python
def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()  # Unregister dispatches
    if _original_torch_bmm is not None:
        torch.bmm = _original_torch_bmm
        _original_torch_bmm = None
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
```

**Context Manager:**

```python
@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    global _batch_invariant_MODE, _batch_invariant_LIB
    old_data = (_batch_invariant_MODE, _batch_invariant_LIB)

    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()

    yield

    # Restore previous state
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE, _batch_invariant_LIB = old_data
```

**State Restoration Subtlety:**

The context manager saves and restores both `_batch_invariant_MODE` and `_batch_invariant_LIB`. This allows nested contexts:

```python
# Outer context: disabled
with set_batch_invariant_mode(False):
    # Inner context: enabled
    with set_batch_invariant_mode(True):
        # Batch-invariant ops active
        result1 = torch.mm(a, b)
    # Restored to outer state (disabled)
    result2 = torch.mm(a, b)
# Restored to original state
```

**Query Function:**

```python
def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE
```

Simple boolean check for debugging and conditional logic.

### 3.6 Attention Block Size Configuration

**File:** `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:572-576`

**Data Structure:**

```python
AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])

def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)
```

**Purpose:**

Provides fixed block sizes for FlashAttention-style attention implementations that need to be batch-invariant. The `(16, 16)` block size is chosen for:
- Good occupancy on modern GPUs (sufficient parallelism)
- Small enough to ensure deterministic tile processing order
- Compatible with typical attention head dimensions (64, 128)

**Integration Point:**

Attention backends query this function when `enable_deterministic_inference` is active:

```python
# In attention backend initialization
if is_batch_invariant_mode_enabled():
    block_m, block_n = get_batch_invariant_attention_block_size()
    # Use fixed block sizes instead of autotuning
```

### 3.7 Integration with Model Runner

**File:** `python/sglang/srt/model_executor/model_runner.py` (approximate line 150)

**Activation Point:**

```python
# Enable batch invariant mode
if server_args.enable_deterministic_inference:
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()
```

This happens during `ModelRunner.__init__()`, before model loading and memory pool initialization.

**Server Argument:**

Users enable via command line:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --enable-deterministic-inference
```

**System-Wide Effect:**

Once enabled, all subsequent PyTorch operations on CUDA tensors will use batch-invariant implementations for the registered ops. This affects:
- Model forward pass (attention, MLP)
- Post-processing (log_softmax for sampling)
- Any custom CUDA operations that use PyTorch tensor ops

### 3.8 Testing Strategy

**File:** `test/srt/batch_invariant/test_batch_invariant_ops.py`

**Test Philosophy:**

Verify that `torch.mm(A[:1], B)` produces **exactly** identical results to `torch.mm(A, B)[:1]` when batch-invariant mode is enabled.

**Test Structure:**

```python
def _test_batch_invariance(self, M, K, N, dtype):
    a = torch.linspace(-100, 100, M * K, dtype=dtype).reshape(M, K)
    b = torch.linspace(-100, 100, K * N, dtype=dtype).reshape(N, K).transpose(0, 1)

    # Method 1: Batch size 1
    out1 = torch.mm(a[:1], b)

    # Method 2: Full batch, then slice
    out2_pre = torch.mm(a, b)
    out2 = out2_pre[:1]

    # Must be exactly zero difference
    diff = (out1 - out2).abs().max()
    return diff.item()
```

**Assertion Strategy:**

```python
def _assert_batch_invariant_results(self, difflist, dtype, test_name):
    max_diff = max(difflist)
    min_diff = min(difflist)
    diff_range = max_diff - min_diff

    # All must be exactly 0.0
    self.assertEqual(max_diff, 0.0, ...)
    self.assertEqual(min_diff, 0.0, ...)
    self.assertEqual(diff_range, 0.0, ...)
```

**Test Cases:**

Three size categories:
- Small: `(8x64x128)`, `(16x128x256)`, `(4x32x64)`
- Medium: `(32x128x1024)`, `(64x512x2048)`, `(24x192x768)`
- Large: `(128x1024x4096)`, `(256x2048x8192)`, `(96x768x3072)`

Each tested with:
- `dtype` in `[torch.float32, torch.bfloat16]`
- 5 iterations per configuration
- Both enabled and disabled modes

**Negative Test:**

```python
def test_without_batch_invariant_mode(self):
    with set_batch_invariant_mode(False):
        difflist = self._run_multiple_iterations(iters=5, M=32, K=128, N=1024, dtype=torch.float32)
        print(f"Without batch-invariant mode, we get diffs: {difflist}")
```

This documents that standard PyTorch is non-deterministic (diffs typically in 1e-5 to 1e-3 range).

---

## 4. Implementation Plan

### 4.1 Work Division

**Phase 1: Core Kernels (2 weeks)**
- Person A: Implement persistent matmul kernel with bias support
  - Days 1-3: Basic kernel structure and tile computation
  - Days 4-6: Dtype configurations and large tensor support
  - Days 7-9: Testing and performance tuning
- Person B: Implement log_softmax and mean kernels
  - Days 1-4: Log_softmax with three-pass reduction
  - Days 5-7: Mean kernel with multi-dim support
  - Days 8-9: Testing and edge case handling

**Phase 2: Integration (1 week)**
- Person A: PyTorch dispatch registration and mode management
  - Days 1-2: Library registration and state management
  - Days 3-4: Context manager and bmm fallback
- Person B: Testing infrastructure
  - Days 1-2: Batch invariance test suite
  - Days 3-4: Integration testing with model runner

**Phase 3: Optimization (1 week)**
- Both: Performance analysis and optimization
  - Days 1-2: Benchmark against cuBLAS baseline
  - Days 3-4: Tune block sizes and occupancy
  - Day 5: Documentation and final testing

### 4.2 Integration Milestones

**Milestone 1 (End of Week 1):**
- Basic persistent matmul kernel working for small matrices
- Log_softmax kernel passing correctness tests

**Milestone 2 (End of Week 2):**
- All kernels implemented and tested independently
- Support for all required dtypes

**Milestone 3 (End of Week 3):**
- PyTorch dispatch integration complete
- Batch invariance tests passing
- Integration with model runner

**Milestone 4 (End of Week 4):**
- Performance optimization complete
- Documentation written
- Ready for production use

### 4.3 Testing Plan

**Unit Tests:**
- Each kernel tested independently with synthetic data
- Edge cases: empty tensors, single element, very large tensors
- Dtype coverage: float16, bfloat16, float32

**Integration Tests:**
- Batch invariance property verification
- Mode enable/disable state management
- Context manager nesting behavior

**End-to-End Tests:**
- Full model inference with deterministic mode
- Verify identical outputs for different batch sizes
- Performance regression tests

### 4.4 Dependencies

**External:**
- PyTorch >= 2.0 (for `torch.library.Library` API)
- Triton >= 2.1 (for persistent kernel support)
- CUDA GPU with compute capability >= 7.0

**Internal:**
- `model_runner.py` integration point
- Server argument parsing for `--enable-deterministic-inference`
- Attention backend configuration

---

## 5. Performance Characteristics

### Expected Performance

**Matmul Performance:**

Persistent kernel performance relative to cuBLAS:
- Small matrices (M, N, K < 512): 80-120% of cuBLAS (overhead amortized)
- Medium matrices (512 ≤ M, N, K < 2048): 60-80% of cuBLAS (determinism overhead)
- Large matrices (M, N, K ≥ 2048): 50-70% of cuBLAS (reduced parallelism)

**Log Softmax Performance:**

Three-pass approach is 20-40% slower than PyTorch fused kernel, but guarantees numerical stability.

**Overall Inference Impact:**

End-to-end inference with `--enable-deterministic-inference`:
- Throughput reduction: 15-30% depending on model architecture
- Latency increase: 10-20% for single batch
- Memory usage: Same as baseline (no additional buffers)

### Optimization Opportunities

**Future Work:**
1. Implement batch-aware persistent matmul to improve bmm performance
2. Add autotuning for block sizes based on input shape distribution
3. Fuse more operations (e.g., matmul + GELU for MLP)
4. Explore multi-stage pipelines for better occupancy

---

## 6. References

**Original Implementation:**
- https://github.com/thinking-machines-lab/batch_invariant_ops

**Related Documentation:**
- PyTorch dispatch documentation: https://pytorch.org/docs/stable/library.html
- Triton programming guide: https://triton-lang.org/main/programming-guide/index.html
- NVIDIA persistent kernels: https://developer.nvidia.com/blog/cooperative-groups/

**Code Locations:**
- Main implementation: `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- Public API: `python/sglang/srt/batch_invariant_ops/__init__.py`
- Tests: `test/srt/batch_invariant/test_batch_invariant_ops.py`
- Integration: `python/sglang/srt/model_executor/model_runner.py`
