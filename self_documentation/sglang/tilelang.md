# TileLang Support in SGLang – Current State Report

## Overview

TileLang is integrated into SGLang as an alternative attention kernel backend, primarily for DeepSeek models with Native Sparse Attention (NSA). It enables cross-platform compatibility, supporting GPU, HPU, and NPU devices.

### Key Integration Points

- **NSA backend**: Native Sparse Attention for DeepSeek V3.2/R1 models
- **Activation quantization kernels**: FP8 quantization
- **Sparse attention forward kernels**
- **FP8 indexing operations**

---

## Implementation Locations

### Core Kernel File

**File:** `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` (786 lines)

Implements three main kernel families:

1. **`act_quant_kernel`** (lines 40–88)
    - Block-wise activation quantization to FP8
    - Configurable group size (default 128)
    - Optional round scaling
    - Returns quantized tensor and scaling factors
2. **`fp8_index_kernel`** (lines 118–170)
    - FP8 precision index scoring for sparse attention
    - Computes Q@K logits with per-head/per-token scaling
    - Pipelined execution (2 stages)
3. **`sparse_attention_fwd_kernel_v1`** (lines 206–365) and **`sparse_attention_fwd_kernel_v2`** (lines 383–761)
    - V1: Basic sparse attention with configurable staging
    - V2: Optimized with double buffering, barrier synchronization, and producer-consumer thread specialization (three thread groups)
    - Both support causal masking and custom softmax scaling

**TileLang Kernel Configuration:**
```python
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}
```

---

## Integration in NSA Backend

**File:** `python/sglang/srt/layers/attention/nsa_backend.py`

TileLang is integrated as a backend option (lines 143–147):

- `_NSA_IMPL_T: TypeAlias = Literal["flashmla_sparse", "flashmla_kv", "fa3", "tilelang"]`
- `NSA_PREFILL_IMPL: _NSA_IMPL_T`
- `NSA_DECODE_IMPL: _NSA_IMPL_T`

**Usage Paths:**

- `forward_extend()` (line 794): Calls `_forward_tilelang()` for prefill when `NSA_PREFILL_IMPL == "tilelang"`
- `forward_decode()` (line 921): Calls `_forward_tilelang()` for decode when `NSA_DECODE_IMPL == "tilelang"`
- `_forward_tilelang()` (lines 1063–1079): Wrapper that calls `tilelang_sparse_fwd()`

**Platform-Specific Dispatch (`nsa_indexer.py`, lines 387, 470):**

- CUDA/default: Uses Triton `act_quant`
- HIP (AMD): Uses TileLang `act_quant`
- NPU: Uses custom ops, but imports TileLang's `fp8_index`

---

## Server Configuration

**File:** `python/sglang/srt/server_args.py`

Command-line arguments for TileLang backend (lines 245–246):

- `--nsa-prefill-backend tilelang`  (prefill stage)
- `--nsa-decode-backend tilelang`   (decode stage)

---

## Documentation

**File:** `docs/basic_usage/deepseek_v32.md` (line 56)

- TileLang is presented as a cross-platform backend:
  > tilelang: tilelang implementation that can run on GPU, HPU and NPU.

- Not the default for H200/B200:
    - H200 default: `flashmla_sparse` prefill + `fa3` decode
    - B200 default: `flashmla_kv` prefill + `flashmla_kv` decode

---

## Test Coverage

**File:** `test/srt/layers/attention/nsa/test_act_quant_triton.py` (282 lines)

Comprehensive benchmarks compare TileLang and Triton:

- Tests a wide range of tensor shapes: (128,512) to (4096,16384)
- Accuracy validation with configurable tolerances
- Performance benchmarking with CUDA graphs
- Tests with both `scale_fmt=None` and set

**Key Metrics Tracked:**

- Execution time (ms)
- Speedup ratio
- Max/mean diff between implementations
- Match percentage

---

## Additional References

- **Benchmark script:** `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`
    - Implements TileLang DeepGEMM FP8, adapted from upstream
    - Line 19: "Adapted from https://github.com/tile-ai/tilelang/..."

- **CUDA kernel:** `sgl-kernel/csrc/elementwise/topk.cu`
    - Lines 5, 76 indicate topk kernel was "adapted from tilelang to pure cuda"

---

## Current Limitations

1. **Not installed in this environment**: `ModuleNotFoundError: No module named 'tilelang'`
2. **Not default backend**: FlashMLA and FA3 used for H200/B200 for better performance
3. **NSA model limitation**: Only integrated for DeepSeek V3.2/R1 sparse attention
4. **HIP/NPU focus**: Primary use is for non-CUDA platforms (where FlashMLA/FA3 may be unavailable)

---

## Architecture Patterns

### Kernel Design

- Uses TileLang's symbolic programming (`T.symbolic()`)
- Explicit memory hierarchy management (shared/fragment allocation)
- Pipelined execution with configurable stages
- GEMM operations via `T.gemm()` intrinsics

### Integration Pattern

- Backend selection via server args sets global `NSA_PREFILL_IMPL`/`NSA_DECODE_IMPL`
- Attention backend dispatches to `_forward_tilelang()` wrapper
- Wrapper calls JIT-compiled TileLang kernels
- Platform-specific imports (conditional on `is_hip()`, `is_npu()`)

---

## Future Considerations

Potential areas for expansion, based on the file structure:

- More attention variants (currently only sparse causal supported)
- Additional quantization formats beyond FP8 E4M3
- Support for a broader range of models (currently only DeepSeek NSA)
- Performance tuning for specific hardware (kernel v2 offers optimization possibility)
