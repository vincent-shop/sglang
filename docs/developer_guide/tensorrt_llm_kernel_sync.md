# TensorRT-LLM Kernel Sync Guide

This document tracks kernels in SGLang that are adapted from TensorRT-LLM and provides guidance for syncing with upstream.

## Current State

**TensorRT-LLM Submodule Version**: v1.2.0rc5 (commit `246a8775712f1e1503933a6665e824d3c680c461`, Dec 12, 2025)

**CUTLASS Versions**:
- `sgl-kernel/CMakeLists.txt`: SHA `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` (~v4.2.0-pre)
- `python/pyproject.toml`: `nvidia-cutlass-dsl==4.3.0`
- FlashInfer requires: `nvidia-cutlass-dsl>=4.3.2`

---

## Files Adapted from TensorRT-LLM

### 1. MoE TopK Softmax/Sigmoid Kernels

| SGLang File | TRT-LLM Source | Version Gap |
|-------------|----------------|-------------|
| `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu` | `cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu` | **v0.7.1 → v1.2.0** |
| `sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu` | `cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu` | **v0.7.1 → v1.2.0** |

**Notes**:
- These were originally adapted via vLLM
- SGLang version: Simple softmax/topk kernels (~700 lines)
- TRT-LLM current: Full MoE GEMM with CUTLASS (~4700 lines), min-latency routing, CuTe tensors
- **Major update needed** - significant architectural changes

---

### 2. FP8 GEMM Kernels

| SGLang File | TRT-LLM Source | Version Gap |
|-------------|----------------|-------------|
| `sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu` | `cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_template.h` | **v0.16.0 → v1.2.0** |
| | `cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm89.h` | |
| | `cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm90.h` | |

**Missing in SGLang**:
- `fp8_rowwise_gemm_kernel_template_sm100.h` - **Blackwell SM100 support**

**SGLang currently has**: SM89 + SM90 templates
**TRT-LLM now has**: SM89 + SM90 + **SM100** (Blackwell)

---

### 3. Fused QK Norm RoPE Kernel

| SGLang File | TRT-LLM Source | Status |
|-------------|----------------|--------|
| `sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu` | `cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu` | Relatively current |

**Notes**:
- Both use similar structure
- TRT-LLM has 2025 copyright, `TRTLLM_NAMESPACE_BEGIN` macros
- Minor differences in includes and namespace handling

---

### 4. DeepSeek v3 Router GEMM Kernels

| SGLang File | TRT-LLM Source | Status |
|-------------|----------------|--------|
| `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` | `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` | Current |
| `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` | `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` | Current |
| `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` | `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu` | Current |
| `sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu` | `cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu` | SHA 619709fc |

**Notes**:
- Both have similar PTX assembly for FMA operations
- Appears relatively synced

---

### 5. Quantization Utilities

| SGLang File | TRT-LLM Source | Status |
|-------------|----------------|--------|
| `sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh` | `cpp/tensorrt_llm/kernels/quantization.cuh` | Referenced (main) |

---

### 6. CUTLASS Extensions

| SGLang File | TRT-LLM Source | Version |
|-------------|----------------|---------|
| `sgl-kernel/csrc/cutlass_extensions/gemm/gemm_with_epilogue_visitor.h` | `cpp/tensorrt_llm/cutlass_extensions/.../gemm_with_epilogue_visitor.h` | SHA be178810 |
| `sgl-kernel/csrc/cutlass_extensions/gemm/gemm_universal_base_compat.h` | `cpp/tensorrt_llm/cutlass_extensions/.../gemm_universal_base_compat.h` | SHA be178810 |
| `sgl-kernel/csrc/cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h` | `cpp/tensorrt_llm/cutlass_extensions/.../epilogue_per_row_per_col_scale.h` | SHA be178810 |

**Notes**:
- These were copied from an old commit
- May need updates for CUTLASS 4.x compatibility

---

## Blackwell (SM100/SM103/SM120) Kernel Inventory in TensorRT-LLM

### Dedicated SM1XX Kernel Files

| File | SM Version | Purpose |
|------|------------|---------|
| `cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm100.h` | SM100 | FP8 rowwise GEMM for Blackwell |
| `cutlass_kernels/fp4_gemm/nvfp4_nvfp4_gemm_template_sm100.h` | SM100 | NVFP4 GEMM |
| `cutlass_kernels/fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h` | SM120 | NVFP4 GEMM (GeForce Blackwell) |
| `cutlass_kernels/fp4_gemm/mxfp8_mxfp4_gemm_template_sm100.h` | SM100 | MXFP8→MXFP4 mixed precision GEMM |
| `cutlass_kernels/fp8_blockscale_gemm/6kd_blockwise_gemm/sm120_fp8_gemm_1d2d.cuh` | SM120 | FP8 blockscale GEMM |
| `cutlass_kernels/fp8_blockscale_gemm/6kd_blockwise_gemm/sm120_utils.cuh` | SM120 | SM120 utilities |
| `cutlass_kernels/allreduce_gemm/allreduce_gemm_impl_sm100.h` | SM100 | AllReduce + GEMM fused kernel |
| `cutlass_kernels/allreduce_gemm/kernel/sm100_gemm_allreduce_tma_warpspecialized.hpp` | SM100 | TMA warp-specialized AllReduce GEMM |
| `cutlass_kernels/allreduce_gemm/epilogue/sm100_visitor_allreduce_tma_warpspecialized.hpp` | SM100 | SM100 epilogue visitor |

### trtllmGenKernels (Precompiled Cubins)

TensorRT-LLM ships precompiled CUDA binary (cubin) kernels for various SM architectures:

| Category | File | SM100 Cubins | SM103 Cubins |
|----------|------|--------------|--------------|
| Batched GEMM | `trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/KernelMetaInfo.h` | ~1347 refs | Many |
| GEMM | `trtllmGenKernels/gemm/trtllmGen_gemm_export/KernelMetaInfo.h` | ~327 refs | Some |
| Gated Act GEMM | `trtllmGenKernels/gemmGatedAct/trtllmGen_gatedAct_export/KernelMetaInfo.h` | ~39 refs | Some |

### Key Blackwell-Specific Features in TRT-LLM

1. **FP4/NVFP4 GEMM** - Block-scaled FP4 operations for SM100/SM120
2. **MXFP8 to MXFP4 conversion** - Mixed precision quantization on SM100
3. **TMA Warp-Specialized kernels** - New Blackwell memory access patterns
4. **AllReduce GEMM Fusion** - Communication + compute overlap on SM100
5. **SM120-specific GEMM** - Optimized for GeForce Blackwell (RTX 50 series)
6. **SM103 kernel variants** - B300 GPU specific optimizations

### Total SM1XX References

```
SM100/SM103/SM120 references in TensorRT-LLM kernels: 1,786 matches across 20+ files
```

---

## Gap Analysis: SGLang vs TensorRT-LLM

### Missing Blackwell Support in SGLang

| Feature | TRT-LLM | SGLang | Priority |
|---------|---------|--------|----------|
| SM100 FP8 GEMM | ✅ `fp8_rowwise_gemm_kernel_template_sm100.h` | ❌ Only SM89/SM90 | High |
| SM100 NVFP4 GEMM | ✅ `nvfp4_nvfp4_gemm_template_sm100.h` | ❌ Missing | High |
| SM120 FP8 blockscale | ✅ `sm120_fp8_gemm_1d2d.cuh` | ❌ Missing | Medium |
| SM100 AllReduce GEMM | ✅ Full implementation | ❌ Missing | Medium |
| SM103 kernel variants | ✅ Precompiled cubins | ❌ Missing | Low (B300 specific) |
| TMA warp-specialized | ✅ Full support | ⚠️ Partial (via FlashInfer) | Medium |
| Modern MoE GEMM | ✅ CUTLASS-based, 4700 lines | ⚠️ Legacy v0.7.1 style | High |

### Current Blackwell Path in SGLang

SGLang currently handles Blackwell through:
1. **FlashInfer CuTe DSL** - `flashinfer.cute_dsl.blockscaled_gemm` for SM100/SM103 MoE
2. **sgl-kernel CMakeLists.txt** - Compiles SM100a/SM103a/SM120a with CUDA 12.8+/13.0+
3. **Python dispatch** - `is_sm100_supported()` treats SM100/SM103 the same (major=10)

---

## Recommendations

### Short-term (High Priority)

1. **Bump CUTLASS version in CMakeLists.txt**
   - Current: SHA `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` (~v4.2.0-pre)
   - Recommended: v4.3.0+ for SM103 C++ support
   - Location: `sgl-kernel/CMakeLists.txt` line 50

2. **Add SM100 FP8 GEMM kernel**
   - Port `fp8_rowwise_gemm_kernel_template_sm100.h` from TRT-LLM

3. **Update MoE kernels**
   - Current v0.7.1 code is significantly outdated
   - Consider if modern CUTLASS-based approach is needed

### Medium-term

4. **Add SM100 AllReduce GEMM**
   - Useful for multi-GPU Blackwell deployments

5. **Update CUTLASS extensions**
   - Current SHA be178810 is old
   - May have compatibility issues with CUTLASS 4.x

### Long-term

6. **Evaluate trtllmGenKernels cubins**
   - TRT-LLM has extensive precompiled kernel coverage
   - May be worth integrating for guaranteed performance

---

## File Locations Reference

### SGLang Kernel Sources
```
sgl-kernel/
├── csrc/
│   ├── moe/
│   │   ├── moe_topk_softmax_kernels.cu     # TRT-LLM v0.7.1
│   │   ├── moe_topk_sigmoid_kernels.cu     # TRT-LLM v0.7.1
│   │   └── fused_qknorm_rope_kernel.cu     # TRT-LLM main
│   ├── gemm/
│   │   ├── fp8_gemm_kernel.cu              # TRT-LLM v0.16.0
│   │   ├── dsv3_router_gemm_*.cu           # TRT-LLM main
│   │   └── dsv3_fused_a_gemm.cu            # TRT-LLM SHA 619709fc
│   ├── cutlass_extensions/
│   │   ├── gemm/*.h                        # TRT-LLM SHA be178810
│   │   └── epilogue/*.h                    # TRT-LLM SHA be178810
│   └── expert_specialization/
│       └── es_sm100_mxfp8_blockscaled_*.cuh # TRT-LLM main (refs)
└── CMakeLists.txt                          # CUTLASS SHA 57e3cfb...
```

### TensorRT-LLM Kernel Sources
```
TensorRT-LLM/cpp/tensorrt_llm/kernels/
├── cutlass_kernels/
│   ├── fp8_rowwise_gemm/
│   │   ├── fp8_rowwise_gemm_kernel_template_sm89.h
│   │   ├── fp8_rowwise_gemm_kernel_template_sm90.h
│   │   └── fp8_rowwise_gemm_kernel_template_sm100.h  # NEW for Blackwell
│   ├── fp4_gemm/
│   │   ├── nvfp4_nvfp4_gemm_template_sm100.h         # NEW for Blackwell
│   │   └── nvfp4_nvfp4_gemm_template_sm120.h         # NEW for GeForce Blackwell
│   ├── fp8_blockscale_gemm/
│   │   └── 6kd_blockwise_gemm/
│   │       ├── sm120_fp8_gemm_1d2d.cuh               # NEW for SM120
│   │       └── sm120_utils.cuh
│   ├── allreduce_gemm/
│   │   ├── allreduce_gemm_impl_sm100.h               # NEW for Blackwell
│   │   └── kernel/sm100_gemm_allreduce_tma_warpspecialized.hpp
│   └── moe_gemm/
│       └── moe_kernels.cu                            # Significantly updated
├── dsv3MinLatencyKernels/
│   ├── dsv3RouterGemm.cu
│   └── dsv3FusedAGemm.cu
├── fusedQKNormRopeKernel.cu
└── quantization.cuh
```

---

## Version History

| Date | Change |
|------|--------|
| 2025-12-19 | Initial document created |
| | Identified 12 files adapted from TensorRT-LLM |
| | Documented Blackwell kernel gaps |
| | Bumped nvidia-cutlass-dsl to 4.3.0 |
