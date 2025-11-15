# ModelOpt Features Analysis for SGLang Integration

This document analyzes NVIDIA ModelOpt's capabilities and identifies opportunities for deeper SGLang integration.

## Executive Summary

**FP4 Support Status**: ✅ **COMPREHENSIVE** - ModelOpt has extensive FP4 support with multiple variants and advanced algorithms.

**Current SGLang Integration**: Basic FP8/FP4 quantization via export workflow
**Opportunity**: Many advanced features could be integrated to enhance SGLang's quantization capabilities

---

## Current SGLang ModelOpt Integration

### What SGLang Currently Supports

1. **FP8 Quantization** (`ModelOptFp8Config`)
   - Per-tensor weight and activation scaling
   - Static quantization (scales from calibration)
   - FP8 KV cache quantization
   - Export to HuggingFace format

2. **FP4 Quantization** (`ModelOptFp4Config`)
   - NVFP4 block-wise quantization (16-element blocks)
   - Per-tensor global scales + FP8 block scales
   - MoE support with multiple backends
   - Export to HuggingFace format

3. **Workflow**
   - Quantize → Export → Deploy pattern
   - Checkpoint save/restore for reuse
   - Automatic export during model loading

---

## ModelOpt FP4 Capabilities (Comprehensive Overview)

### 1. FP4 Quantization Formats

ModelOpt supports **TWO DISTINCT FP4 formats**:

#### **NVFP4** (2-bit exponent, 1-bit mantissa)
```python
# Available configs in ModelOpt:
NVFP4_DEFAULT_CFG                    # Basic NVFP4 quantization
NVFP4_AWQ_LITE_CFG                   # NVFP4 + AWQ Lite calibration
NVFP4_AWQ_CLIP_CFG                   # NVFP4 + AWQ Clip calibration
NVFP4_AWQ_FULL_CFG                   # NVFP4 + Full AWQ calibration
NVFP4_AFFINE_KV_CFG                  # NVFP4 KV cache with affine quantization
NVFP4_KV_CFG                         # NVFP4 KV cache quantization
NVFP4_FP8_MHA_CONFIG                 # NVFP4 weights + FP8 MHA
NVFP4_KV_ROTATE_CFG                  # NVFP4 KV with rotation
NVFP4_SVDQUANT_DEFAULT_CFG          # NVFP4 + SVDQuant algorithm
W4A8_NVFP4_FP8_CFG                  # NVFP4 weights + FP8 activations
NVFP4_MLP_WEIGHT_ONLY_CFG           # NVFP4 for MLP layers only
NVFP4_MLP_ONLY_CFG                  # NVFP4 for MLP layers (full)
```

**Key Features**:
- Block size: 16 elements
- Block scales: FP8 E4M3
- Global scales: FP32
- Two-level scaling: `alpha = input_scale * weight_scale_2`

#### **MXFP4** (Microscaling FP4)
```python
# Available configs in ModelOpt:
MXFP4_DEFAULT_CFG                    # Basic MXFP4 quantization
W4A8_MXFP4_FP8_CFG                  # MXFP4 weights + FP8 activations
MXFP4_MLP_WEIGHT_ONLY_CFG           # MXFP4 for MLP layers only
```

**Key Features**:
- Block size: 32 elements (vs NVFP4's 16)
- Scale bits: (8, 0) - 8-bit scales
- Dynamic block quantization
- Different scale format than NVFP4

### 2. Advanced Calibration Algorithms

ModelOpt provides multiple calibration algorithms that SGLang doesn't currently leverage:

#### **AWQ (Activation-aware Weight Quantization)**
```python
def awq(model, forward_loop, ...):
    """
    Full AWQ algorithm with per-channel scaling
    - Analyzes activation magnitudes
    - Scales weights based on activation importance
    - Preserves important channels
    """
```

**Variants in ModelOpt**:
- `awq` - Full AWQ with grid search
- `awq_lite` - Fast AWQ without search (90% quality, 10x faster)
- `awq_clip` - AWQ with clipping optimization

**SGLang Integration Opportunity**: ⭐⭐⭐⭐⭐
- Could significantly improve FP4 accuracy
- Relatively easy to integrate (use ModelOpt's calibration, then export)
- Major user value: better quality quantization

#### **SmoothQuant**
```python
def smoothquant(model, forward_loop, alpha=1.0):
    """
    Smooths activation outliers for better quantization
    - Migrates quantization difficulty from activations to weights
    - Applies channel-wise scaling
    """
```

**SGLang Integration Opportunity**: ⭐⭐⭐⭐
- Useful for INT8 quantization
- Could improve FP8 quantization for outlier-heavy models
- Requires activation statistics collection

#### **SVDQuant**
```python
def svdquant(model, forward_loop, ...):
    """
    SVD-based quantization for improved accuracy
    - Decomposes outlier channels
    - Handles structured quantization errors
    """
```

**SGLang Integration Opportunity**: ⭐⭐⭐
- Advanced technique, complex integration
- Best for extremely low precision (W2A8, W3A8)
- May be overkill for FP4/FP8

#### **Rotation-based Quantization**
```python
NVFP4_KV_ROTATE_CFG  # Applies rotation to KV cache
```

**SGLang Integration Opportunity**: ⭐⭐⭐
- Improves KV cache quantization
- Requires special handling in attention kernels
- Medium complexity

### 3. Mixed Precision Quantization

#### **W4A8 Quantization**
```python
W4A8_NVFP4_FP8_CFG    # FP4 weights + FP8 activations
W4A8_MXFP4_FP8_CFG    # MXFP4 weights + FP8 activations
W4A8_AWQ_BETA_CFG     # INT4 weights + FP8 activations
```

**Current SGLang Status**: ❌ Not supported
- SGLang only supports uniform quantization (W4A4, W8A8)
- W4A8 could provide better speedup/accuracy tradeoff

**Integration Opportunity**: ⭐⭐⭐⭐⭐
- High value: better than W4A4, more memory-efficient than W8A8
- Requires kernel support (partially available in FlashInfer)
- ModelOpt handles export format

### 4. Selective Layer Quantization

ModelOpt supports **fine-grained layer exclusion**:

```python
NVFP4_MLP_WEIGHT_ONLY_CFG    # Only quantize MLP layers
NVFP4_MLP_ONLY_CFG           # Only quantize MLP, skip attention
FP8_AFFINE_KV_CFG            # Only quantize KV cache with affine
```

**Current SGLang Status**: ⚠️ Partial support
- SGLang supports basic layer exclusion via `ignore` field
- No support for MLP-only or attention-only quantization

**Integration Opportunity**: ⭐⭐⭐⭐
- Easy to integrate (just export with right config)
- Could improve accuracy for attention-heavy workloads
- Example: Quantize MLP to FP4, keep attention in FP8

### 5. AutoQuantize (Automated Mixed Precision)

```python
model, search_state = mtq.auto_quantize(
    model,
    constraints={"auto_quantize_bits": 4.8},  # Effective bits target
    quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
    data_loader=calib_dataloader,
    forward_step=forward_step,
    loss_func=loss_func,
)
```

**What it does**:
- Automatically searches for best per-layer quantization precision
- Balances accuracy vs. model size
- Example: 4.8-bit target = some layers FP8, some FP4, some unquantized

**Current SGLang Status**: ❌ Not supported

**Integration Opportunity**: ⭐⭐⭐⭐⭐
- **HIGHEST VALUE** feature for users
- Eliminates manual tuning
- Requires search infrastructure (could run offline, export result)
- Perfect for production deployments

### 6. KV Cache Quantization Variants

ModelOpt offers multiple KV cache strategies:

```python
FP8_KV_CFG              # FP8 KV cache (simple)
FP8_AFFINE_KV_CFG       # FP8 KV + affine quantization (better accuracy)
NVFP4_KV_CFG            # FP4 KV cache
NVFP4_AFFINE_KV_CFG     # FP4 KV + affine quantization
NVFP4_KV_ROTATE_CFG     # FP4 KV + rotation (best accuracy)
```

**Affine Quantization**: Uses bias terms for better range coverage
```python
quantized = (value - bias) * scale
# vs standard: quantized = value * scale
```

**Current SGLang Status**: ⚠️ Partial support
- Only supports basic FP8 KV cache
- No affine quantization
- No FP4 KV cache

**Integration Opportunity**: ⭐⭐⭐⭐
- Affine quantization could improve long-context quality
- FP4 KV cache = 4x memory reduction (vs FP8 2x)
- Requires attention kernel changes

### 7. Quantization-Aware Training (QAT)

ModelOpt supports QAT for accuracy recovery:

```python
# After PTQ, refine with QAT
model = mtq.quantize(model, config, forward_loop)
# ... train model for a few steps
```

**Current SGLang Status**: ❌ Not supported (out of scope)

**Integration Opportunity**: ⭐⭐
- Not applicable to SGLang (inference-only)
- Users can do QAT with ModelOpt externally, then export to SGLang

---

## Features NOT Currently in SGLang

### High Priority (⭐⭐⭐⭐⭐)

1. **AWQ Calibration for FP4**
   - **What**: Activation-aware weight quantization
   - **Why**: Dramatically improves FP4 accuracy (often 2-3% perplexity improvement)
   - **Effort**: Medium (use ModelOpt's awq_lite, export to SGLang)
   - **Status**: ✅ ModelOpt has full implementation

2. **AutoQuantize (Mixed Precision Search)**
   - **What**: Automatically find best per-layer precision
   - **Why**: Eliminates manual tuning, optimal accuracy/size tradeoff
   - **Effort**: High (need to run search, export mixed-precision model)
   - **Status**: ✅ ModelOpt has full implementation

3. **W4A8 Mixed Precision**
   - **What**: FP4 weights + FP8 activations
   - **Why**: Better than W4A4 accuracy, faster than W8A8
   - **Effort**: High (need kernel support)
   - **Status**: ✅ ModelOpt exports it, kernels partially available

### Medium Priority (⭐⭐⭐⭐)

4. **SmoothQuant for FP8**
   - **What**: Smooth activation outliers before quantization
   - **Why**: Better FP8 quality for models with outliers (e.g., BERT, T5)
   - **Effort**: Medium (use ModelOpt calibration, export)
   - **Status**: ✅ ModelOpt has implementation

5. **Selective Layer Quantization**
   - **What**: MLP-only, attention-only, etc.
   - **Why**: Better accuracy by keeping sensitive layers in higher precision
   - **Effort**: Low (just export with right config)
   - **Status**: ✅ ModelOpt supports via configs

6. **Affine KV Cache Quantization**
   - **What**: KV cache quantization with bias terms
   - **Why**: Better long-context quality
   - **Effort**: Medium (need attention kernel changes)
   - **Status**: ✅ ModelOpt exports it

### Lower Priority (⭐⭐⭐)

7. **MXFP4 Format Support**
   - **What**: Alternative FP4 format (32-element blocks)
   - **Why**: May have better quality/speed tradeoff for some models
   - **Effort**: High (need new kernels)
   - **Status**: ✅ ModelOpt supports it

8. **FP4 KV Cache**
   - **What**: Quantize KV cache to FP4
   - **Why**: 4x memory reduction vs unquantized, 2x vs FP8
   - **Effort**: High (need attention kernel changes)
   - **Status**: ✅ ModelOpt supports it

9. **SVDQuant**
   - **What**: SVD-based outlier handling
   - **Why**: Best for ultra-low precision (W2A8)
   - **Effort**: Very High (complex algorithm)
   - **Status**: ✅ ModelOpt has implementation

---

## Recommended Integration Roadmap

### Phase 1: Quick Wins (1-2 weeks)

1. **Add AWQ calibration support** (⭐⭐⭐⭐⭐)
   ```python
   # In modelopt_quantize_and_export.py
   model = mtq.quantize(model, mtq.NVFP4_AWQ_LITE_CFG, forward_loop)
   ```
   - Expose `--calibration-algorithm` flag: `max`, `awq_lite`, `awq_clip`
   - Update docs with accuracy comparisons
   - No kernel changes needed!

2. **Document selective layer quantization** (⭐⭐⭐⭐)
   - Show users how to use `NVFP4_MLP_ONLY_CFG`
   - Provide accuracy/speed tradeoffs
   - Add example configs to docs

### Phase 2: Major Features (1-2 months)

3. **Implement W4A8 support** (⭐⭐⭐⭐⭐)
   - Add `W4A8_NVFP4_FP8_CFG` export support
   - Integrate FlashInfer's W4A8 kernels
   - Update SGLang's linear layers to handle mixed precision
   - Document when to use W4A8 vs W4A4 vs W8A8

4. **Add AutoQuantize workflow** (⭐⭐⭐⭐⭐)
   ```bash
   python examples/modelopt_auto_quantize.py \
       --model-path meta-llama/Llama-3.2-70B \
       --target-bits 4.5 \
       --export-dir ./auto_quantized
   ```
   - Create separate script for AutoQuantize
   - Save search results for reproducibility
   - Export mixed-precision config to SGLang

### Phase 3: Advanced Features (2-3 months)

5. **Affine KV cache quantization** (⭐⭐⭐⭐)
   - Modify attention kernels to support bias terms
   - Add config flag for affine vs standard KV quant
   - Benchmark long-context quality improvements

6. **SmoothQuant integration** (⭐⭐⭐⭐)
   - Add SmoothQuant calibration option for FP8
   - Update export to preserve smooth scales
   - Document use cases (outlier-heavy models)

---

## Specific ModelOpt Features to Leverage

### 1. Calibration Data Handling

ModelOpt has excellent calibration utilities:

```python
from modelopt.torch.utils import create_data_loader

# ModelOpt can use various datasets
calib_loader = create_data_loader(
    dataset_name="cnn_dailymail",  # or "wikitext", "c4", etc.
    batch_size=1,
    num_samples=512,
)
```

**SGLang Opportunity**: Use ModelOpt's data loading for consistent calibration

### 2. Export Flexibility

ModelOpt supports multiple export formats:

```python
# Unified HF format (SGLang uses this)
export_hf_checkpoint(model, export_dir)

# TensorRT-LLM format (legacy)
export_tensorrt_llm_checkpoint(model, ...)
```

**Current SGLang**: ✅ Already uses unified HF format

### 3. Distributed Quantization

ModelOpt supports multi-GPU quantization:

```python
# Automatically handles TP/PP during quantization
quantized_model = mtq.quantize(
    model,
    config,
    forward_loop,
    # Works with torch.distributed
)
```

**SGLang Opportunity**: Could speed up quantization of large models

---

## FP4 Support Confirmation

### ✅ YES, ModelOpt FP4 support is COMPREHENSIVE

**Evidence**:

1. **Two FP4 formats supported**: NVFP4 and MXFP4
2. **13+ FP4 configurations** covering various use cases
3. **Multiple calibration algorithms**: max, AWQ (lite/clip/full), SVDQuant
4. **Special variants**:
   - MLP-only quantization
   - KV cache quantization (with rotation, affine)
   - Mixed precision (W4A8)
   - MHA-specific configs
5. **Production-ready**: Used for DeepSeek-R1, Llama 4, etc.
6. **Export support**: HF format, TensorRT-LLM format
7. **Hardware optimized**: Tuned for Blackwell (GB200, B100, B200)

**What SGLang is missing**:
- Only uses `NVFP4_DEFAULT_CFG` (basic max calibration)
- Doesn't leverage advanced algorithms (AWQ, SVDQuant)
- No mixed precision support
- No specialized configs (MLP-only, etc.)

---

## Conclusion

### Key Takeaways

1. **FP4 Status**: ModelOpt's FP4 support is **comprehensive and production-ready**
   - SGLang uses only a small subset of available features
   - Many opportunities for accuracy/speed improvements

2. **Low-Hanging Fruit**:
   - AWQ calibration (huge accuracy wins, easy integration)
   - Selective layer quantization (documentation update)
   - Mixed precision configs (expose existing ModelOpt configs)

3. **High-Impact Features**:
   - AutoQuantize (automated mixed precision)
   - W4A8 support (better tradeoff than W4A4)
   - Affine KV cache (long-context quality)

4. **Integration Strategy**:
   - Phase 1: Expose existing ModelOpt features (calibration algorithms, configs)
   - Phase 2: Add new inference paths (W4A8, affine KV cache)
   - Phase 3: Build advanced workflows (AutoQuantize, distributed quantization)

### Next Steps

1. **Immediate**: Add AWQ calibration support to `modelopt_quantize_and_export.py`
2. **Short-term**: Document selective layer quantization patterns
3. **Medium-term**: Implement W4A8 mixed precision
4. **Long-term**: Build AutoQuantize workflow for SGLang

---

## References

- ModelOpt Repo: `/Users/vincentzhong/src/github.com/sgl-project/sglang/TensorRT-Model-Optimizer`
- Config definitions: `modelopt/torch/quantization/config.py`
- Calibration algorithms: `modelopt/torch/quantization/model_calib.py`
- LLM PTQ examples: `examples/llm_ptq/`
- SGLang integration: `python/sglang/srt/layers/quantization/modelopt_quant.py`
