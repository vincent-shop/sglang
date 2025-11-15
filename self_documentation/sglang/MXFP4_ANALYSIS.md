# MXFP4 Analysis: SGLang vs ModelOpt

**Status**: SGLang has **PARTIAL MXFP4 support** - MoE only, missing linear layers
**GPT OSS Support**: ✅ Has MoE MXFP4 quantization via FlashInfer and Triton kernels

---

## Executive Summary

### What SGLang Currently Has

✅ **MXFP4 MoE Quantization** - Comprehensive support
- `Mxfp4MoEMethod`: Static MXFP4 quantization for MoE layers
- `Mxfp4DynamicQuantMoEMethod`: Dynamic MXFP4 quantization for MoE layers
- Multiple backend support:
  - FlashInfer MXFP4 kernel (Blackwell optimized)
  - Triton kernels (OpenAI implementation)
  - ROCm support via AITER library

❌ **MISSING: MXFP4 Linear Layer Quantization**
- No `Mxfp4LinearMethod` class
- Cannot quantize attention layers, FFN gate/up/down projections (non-MoE)
- Only MoE weights can be MXFP4 quantized

### What ModelOpt Provides

ModelOpt has **3 MXFP4 configurations**:
1. `MXFP4_DEFAULT_CFG` - W4A4 (weights + activations)
2. `W4A8_MXFP4_FP8_CFG` - W4A8 (MXFP4 weights + FP8 activations)
3. `MXFP4_MLP_WEIGHT_ONLY_CFG` - Weight-only MXFP4 for MLP layers

All support **full model quantization** including linear layers.

---

## Technical Comparison

### MXFP4 Format Specification

Both SGLang and ModelOpt use the same MXFP4 format:
- **Data type**: E2M1 (2-bit exponent, 1-bit mantissa)
- **Block size**: 32 elements (vs NVFP4's 16)
- **Scale format**: E8M0 (8-bit exponent, 0-bit mantissa)
- **Packing**: 2 MXFP4 values per uint8 byte
- **Total bits per weight**: 4 bits

```
Weight storage format:
weight:         [N, K/2]  uint8 (packed E2M1)
weight_scale:   [N, K/32] uint8 (E8M0 scales)
```

### SGLang MXFP4 MoE Implementation

#### Class 1: Mxfp4MoEMethod (Static Quantization)

**What it does**: Loads pre-quantized MXFP4 MoE weights

**Key Features**:
- **Weight format**: Uint8 packed MXFP4
- **Scale format**: Uint8 E8M0 (32-element blocks)
- **Bias support**: ✅ Yes (per-expert per-channel in BF16)
- **Backends**:
  - FlashInfer MXFP4 MoE (Blackwell optimized)
  - Triton kernels (OpenAI implementation)
  - Triton (fallback, dequantizes to BF16)

**FlashInfer Path** (`use_flashinfer=True`):
```python
# Weight shuffling for transposed MMA
gemm1_weights_shuffled = shuffle_matrix_a(w13_weight)
gemm1_scales_shuffled = shuffle_matrix_sf_a(w13_weight_scale)

# Swaps w1 and w3 for TRT-LLM compatibility
w13_weight_scale = swap_every_two_rows(w13_weight_scale)

# Runtime: Uses trtllm_fp4_block_scale_moe kernel
output = trtllm_fp4_block_scale_moe(
    router_logits,
    x_quant,           # Input quantized to MXFP8 or BF16
    x_scale,           # Input scales
    w13_weight,        # MXFP4 weights
    w13_weight_scale,  # E8M0 block scales
    w13_weight_bias,   # Per-expert per-channel bias
    gemm1_alpha,       # Hardcoded [1.702] * num_experts
    gemm1_beta,        # Hardcoded [1.0] * num_experts
    gemm1_clamp_limit, # Hardcoded [7.0] * num_experts
    ...
)
```

**Precision modes**:
- `default`: Quantizes input to MXFP8, uses FlashInfer MXFP4 kernel
- `bf16`: Keeps input in BF16, kernel handles quantization internally (better pipeline)

**Triton Kernels Path** (`use_triton_kernels=True`):
```python
# Uses OpenAI's triton_kernels implementation
# Weight swizzling for optimal memory access
w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
    layer.w13_weight, layer.w13_weight_scale, num_warps=8
)

# Creates precision config with flex context
w13_precision_config = PrecisionConfig(
    weight_scale=w13_scale,
    flex_ctx=FlexCtx(rhs_data=w13_flex)
)

# Runtime: Uses triton_kernels MoE
output = runner.run(dispatch_output, quant_info)
```

**Triton Fallback Path** (neither flashinfer nor triton_kernels):
```python
# Dequantizes to BF16 at load time
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

w13_weight = upcast_from_mxfp(
    layer.w13_weight, layer.w13_weight_scale,
    dtype=torch.bfloat16, axis=-1
)
# Then runs standard BF16 MoE
```

#### Class 2: Mxfp4DynamicQuantMoEMethod

**What it does**: Dynamically quantizes unquantized MoE weights at runtime

**Key Features**:
- **Input**: BF16/FP16 weights
- **Quantization**: On-the-fly during model loading
- **Backend**: ROCm only (AITER library)
- **Use case**: Models without pre-quantized checkpoints

```python
def process_weights_after_loading(self, layer):
    # Dynamic quantization using AITER
    w13, w13_mx_scales = self.mxfp4_quantize(layer.w13_weight.data)
    w2, w2_mx_scales = self.mxfp4_quantize(layer.w2_weight.data)

    # Uses dynamic_mxfp4_quant from AITER
    # Block size: 32 elements (1x32 quantization)
```

### ModelOpt MXFP4 Configurations

#### 1. MXFP4_DEFAULT_CFG (W4A4)

```python
{
    "*weight_quantizer": {
        "num_bits": (2, 1),  # E2M1
        "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        "enable": True,
    },
    "*input_quantizer": {
        "num_bits": (2, 1),  # E2M1
        "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        "enable": True,
    },
}
```

**Quantizes**:
- ✅ All linear layers (attention QKV, O, MLP gate/up/down)
- ✅ MoE expert layers
- ✅ Both weights AND activations

**Export format**: HuggingFace checkpoint with:
- Weight dtype: `uint8` (packed MXFP4)
- Scale dtype: `uint8` (E8M0)
- Config: `{"quant_method": "mxfp4", "block_size": 32}`

#### 2. W4A8_MXFP4_FP8_CFG (W4A8 Mixed Precision)

```python
{
    "*weight_quantizer": {
        "num_bits": (2, 1),  # MXFP4 weights
        "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        "enable": True,
    },
    "*input_quantizer": {
        "num_bits": (4, 3),  # FP8 E4M3 activations
        "axis": None,
    },
}
```

**Benefits**:
- Better accuracy than W4A4 (FP8 activations vs MXFP4)
- Still 4x weight compression
- ~2x speedup on Blackwell (vs unquantized)

**Use case**: Production deployments where accuracy matters

#### 3. MXFP4_MLP_WEIGHT_ONLY_CFG

```python
{
    "*mlp*weight_quantizer": {  # Only matches MLP layers
        "num_bits": (2, 1),
        "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        "enable": True,
        "pass_through_bwd": True,  # For QAT
    },
}
```

**Quantizes**:
- ✅ Only MLP/FFN layers (gate_proj, up_proj, down_proj)
- ❌ Skips attention layers (keeps QKV, O in higher precision)

**Benefits**:
- Better accuracy (attention is often more sensitive)
- Still significant memory savings (MLPs are 2/3 of model parameters)

---

## Missing Features in SGLang

### 1. MXFP4 Linear Layer Quantization (⭐⭐⭐⭐⭐ CRITICAL)

**What's missing**:
```python
# This class DOES NOT EXIST in SGLang
class Mxfp4LinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size, output_sizes, ...):
        # Create MXFP4 quantized weights
        weight = torch.zeros(output_size, input_size // 2, dtype=torch.uint8)
        weight_scale = torch.zeros(output_size, input_size // 32, dtype=torch.uint8)

    def apply(self, layer, x, bias=None):
        # MXFP4 GEMM for linear layers
        output = mxfp4_gemm(x, layer.weight, layer.weight_scale, ...)
```

**Impact**: Cannot quantize non-MoE models with MXFP4
- ❌ Attention layers (QKV projection, O projection)
- ❌ Standard MLP layers (in non-MoE models)
- ❌ LM head projection
- ❌ Vision encoder projections (for VLMs)

**Why it matters**:
- **GPT OSS**: Uses MXFP4 for MoE layers ONLY (because it's MoE-based)
- **Other models**: Need MXFP4 for all layers to get full benefit

**ModelOpt support**: ✅ Full support via quantization configs

### 2. W4A8 Mixed Precision (⭐⭐⭐⭐⭐ HIGH VALUE)

**What's missing**: MXFP4 weights + FP8 activations

**Current SGLang**: Only supports W4A4 (MXFP4 weights + MXFP4 activations)

**ModelOpt**: Has `W4A8_MXFP4_FP8_CFG` for mixed precision

**Benefits of W4A8**:
- **Better accuracy**: FP8 activations >> MXFP4 activations
- **Same memory**: Still 4x weight compression
- **Hardware support**: Blackwell has dedicated FP8 tensor cores

**Implementation needs**:
- Modify `Mxfp4MoEMethod` to support FP8 activation quantization
- Add FP8 activation scales to forward pass
- Update export format to include activation precision metadata

### 3. Selective Layer Quantization (⭐⭐⭐⭐)

**What's missing**: Ability to quantize only specific layer types

**ModelOpt example**:
```python
# Only quantize MLP layers, keep attention in BF16
config = MXFP4_MLP_WEIGHT_ONLY_CFG
model = mtq.quantize(model, config, forward_loop)
```

**Current SGLang workaround**:
```python
# Can use "ignored_layers" but requires manual specification
config = {
    "quant_method": "mxfp4",
    "ignored_layers": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]
}
```

**Better approach**: Pattern-based layer selection
```python
config = {
    "quant_method": "mxfp4",
    "quantize_patterns": ["*mlp*"],  # Only MLP layers
}
```

### 4. Dynamic Calibration Algorithms (⭐⭐⭐)

**What's missing**: Advanced calibration beyond "max"

**Current SGLang**: Uses "max" calibration (simple min/max range)

**ModelOpt algorithms**:
- `awq_lite`: Activation-aware weight quantization (fast)
- `awq_clip`: AWQ with clipping optimization
- `smoothquant`: Smooth activation outliers

**Impact**: Better MXFP4 accuracy, especially for attention-heavy models

**Note**: Current SGLang MXFP4 loads pre-quantized models, so calibration happens externally via ModelOpt

### 5. MXFP4 Attention Layer Support (⭐⭐⭐)

**What's missing**: MXFP4 quantization for attention ops

```python
# This is explicitly NOT implemented
if self.is_checkpoint_mxfp4_serialized:
    raise NotImplementedError("Mxfp4 attention layer is not implemented")
```

**Includes**:
- QKV projection quantization
- Attention output projection quantization
- KV cache quantization (MXFP4 format)

**ModelOpt**: Can quantize attention layers via `MXFP4_DEFAULT_CFG`

**Challenge**: Attention ops are more sensitive to quantization
- May need affine quantization (bias terms)
- May need rotation-based quantization

---

## Integration with ModelOpt

### Current Export Workflow

**ModelOpt → SGLang**:
```bash
# Step 1: Quantize with ModelOpt (MoE only currently works)
python -m modelopt.torch.quantization.hf_ptq \
    --model-path meta-llama/Llama-3.1-405B-Instruct \
    --config MXFP4_DEFAULT_CFG \
    --export-dir ./mxfp4_model

# Step 2: Deploy with SGLang
python -m sglang.launch_server \
    --model-path ./mxfp4_model \
    --quantization mxfp4
```

**What happens**:
1. ModelOpt quantizes ALL layers (linear + MoE)
2. Exports to HuggingFace format
3. SGLang loads MoE layers as MXFP4 ✅
4. SGLang **ignores** linear layer quantization ❌ (no LinearMethod)
5. Linear layers loaded as BF16 (fallback)

**Result**: Only MoE layers are quantized, rest is unquantized!

### What's Needed for Full Integration

#### Phase 1: Add Linear Layer Support (⭐⭐⭐⭐⭐)

```python
class Mxfp4LinearMethod(LinearMethodBase):
    """Linear method for MXFP4 quantization.

    Supports block-wise quantization with 32-element blocks and E8M0 scales.
    """

    def create_weights(self, layer, input_size, output_sizes, ...):
        output_size = sum(output_sizes)
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        block_size = 32

        # Packed MXFP4 weights (2 values per byte)
        weight = Parameter(torch.zeros(
            output_size, input_size // 2, dtype=weight_dtype
        ))

        # E8M0 block scales
        weight_scale = Parameter(torch.zeros(
            output_size, input_size // block_size, dtype=scale_dtype
        ))

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)

    def apply(self, layer, x, bias=None):
        # Option 1: Use FlashInfer MXFP4 GEMM
        if has_flashinfer_mxfp4_gemm():
            output = mxfp4_gemm(x, layer.weight, layer.weight_scale)

        # Option 2: Use Triton kernel
        elif has_triton_mxfp4_gemm():
            output = triton_mxfp4_gemm(x, layer.weight, layer.weight_scale)

        # Option 3: Dequantize and use BF16 GEMM (fallback)
        else:
            from triton_kernels.numerics_details.mxfp import upcast_from_mxfp
            weight_bf16 = upcast_from_mxfp(
                layer.weight, layer.weight_scale, dtype=torch.bfloat16
            )
            output = torch.matmul(x, weight_bf16.t())

        if bias is not None:
            output = output + bias
        return output
```

**Kernel requirements**:
- FlashInfer: Already has `mm_mxfp4` for MXFP4 GEMM
- Triton: Can use OpenAI's triton_kernels
- Fallback: Dequantize to BF16

**Effort**: Medium (2-3 weeks)
- Implement LinearMethod class
- Test on various model architectures
- Benchmark performance vs BF16

#### Phase 2: Add W4A8 Support (⭐⭐⭐⭐⭐)

```python
class Mxfp4Fp8LinearMethod(LinearMethodBase):
    """MXFP4 weights + FP8 activations."""

    def create_weights(self, layer, ...):
        # MXFP4 weights (same as before)
        ...

        # FP8 activation scales
        input_scale = Parameter(torch.ones(1, dtype=torch.float32))
        layer.register_parameter("input_scale", input_scale)

    def apply(self, layer, x, bias=None):
        # Quantize activation to FP8
        x_fp8, x_scale = quantize_fp8(x, layer.input_scale)

        # MXFP4 x FP8 GEMM
        output = mxfp4_fp8_gemm(
            x_fp8, layer.weight, layer.weight_scale, x_scale
        )
        return output
```

**Kernel requirements**:
- New kernel for MXFP4 weights × FP8 activations
- May need to implement or wait for FlashInfer support

**Effort**: High (1-2 months)
- Kernel development/integration
- Calibration workflow for activation scales
- Export format updates

#### Phase 3: Add Selective Quantization (⭐⭐⭐⭐)

```python
class Mxfp4Config(QuantizationConfig):
    def __init__(
        self,
        quantize_patterns: Optional[List[str]] = None,  # NEW
        ignore_patterns: Optional[List[str]] = None,
        ...
    ):
        self.quantize_patterns = quantize_patterns or ["*"]  # All by default
        self.ignore_patterns = ignore_patterns or []

    def should_quantize_layer(self, prefix: str) -> bool:
        # Check if matches quantize patterns
        if not any(fnmatch(prefix, p) for p in self.quantize_patterns):
            return False

        # Check if matches ignore patterns
        if any(fnmatch(prefix, p) for p in self.ignore_patterns):
            return False

        return True
```

**Example usage**:
```python
# Only quantize MLP layers
config = Mxfp4Config(quantize_patterns=["*mlp*"])

# Quantize everything except attention
config = Mxfp4Config(ignore_patterns=["*attn*"])
```

**Effort**: Low (1 week)
- Update config class
- Add pattern matching logic
- Update documentation

---

## Recommended Implementation Plan

### Immediate (Phase 1): Linear Layer Support - 2-3 weeks

**Goal**: Enable MXFP4 quantization for non-MoE models

**Tasks**:
1. Implement `Mxfp4LinearMethod` class
2. Add weight creation and loading logic
3. Implement apply() with FlashInfer/Triton/fallback paths
4. Test on Llama-2, Llama-3 (non-MoE models)
5. Benchmark vs BF16 and NVFP4

**Deliverables**:
- `Mxfp4LinearMethod` in `mxfp4.py`
- Unit tests for weight loading
- Performance benchmarks
- Documentation update

### Short-term (Phase 2): W4A8 Mixed Precision - 1-2 months

**Goal**: Support MXFP4 weights + FP8 activations

**Tasks**:
1. Design W4A8 export format (coordinate with ModelOpt)
2. Implement `Mxfp4Fp8LinearMethod` and `Mxfp4Fp8MoEMethod`
3. Add FP8 activation quantization in forward pass
4. Integrate or implement MXFP4×FP8 kernels
5. Test accuracy vs W4A4

**Deliverables**:
- W4A8 linear and MoE methods
- Export format specification
- Accuracy benchmarks
- User guide for when to use W4A8

### Medium-term (Phase 3): Advanced Features - 1-2 months

**Goal**: Complete MXFP4 feature parity with NVFP4

**Tasks**:
1. Add selective layer quantization patterns
2. Implement MXFP4 attention layer support
3. Add MXFP4 KV cache quantization
4. Integrate ModelOpt calibration algorithms (AWQ, etc.)

**Deliverables**:
- Pattern-based layer selection
- MXFP4 attention layers
- MXFP4 KV cache
- Calibration algorithm support

---

## Performance Expectations

### MXFP4 vs NVFP4

| Aspect | MXFP4 | NVFP4 |
|--------|-------|-------|
| Block size | 32 | 16 |
| Scale format | E8M0 (8-bit) | FP8 E4M3 (8-bit) |
| Scale granularity | Coarser | Finer |
| Accuracy | Potentially worse | Potentially better |
| Hardware support | Blackwell | Blackwell |
| Kernel maturity | Less mature | More mature |

**General guidance**:
- **NVFP4**: Better for accuracy-critical workloads (default recommendation)
- **MXFP4**: Good for MoE layers, may have better kernel support (OpenAI, etc.)

### Current SGLang MXFP4 Performance

**GPT OSS with MXFP4 MoE**:
- FlashInfer MXFP4 MoE: ~3-4x speedup vs BF16 MoE
- Memory: 4x reduction for MoE weights
- Accuracy: Minimal degradation (<1% perplexity)

**Note**: This is ONLY for MoE layers. Non-MoE layers still run in BF16.

---

## Conclusion

### Key Findings

1. **Current Status**: SGLang has excellent MXFP4 MoE support, zero MXFP4 linear support

2. **GPT OSS**: Works well because it's MoE-based, but still runs attention/embeddings in BF16

3. **ModelOpt Integration Gap**: ModelOpt can quantize full models, SGLang can only use MoE quantization

4. **Biggest Missing Feature**: `Mxfp4LinearMethod` for non-MoE layers (⭐⭐⭐⭐⭐)

5. **High-Value Addition**: W4A8 mixed precision (MXFP4 weights + FP8 activations)

### Prioritized Recommendations

**P0 (Critical)**: Add `Mxfp4LinearMethod`
- Enables full-model MXFP4 quantization
- Required for non-MoE models
- ~2-3 weeks effort

**P1 (High Value)**: Add W4A8 support
- Significantly better accuracy than W4A4
- Leverages Blackwell FP8 tensor cores
- ~1-2 months effort

**P2 (Nice to Have)**: Selective layer quantization
- Easier accuracy tuning
- Pattern-based configuration
- ~1 week effort

### Next Steps

1. **Immediate**: Implement `Mxfp4LinearMethod` for full model support
2. **Short-term**: Add W4A8 mixed precision for better accuracy
3. **Medium-term**: Add attention layer and KV cache MXFP4 support
4. **Long-term**: Integrate ModelOpt calibration algorithms (AWQ, SmoothQuant)

---

## References

- SGLang MXFP4 Implementation: `python/sglang/srt/layers/quantization/mxfp4.py`
- ModelOpt Config: `TensorRT-Model-Optimizer/modelopt/torch/quantization/config.py`
- FlashInfer MXFP4 MoE: `trtllm_fp4_block_scale_moe` kernel
- OpenAI Triton Kernels: `triton_kernels` package
- MXFP4 Format Spec: Microscaling Formats specification
