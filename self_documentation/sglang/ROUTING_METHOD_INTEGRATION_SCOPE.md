# Integration Scope: Multiple Routing Methods for FP8 FlashInfer TRT-LLM MOE in SGLang

## Executive Summary

This document outlines the work required to integrate multiple routing methods for FP8 FlashInfer TRT-LLM MOE in SGLang, similar to the implementation in vLLM PR #27492. The changes will enable proper routing method selection for different model architectures (Qwen3, Qwen3-Next, DeepSeek, Llama4) when using the FlashInfer TRT-LLM backend.

## Current State Analysis

### What SGLang Currently Has

1. **FlashInfer TRT-LLM MOE Backend Support**
   - Location: `python/sglang/srt/layers/moe/utils.py:73-74, 212-218`
   - Backend selection via `should_use_flashinfer_trtllm_moe()`
   - Requires FlashInfer >= 0.2.9rc1

2. **FP8 Block Scale MOE Implementation**
   - Location: `python/sglang/srt/layers/quantization/fp8.py:1180-1239`
   - Uses `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`
   - Currently hardcoded to `routing_method_type=2` (DeepSeekV3)

3. **FP4 Block Scale MOE Implementation**
   - Location: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:989-1078`
   - Uses `flashinfer.fused_moe.trtllm_fp4_block_scale_moe`
   - Uses `RoutingMethodType.DeepSeekV3` from FlashInfer enum

4. **MxFP4 MOE Implementation**
   - Location: `python/sglang/srt/layers/quantization/mxfp4.py:620-670`
   - Uses `trtllm_fp4_block_scale_moe`
   - Uses `routing_method_type=1` (Renormalize)

5. **Model Support**
   - Qwen3 MoE: `python/sglang/srt/models/qwen3_moe.py`
   - Qwen3-Next: `python/sglang/srt/models/qwen3_next.py`
   - Both use `Qwen2MoeSparseMoeBlock` for MoE layers
   - Both use FusedMoE implementation via `get_moe_impl_class(quant_config)`

6. **TopK Configuration**
   - Location: `python/sglang/srt/layers/moe/topk.py:88-100`
   - Has `TopKConfig` dataclass with renormalize flag
   - No routing_method_type field currently

### What is Missing Compared to vLLM

1. **RoutingMethodType Enum in SGLang**
   - vLLM defines it in `vllm/model_executor/layers/fused_moe/config.py:94-111`
   - SGLang relies on FlashInfer's enum but doesn't define its own
   - Need to decide: import from FlashInfer or define locally

2. **Configurable routing_method_type Parameter**
   - FP8 implementation hardcodes routing_method_type=2
   - No parameter passing mechanism from model layer to kernel invocation
   - No model-specific routing method configuration

3. **Layer-Level routing_method_type Storage**
   - vLLM stores it in FusedMoE layer: `self.routing_method_type`
   - SGLang FusedMoE layer has no such field
   - Need to add this to `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

4. **Model-Specific Routing Method Assignment**
   - vLLM Qwen3/Qwen3-Next models set `routing_method_type=RoutingMethodType.RenormalizeNaive`
   - SGLang models don't specify routing method
   - Need to update Qwen3MoeSparseMoeBlock initialization

5. **Relaxed Kernel Constraints**
   - vLLM updated constraints: top_k <= 10, num_experts <= 512
   - SGLang may still have old constraints (need verification in FlashInfer)
   - This is mostly a FlashInfer kernel change, but need validation

6. **Optional num_expert_group and topk_group**
   - vLLM made these parameters optional (can be None)
   - SGLang FP8 implementation asserts they must not be None
   - Need to relax these constraints

7. **Dynamic Tile Size Calculation**
   - vLLM added imbalance factor of 1.3 to tile size calculation
   - SGLang may need similar update (need to check current implementation)

8. **Conditional Routing Logits Casting**
   - vLLM only casts to FP32 for DeepSeekV3 routing (routing_method_type=2)
   - SGLang always casts to FP32
   - Need conditional casting based on routing method

## Detailed Implementation Plan

### Phase 1: Add RoutingMethodType Support

#### Task 1.1: Define or Import RoutingMethodType Enum

**Option A: Import from FlashInfer (Recommended)**
- File: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:56`
- Already imports: `from flashinfer import RoutingMethodType, fp4_quantize`
- Action: Re-export from a central location for consistency

**Option B: Define Local Enum**
- File: Create `python/sglang/srt/layers/moe/routing_method.py`
- Mirror vLLM implementation
- Content:
```python
from enum import IntEnum

class RoutingMethodType(IntEnum):
    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    Unspecified = 6
```

**Recommendation**: Use Option A (import from FlashInfer) since:
- Already imported in FP4 implementation
- Ensures compatibility with FlashInfer kernel expectations
- Reduces duplication
- SGLang is tightly coupled to FlashInfer for TRT-LLM backend

**Files to modify**:
1. `python/sglang/srt/layers/moe/__init__.py` - Add re-export
2. All files importing routing method types

#### Task 1.2: Update TopKConfig to Include Routing Method

**File**: `python/sglang/srt/layers/moe/topk.py:88-100`

**Changes**:
```python
@dataclass
class TopKConfig:
    top_k: int
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    torch_native: bool = False
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False
    routing_method_type: Optional[int] = None  # NEW FIELD
```

**Investigation needed**:
- Determine if routing_method_type should be passed through TopKConfig
- Alternative: Pass directly to expert forward method
- Consider backward compatibility

### Phase 2: Update FusedMoE Layer

#### Task 2.1: Add routing_method_type to FusedMoE.__init__

**File**: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:123-145`

**Changes**:
```python
def __init__(
    self,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    layer_id: int,
    top_k: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    params_dtype: Optional[torch.dtype] = None,
    reduce_results: bool = False,
    quant_config: Optional[QuantizationConfig] = None,
    prefix: str = "",
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_presharded_weights: bool = False,
    inplace: bool = True,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
    use_weight_loader_fused: bool = False,
    with_bias=False,
    routing_method_type: int = 2,  # NEW PARAMETER, default DeepSeekV3
):
    super().__init__()
    # ... existing code ...
    self.routing_method_type = routing_method_type  # NEW FIELD
```

**Investigation needed**:
- Check all places where FusedMoE is instantiated
- Ensure default value (2 = DeepSeekV3) maintains backward compatibility
- Verify EP MOE layer also needs this parameter

#### Task 2.2: Store routing_method_type in FlashInferFP4MoE

**File**: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:989-1078`

**Changes**:
```python
class FlashInferFP4MoE(FusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # routing_method_type now inherited from parent
```

No additional changes needed if parent class stores it correctly.

### Phase 3: Update FP8 Quantization Layer

#### Task 3.1: Make routing_method_type Configurable in FP8 MOE

**File**: `python/sglang/srt/layers/quantization/fp8.py:1180-1239`

**Current code**:
```python
return trtllm_fp8_block_scale_moe(
    routing_logits=router_logits.to(torch.float32),  # Always FP32
    routing_bias=correction_bias,
    # ... other params ...
    routing_method_type=2,  # Hardcoded
    use_shuffled_weight=False,
)
```

**Updated code**:
```python
# Get routing_method_type from layer, default to DeepSeekV3
routing_method_type = getattr(layer, "routing_method_type", 2)

# Only cast to FP32 for DeepSeekV3 routing
router_logits_input = (
    router_logits.to(torch.float32)
    if routing_method_type == 2
    else router_logits
)

return trtllm_fp8_block_scale_moe(
    routing_logits=router_logits_input,
    routing_bias=correction_bias,
    # ... other params ...
    routing_method_type=routing_method_type,  # Use dynamic value
    use_shuffled_weight=False,
)
```

**Investigation needed**:
- Verify if other routing methods require specific precision
- Check if FlashInfer kernel handles different input precisions correctly
- Test performance impact of not casting to FP32

#### Task 3.2: Relax num_expert_group and topk_group Constraints

**File**: `python/sglang/srt/layers/quantization/fp8.py:1204-1207`

**Current code**:
```python
assert (
    topk_config.num_expert_group is not None
    and topk_config.topk_group is not None
), "Current trtllm_fp8_block_scale_moe kernel does not support these two arguments as None"
```

**Updated code**:
```python
# Make parameters optional, defaulting to 0 if None
num_expert_group = topk_config.num_expert_group if topk_config.num_expert_group is not None else 0
topk_group = topk_config.topk_group if topk_config.topk_group is not None else 0
```

**Investigation needed**:
- Verify FlashInfer kernel behavior when these are 0
- Check if this matches vLLM's approach
- Test with models that don't use grouped topk

### Phase 4: Update Model Implementations

#### Task 4.1: Update Qwen3MoeSparseMoeBlock

**File**: `python/sglang/srt/models/qwen3_moe.py:82-122`

**Current code**:
```python
self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    use_grouped_topk=False,
)

self.experts = get_moe_impl_class(quant_config)(
    num_experts=config.num_experts + get_global_server_args().ep_num_redundant_experts,
    top_k=config.num_experts_per_tok,
    layer_id=layer_id,
    hidden_size=config.hidden_size,
    intermediate_size=config.moe_intermediate_size,
    quant_config=quant_config,
    prefix=add_prefix("experts", prefix),
)
```

**Updated code**:
```python
from flashinfer import RoutingMethodType

self.topk = TopK(
    top_k=config.num_experts_per_tok,
    renormalize=config.norm_topk_prob,
    use_grouped_topk=False,
)

# Determine routing method type based on model config
# Qwen3 models use RenormalizeNaive when renormalize is True
routing_method_type = (
    RoutingMethodType.RenormalizeNaive
    if config.norm_topk_prob
    else RoutingMethodType.Default
)

self.experts = get_moe_impl_class(quant_config)(
    num_experts=config.num_experts + get_global_server_args().ep_num_redundant_experts,
    top_k=config.num_experts_per_tok,
    layer_id=layer_id,
    hidden_size=config.hidden_size,
    intermediate_size=config.moe_intermediate_size,
    quant_config=quant_config,
    prefix=add_prefix("experts", prefix),
    routing_method_type=routing_method_type,  # NEW PARAMETER
)
```

**Investigation needed**:
- Verify Qwen3 model's expected routing behavior
- Check if `norm_topk_prob` config field correctly indicates RenormalizeNaive
- Confirm vLLM's Qwen3 implementation details

#### Task 4.2: Update Qwen2MoeSparseMoeBlock (for Qwen3-Next)

**File**: `python/sglang/srt/models/qwen2_moe.py:150-174`

**Apply same changes as Task 4.1**

**Files that use Qwen2MoeSparseMoeBlock**:
1. `qwen2_moe.py` - Original implementation
2. `qwen3_next.py:41, 518, 672` - Imports and uses it

**Investigation needed**:
- Determine if Qwen3-Next uses different routing than Qwen3 MoE
- Check if both should use RenormalizeNaive

#### Task 4.3: Verify DeepSeek Models

**Files**:
1. `python/sglang/srt/models/deepseek_v2.py`
2. `python/sglang/srt/models/deepseek.py`

**Investigation needed**:
- Check if DeepSeek models explicitly set routing_method_type
- Verify default value (2 = DeepSeekV3) is correct for these models
- No changes needed if default is appropriate

#### Task 4.4: Add Llama4 Support (if needed)

**File**: `python/sglang/srt/models/llama4.py:75-120`

**Current code**:
```python
renormalize: bool,
# ...
self.topk = TopK(
    top_k=top_k,
    renormalize=False,
    use_grouped_topk=False,
    custom_routing_function=custom_routing_function,
)
# ...
apply_router_weight_on_input=True,
```

**Investigation needed**:
- Check if Llama4 MoE already exists in SGLang
- Verify if it needs RoutingMethodType.Llama4
- vLLM uses apply_router_weight_on_input=True for Llama4
- Determine if custom_routing_function is equivalent

**Updated code (if needed)**:
```python
from flashinfer import RoutingMethodType

routing_method_type = RoutingMethodType.Llama4

self.experts = FusedMoE(
    # ... existing params ...
    routing_method_type=routing_method_type,
    apply_router_weight_on_input=True,
)
```

### Phase 5: Update MxFP4 Implementation

#### Task 5.1: Make MxFP4 routing_method_type Configurable

**File**: `python/sglang/srt/layers/quantization/mxfp4.py:641-669`

**Current code**:
```python
trtllm_gen_output = trtllm_fp4_block_scale_moe(
    router_logits.to(torch.bfloat16),
    None,  # routing_bias
    # ... other params ...
    routing_method_type=1,  # Hardcoded Renormalize
    do_finalize=True,
)[0]
```

**Updated code**:
```python
# Get routing_method_type from layer, default to Renormalize
routing_method_type = getattr(layer, "routing_method_type", 1)

trtllm_gen_output = trtllm_fp4_block_scale_moe(
    router_logits.to(torch.bfloat16),
    None,  # routing_bias
    # ... other params ...
    routing_method_type=routing_method_type,  # Use dynamic value
    do_finalize=True,
)[0]
```

**Investigation needed**:
- Verify MxFP4 models' expected routing behavior
- Check if Renormalize (1) is always correct for MxFP4

### Phase 6: Update EP MOE Layer

#### Task 6.1: Add routing_method_type to EP MOE Layer

**File**: `python/sglang/srt/layers/moe/ep_moe/layer.py`

**Investigation needed**:
- Check if EP MOE implementation needs routing_method_type parameter
- Verify EP MOE uses same kernel invocations as standard MOE
- Grep for EP MOE usage of TRT-LLM kernels

**Search for**:
- `trtllm_fp8_block_scale_moe` usage
- `trtllm_fp4_block_scale_moe` usage
- EP MOE initialization

**Likely changes**: Similar parameter passing as in FusedMoE

### Phase 7: Testing and Validation

#### Task 7.1: Create Test Cases

**File**: Create `python/sglang/test/test_routing_methods.py`

**Test scenarios**:
1. Qwen3 MoE with RenormalizeNaive routing
2. Qwen3-Next with RenormalizeNaive routing
3. DeepSeek with DeepSeekV3 routing
4. Llama4 with Llama4 routing (if supported)
5. Verify backward compatibility with existing models
6. Test optional num_expert_group/topk_group

**Test structure**:
```python
import pytest
from flashinfer import RoutingMethodType

@pytest.mark.parametrize("model_type,expected_routing_method", [
    ("qwen3-moe", RoutingMethodType.RenormalizeNaive),
    ("qwen3-next", RoutingMethodType.RenormalizeNaive),
    ("deepseek-v3", RoutingMethodType.DeepSeekV3),
])
def test_routing_method_selection(model_type, expected_routing_method):
    # Test that correct routing method is selected
    pass

def test_fp8_routing_logits_casting():
    # Test conditional FP32 casting
    pass

def test_optional_expert_group_params():
    # Test None values for num_expert_group and topk_group
    pass
```

#### Task 7.2: Integration Testing

**Models to test**:
1. Qwen3-MoE-8x7B (or available variant)
2. Qwen3-Next-MoE (if available)
3. DeepSeek-V3 (verify no regression)

**Test dimensions**:
- Quantization: FP8, FP4, MxFP4
- Backend: flashinfer_trtllm
- Parallel: TP, EP, DP combinations
- Batch sizes: 1, 8, 32, 256

**Metrics to verify**:
- Correctness: output matches reference
- Performance: no regression compared to hardcoded routing
- Memory: no increase in memory usage

#### Task 7.3: Kernel Constraint Verification

**Investigate and test**:
1. Verify FlashInfer version supports top_k <= 10
2. Verify FlashInfer version supports num_experts <= 512
3. Test with models exceeding old constraints (top_k=9, num_experts=400)

**File to check**: `flashinfer` kernel documentation or source

### Phase 8: Documentation and Code Quality

#### Task 8.1: Update Documentation

**Files to update**:
1. `docs/backend_guide.md` (if exists) - Add routing method section
2. `python/sglang/srt/layers/moe/README.md` (if exists) - Document routing options
3. Model-specific docs for Qwen3/Qwen3-Next

**Content to add**:
- Explanation of routing methods
- When to use which routing method
- How to specify custom routing method
- Performance implications

#### Task 8.2: Add Inline Comments

**Files needing documentation**:
1. `fp8.py` - Explain routing_method_type logic
2. `fused_moe_triton/layer.py` - Document routing_method_type parameter
3. Model files - Explain routing method choice

**Example**:
```python
# Qwen3 models use RenormalizeNaive routing method which performs:
# TopK selection -> Softmax normalization -> Renormalize
# This differs from DeepSeekV3 which uses:
# Sigmoid -> RoutingBiasAdd -> Grouped TopK
routing_method_type = RoutingMethodType.RenormalizeNaive
```

#### Task 8.3: Add Type Hints

**Files needing type hints**:
1. All modified function signatures
2. New routing_method_type parameters

**Example**:
```python
from typing import Optional
from flashinfer import RoutingMethodType

def __init__(
    self,
    # ... existing params ...
    routing_method_type: Optional[int] = RoutingMethodType.DeepSeekV3,
):
```

## Code Similarity with vLLM

### High Similarity (>80%)

1. **RoutingMethodType Enum**
   - vLLM defines it locally
   - SGLang can import from FlashInfer (simpler)
   - Logic is identical

2. **FusedMoE Layer Parameter Addition**
   - vLLM: Add `routing_method_type` parameter
   - SGLang: Same parameter, same location
   - ~95% code reuse

3. **FP8 Kernel Invocation Updates**
   - vLLM: Conditional FP32 casting + dynamic routing_method_type
   - SGLang: Identical logic
   - ~90% code reuse

4. **Optional Parameter Handling**
   - vLLM: Made num_expert_group/topk_group optional
   - SGLang: Same approach needed
   - 100% code reuse

### Moderate Similarity (50-80%)

1. **Model-Specific Routing Assignment**
   - vLLM: Qwen3/Qwen3-Next set routing_method_type in model __init__
   - SGLang: Same location, but model structure differs slightly
   - ~70% concept reuse, needs adaptation

2. **TopKConfig Integration**
   - vLLM: May store in different location
   - SGLang: Has TopKConfig, needs routing_method_type field
   - ~60% similarity

### Low Similarity (<50%)

1. **EP MOE Integration**
   - vLLM: Different EP architecture
   - SGLang: Has unique EP MOE layer
   - ~30% concept reuse, significant investigation needed

2. **Model Registry**
   - vLLM: Different model registration system
   - SGLang: Different architecture
   - ~20% similarity

## Investigation Checklist

Before implementation, investigate:

### High Priority

- [ ] Verify FlashInfer version supports all routing methods
- [ ] Check FlashInfer kernel constraints (top_k, num_experts)
- [ ] Determine if TopKConfig or layer stores routing_method_type
- [ ] Verify EP MOE needs routing_method_type parameter
- [ ] Check if Qwen3 and Qwen3-Next both use RenormalizeNaive
- [ ] Test performance impact of not casting to FP32

### Medium Priority

- [ ] Verify backward compatibility with existing models
- [ ] Check if MxFP4 always uses Renormalize routing
- [ ] Determine default routing_method_type for new models
- [ ] Investigate tile size calculation differences
- [ ] Check if Llama4 MOE exists in SGLang

### Low Priority

- [ ] Compare dynamic vs static routing method selection
- [ ] Investigate if routing_method_type should be per-layer
- [ ] Check if routing method affects expert load balancing
- [ ] Verify if custom_routing_function is compatible

## Risk Assessment

### High Risk

1. **Breaking Changes**
   - Risk: New parameter breaks existing code
   - Mitigation: Use default value matching current behavior
   - Test: Comprehensive backward compatibility tests

2. **Performance Regression**
   - Risk: Dynamic routing method selection adds overhead
   - Mitigation: Use attribute lookup, no runtime logic
   - Test: Benchmark against hardcoded version

3. **Incorrect Routing for Existing Models**
   - Risk: Default routing_method_type=2 wrong for some models
   - Mitigation: Audit all model types, set correct defaults
   - Test: Validation against known-good outputs

### Medium Risk

1. **EP MOE Compatibility**
   - Risk: EP MOE layer needs different integration
   - Mitigation: Investigate EP MOE thoroughly before changes
   - Test: EP-specific test cases

2. **FlashInfer Version Compatibility**
   - Risk: Older FlashInfer versions don't support all routing methods
   - Mitigation: Add version check, fallback behavior
   - Test: Test with minimum required FlashInfer version

### Low Risk

1. **Documentation Gaps**
   - Risk: Users don't understand routing methods
   - Mitigation: Comprehensive documentation
   - Test: User acceptance testing

2. **Type Hint Issues**
   - Risk: Type checkers fail on new parameters
   - Mitigation: Add proper type hints from start
   - Test: Run mypy/pyright

## Timeline Estimation

### Optimistic (assuming high code reuse)

- Phase 1: 2 days (RoutingMethodType support)
- Phase 2: 1 day (FusedMoE layer)
- Phase 3: 2 days (FP8 quantization)
- Phase 4: 3 days (Model implementations)
- Phase 5: 1 day (MxFP4)
- Phase 6: 2 days (EP MOE investigation + implementation)
- Phase 7: 3 days (Testing)
- Phase 8: 1 day (Documentation)

**Total: 15 days**

### Realistic (accounting for investigation)

- Phase 1: 3 days (includes enum decision investigation)
- Phase 2: 2 days (includes backward compatibility testing)
- Phase 3: 3 days (includes constraint relaxation testing)
- Phase 4: 5 days (includes model-specific routing determination)
- Phase 5: 2 days (includes MxFP4 routing verification)
- Phase 6: 4 days (EP MOE investigation + implementation + testing)
- Phase 7: 5 days (comprehensive testing across models)
- Phase 8: 2 days (documentation + code review feedback)

**Total: 26 days**

### Pessimistic (if major issues found)

- Add 1 week for EP MOE complications
- Add 1 week for FlashInfer kernel issues
- Add 1 week for unexpected model compatibility issues

**Total: 47 days**

## Success Criteria

### Functional Requirements

1. Qwen3 MoE models use RenormalizeNaive routing
2. Qwen3-Next MoE models use RenormalizeNaive routing
3. DeepSeek models continue to work (DeepSeekV3 routing)
4. FP8 routing logits only cast to FP32 when needed
5. num_expert_group and topk_group can be None
6. All existing models maintain correctness

### Non-Functional Requirements

1. No performance regression (< 2% slower)
2. No memory increase
3. Code passes existing tests
4. New code has > 80% test coverage
5. Documentation covers all routing methods

## Open Questions

1. Should routing_method_type be stored in TopKConfig or FusedMoE layer?
   - vLLM: Stores in layer
   - Recommendation: Store in layer, pass to kernel directly

2. Should SGLang define its own RoutingMethodType enum or import from FlashInfer?
   - vLLM: Defines locally
   - Recommendation: Import from FlashInfer for consistency

3. How to handle models that need routing method inference?
   - Option A: Explicit configuration required
   - Option B: Infer from renormalize flag
   - Recommendation: Start with explicit, add inference later

4. Should routing_method_type be configurable via server args?
   - vLLM: No, model-specific
   - Recommendation: Model-specific, no server arg override

5. What is the priority of Llama4 MoE support?
   - Need stakeholder input
   - Recommendation: Phase 2 feature if Llama4 MOE models exist

## Files to Modify (Summary)

### Core Implementation (12 files)

1. `python/sglang/srt/layers/moe/__init__.py` - Export RoutingMethodType
2. `python/sglang/srt/layers/moe/topk.py` - Add routing_method_type to TopKConfig
3. `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` - Add parameter to FusedMoE
4. `python/sglang/srt/layers/quantization/fp8.py` - Make routing configurable
5. `python/sglang/srt/layers/quantization/mxfp4.py` - Make routing configurable
6. `python/sglang/srt/models/qwen3_moe.py` - Set RenormalizeNaive routing
7. `python/sglang/srt/models/qwen2_moe.py` - Set RenormalizeNaive routing
8. `python/sglang/srt/models/qwen3_next.py` - Verify routing (uses Qwen2MoeSparseMoeBlock)
9. `python/sglang/srt/layers/moe/ep_moe/layer.py` - Add parameter (if needed)
10. `python/sglang/srt/models/llama4.py` - Add Llama4 routing (if needed)
11. `python/sglang/srt/models/deepseek_v2.py` - Verify default routing
12. `python/sglang/srt/models/deepseek.py` - Verify default routing

### Testing (2 files)

1. Create `python/sglang/test/test_routing_methods.py` - New test file
2. Update existing MOE tests to cover routing methods

### Documentation (3+ files)

1. Update `README.md` - Mention routing method support
2. Update `docs/backend_guide.md` (if exists) - Document routing options
3. Add inline documentation in modified files

## Conclusion

Integrating multiple routing methods for FP8 FlashInfer TRT-LLM MOE in SGLang is a well-scoped task with high code reuse potential from the vLLM implementation. The main challenges are:

1. EP MOE integration (requires investigation)
2. Model-specific routing determination (requires testing)
3. Backward compatibility (requires careful default selection)

With proper investigation and testing, this can be completed in 3-5 weeks with high confidence in correctness and performance.
