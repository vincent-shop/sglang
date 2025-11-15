# MOE Runner System

## Overview

The MOE (Mixture of Experts) runner system is a pluggable execution engine for running expert computations in transformer models with MoE layers. It provides multiple backend implementations optimized for different hardware and quantization configurations, with a flexible registry system for format transformations between dispatch, execution, and combine stages.

## Architecture

### High-Level Pipeline

```
Token Dispatch → Pre-Permute → Expert Execution → Post-Permute → Combine Results
     │              │                 │                │              │
     └──────────────┴─────────────────┴────────────────┴──────────────┘
                              OR
                      Fused Operation (optimized path)
```

The MOE runner sits between the token dispatcher and the combiner, handling the actual expert computations. It can either:
1. Use a **three-stage pipeline** (pre-permute → run → post-permute)
2. Use a **fused operation** that combines all stages for better performance

### Core Components

#### 1. MoeRunner (`runner.py`)

Main orchestrator that manages the execution flow.

**Key responsibilities:**
- Select appropriate runner backend based on configuration
- Check for fused operation availability and use it if present
- Fall back to three-stage pipeline otherwise
- Handle environment-based feature toggles

**Code flow:**
```python
def run(dispatch_output, quant_info):
    if fused_func exists:
        return fused_func(dispatch_output, quant_info, config)

    # Three-stage pipeline
    runner_input = pre_permute(dispatch_output)
    runner_output = runner_core.run(runner_input, quant_info)
    combine_input = post_permute(runner_output)
    return combine_input
```

**Location:** `python/sglang/srt/layers/moe/moe_runner/runner.py:25-86`

#### 2. Base Classes and Registries (`base.py`)

Abstract base classes and registry system for extensibility.

**Core abstractions:**

- `MoeRunnerConfig` - Configuration for runner behavior (activation, quantization, scaling)
- `RunnerInput` - Input format to runner backend (abstract)
- `RunnerOutput` - Output format from runner backend (abstract)
- `MoeQuantInfo` - Quantization metadata (weights, scales, precision configs)
- `MoeRunnerCore` - Abstract runner backend interface

**Registry pools:**

- `FusedOpPool` - Registers end-to-end fused operations
  - Key: `(a2a_backend_name, runner_backend_name)`
  - Bypasses permute stages for performance

- `PermuteMethodPool` - Registers format transformation functions
  - Pre-permute: `(dispatch_format, runner_backend)` → transformation function
  - Post-permute: `(runner_backend, combine_format)` → transformation function

**Decorators:**
- `@register_fused_func(a2a_backend, runner_backend)` - Register fused operation
- `@register_pre_permute(dispatch_format, runner_backend)` - Register input transformation
- `@register_post_permute(runner_backend, combine_format)` - Register output transformation

**Location:** `python/sglang/srt/layers/moe/moe_runner/base.py:1-287`

#### 3. Runner Backends

Three specialized backends implementing `MoeRunnerCore`:

##### Triton Backend (`triton.py`)

Standard Triton-based MOE execution using custom fused kernels.

**Input format:** `TritonRunnerInput`
- hidden_states: Input activations
- topk_weights: Router weights for top-k experts
- topk_ids: Expert IDs selected per token
- sorted_token_ids: Token ordering for efficient batching
- expert_ids: Expert indices
- num_tokens_post_padded: Padding information

**Quantization support:**
- FP8 W8A8 (weights and activations)
- INT8 W8A8, W8A16
- INT4 W4A16
- Per-channel and block-wise quantization

**Execution flow:**
1. First GEMM: `hidden_states @ w13 → intermediate_cache1`
2. Activation: SiLU/GELU with optional alpha/clamp (`silu_and_mul`, `gelu_and_mul`)
3. Second GEMM: `intermediate_cache2 @ w2 → intermediate_cache3`
4. Reduce: Sum across top-k experts with scaling

**Optimization paths:**
- Top-k=1 with scaling=1.0: Direct write to output
- Top-k=2 with scaling=1.0: Simple tensor addition
- Small batch (M≤32): Torch compile reduction
- Large batch: Triton reduction kernel
- HIP backend: Special handling with aiter or vllm_ops

**Fused operation:** `@register_fused_func("none", "triton")` combines all stages for standard dispatch format.

**Location:** `python/sglang/srt/layers/moe/moe_runner/triton.py:1-451`

##### Triton Kernels Backend (`triton_kernels.py`)

External triton_kernels package integration for specialized hardware.

**Input format:** `TritonKernelsRunnerInput`
- hidden_states: Input activations
- routing_data: Routing metadata from triton_kernels
- gather_indx: Gather indices for token reordering
- scatter_indx: Scatter indices for result accumulation

**Quantization support:**
- Flexible precision configuration via `PrecisionConfig`
- Support for biases (w13_bias, w2_bias)
- Global expert indexing for distributed setups

**Execution flow:**
- Single kernel call to `triton_kernel_fused_experts` or `triton_kernel_fused_experts_with_bias`
- Handles gather/scatter internally
- Optional bias addition
- Optional router weight application on input
- Alpha scaling and clamping for GEMM1

**Special features:**
- `no_combine` mode: Returns per-expert outputs without reduction
- `apply_router_weight_on_input`: Apply routing weights before or after expert computation
- `routed_scaling_factor`: Additional scaling for routed experts

**Location:** `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py:1-195`

##### Deep GEMM Backend (`deep_gemm.py`)

High-performance FP8 grouped GEMM backend with two execution modes.

**Input format:** `DeepGemmRunnerInput`
- hidden_states: FP8 input activations
- hidden_states_scale: Per-token group scales
- use_masked_gemm: Boolean for execution mode selection
- masked_m: Mask for valid tokens (masked mode)
- expected_m: Expected M dimension (masked mode)
- m_indices: Expert indices per token (contiguous mode)

**Execution modes:**

1. **Contiguous GEMM** (`_run_contiguous_gemm`)
   - Used with `deepep_normal` dispatch format
   - Tokens pre-sorted and packed by expert
   - Uses `grouped_gemm_nt_f8f8bf16_contig`
   - More efficient for distributed EP (expert parallelism)

2. **Masked GEMM** (`_run_masked_gemm`)
   - Used with `standard` and `deepep_ll` dispatch formats
   - Tokens organized in groups with masks
   - Uses `grouped_gemm_nt_f8f8bf16_masked`
   - Better memory access patterns for certain workloads

**Quantization:**
- FP8 E4M3 for activations
- FP8 for weights
- Per-token group scaling (block_size=128)
- Optional E8M0 scale format (UE8M0)
- TMA-aligned scales for Hopper architecture

**Execution flow:**
```
1. GroupGemm-0: hidden_states @ w13 → gateup_output
2. Activation: silu_and_mul with optional fusion
3. Quantize: down_input → FP8 with scales
4. GroupGemm-1: down_input @ w2 → down_output
```

**Optimizations:**
- Fast activation path: Fused quantization with SiLU (`SGLANG_MASKED_GEMM_FAST_ACT`)
- E8M0 scale format: Compact scale representation
- Copy engine control: `forbid_copy_engine_usage` for kernel coordination
- Tensor disposal: Explicit memory cleanup

**Location:** `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:1-590`

### Format Transformations

The system supports multiple dispatch and combine formats with registered transformations:

#### Standard Format

**Dispatch → Triton:**
- Extract topk_output (weights, ids)
- Call `moe_align_block_size` to align tokens to block size
- Generate sorted_token_ids, expert_ids, num_tokens_post_padded
- Store config in running_state

**Triton → Standard:**
- Pass through hidden_states
- No additional transformation needed

**Dispatch → Deep GEMM:**
- Call `moe_ep_deepgemm_preprocess` for reordering
- Quantize to FP8 with per-token group scales
- Generate masked_m, expected_m, src2dst indices
- Store metadata in running_state

**Deep GEMM → Standard:**
- Call `post_reorder_triton_kernel` to restore token order
- Apply topk_weights for combining experts
- Apply routed_scaling_factor if configured

#### Triton Kernels Format

**Dispatch → Triton Kernels:**
- Verify format is TritonKernelTopKOutput
- Extract routing_data, gather_indx, scatter_indx
- Direct pass-through of pre-computed indices

**Triton Kernels → Standard:**
- Apply routed_scaling_factor if configured and not in no_combine mode
- Wrap in StandardCombineInput

#### DeepEP Formats

Two specialized formats for expert parallelism:

**DeepEP-LL (Low Latency):**
- Pre-permute: Pass through FP8 hidden states with scales
- Post-permute: Preserve grouped output with topk metadata

**DeepEP-Normal:**
- Pre-permute: Call `ep_scatter` to distribute tokens across experts
- Post-permute: Call `ep_gather` to collect results with weighted reduction

### Configuration

`MoeRunnerConfig` controls runner behavior:

**MOE parameters:**
- `num_experts`: Total experts in model
- `num_local_experts`: Experts on current device
- `hidden_size`: Model hidden dimension
- `intermediate_size_per_partition`: Expert FFN size
- `top_k`: Number of experts per token
- `num_fused_shared_experts`: Shared experts count
- `layer_id`: Layer index for debugging

**Runner configuration:**
- `activation`: "silu" or "gelu"
- `apply_router_weight_on_input`: Apply routing weights early vs late
- `inplace`: Reuse input buffer for output
- `no_combine`: Return per-expert outputs without reduction
- `routed_scaling_factor`: Scale factor for routed expert outputs
- `gemm1_alpha`: Alpha parameter for GEMM1 activation scaling
- `gemm1_clamp_limit`: Clamp limit for GEMM1 outputs

### Quantization Support

Different backends support different quantization schemes:

**Triton Backend:**
- FP8 W8A8: Float8 weights and activations
- INT8 W8A8: INT8 weights and activations
- INT8 W8A16: INT8 weights, INT16 activations
- INT4 W4A16: INT4 weights, INT16 activations
- Per-channel or block-wise quantization
- Dynamic activation scaling

**Triton Kernels Backend:**
- Flexible precision via `PrecisionConfig`
- Supports arbitrary weight/activation bit-widths
- Hardware-specific optimizations

**Deep GEMM Backend:**
- FP8 E4M3 only (weights and activations)
- Per-token group scaling (block_size=128)
- Optional E8M0 scale format for Hopper
- TMA-aligned scales for optimal memory access

## Key Design Patterns

### 1. Strategy Pattern

Multiple runner backends implement the same `MoeRunnerCore` interface, allowing runtime selection based on hardware and configuration.

### 2. Registry Pattern

`FusedOpPool` and `PermuteMethodPool` enable registration of transformations without modifying core code. New backends can register their own transformations.

### 3. Pipeline with Bypass

The three-stage pipeline (pre → run → post) provides flexibility, while fused operations bypass stages for performance when possible.

### 4. Type-Safe Formats

Dataclasses with `@property runner_backend` ensure format compatibility at the type level and provide type guards for safe casting.

### 5. Running State

Dictionary passed through pipeline stages enables stateful transformations without modifying data structures.

## Performance Optimizations

### Fused Operations

Registered with `@register_fused_func`, these combine dispatch → run → combine into a single operation:
- Eliminates intermediate data transformations
- Reduces memory allocations
- Better kernel fusion opportunities

**Example:** `fused_experts_none_to_triton` in `triton.py:326-359`

### Kernel Selection

**Triton backend** dynamically selects reduction strategy:
- Top-k=1: No reduction needed
- Top-k=2: Simple addition
- Small batch: Torch compile
- Large batch: Triton kernel

**Deep GEMM backend** selects execution mode:
- Contiguous mode: Pre-sorted tokens
- Masked mode: Grouped with masks

### Memory Management

- Explicit tensor disposal: `dispose_tensor()` releases memory early
- In-place operations: Reuse buffers when safe
- Optimal padding: Environment-controlled padding for alignment
- Scale format optimization: E8M0 for compact scales

### Hardware-Specific Paths

- CUDA: sgl_kernel implementations
- HIP: vllm_ops or aiter alternatives
- CPU with AMX: Specialized path
- NPU: Device-specific optimizations

## Environment Variables

- `SGLANG_CI_DISABLE_MOE_FUSED_FUNC`: Disable fused operations (testing)
- `SGLANG_MOE_USE_AITER`: Use aiter library on HIP
- `SGLANG_MOE_PADDING`: Enable padding for alignment
- `SGLANG_MASKED_GEMM_FAST_ACT`: Fast activation path in Deep GEMM

## Usage Example

```python
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend

# Create configuration
config = MoeRunnerConfig(
    num_experts=8,
    num_local_experts=8,
    hidden_size=4096,
    intermediate_size_per_partition=14336,
    top_k=2,
    activation="silu",
    routed_scaling_factor=1.0,
)

# Initialize runner with backend
runner = MoeRunner(
    runner_backend=MoeRunnerBackend.TRITON,
    config=config,
)

# Execute MOE computation
combine_input = runner.run(
    dispatch_output=dispatch_output,  # From token dispatcher
    quant_info=quant_info,            # Quantization metadata
)
```

## Adding a New Backend

1. **Define data structures:**
   ```python
   @dataclass
   class MyRunnerInput(RunnerInput):
       @property
       def runner_backend(self) -> MoeRunnerBackend:
           return MoeRunnerBackend.MY_BACKEND
   ```

2. **Implement runner core:**
   ```python
   class MyRunnerCore(MoeRunnerCore):
       def run(self, runner_input, quant_info, running_state):
           # Your implementation
           return MyRunnerOutput(...)
   ```

3. **Register transformations:**
   ```python
   @register_pre_permute("standard", "my_backend")
   def pre_permute_standard_to_my_backend(dispatch_output, ...):
       return MyRunnerInput(...)

   @register_post_permute("my_backend", "standard")
   def post_permute_my_backend_to_standard(runner_output, ...):
       return StandardCombineInput(...)
   ```

4. **Optional fused operation:**
   ```python
   @register_fused_func("none", "my_backend")
   def fused_experts_none_to_my_backend(dispatch_output, ...):
       # Optimized end-to-end implementation
       return StandardCombineInput(...)
   ```

5. **Update runner.py:**
   ```python
   elif runner_backend.is_my_backend():
       self.runner_core = MyRunnerCore(config)
   ```

## Integration Points

### Upstream: Token Dispatcher

The runner receives `DispatchOutput` from the token dispatcher, which contains:
- `hidden_states`: Input activations
- `topk_output`: Router decisions (weights, expert IDs)
- Format-specific metadata

### Downstream: Combiner

The runner produces `CombineInput` for the combiner, which contains:
- `hidden_states`: Expert outputs (combined or per-expert)
- Optional metadata for final reduction

## Testing Considerations

- **Fused vs Non-Fused:** Test both paths by toggling `SGLANG_CI_DISABLE_MOE_FUSED_FUNC`
- **Format Combinations:** Verify all registered pre/post permute functions
- **Quantization Modes:** Test each quantization scheme with appropriate hardware
- **Top-K Values:** Validate behavior for top_k=1, 2, and higher
- **Edge Cases:** Empty experts, padding, small/large batch sizes
- **Memory:** Check for leaks with explicit tensor disposal

## References

- Base MOE layer: `python/sglang/srt/layers/moe/`
- Token dispatcher: `python/sglang/srt/layers/moe/token_dispatcher/`
- Triton kernels: `python/sglang/srt/layers/moe/fused_moe_triton/`
- Deep GEMM kernels: `python/sglang/srt/layers/moe/ep_moe/kernels.py`
- Quantization: `python/sglang/srt/layers/quantization/`
