# LayerCommunicator System Documentation

## Overview

The `LayerCommunicator` system manages distributed communication patterns and tensor scatter/gather operations across model layers in SGLang. It orchestrates data movement between different parallelism groups (tensor parallel, data parallel, attention groups) while handling MoE (Mixture of Experts) layers and dense layers with different scatter modes.

**Location**: `python/sglang/srt/layers/communicator.py`

## Core Concepts

### ScatterMode

Defines how data is distributed across ranks in distributed training/inference scenarios.

**Definition** (`communicator.py:69-86`):
```python
class ScatterMode(Enum):
    SCATTERED = auto()      # Data split across ranks: [a, b, c, d]
    TP_ATTN_FULL = auto()   # Full data within TP attention group: [ab, ab, cd, cd]
    FULL = auto()           # Full data on all ranks: [abcd, abcd, abcd, abcd]
```

**Example scenario** (TP=4, DP=2, enable-dp-attention, sequences a,b,c,d):
- `SCATTERED`: Each rank has one sequence: [a, b, c, d]
- `TP_ATTN_FULL`: Ranks within TP attention group have shared data: [ab, ab, cd, cd]
- `FULL`: All ranks have all data: [abcd, abcd, abcd, abcd]

**Usage**: The system transitions between these modes throughout the forward pass to optimize communication.

### LayerScatterModes

Tracks scatter modes for different stages within a single transformer layer.

**Definition** (`communicator.py:105-169`):
```python
@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode       # Data mode entering the layer
    attn_mode: ScatterMode              # Data mode during attention (always TP_ATTN_FULL)
    mlp_mode: ScatterMode               # Data mode during MLP/MoE
    middle_residual_mode: ScatterMode   # Residual data mode between attention and MLP
    layer_output_mode: ScatterMode      # Data mode exiting the layer
```

**Initialization** (`communicator.py:114-123`):
- Created via `LayerScatterModes.init_new(layer_id, num_layers, is_layer_sparse, is_previous_layer_sparse)`
- Automatically computes optimal modes based on layer type (sparse/dense) and position

**Mode computation rules**:
1. **layer_input_mode**: If first layer, uses `model_input_output()`, else inherits previous layer's output mode
2. **attn_mode**: Always `TP_ATTN_FULL` (attention requires full data within TP group)
3. **mlp_mode**:
   - Sparse (MoE) layers: `SCATTERED` if using A2A backend or FP4 allgather, else `FULL`
   - Dense layers: `SCATTERED` if `moe_dense_tp_size==1`, else `FULL`
4. **middle_residual_mode**: `SCATTERED` if mlp is scattered, else `TP_ATTN_FULL`
5. **layer_output_mode**: Last layer uses `model_input_output()`, else follows mlp_mode patterns

### CommunicateContext

Stores parallelism configuration and group sizes used throughout communication.

**Definition** (`communicator.py:377-407`):
```python
@dataclass
class CommunicateContext:
    process_group_sizes: Dict[ScatterMode, int]  # Group size for each scatter mode
    attn_tp_rank: int                             # Rank within attention TP group
    attn_tp_size: int                             # Size of attention TP group
    attn_dp_size: int                             # Size of attention DP group
    tp_size: int                                  # Total tensor parallel size
```

**Initialization** (`communicator.py:390-407`):
- Queries distributed configuration via `get_attention_tp_rank()`, `get_attention_tp_size()`, etc.
- Maps scatter modes to process group sizes:
  - `SCATTERED`: 1 (single rank)
  - `TP_ATTN_FULL`: attn_tp_size
  - `FULL`: tp_size

## Main Class: LayerCommunicator

The central orchestrator that manages all communication within a transformer layer.

**Definition** (`communicator.py:175-375`):

### Initialization

**Constructor** (`communicator.py:176-217`):
```python
def __init__(
    self,
    layer_scatter_modes: LayerScatterModes,
    input_layernorm: torch.nn.Module,
    post_attention_layernorm: torch.nn.Module,
    allow_reduce_scatter: bool = False,
    is_last_layer: bool = False,
)
```

**Parameters**:
- `layer_scatter_modes`: Defines scatter modes for this layer
- `input_layernorm`: RMSNorm applied before attention
- `post_attention_layernorm`: RMSNorm applied before MLP
- `allow_reduce_scatter`: Enable reduce-scatter optimization (requires model support)
- `is_last_layer`: Whether this is the final transformer layer

**Internal state**:
- `_context`: CommunicateContext instance
- `_communicate_simple_fn`: Function for simple scatter/gather operations
- `_communicate_with_all_reduce_and_layer_norm_fn`: Function combining all-reduce + layernorm
- `_communicate_summable_tensor_pair_fn`: Function for handling (hidden_states, residual) pairs

### Core Methods

#### prepare_attn

Prepares hidden states before attention computation.

**Signature** (`communicator.py:241-299`):
```python
def prepare_attn(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    qaunt_format: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Operations**:
1. Handles residual connection: `residual = hidden_states` if None
2. Applies input layernorm (with optional allreduce fusion)
3. Optionally supports MXFP4 quantization fusion (ROCm GFX95)
4. Communicates hidden states via `_communicate_simple_fn` to reach `attn_mode`
5. Returns transformed `(hidden_states, residual)`

**Special case**: `prepare_attn_and_capture_last_layer_outputs` variant captures outputs for EAGLE3 speculative decoding (`communicator.py:219-239`)

#### prepare_mlp

Prepares hidden states before MLP/MoE computation.

**Signature** (`communicator.py:301-317`):
```python
def prepare_mlp(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    cache=None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Operations**:
1. Performs all-reduce on attention output if needed
2. Adds residual connection
3. Applies post-attention layernorm
4. Transforms data to `mlp_mode`
5. Returns prepared `(hidden_states, residual)`

**Optimization**: Can fuse allreduce with layernorm on SM90/SM100 GPUs (`communicator.py:555-566`)

#### postprocess_layer

Finalizes layer computation and prepares output for next layer.

**Signature** (`communicator.py:319-331`):
```python
def postprocess_layer(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Operations**:
1. Adds MLP output to residual
2. Transforms data from `mlp_mode` to `layer_output_mode`
3. Optionally uses reduce-scatter instead of scatter for DP padding mode
4. Returns final `(hidden_states, residual)` for next layer

#### Helper Methods

**should_use_reduce_scatter** (`communicator.py:333-339`):
- Returns True if reduce-scatter optimization is applicable
- Requires: `allow_reduce_scatter=True`, scatter operation, and max-length DP padding

**should_fuse_mlp_allreduce_with_next_layer** (`communicator.py:341-374`):
- Determines if allreduce can be fused with next layer's layernorm
- Conditions: FlashInfer available, SM90+ GPU, batch size ≤ 2048, TP size > 1, not last layer
- Disabled for DP attention or EAGLE speculative decoding

## Communication Function Classes

These classes provide the actual implementation of scatter/gather/all-reduce operations.

### CommunicateSimpleFn

Simple scatter/gather operations without residual handling.

**Definition** (`communicator.py:410-449`):

**Supported transformations**:
1. `_trivial`: No-op when input and output modes have same group size
2. `_scattered_to_tp_attn_full`: All-gather from scattered to TP attention group

**Usage**: Called during `prepare_attn` to transform layer input to attention input mode.

### CommunicateWithAllReduceAndLayerNormFn

Combines communication with all-reduce and layernorm application.

**Definition** (`communicator.py:452-593`):

**Supported transformations**:

1. **_simple** (`communicator.py:506-517`):
   - Used when no communication needed (attn_tp_size == 1)
   - Just applies layernorm

2. **_gather_hidden_states_and_residual** (`communicator.py:519-572`):
   - Gathers hidden states from `TP_ATTN_FULL` to `FULL` mode
   - Performs all-reduce within attention TP group
   - Applies layernorm
   - Optionally fuses allreduce with layernorm on SM90/SM100
   - Handles DP attention case with gather/scatter operations

3. **_scatter_hidden_states_and_residual** (`communicator.py:574-593`):
   - Scatters from `TP_ATTN_FULL` to `SCATTERED` mode
   - Uses reduce-scatter for efficient communication
   - Applies layernorm

**Usage**: Called in `prepare_mlp` to transition from attention output to MLP input.

### CommunicateSummableTensorPairFn

Handles (hidden_states, residual) pairs that can be summed if needed.

**Definition** (`communicator.py:596-711`):

**Supported transformations**:

1. **_trivial** (`communicator.py:652-660`):
   - No-op when modes match

2. **_scatter_hidden_states** (`communicator.py:662-679`):
   - Scatters from `FULL` to `TP_ATTN_FULL` mode
   - Can use reduce-scatter optimization when `allow_reduce_scatter=True`
   - Used in MoE layers

3. **_gather** (`communicator.py:681-699`):
   - Gathers from `SCATTERED` to `TP_ATTN_FULL` mode
   - Sums hidden_states and residual before gathering
   - Sets residual to None after sum

4. **_scatter** (`communicator.py:701-711`):
   - Scatters from `TP_ATTN_FULL` to `SCATTERED` mode
   - Simple tensor split operation

**Usage**: Called in `postprocess_layer` to prepare output for next layer.

## Integration with Models

### Dense Model (e.g., Llama4)

**Example**: `python/sglang/srt/models/llama4.py:352-467`

**Setup** (`llama4.py:407-419`):
```python
self.layer_scatter_modes = LayerScatterModes.init_new(
    layer_id=layer_id,
    num_layers=config.num_hidden_layers,
    is_layer_sparse=is_moe_layer,
    is_previous_layer_sparse=is_previous_moe_layer,
)

self.layer_communicator = LayerCommunicator(
    layer_scatter_modes=self.layer_scatter_modes,
    input_layernorm=self.input_layernorm,
    post_attention_layernorm=self.post_attention_layernorm,
    allow_reduce_scatter=True,
)
```

**Forward pass** (`llama4.py:432-467`):
```python
def forward(self, positions, hidden_states, forward_batch, residual):
    # Prepare for attention
    hidden_states, residual = self.layer_communicator.prepare_attn(
        hidden_states, residual, forward_batch
    )

    # Attention computation
    hidden_states = self.self_attn(positions, hidden_states, forward_batch)

    # Prepare for MLP
    hidden_states, residual = self.layer_communicator.prepare_mlp(
        hidden_states, residual, forward_batch
    )

    # MLP computation with optional reduce-scatter
    use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(forward_batch)
    hidden_states = self.feed_forward(hidden_states, forward_batch, use_reduce_scatter)

    # Postprocess layer
    hidden_states, residual = self.layer_communicator.postprocess_layer(
        hidden_states, residual, forward_batch
    )

    return hidden_states, residual
```

### MoE Model (e.g., Qwen2MoE)

**Example**: `python/sglang/srt/models/qwen2_moe.py:394-510`

**Setup** (`qwen2_moe.py:436-468`):
```python
self.layer_scatter_modes = LayerScatterModes.init_new(
    layer_id=layer_id,
    num_layers=config.num_hidden_layers,
    is_layer_sparse=True,  # All layers are sparse in Qwen2MoE
    is_previous_layer_sparse=True,
)

self.layer_communicator = LayerCommunicator(
    layer_scatter_modes=self.layer_scatter_modes,
    input_layernorm=self.input_layernorm,
    post_attention_layernorm=self.post_attention_layernorm,
    allow_reduce_scatter=True,
)
```

**Forward pass pattern**: Same as dense model (uses same three-method pattern)

**Special feature**: `prepare_attn_and_capture_last_layer_outputs` for EAGLE3 speculative decoding (`qwen2_moe.py:479-486`)

## Integration with Two-Batch Overlap (TBO)

**Location**: `python/sglang/srt/two_batch_overlap.py`

The TBO system uses `ScatterMode` and `CommunicateSummableTensorPairFn` to split batches for overlapped execution.

**Key usage** (`two_batch_overlap.py:870-881`):
```python
context = CommunicateContext.init_new()

# Transform to TBO splitter mode
hidden_states, residual = CommunicateSummableTensorPairFn.execute(
    hidden_states_input_mode=input_data_scatter_mode,
    residual_input_mode=input_data_scatter_mode,
    output_mode=ScatterMode.TP_ATTN_FULL,  # TBO works in TP_ATTN_FULL mode
    hidden_states=hidden_states,
    residual=residual,
    forward_batch=forward_batch,
    context=context,
)
```

**TBO operation flow**:
1. Split batch into two sub-batches
2. Transform data to `TP_ATTN_FULL` mode for splitting
3. Execute sub-batches with pipeline overlap
4. Merge outputs back together

## Caller Map

### Direct Callers of LayerCommunicator

1. **Model decoder layers** (14+ models):
   - `llama4.py:414` - Llama4DecoderLayer
   - `qwen2_moe.py:463` - Qwen2MoeDecoderLayer
   - `qwen3_moe.py` - Qwen3MoeDecoderLayer
   - `deepseek_v2.py` - DeepseekV2DecoderLayer
   - `minimax_m2.py` - MinimaxDecoderLayer
   - `qwen3.py`, `qwen3_next.py` - Qwen3/Next decoder layers
   - `longcat_flash.py`, `longcat_flash_nextn.py` - LongCat models
   - `gpt_oss.py` - GPT decoder
   - `glm4_moe.py` - GLM4 MoE
   - `bailing_moe.py` - Bailing MoE
   - `falcon_h1.py` - Falcon H1
   - `step3_vl.py` - Step3 VL model

### Callers of ScatterMode

1. **Two-batch overlap system**:
   - `two_batch_overlap.py:16` - Import
   - `two_batch_overlap.py:799,870,876` - Used for TBO splitting/merging

2. **All model files** (transitively through LayerScatterModes)

### Callers of CommunicateContext

1. **LayerCommunicator**: Internal usage in all communication functions
2. **two_batch_overlap.py:871** - Creates context for TBO transformations

### Callers of CommunicateSummableTensorPairFn

1. **LayerCommunicator**: Used in `postprocess_layer` method
2. **two_batch_overlap.py:873,892** - Used for TBO batch splitting/merging

### Helper Function Callers

**enable_moe_dense_fully_dp()** (`communicator.py:171-172`):
- Called by `LayerScatterModes._compute_mlp_mode`
- Also imported and used in `ernie4.py:30`

## Dependencies

### From sglang.srt.distributed
- `get_tensor_model_parallel_world_size()`: Gets total TP size
- `tensor_model_parallel_all_reduce()`: All-reduce within TP group

### From sglang.srt.layers.dp_attention
- `attn_tp_all_gather_into_tensor()`: Gather within attention TP group
- `attn_tp_reduce_scatter_tensor()`: Reduce-scatter within attention TP group
- `dp_gather_partial()`, `dp_scatter()`, `dp_reduce_scatter_tensor()`: DP operations
- `get_attention_tp_rank/size()`, `get_attention_dp_size()`: Query DP attention config
- `get_global/local_dp_buffer()`: Get pre-allocated buffers
- `is_dp_attention_enabled()`: Check if DP attention is enabled

### From sglang.srt.layers.moe
- `get_moe_a2a_backend()`: Get all-to-all backend configuration
- `should_use_flashinfer_cutlass_moe_fp4_allgather()`: Check FP4 optimization

### From sglang.srt.model_executor.forward_batch_info
- `ForwardBatch`: Main batch information dataclass

### From sglang.srt.server_args
- `get_global_server_args()`: Access server configuration

### From sglang.srt.utils
- `is_cuda()`, `is_hip()`, `is_sm90_supported()`, etc.: Platform detection
- `get_bool_env_var()`: Environment variable checking
- `prepare_weight_cache()`: Weight cache preparation

## Performance Optimizations

### 1. Allreduce Fusion

**Location**: `communicator.py:253-260`, `communicator.py:555-566`

Fuses all-reduce with layernorm on SM90/SM100 GPUs when batch size ≤ 4096.

**Benefits**:
- Reduces kernel launch overhead
- Better memory bandwidth utilization
- Lower latency

**Conditions**:
- FlashInfer available
- SM90 or SM100 GPU
- `enable_flashinfer_allreduce_fusion` flag set
- Appropriate batch size

### 2. Reduce-Scatter

**Location**: `communicator.py:333-339`, `communicator.py:674-676`

Uses reduce-scatter instead of all-reduce + scatter when padding is enabled.

**Benefits**:
- Reduces communication volume
- Better load balancing

**Requirements**:
- Model must skip all-reduce after MLP/MoE
- `allow_reduce_scatter=True` in LayerCommunicator
- DP padding mode is `MAX_LEN`

### 3. Weight Cache Preparation

**Location**: `communicator.py:569-570`

Pre-fetches weights into cache after all-reduce.

**Benefits**:
- Hides memory latency
- Better cache utilization

### 4. DP Attention Optimization

**Location**: `communicator.py:535-553`

Performs layernorm before gather when `attn_tp_size == 1`.

**Benefits**:
- Reduces communication volume (smaller normalized data)
- Lower latency

### 5. MXFP4 Quantization Fusion

**Location**: `communicator.py:265-286`

Fuses RMSNorm with MXFP4 quantization on AMD GFX95 GPUs.

**Benefits**:
- Single kernel for norm + quantization
- Reduced memory traffic

## Configuration

### Server Arguments

Relevant flags from `get_global_server_args()`:
- `moe_dense_tp_size`: TP size for dense layers (affects scatter mode)
- `enable_flashinfer_allreduce_fusion`: Enable allreduce fusion optimization
- `attention_backend`: Backend for attention computation
- `device`: Target device (cuda/hip)
- `enable_dp_lm_head`: Enable DP for LM head
- `speculative_algorithm`: Speculative decoding algorithm (affects fusion decisions)
- `enable_piecewise_cuda_graph`: Enable piecewise CUDA graphs

### Environment Variables

- `SGLANG_USE_AITER`: Enable AITER for AMD GPUs
- Various platform detection variables (used by imported utilities)

## Data Flow Example

Example for a dense layer with DP attention enabled:

```
Input: [SCATTERED] hidden_states, residual

1. prepare_attn():
   [SCATTERED] -> all_gather -> [TP_ATTN_FULL]
   Apply input_layernorm

2. self_attn():
   [TP_ATTN_FULL] attention computation

3. prepare_mlp():
   all_reduce (if TP size > 1)
   Add residual
   Apply post_attention_layernorm
   [TP_ATTN_FULL] -> gather -> [FULL]

4. mlp/moe():
   [FULL] MLP computation
   all_reduce (unless using reduce_scatter)

5. postprocess_layer():
   [FULL] -> scatter -> [TP_ATTN_FULL]
   Add residual

Output: [TP_ATTN_FULL] hidden_states, residual (for next layer)
```

## Testing and Debugging

### Enable Debug Logging

```bash
export SGLANG_TBO_DEBUG=1  # Enable TBO debug logging
```

### Key Invariants

1. `attn_mode` is always `TP_ATTN_FULL`
2. Model input/output uses `ScatterMode.model_input_output()` (which returns `TP_ATTN_FULL`)
3. Layer output mode matches next layer's input mode
4. Communication functions handle empty tensors (shape[0] == 0) gracefully
5. Residual can be None (first layer) or tensor (subsequent layers)

### Common Issues

1. **Mode mismatch**: If layer output mode doesn't match next layer input mode, check `is_layer_sparse` and `is_previous_layer_sparse` parameters
2. **DP attention errors**: Verify DP attention is properly initialized via `is_dp_attention_enabled()`
3. **Fusion not applying**: Check GPU architecture, batch size, and server args
4. **TBO issues**: Verify batch splitting logic in `TboForwardBatchPreparer`

## Future Enhancements

Potential areas for improvement noted in code:

1. **TODO** (`communicator.py:398`): Support `moe_dense_tp_size > 1`
2. **TODO** (`communicator.py:514`): Move empty tensor checks into LayerNorm itself
3. **TODO** (`communicator.py:692`): Improve DP buffer length calculation
4. **TODO** (`communicator.py:713`): Handle residual != None in scatter operation
5. Expand reduce-scatter support to all models (currently opt-in via `allow_reduce_scatter`)
6. More granular control over fusion heuristics

## Related Documentation

- Attention backends: `BASE_ATTENTION_BACKEND_DEEP_DIVE.md`
- MoE system: `MOE_RUNNER_SYSTEM.md`
- Two-batch overlap: `TBO_BACKEND_DEEP_DIVE.md`
- Piecewise CUDA graphs: `PIECEWISE_CUDA_GRAPH_ONBOARDING.md`
