# Tensor Parallelism (TP) in SGLang

## Overview

Tensor Parallelism (TP) is a model parallelization technique that distributes model weights and computations across multiple GPUs. In SGLang, TP enables efficient inference for large language models by partitioning model layers across GPU devices within a single node or across nodes.

### Key Concepts

- **TP Size**: Number of GPUs participating in tensor parallel execution
- **TP Rank**: Identifier (0 to tp_size-1) for each GPU in the TP group
- **Column Parallelism**: Splits weight matrices along the output dimension
- **Row Parallelism**: Splits weight matrices along the input dimension
- **All-Reduce**: Synchronization operation that sums tensors across all TP ranks

## Architecture Components

### 1. TpModelWorker (`python/sglang/srt/managers/tp_worker.py:197`)

Main worker class that manages tensor parallel execution on each GPU.

**Key Responsibilities:**
- Initialize TP groups and communication channels
- Manage model runner and memory pools
- Handle forward passes for generation and embedding
- Coordinate weight updates and LoRA adapters
- Sync random seeds across TP workers

**Initialization Flow:**
```python
TpModelWorker(
    server_args=...,
    gpu_id=...,      # Physical GPU device ID
    tp_rank=...,     # Rank in TP group (0 to tp_size-1)
    tp_size=...,     # Total TP group size
    pp_rank=...,     # Pipeline parallel rank
    nccl_port=...,   # Port for NCCL communication
)
```

### 2. ModelRunner (`python/sglang/srt/model_executor/model_runner.py:226`)

Executes forward passes and manages model state.

**TP-Related Functions:**
- `init_torch_distributed()`: Sets up distributed backend (NCCL/GLOO)
- `initialize()`: Loads model and initializes TP groups
- `forward()`: Runs forward pass with TP communication

**Key Initialization Code:**
```python
# python/sglang/srt/model_executor/model_runner.py:630
initialize_model_parallel(
    tensor_model_parallel_size=self.tp_size,
    pipeline_model_parallel_size=self.pp_size,
    expert_model_parallel_size=self.moe_ep_size,
    duplicate_tp_group=self.server_args.enable_pdmux,
    torch_compile=self.server_args.enable_piecewise_cuda_graph,
)
```

### 3. GroupCoordinator (`python/sglang/srt/distributed/parallel_state.py:191`)

Manages distributed process groups and communication operations.

**Features:**
- Wraps PyTorch ProcessGroup for NCCL/GLOO communication
- Routes operations to optimized implementations:
  - **PyNccl**: For CUDA graph capture
  - **Custom AllReduce**: For small tensors
  - **Quick AllReduce**: AMD GPU optimization (ROCm)
  - **MSCCLPP**: Multi-scale communication
  - **SymmMem**: Symmetric memory communication

**Communication Operations:**
- `all_reduce()`: Sum tensors across all ranks
- `all_gather()`: Gather tensors from all ranks
- `reduce_scatter()`: Reduce and distribute results
- `broadcast()`: Send tensor from one rank to all
- `send_tensor_dict()`: P2P tensor dictionary transfer

### 4. Parallel State (`python/sglang/srt/distributed/parallel_state.py`)

Global state management for distributed groups.

**Key Groups:**
- `_TP`: Tensor parallel group
- `_PP`: Pipeline parallel group
- `_MOE_EP`: MoE expert parallel group
- `_MOE_TP`: MoE tensor parallel group
- `_WORLD`: World group (all ranks)

**Initialization:**
```python
# python/sglang/srt/distributed/parallel_state.py:1475
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    torch_compile: Optional[bool] = None,
) -> None:
    """
    Creates TP groups by partitioning ranks.

    Example with 8 GPUs, tp_size=2, pp_size=4:
    - TP groups: [g0, g1], [g2, g3], [g4, g5], [g6, g7]
    - PP groups: [g0, g2, g4, g6], [g1, g3, g5, g7]
    """
```

## Initialization Flow

### Step-by-Step Initialization

1. **Distributed Environment Setup** (`model_runner.py:557`)
   ```python
   init_distributed_environment(
       world_size=tp_size * pp_size,
       rank=global_rank,
       distributed_init_method="tcp://...",
       backend="nccl",  # or "gloo", "xccl", "hccl"
   )
   ```

2. **Model Parallel Groups** (`model_runner.py:630`)
   ```python
   initialize_model_parallel(
       tensor_model_parallel_size=tp_size,
       pipeline_model_parallel_size=pp_size,
       expert_model_parallel_size=ep_size,
   )
   ```

3. **GroupCoordinator Creation** (`parallel_state.py:1529`)
   - Creates TP process groups
   - Initializes communicators (PyNccl, Custom AllReduce, etc.)
   - Sets up CPU and device groups

4. **Model Loading** (`model_runner.py:387`)
   - Each rank loads its shard of weights
   - Column/Row parallel layers partition automatically

5. **Random Seed Sync** (`tp_worker.py:303`)
   ```python
   self.random_seed = broadcast_pyobj(
       [server_args.random_seed],
       self.tp_size * self.pp_rank + tp_rank,
       self.world_group.cpu_group,
       src=self.world_group.ranks[0],
   )[0]
   set_random_seed(self.random_seed)
   ```

## Parallelization Strategy

### Column Parallel Linear (`layers/linear.py:263`)

**Weight Partitioning:**
```
Original Weight: [input_size, output_size]
                           ↓
      Partition along output dimension
                           ↓
Rank 0: [input_size, output_size/tp_size]
Rank 1: [input_size, output_size/tp_size]
...
```

**Forward Pass:**
- Input: `X` (replicated across all ranks)
- Weight shard: `A_i` on rank `i`
- Output: `Y_i = X @ A_i` (no communication needed)
- Optional `gather_output`: All-gather to get full `Y = [Y_0, Y_1, ...]`

**Example Usage in Attention:**
```python
# QKV projection is column parallel
self.qkv_proj = ColumnParallelLinear(
    hidden_size,
    (num_heads + 2 * num_kv_heads) * head_dim,
    gather_output=False,  # Keep partitioned
)
```

### Row Parallel Linear (`layers/linear.py`)

**Weight Partitioning:**
```
Original Weight: [input_size, output_size]
                           ↓
      Partition along input dimension
                           ↓
Rank 0: [input_size/tp_size, output_size]
Rank 1: [input_size/tp_size, output_size]
...
```

**Forward Pass:**
- Input: `X_i` (partitioned, from column parallel layer)
- Weight shard: `A_i` on rank `i`
- Local output: `Y_i = X_i @ A_i`
- All-reduce: `Y = sum(Y_0, Y_1, ...)` across all ranks

**Example Usage in Attention:**
```python
# Output projection is row parallel
self.o_proj = RowParallelLinear(
    num_heads * head_dim,
    hidden_size,
    input_is_parallel=True,  # Input comes from column parallel
)
```

### Communication Patterns

#### 1. Standard MLP Block
```
Input (replicated)
    ↓
ColumnParallelLinear (gate/up projections)
    ↓ (no communication)
Activation (replicated on each rank)
    ↓ (no communication)
RowParallelLinear (down projection)
    ↓ (all-reduce)
Output (replicated)
```

#### 2. Attention Block
```
Input (replicated)
    ↓
ColumnParallelLinear (QKV projection)
    ↓ (no communication)
Attention Computation (partitioned heads)
    ↓ (no communication)
RowParallelLinear (output projection)
    ↓ (all-reduce)
Output (replicated)
```

## LayerCommunicator (`layers/communicator.py:175`)

Orchestrates communication between layers with support for different scatter modes.

### Scatter Modes

**ScatterMode.SCATTERED**: Each rank has different data (1/tp_size of full data)

**ScatterMode.TP_ATTN_FULL**: Each rank in attention TP group has full data for its group

**ScatterMode.FULL**: All ranks have identical full data

### Layer Communication Workflow

```python
class LayerCommunicator:
    def prepare_attn(self, hidden_states, residual, forward_batch):
        # 1. Apply layer norm
        # 2. Convert between scatter modes if needed
        # 3. Return prepared tensors for attention

    def prepare_mlp(self, hidden_states, residual, forward_batch):
        # 1. All-reduce attention output
        # 2. Apply layer norm
        # 3. Convert scatter modes for MLP input

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        # 1. Combine MLP output with residual
        # 2. Convert to output scatter mode
```

### Optimizations

**AllReduce Fusion** (`communicator.py:341`):
- Fuses all-reduce with next layer's RMSNorm
- Reduces synchronization overhead
- Enabled with `--enable-flashinfer-allreduce-fusion`
- Requires FlashInfer and batch_size ≤ 2048

**Reduce Scatter** (`communicator.py:333`):
- Replaces scatter + all-reduce with reduce-scatter
- More efficient for certain data patterns
- Used when `dp_padding_mode.is_max_len()`

## Data Parallel Attention (DP-Attention)

SGLang supports DP-Attention for better scaling with large TP sizes.

### Concept

Instead of keeping attention computation fully replicated:
- Split attention across DP groups within TP group
- Each DP rank handles subset of attention computation
- Additional all-gather/reduce-scatter operations

### Configuration

```python
initialize_dp_attention(
    server_args=server_args,
    model_config=model_config,
)
```

**Attention TP Group Size** = tp_size / dp_size

## Pipeline Parallelism Integration

TP workers integrate with pipeline parallelism:

### Pipeline Communication (`tp_worker.py:360`)

```python
def forward_batch_generation(self, model_worker_batch, forward_batch):
    # Non-first PP ranks receive tensors from previous stage
    if not self.pp_group.is_first_rank:
        pp_proxy_tensors = PPProxyTensors(
            self.pp_group.recv_tensor_dict(
                all_gather_group=self.get_attention_tp_group()
            )
        )

    # Forward pass
    logits_output, can_run_cuda_graph = self.model_runner.forward(
        forward_batch,
        pp_proxy_tensors=pp_proxy_tensors,
    )

    # Non-last PP ranks send tensors to next stage
    if not self.pp_group.is_last_rank:
        self.pp_group.send_tensor_dict(
            pp_proxy_tensors,
            all_gather_group=self.get_attention_tp_group()
        )
```

## Memory Pools

TP workers manage shared memory pools:

### ReqToTokenPool
- Maps request IDs to token positions
- Shared across TP ranks (same view)

### TokenToKVPool
- Stores KV cache for attention
- Partitioned by num_heads (each TP rank stores subset of heads)

### Memory Pool Initialization

```python
# python/sglang/srt/managers/tp_worker.py:95
def get_memory_pool(self):
    return (
        self.model_runner.req_to_token_pool,       # Shared
        self.model_runner.token_to_kv_pool_allocator,  # Partitioned
    )
```

## Weight Updates and Dynamic Loading

### Weight Update Methods

**From Disk** (`tp_worker.py:101`):
```python
def update_weights_from_disk(self, recv_req):
    success, message = self.model_runner.update_weights_from_disk(
        recv_req.model_path,
        recv_req.load_format
    )
```

**From Distributed Source** (`tp_worker.py:149`):
```python
def update_weights_from_distributed(self, recv_req):
    success, message = self.model_runner.update_weights_from_distributed(
        recv_req.names,
        recv_req.dtypes,
        recv_req.shapes,
        recv_req.group_name
    )
```

### LoRA Adapter Support

```python
def load_lora_adapter(self, recv_req):
    result = self.model_runner.load_lora_adapter(recv_req.to_ref())
    return result

def unload_lora_adapter(self, recv_req):
    result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
    return result

def can_run_lora_batch(self, lora_ids: list[str]) -> bool:
    return self.model_runner.lora_manager.validate_lora_batch(lora_ids)
```

## Communication Backend Optimization

### Communicator Selection (`parallel_state.py:513`)

**Graph Capture Mode** (CUDA graphs):
- PyNccl: Enabled
- MSCCLPP: Enabled
- SymmMem: Enabled
- torch.distributed: Disabled

**Eager Mode** (regular execution):
- Quick AllReduce: Enabled (if tensor small enough)
- Custom AllReduce: Enabled (if tensor within size limits)
- torch.distributed: Enabled (fallback)

### Custom AllReduce

Fast all-reduce for small tensors using:
- IPC memory mapping
- Custom CUDA kernels
- P2P GPU communication

**Enable/Disable:**
```bash
# Enable (default)
python -m sglang.launch_server --model ... --tp-size 4

# Disable
python -m sglang.launch_server --model ... --tp-size 4 \
    --disable-custom-all-reduce
```

## Debugging and Monitoring

### Useful Environment Variables and Flags

```bash
# Detect slow ranks during init
export SGLANG_DETECT_SLOW_RANK=1

# Enable message queue broadcaster (default: True)
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=true

# Block non-rank-0 processes from spawning children (default: True)
export SGLANG_BLOCK_NONZERO_RANK_CHILDREN=1

# Skip TP memory imbalance check
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

# Synchronize token IDs across TP ranks (debugging)
export SGLANG_SYNC_TOKEN_IDS_ACROSS_TP=1

# Skip P2P connectivity check for custom all-reduce
export SGLANG_SKIP_P2P_CHECK=1

# Disable custom all-reduce for debugging (CLI flag)
--disable-custom-all-reduce
```

### Checking TP Group Status

```python
# Get TP group info
tp_group = get_tp_group()
print(f"TP Size: {tp_group.world_size}")
print(f"TP Rank: {tp_group.rank_in_group}")
print(f"Ranks: {tp_group.ranks}")

# Check communication backend
print(f"PyNccl: {tp_group.pynccl_comm is not None}")
print(f"Custom AllReduce: {tp_group.ca_comm is not None}")
```

## Key Files Reference

### Core Implementation
- `python/sglang/srt/managers/tp_worker.py` - TP worker implementation
- `python/sglang/srt/model_executor/model_runner.py` - Model execution
- `python/sglang/srt/distributed/parallel_state.py` - Distributed state management
- `python/sglang/srt/layers/communicator.py` - Layer communication orchestration
- `python/sglang/srt/layers/linear.py` - TP linear layers

### Communication Backends
- `python/sglang/srt/distributed/device_communicators/pynccl.py` - PyNccl wrapper
- `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` - Custom AR
- `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py` - AMD optimization
- `python/sglang/srt/distributed/device_communicators/symm_mem.py` - Symmetric memory

### Attention Integration
- `python/sglang/srt/layers/dp_attention.py` - Data parallel attention
- `python/sglang/srt/layers/attention/` - Various attention backends

## Performance Considerations

### Communication Overhead

**Minimize All-Reduce:**
- Use larger TP sizes only when necessary
- Enable AllReduce fusion when possible
- Consider DP-Attention for very large TP

**Optimize Bandwidth:**
- Use NVLink/NVSwitch for intra-node TP
- Use InfiniBand for multi-node TP
- Enable MSCCLPP for multi-node scenarios

### Memory Efficiency

**KV Cache Partitioning:**
- Each TP rank stores 1/tp_size of KV heads
- Total KV cache memory = full_cache_size / tp_size

**Activation Memory:**
- Column parallel output: partitioned (1/tp_size)
- Row parallel input: partitioned (1/tp_size)
- Intermediate activations: mostly partitioned

### Scaling Characteristics

**Strong Scaling:**
- Linear speedup for compute-bound models
- Communication overhead increases with TP size
- Sweet spot typically at tp_size = 2, 4, or 8

**Weak Scaling:**
- Larger models scale better with TP
- Models > 70B often require tp_size ≥ 4
- Models > 405B may need tp_size = 8 or 16

## Example Configurations

### Single Node (8x A100)

```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tp-size 8 \
    --port 30000
```

### Multi-Node (2 nodes, 8 GPUs each)

```bash
# Node 0
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-405B-Instruct \
    --tp-size 16 \
    --nccl-init-addr <node0-ip> \
    --nnodes 2 \
    --node-rank 0

# Node 1
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-405B-Instruct \
    --tp-size 16 \
    --nccl-init-addr <node0-ip> \
    --nnodes 2 \
    --node-rank 1
```

### With Pipeline Parallelism

```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-405B-Instruct \
    --tp-size 8 \
    --pp-size 2 \
    --port 30000
```

## Advanced Features

### Speculative Decoding with TP

```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --speculative-algorithm eagle \
    --speculative-draft-model-path <draft-model> \
    --tp-size 4
```

### MoE with Expert Parallelism

```bash
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp-size 8 \
    --ep-size 4  # Expert parallel size
```

### Dynamic LoRA with TP

```bash
# Load base model with TP
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3-70B \
    --tp-size 8 \
    --enable-lora

# LoRA adapters are automatically partitioned across TP ranks
```

## Configuration Reference

### Command-Line Arguments

SGLang provides extensive configuration options for tensor parallelism and distributed execution. Below is a comprehensive reference of all TP-related CLI flags:

#### Basic Parallelism Configuration

**`--tp-size`, `--tensor-parallel-size`** (default: `1`)
- Number of GPUs for tensor parallelism
- Distributes model weights across GPUs
- Example: `--tp-size 8`

**`--pp-size`, `--pipeline-parallel-size`** (default: `1`)
- Number of pipeline stages for pipeline parallelism
- Splits model vertically into stages
- Example: `--pp-size 2`

**`--pp-max-micro-batch-size`** (default: `None`)
- Maximum micro-batch size for pipeline parallelism
- Controls granularity of pipeline execution

**`--ep-size`, `--ep`, `--expert-parallel-size`** (default: `1`)
- Expert parallelism size for MoE models
- Distributes experts across GPUs
- Example: `--ep-size 4`

#### Advanced Parallelism Options

**`--enable-dp-attention`** (default: `False`)
- Enable data parallel attention with tensor parallel FFN
- Splits attention computation across DP groups
- DP size must equal TP size
- Supported models: DeepSeek-V2, Qwen 2/3 MoE
- Example: `--enable-dp-attention --tp-size 8`

**`--enable-dp-lm-head`** (default: `False`)
- Enable vocabulary parallelism across attention TP group
- Avoids all-gather across DP groups under DP attention
- Optimizes performance with DP attention enabled

**`--moe-dense-tp-size`** (default: `None`)
- TP size for MoE dense MLP layers
- Useful when main TP size is large but MLP dimensions are small
- Example: `--tp-size 8 --moe-dense-tp-size 4`

**`--ep-num-redundant-experts`** (default: `0`)
- Number of redundant experts in expert parallel
- Improves load balancing for MoE models

**`--ep-dispatch-algorithm`** (default: `None`)
- Algorithm for dispatching to redundant experts
- Choices: `static`, `dynamic`, `fake`

**`--init-expert-location`** (default: `"trivial"`)
- Initial placement strategy for EP experts

#### Communication Backend Configuration

**`--nccl-port`** (default: `None`)
- Port for NCCL initialization
- Auto-assigned if not specified

**`--disable-custom-all-reduce`** (default: `False`)
- Disable custom all-reduce kernel
- Falls back to NCCL for all communication
- Use for debugging communication issues

**`--enable-symm-mem`** (default: `False`)
- Enable NCCL symmetric memory (CUMEM)
- Provides fast collectives on supported hardware
- Requires CUDA 12.4+ and Hopper+ GPUs
- Sets `NCCL_CUMEM_ENABLE=1`

**`--enable-torch-symm-mem`** (default: `False`)
- Enable PyTorch symmetric memory for all-reduce
- Requires CUDA SM90+ (Hopper) or SM100+
- Supported world sizes: 4/6/8 (SM90), 6/8 (SM100)

**`--enable-nccl-nvls`** (default: `False`)
- Enable NCCL NVLS (NVLink Sharp)
- Optimizes prefill-heavy workloads
- Sets `NCCL_NVLS_ENABLE=1` when enabled

**`--enable-mscclpp`** (default: `False`)
- Enable MSCCLPP for small message all-reduce
- Multi-scale collective communication
- Falls back to NCCL for unsupported cases
- Useful for multi-node setups

**`--enable-flashinfer-allreduce-fusion`** (default: `False`)
- Fuse all-reduce with Residual RMSNorm
- Reduces synchronization overhead
- Requires FlashInfer backend
- Works best with batch_size ≤ 2048

#### CUDA Graph Configuration

**`--disable-cuda-graph`** (default: `False`)
- Completely disable CUDA graph capture
- Useful for debugging or profiling

**`--cuda-graph-max-bs`** (default: `None`)
- Maximum batch size for CUDA graph capture
- Auto-configured based on GPU memory if not set
- Example: `--cuda-graph-max-bs 256`

**`--cuda-graph-bs`** (default: `None`)
- Explicit list of batch sizes to capture
- Example: `--cuda-graph-bs 1 2 4 8 16 32 64 128`

**`--disable-cuda-graph-padding`** (default: `False`)
- Disable CUDA graph when padding is needed
- Still uses CUDA graph for non-padded batches

**`--enable-profile-cuda-graph`** (default: `False`)
- Enable profiling during CUDA graph capture

**`--enable-cudagraph-gc`** (default: `False`)
- Enable garbage collection during capture
- Disabled by default to speed up capture

#### Piecewise CUDA Graph (Experimental)

**`--enable-piecewise-cuda-graph`** (default: `False`)
- Enable piecewise CUDA graph optimization
- Optimizes extend/prefill operations
- Experimental feature for advanced users
- Uses torch.compile under the hood

**`--piecewise-cuda-graph-tokens`** (default: `None`)
- List of token counts for piecewise capture
- Example: `--piecewise-cuda-graph-tokens 128 256 512 1024`

**`--piecewise-cuda-graph-max-tokens`** (default: `4096`)
- Maximum tokens for piecewise CUDA graph

**`--piecewise-cuda-graph-compiler`** (default: `"eager"`)
- Compiler backend for piecewise graphs
- Choices: `eager`, `inductor`

#### MoE Communication Backend

**`--moe-a2a-backend`** (default: `"none"`)
- Backend for MoE all-to-all communication
- Choices: `none`, `deepep`, `mooncake`

**`--deepep-mode`** (default: `"auto"`)
- Mode for DeepEP MoE backend
- Choices: `auto`, `normal`, `low_latency`
- `auto` = `low_latency` for decode, `normal` for prefill

**`--deepep-config`** (default: `None`)
- Tuned DeepEP config (JSON string or file path)

**`--elastic-ep-backend`** (default: `None`)
- Backend for elastic expert parallelism
- Currently supports: `mooncake`

**`--mooncake-ib-device`** (default: `None`)
- InfiniBand devices for Mooncake (comma-separated)
- Auto-detected if not specified

#### Overlap and Scheduling

**`--enable-two-batch-overlap`** (default: `False`)
- Enable two micro-batch overlap
- Overlaps computation of two batches

**`--enable-single-batch-overlap`** (default: `False`)
- Enable overlap within single batch
- Overlaps computation and communication

**`--tbo-token-distribution-threshold`** (default: `0.48`)
- Token distribution threshold for two-batch-overlap
- Controls two-batch vs two-chunk overlap behavior
- Set to 0 to disable two-chunk-overlap

**`--disable-overlap-schedule`** (default: `False`)
- Disable overlap of CPU scheduler with GPU worker

### Environment Variables

SGLang respects numerous environment variables for fine-tuning distributed behavior:

#### Core TP/Distributed Variables

**`SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK`** (default: `False`)
- Skip memory balance check across TP ranks
- Useful when ranks intentionally have different memory usage

**`SGLANG_SYNC_TOKEN_IDS_ACROSS_TP`** (default: `False`)
- Synchronize token IDs across TP ranks
- Debugging tool for distributed inference

**`SGLANG_USE_MESSAGE_QUEUE_BROADCASTER`** (default: `True`)
- Use message queue for broadcasting to workers
- Improves efficiency of rank-to-rank communication

**`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS`** (default: `False`)
- Set CUDA_VISIBLE_DEVICES to single GPU per process
- Useful for certain deployment scenarios

**`SGLANG_BLOCK_NONZERO_RANK_CHILDREN`** (default: `True`)
- Block non-rank-0 processes from spawning children
- Set to `0` to allow child processes from all ranks

**`SGLANG_DETECT_SLOW_RANK`** (default: `False`)
- Detect slow ranks during initialization
- Useful for debugging performance issues

#### NCCL Configuration

**`NCCL_CUMEM_ENABLE`** (auto-set by `--enable-symm-mem`)
- Enable NCCL symmetric memory
- Automatically set to `1` when `--enable-symm-mem` is used

**`NCCL_NVLS_ENABLE`** (auto-set by `--enable-nccl-nvls`)
- Enable NCCL NVLink Sharp
- Automatically configured based on `--enable-nccl-nvls`

**`NCCL_ALGO`** (auto-set in some configurations)
- NCCL algorithm selection
- SGLang may set to `"allreduce:tree"` in certain cases

**`SGLANG_NCCL_SO_PATH`**
- Path to custom NCCL library (.so file)
- Overrides default NCCL discovery

**`SGLANG_TMP_NCCL_COMM_VALUE`** (internal)
- Temporary value for NCCL communicator sharing
- Used internally by PyNccl allocator

#### Custom AllReduce Configuration

**`SGLANG_SKIP_P2P_CHECK`** (default: `False`)
- Skip P2P connectivity check for custom all-reduce
- Use when you know P2P is available but check fails
- Set to `1` to skip check

#### ROCm-Specific Variables

**`ROCM_QUICK_REDUCE_QUANTIZATION`** (default: `"NONE"`)
- Quantization mode for ROCm quick all-reduce
- Choices: `NONE`, `FP8`, `INT8`

**`ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16`** (default: `True`)
- Cast BF16 to FP16 for ROCm quick reduce
- Set to `0` to disable

**`ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB`** (default: `0`)
- Maximum tensor size (MB) for quick reduce
- `0` means no limit

#### CUDA Graph Variables

**`SGLANG_MEMORY_SAVER_CUDA_GRAPH`** (default: `False`)
- Enable memory-saving mode for CUDA graphs
- Releases unused memory during graph capture

**`SGLANG_ENABLE_CUDAGRAPH_GC`** (env var, see also `--enable-cudagraph-gc`)
- Alternative way to enable GC during CUDA graph capture

#### CUDA Runtime Configuration

**`CUDA_DEVICE_MAX_CONNECTIONS`** (auto-set to `8`)
- Maximum concurrent kernel launches per device
- Automatically set by SGLang for optimal parallelism

**`CUDA_MODULE_LOADING`** (auto-set to `"AUTO"`)
- CUDA module loading mode
- Automatically configured by SGLang

#### Other Environment Variables

**`LOCAL_SIZE`**
- Set to `tp_size` in certain contexts
- Used for local rank configuration

**`RANK`**, **`WORLD_SIZE`**, **`MASTER_ADDR`**, **`MASTER_PORT`**
- Standard PyTorch distributed environment variables
- Used for distributed initialization

**`LOCAL_RANK`**
- Local rank within the node
- Used by distributed frameworks

### Runtime State Configuration

Beyond CLI arguments and environment variables, SGLang's TP behavior is influenced by:

#### Model Configuration
- Number of attention heads and KV heads
- Hidden dimension size
- MoE expert count and configuration
- Layer count and architecture

#### Hardware Detection
- GPU memory size (affects default CUDA graph batch sizes)
- GPU compute capability (SM version)
- NVLink/NVSwitch availability
- InfiniBand availability for multi-node

#### Auto-Configured Defaults
- CUDA graph batch sizes based on GPU memory and TP size
- Communication backend selection (custom vs NCCL)
- Memory pool sizes
- Attention backend selection

### Debugging TP Issues

When troubleshooting TP-related problems, try these steps:

1. **Disable custom all-reduce**: `--disable-custom-all-reduce`
2. **Disable CUDA graphs**: `--disable-cuda-graph`
3. **Enable verbose logging**: Check scheduler and worker logs
4. **Check P2P connectivity**: Verify NVLink/PCIe connectivity
5. **Verify environment**: Ensure `CUDA_VISIBLE_DEVICES` is correct
6. **Test with simpler config**: Start with `--tp-size 2` before scaling up

### Common Configuration Patterns

#### Single Node, 8x A100
```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tp-size 8 \
    --port 30000
```

#### Multi-Node, 16 GPUs Total
```bash
# Node 0
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-405B-Instruct \
    --tp-size 16 \
    --nnodes 2 \
    --node-rank 0 \
    --nccl-init-addr <node0-ip> \
    --port 30000

# Node 1
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-405B-Instruct \
    --tp-size 16 \
    --nnodes 2 \
    --node-rank 1 \
    --nccl-init-addr <node0-ip> \
    --port 30000
```

#### With DP Attention (DeepSeek-V2/V3)
```bash
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp-size 8 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --port 30000
```

#### With Piecewise CUDA Graph
```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tp-size 4 \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-max-tokens 8192 \
    --port 30000
```

#### With Custom Communication Optimizations
```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tp-size 8 \
    --enable-symm-mem \
    --enable-flashinfer-allreduce-fusion \
    --enable-mscclpp \
    --port 30000
```

## Conclusion

SGLang's TP implementation provides:
- Efficient weight partitioning with column/row parallelism
- Flexible communication backend selection with extensive configuration
- Integration with pipeline and expert parallelism
- Support for advanced features (LoRA, speculative decoding, DP attention)
- Optimizations for various hardware configurations
- Comprehensive tuning knobs via CLI arguments and environment variables

The modular design allows for easy extension and customization while maintaining high performance across different deployment scenarios. With the extensive configuration options documented above, users can fine-tune behavior for their specific hardware and workload requirements.
