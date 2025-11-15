# SGLang Scheduler System Documentation

## Overview

The SGLang scheduler system is a sophisticated request management and batching system for large language model inference. It orchestrates the flow of requests from input to output, managing GPU memory, KV cache, and continuous batching to maximize throughput and efficiency.

## Architecture

The scheduler system consists of four main components:

1. **Scheduler** (`scheduler.py`) - The main orchestrator
2. **Schedule Batch** (`schedule_batch.py`) - Request and batch data structures
3. **Schedule Policy** (`schedule_policy.py`) - Request prioritization and selection strategies
4. **Scheduler Input Blocker** (`scheduler_input_blocker.py`) - Input flow control

---

## 1. Scheduler (`scheduler.py`)

**Location**: `python/sglang/srt/managers/scheduler.py`

### Purpose
The Scheduler is the central component that manages:
- Request lifecycle from tokenization to detokenization
- Batch creation and execution
- Memory pool and KV cache management
- Communication between tokenizer, model workers, and detokenizer
- Support for various features (LoRA, multimodal, speculative decoding, disaggregation)

### Key Components

#### Initialization (`__init__`)
- Sets up communication sockets (ZMQ) with tokenizer and detokenizer
- Initializes model workers (tensor parallel, draft workers for speculation)
- Configures memory pools and KV cache
- Sets up scheduling policies and metrics collection

#### Event Loops
Three main event loop modes:

1. **Normal Mode** (`event_loop_normal`)
   - Sequential processing: receive → process → run batch → process results
   - Simple but less optimized

2. **Overlap Mode** (`event_loop_overlap`)
   - Overlaps CPU processing with GPU computation
   - Uses separate CUDA streams for forward pass and data copying
   - Higher throughput via pipelining

3. **Pipeline Parallel Mode** (`event_loop_pp`)
   - For multi-stage pipeline parallelism
   - Manages micro-batches across pipeline stages

#### Request Processing

**Request Reception** (`recv_requests`)
- Receives requests from tokenizer via ZMQ sockets
- Broadcasts requests across TP (tensor parallel) ranks
- Supports DP (data parallel) attention distribution

**Request Handling**
- `handle_generate_request`: Text generation requests
- `handle_embedding_request`: Embedding requests
- `handle_batch_generate_request`: Batch generation
- `handle_batch_embedding_request`: Batch embedding

**Request Validation**
- Input length validation (schedule_batch.py:1317-1325)
- Priority assignment (schedule_policy.py:104-141)
- Grammar/regex validation for constrained generation
- Multimodal input processing

#### Batch Management

**Batch Creation** (`get_next_batch_to_run`)
Flow:
1. Merge last prefill batch into running batch
2. Get new prefill batch from waiting queue
3. If no prefill, prepare decode batch
4. Handle DP attention preparation if needed

**Prefill Batching** (`get_new_batch_prefill`)
- Checks grammar queue readiness (scheduler.py:1706-1707)
- Applies scheduling policy to waiting queue (scheduler.py:1737)
- Uses `PrefillAdder` to select requests (scheduler.py:1746-1813)
- Handles chunked prefill for long contexts
- Supports mixed prefill+decode batches

**Decode Batching** (`update_running_batch`)
- Filters finished requests
- Checks memory availability
- Handles retraction if out of memory (scheduler.py:1901-1923)
- Prepares batch for next decode step

#### Memory Management

**Memory Pools**
- `req_to_token_pool`: Maps requests to token positions
- `token_to_kv_pool_allocator`: Manages KV cache memory
- `tree_cache`: Prefix caching (RadixCache, HiRadixCache, etc.)

**Cache Types**
- `RadixCache`: Standard prefix caching
- `HiRadixCache`: Hierarchical cache with host memory
- `SWARadixCache`: Sliding window attention cache
- `MambaRadixCache`: For Mamba/SSM models
- `ChunkCache`: For chunked prefill without radix

**Memory Operations**
- Allocation: `alloc_for_extend`, `alloc_for_decode`
- Eviction: `evict_from_tree_cache`
- Retraction: `retract_decode` (scheduler.py:1448-1518)

#### Special Features

**Speculative Decoding**
- Draft model integration
- Eagle/Eagle3 support
- Verification and acceptance logic

**Disaggregation**
- Prefill/Decode separation
- KV cache transfer between prefill and decode instances
- Bootstrap queue management

**LoRA Support**
- Dynamic adapter loading/unloading
- Per-request LoRA selection
- Batch LoRA capacity limits

**Multimodal**
- Image, video, audio input processing
- Token padding for multimodal embeddings
- Feature caching

**Priority Scheduling**
- Request priority handling
- Preemption support
- Queue management with priorities

### Key Methods

| Method | Line | Purpose |
|--------|------|---------|
| `event_loop_normal` | 960 | Main event loop (normal mode) |
| `event_loop_overlap` | 979 | Main event loop (overlap mode) |
| `recv_requests` | 1009 | Receive and broadcast requests |
| `process_input_requests` | 1119 | Process incoming requests |
| `handle_generate_request` | 1212 | Handle text generation request |
| `get_next_batch_to_run` | 1629 | Select next batch to execute |
| `get_new_batch_prefill` | 1704 | Create new prefill batch |
| `update_running_batch` | 1891 | Update running decode batch |
| `run_batch` | 1943 | Execute a batch on GPU |
| `process_batch_result` | 2068 | Process batch results |
| `flush_cache` | 2386 | Clear memory and cache |

---

## 2. Schedule Batch (`schedule_batch.py`)

**Location**: `python/sglang/srt/managers/schedule_batch.py`

### Purpose
Defines data structures for requests and batches throughout the scheduling pipeline.

### Key Classes

#### `Req` (Line 428)
Represents a single inference request with complete state tracking.

**Core Attributes**:
- `rid`: Request ID
- `origin_input_ids`: Original tokenized input
- `output_ids`: Generated tokens
- `fill_ids`: Combined input + output for next iteration
- `sampling_params`: Sampling configuration
- `finished_reason`: Completion status
- `multimodal_inputs`: Multimodal data if applicable

**Memory Tracking**:
- `req_pool_idx`: Index in request pool
- `prefix_indices`: Cached prefix tokens
- `extend_input_len`: Tokens to compute in next forward
- `last_node`: Tree cache node reference

**Logprob Tracking**:
- `return_logprob`: Whether to return logprobs
- `logprob_start_len`: Start position for logprob computation
- `input_token_logprobs_val/idx`: Input logprobs
- `output_token_logprobs_val/idx`: Output logprobs
- `top_logprobs_val/idx`: Top-k logprobs

**State Tracking**:
- `is_chunked`: Chunked prefill counter
- `is_retracted`: Whether request was preempted
- `cached_tokens`: Number of already cached tokens
- `spec_verify_ct`: Speculative decoding verification count

**Key Methods**:
- `init_next_round_input`: Prepare for next forward pass (line 711)
- `check_finished`: Check stop conditions (line 867)
- `reset_for_retract`: Reset state after preemption (line 899)

#### `ScheduleBatch` (Line 963)
Represents a batch of requests being processed together.

**Core Attributes**:
- `reqs`: List of Req objects
- `forward_mode`: EXTEND, DECODE, MIXED, or IDLE
- `req_to_token_pool`: Memory pool reference
- `token_to_kv_pool_allocator`: KV allocator reference
- `tree_cache`: Prefix cache reference

**Batch Tensors**:
- `input_ids`: Input token IDs (shape: [num_tokens])
- `seq_lens`: Sequence lengths (shape: [batch_size])
- `req_pool_indices`: Request indices (shape: [batch_size])
- `out_cache_loc`: Output KV cache locations
- `output_ids`: Output token IDs

**Prefill-Specific**:
- `prefix_lens`: Cached prefix lengths per request
- `extend_lens`: New tokens to process per request
- `extend_num_tokens`: Total new tokens
- `extend_logprob_start_lens`: Logprob start offsets

**Multimodal**:
- `multimodal_inputs`: List of MultimodalInputs per request

**Key Methods**:
- `init_new`: Factory method to create new batch (line 1076)
- `prepare_for_extend`: Prepare prefill batch (line 1196)
- `prepare_for_decode`: Prepare decode batch (line 1559)
- `mix_with_running`: Mix prefill with decode (line 1389)
- `filter_batch`: Remove finished requests (line 1625)
- `merge_batch`: Merge two batches (line 1695)
- `retract_decode`: Handle OOM by retracting requests (line 1448)

#### `ModelWorkerBatch` (Line 1848)
Simplified batch sent to model worker for forward pass.

Contains subset of ScheduleBatch data needed for inference:
- Forward mode and input tensors
- Sequence lengths and cache locations
- Sampling information
- Multimodal inputs
- LoRA IDs

#### Data Structures

**`MultimodalDataItem` (Line 188)**
- Single modality data (image, video, or audio)
- Contains features or precomputed embeddings
- Model-specific data dictionary

**`MultimodalInputs` (Line 289)**
- Collection of multimodal items
- Token IDs for special tokens (image, video, audio)
- Rope positions for models like Qwen2-VL

**Finish Reasons (Lines 95-165)**
- `FINISH_MATCHED_TOKEN`: Stopped on EOS/stop token
- `FINISH_MATCHED_STR`: Stopped on stop string
- `FINISHED_MATCHED_REGEX`: Stopped on regex match
- `FINISH_LENGTH`: Reached max length
- `FINISH_ABORT`: Aborted with error

### Data Flow

```
Req.origin_input_ids
    ↓
[Prefix Matching]
    ↓
Req.prefix_indices (cached)
Req.extend_input_len (new tokens)
    ↓
[Batch Creation]
    ↓
ScheduleBatch.input_ids (all new tokens concatenated)
ScheduleBatch.seq_lens (total length per request)
    ↓
[Model Forward]
    ↓
ScheduleBatch.output_ids (next tokens)
    ↓
[Update]
    ↓
Req.output_ids.append(new_token)
```

---

## 3. Schedule Policy (`schedule_policy.py`)

**Location**: `python/sglang/srt/managers/schedule_policy.py`

### Purpose
Implements request prioritization and selection strategies for batch creation.

### Components

#### `SchedulePolicy` (Line 79)

**Policies Available**:

**Cache-Aware Policies** (Line 64)
- `LPM` (Longest Prefix Match): Prioritize requests with longest cached prefix
- `DFS_WEIGHT` (Depth-First Search Weight): Tree-based prioritization

**Cache-Agnostic Policies** (Line 71)
- `FCFS` (First Come First Serve): FIFO order
- `LOF` (Longest Output First): Prioritize requests with largest max_new_tokens
- `RANDOM`: Random shuffle

**Key Methods**:
- `calc_priority`: Compute request priorities (line 104)
- `_compute_prefix_matches`: Match prefixes in tree cache (line 167)
- `_sort_by_longest_prefix`: Sort by prefix length (line 216)
- `_sort_by_dfs_weight`: DFS tree traversal (line 229)

**In-Batch Prefix Caching** (Line 167-213)
- Maintains temporary radix tree of waiting queue
- Deprioritizes requests with heavy prefix overlap
- Prevents redundant computation

#### `PrefillAdder` (Line 316)

**Purpose**: Incrementally add requests to a prefill batch while respecting memory and scheduling constraints.

**Initialization Parameters**:
- `page_size`: Memory page size
- `tree_cache`: Prefix cache
- `token_to_kv_pool_allocator`: Memory allocator
- `running_batch`: Current running decode batch
- `new_token_ratio`: Expected token generation ratio
- `rem_input_tokens`: Remaining input token budget
- `rem_chunk_tokens`: Remaining chunk token budget (for chunked prefill)

**Budget Tracking**:
- `rem_total_tokens`: Total available + evictable memory (line 376)
- `cur_rem_tokens`: Current available + evictable memory (line 398)
- `rem_input_tokens`: Remaining prefill token budget
- `rem_chunk_tokens`: Remaining chunk size budget

**Key Methods**:

`add_one_req` (Line 565)
- Check total token budget (input + max_new_tokens)
- Lock tree cache node to prevent eviction
- Handle hierarchical cache loading
- Handle chunked prefill truncation
- Update budgets

`add_one_req_ignore_eos` (Line 483)
- Special path for ignore_eos requests
- Tracks request states for memory estimation
- Pessimistic memory checking

`add_chunked_req` (Line 449)
- Add continuation of chunked request
- Truncate if exceeds chunk size

`preempt_to_schedule` (Line 654)
- Preempt running requests to serve higher priority request
- Release resources from preempted requests

**Return Values** (`AddReqResult`):
- `CONTINUE`: Successfully added, can add more
- `NO_TOKEN`: Out of memory
- `OTHER`: Hit other limits (max prefill tokens, chunk size)

### Configuration

**Environment Variables**:
- `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`: Clip max_new_tokens for conservative estimation (default: 4096)
- `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`: Threshold for in-batch prefix check (default: 32)
- `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`: Threshold for deprioritization (default: 32)

---

## 4. Scheduler Input Blocker (`scheduler_input_blocker.py`)

**Location**: `python/sglang/srt/managers/scheduler_input_blocker.py`

### Purpose
Provides a mechanism to temporarily block the scheduler from accepting new requests, used for coordinated operations across distributed schedulers.

### `SchedulerInputBlocker` (Line 25)

**States** (Line 94):
- `UNBLOCKED`: Normal operation, accepting requests
- `BLOCKED`: Not accepting new requests (queued in pending_reqs)
- `GLOBAL_UNBLOCK_BARRIER`: Waiting for global synchronization before unblock

**Key Methods**:

`handle` (Line 32)
- Process incoming requests and block control messages
- Queue requests if blocked
- Check for global unblock barrier

`_handle_recv_req` (Line 52)
- Route `BlockReqInput` to block/unblock logic
- Queue normal requests if blocked

`_execute_block_req` (Line 69)
- Transition to BLOCKED state

`_execute_unblock_req` (Line 73)
- Signal local arrival at unblock barrier
- Transition to GLOBAL_UNBLOCK_BARRIER state

`_handle_arrive_unblock_barrier` (Line 80)
- Transition back to UNBLOCKED
- Release all pending requests

### `input_blocker_guard_region` (Line 100)

Context manager for blocking/unblocking:
```python
with input_blocker_guard_region(send_to_scheduler):
    # Critical section - scheduler blocked
    pass
# Scheduler unblocked after exit
```

**Usage**: Coordinate batch generation across distributed schedulers.

---

## Request Flow

### End-to-End Flow

```
1. Tokenizer → Scheduler (TokenizedGenerateReqInput)
   ↓
2. Scheduler.handle_generate_request()
   - Create Req object
   - Validate input length
   - Process multimodal inputs
   - Initialize grammar if needed
   ↓
3. Scheduler._add_request_to_queue()
   - Add to waiting_queue or disagg queues
   ↓
4. Scheduler.get_new_batch_prefill()
   - Apply SchedulePolicy.calc_priority()
   - Use PrefillAdder to select requests
   - Create ScheduleBatch
   ↓
5. ScheduleBatch.prepare_for_extend()
   - Match prefixes in tree cache
   - Allocate memory
   - Build input tensors
   ↓
6. Scheduler.run_batch()
   - Call model_worker.forward_batch_generation()
   - Get next tokens
   ↓
7. Scheduler.process_batch_result_prefill()
   - Update Req.output_ids
   - Check finish conditions
   - Update tree cache
   - Stream output to detokenizer
   ↓
8. Merge to running_batch (if not finished)
   ↓
9. Loop decode steps:
   - update_running_batch()
   - prepare_for_decode()
   - run_batch()
   - process_batch_result_decode()
   ↓
10. Detokenizer → Client (finished response)
```

### Memory Flow

```
Request arrives
    ↓
[Prefix Matching in Tree Cache]
    ↓
prefix_indices (reuse cached)
extend_input_len (allocate new)
    ↓
[Allocation]
req_to_token_pool.alloc()
token_to_kv_pool.allocate()
    ↓
[Forward Pass]
Write KV to allocated positions
    ↓
[Cache Update]
tree_cache.insert() or cache_unfinished_req()
    ↓
[Request Completion]
tree_cache.cache_finished_req()
req_to_token_pool.free()
    ↓
[Memory available for reuse]
```

---

## Key Algorithms

### Continuous Batching

**Problem**: Maximize GPU utilization by dynamically batching requests of varying lengths.

**Solution**:
1. Maintain `running_batch` of decode requests
2. Continuously add new prefill requests when memory permits
3. Support mixed prefill+decode batches
4. Filter finished requests each iteration

**Implementation** (scheduler.py:1629-1696):
- Merge last prefill batch into running batch
- Check for new prefill opportunities
- Handle mixed mode if enabled

### Chunked Prefill

**Problem**: Long context prefills can block other requests.

**Solution**:
1. Split long prefill into chunks
2. Process one chunk, yield control
3. Continue in next iteration
4. Cache intermediate KV

**Implementation** (schedule_policy.py:622-650):
- Truncate request to chunk size
- Mark as chunked
- Re-add to batch in next iteration

### Prefix Caching (Radix Cache)

**Problem**: Many requests share common prefixes (system prompts, few-shot examples).

**Solution**:
1. Build radix tree of all cached sequences
2. Match longest prefix for new requests
3. Reuse cached KV for prefix
4. Only compute new tokens

**Implementation** (schedule_batch.py:711-736):
- `tree_cache.match_prefix()` finds cached prefix
- `prefix_indices` stores matched positions
- Only `extend_input_len` tokens computed

### Retraction/Preemption

**Problem**: Running out of memory mid-decode.

**Solution**:
1. Detect insufficient memory for next decode
2. Select requests to evict (longest output first)
3. Save request state, release memory
4. Re-add to waiting queue
5. Adjust new_token_ratio to be more conservative

**Implementation** (schedule_batch.py:1448-1518):
- Sort by output length, retract longest
- Cache KV to tree (or discard)
- Reset request state

### Speculative Decoding

**Problem**: Decode is memory-bound, underutilizes compute.

**Solution**:
1. Draft model generates k candidate tokens quickly
2. Target model verifies all k+1 positions in parallel
3. Accept longest matching prefix
4. Repeat

**Integration**:
- Draft worker generates candidates
- Scheduler prepares verification batch
- Acceptance in `forward_batch_speculative_generation`

---

## Configuration Options

### Server Args (Relevant to Scheduling)

| Argument | Default | Description |
|----------|---------|-------------|
| `schedule_policy` | "lpm" | Request scheduling policy |
| `schedule_conservativeness` | 1.0 | How conservative memory estimation is |
| `chunked_prefill_size` | None | Chunk size for chunked prefill |
| `enable_mixed_chunk` | False | Mix prefill and decode in same batch |
| `disable_overlap_schedule` | False | Disable overlap scheduling |
| `max_running_requests` | Auto | Max concurrent requests |
| `max_queued_requests` | None | Max queued requests (reject if exceeded) |
| `enable_priority_scheduling` | False | Enable request priorities |
| `priority_scheduling_preemption_threshold` | 0 | Priority diff threshold for preemption |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | 4096 | Clip max_new_tokens for scheduling |
| `SGLANG_INIT_NEW_TOKEN_RATIO` | 0.7 | Initial new token ratio estimate |
| `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR` | 0.5 | Minimum ratio factor |
| `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS` | 100 | Decay steps for ratio |
| `SGLANG_RETRACT_DECODE_STEPS` | 20 | Reserved decode steps for retraction |
| `SGLANG_TEST_RETRACT` | 0 | Test retraction mechanism |
| `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD` | 32 | In-batch prefix check threshold |

---

## Performance Optimizations

### Overlap Scheduling
**scheduler.py:979-1007**

Uses separate CUDA streams to overlap:
- Forward pass (forward_stream)
- CPU→GPU/GPU→CPU copy (copy_stream)
- CPU scheduling (default_stream)

Benefits: 10-20% higher throughput

### DP Attention
Splits attention computation across data parallel workers when batch contains both prefill and decode.

### Hierarchical Cache
**HiRadixCache**: Two-tier caching
- GPU: Hot cache
- Host memory: Cold cache
- Automatic promotion/demotion

### CUDA Graphs
Capture decode batches as CUDA graphs for lower kernel launch overhead.

---

## Monitoring and Metrics

### Key Metrics (when `enable_metrics=True`)

**Throughput**:
- `num_generated_tokens`: Total tokens generated
- `forward_ct_decode`: Number of decode steps
- `gen_throughput`: Tokens/second

**Latency**:
- `TimeStats.wait_queue_entry_time`: Time entering queue
- `TimeStats.forward_entry_time`: Time starting forward
- Queueing time, TTFT, E2E latency

**Memory**:
- `token_usage`: KV cache utilization
- `available_size`: Free memory tokens
- `evictable_size`: Evictable cache size

**Retraction**:
- `num_retracted_reqs`: Requests preempted
- `new_token_ratio`: Current estimation ratio

### Logging

Set via server args:
- `disable_log_stats`: Disable periodic stats logging
- `log_level`: Logging level
- Request-level tracing with `enable_trace`

---

## Advanced Features

### Disaggregation (Prefill/Decode Separation)

**Prefill Instance**:
- Receives requests, runs prefill
- Transfers KV cache to decode instance
- `disagg_prefill_bootstrap_queue`: Queue for KV transfer

**Decode Instance**:
- Receives KV cache from prefill
- Runs decode iterations
- `disagg_decode_prealloc_queue`: Queue for preallocation
- `disagg_decode_transfer_queue`: Queue for active transfers

**Transfer**:
- Supports gloo, nccl, ucx backends
- Pipelined with computation

### LoRA Adapters

**Batch Support**:
- `max_loras_per_batch`: Max LoRA adapters per batch
- Per-request `lora_id` selection
- Dynamic loading/unloading

### Grammar-Constrained Generation

**Flow**:
1. Request specifies JSON schema, regex, or EBNF
2. Grammar compiled asynchronously
3. Request waits in `grammar_queue`
4. Grammar applied during sampling

**Backends**:
- Outlines
- XGrammar
- LMFP

---

## Common Patterns

### Adding a New Scheduling Policy

1. Define policy in `SchedulePolicy` class
2. Implement sorting logic (e.g., `_sort_by_X`)
3. Add to `calc_priority` method
4. Test with `--schedule-policy X`

### Modifying Batch Preparation

1. Update `ScheduleBatch.prepare_for_extend/decode`
2. Ensure tensors match model worker expectations
3. Update `ModelWorkerBatch` if needed
4. Test prefill, decode, and mixed modes

### Adding Request Metadata

1. Add field to `Req.__init__`
2. Update `TokenizedGenerateReqInput` in io_struct
3. Handle in `handle_generate_request`
4. Propagate through batch creation

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: Retraction messages, slow throughput

**Solutions**:
- Increase `schedule_conservativeness` (more conservative estimation)
- Enable `chunked_prefill_size`
- Reduce `max_running_requests`
- Check `new_token_ratio` isn't too optimistic

### Low Throughput

**Causes**:
- Not enough batching
- Disabled overlap schedule
- Cache misses

**Solutions**:
- Check `max_prefill_tokens` not too small
- Enable `--enable-mixed-chunk`
- Use better `schedule_policy` (lpm vs fcfs)
- Profile with `--enable-metrics`

### Priority Scheduling Not Working

**Check**:
- `enable_priority_scheduling=True`
- Requests have `priority` field set
- `schedule_low_priority_values_first` matches expectation
- Check logs for preemption events

### Speculative Decoding Low Accept Rate

**Debug**:
- Check `avg_spec_accept_length` in metrics
- Verify draft model matches target model
- Adjust `speculative_accept_threshold`

---

## Code References

### Scheduler Core
- Main event loop: scheduler.py:960-1007
- Request handling: scheduler.py:1212-1567
- Batch creation: scheduler.py:1629-1889
- Memory management: scheduler.py:695-833

### Batch Management
- Req class: schedule_batch.py:428-960
- ScheduleBatch class: schedule_batch.py:963-1845
- Prepare methods: schedule_batch.py:1196-1605

### Policy and Selection
- Policy calculation: schedule_policy.py:104-141
- PrefillAdder: schedule_policy.py:316-710
- Memory budgeting: schedule_policy.py:376-418

---

## Summary

The SGLang scheduler system is a comprehensive request orchestration framework optimized for LLM serving. Key strengths:

1. **Continuous Batching**: Dynamically batch prefill and decode
2. **Prefix Caching**: Reuse computation via radix tree cache
3. **Memory Management**: Sophisticated allocation, eviction, and retraction
4. **Flexibility**: Supports numerous features (LoRA, multimodal, speculation, disaggregation)
5. **Performance**: Overlap scheduling, CUDA graphs, hierarchical cache

The modular design separates concerns:
- **Scheduler**: High-level orchestration
- **ScheduleBatch**: Data structures
- **SchedulePolicy**: Request selection
- **SchedulerInputBlocker**: Flow control

This architecture enables efficient LLM inference at scale while supporting advanced features and optimizations.
