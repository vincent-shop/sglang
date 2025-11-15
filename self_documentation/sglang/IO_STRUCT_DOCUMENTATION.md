# SGLang IO Struct System Documentation

## Overview

The `io_struct.py` module defines the complete data structure layer for inter-process communication in SGLang. These structures flow between three core processes: **TokenizerManager**, **Scheduler**, and **DetokenizerManager**, forming the backbone of request processing in the inference pipeline.

**Location**: `python/sglang/srt/managers/io_struct.py`

## Purpose and Importance

### Why IO Structs Matter

1. **Process Boundary Communication**: SGLang's architecture separates concerns across different processes. IO structs provide a well-defined contract for data exchange between these boundaries.

2. **Type Safety**: Using dataclasses with explicit type annotations ensures type safety across process boundaries and makes the codebase more maintainable.

3. **Serialization Ready**: All structures are designed to be easily serializable for IPC (Inter-Process Communication), allowing efficient data transfer.

4. **Batch Processing**: The structures natively support both single requests and batched requests, enabling efficient throughput optimization.

5. **Extensibility**: New features (multimodal, LoRA, sessions) are added as fields to existing structures, maintaining backward compatibility.

## Data Flow Architecture

```
HTTP/API Request
       ↓
GenerateReqInput / EmbeddingReqInput
       ↓ [TokenizerManager]
TokenizedGenerateReqInput / TokenizedEmbeddingReqInput
       ↓ [Batching]
BatchTokenizedGenerateReqInput / BatchTokenizedEmbeddingReqInput
       ↓ [Scheduler]
BatchTokenIDOutput
       ↓ [DetokenizerManager]
BatchStrOutput / BatchMultimodalOutput
       ↓
HTTP/API Response
```

### Stage-by-Stage Flow

1. **API Layer → TokenizerManager**
   - Input: `GenerateReqInput` or `EmbeddingReqInput`
   - Contains raw text, image data, or token IDs
   - May contain batch or single request
   - Includes sampling parameters, LoRA paths, session info

2. **TokenizerManager → Scheduler**
   - Input: `BatchTokenizedGenerateReqInput` or `BatchTokenizedEmbeddingReqInput`
   - Contains tokenized text as `input_ids`
   - Processed multimodal inputs in `mm_inputs`
   - Normalized sampling parameters as `SamplingParams`
   - All requests are in batch format

3. **Scheduler → DetokenizerManager**
   - Input: `BatchTokenIDOutput`
   - Contains generated token IDs
   - Includes logprobs, finish reasons, token counts
   - Has metadata for incremental decoding

4. **DetokenizerManager → API Layer**
   - Input: `BatchStrOutput` or `BatchMultimodalOutput`
   - Contains decoded strings
   - Includes all metrics and metadata
   - Ready for HTTP response formatting

## Core Base Classes

### BaseReq

Base class for single requests.

```python
@dataclass
class BaseReq(ABC):
    rid: Optional[Union[str, List[str]]]  # Request ID
    http_worker_ipc: Optional[str]        # IPC channel identifier
```

**Key Methods**:
- `regenerate_rid()`: Generates new UUID-based request IDs

**Purpose**: Provides request tracking and routing information across all stages of processing.

### BaseBatchReq

Base class for batched requests.

```python
@dataclass
class BaseBatchReq(ABC):
    rids: Optional[List[str]]           # Batch of request IDs
    http_worker_ipcs: Optional[List[str]]  # IPC channels for batch
```

**Key Methods**:
- `regenerate_rids()`: Generates new UUIDs for all requests in batch

**Purpose**: Enables efficient batch processing while maintaining individual request tracking.

## Input Request Structures

### GenerateReqInput

**Primary input structure for text generation requests from API layer.**

**Key Fields**:

1. **Input Data** (exactly one must be specified):
   - `text`: Raw text string(s)
   - `input_ids`: Pre-tokenized input
   - `input_embeds`: Pre-computed embeddings

2. **Multimodal Data**:
   - `image_data`: Images (PIL, URL, base64, file path)
   - `video_data`: Video inputs
   - `audio_data`: Audio inputs
   - `modalities`: Automatic modality detection (`["image", "multi-images", "video"]`)

3. **Sampling Control**:
   - `sampling_params`: Temperature, top_p, top_k, etc.
   - `custom_logit_processor`: Advanced sampling control

4. **Logprobs Configuration**:
   - `return_logprob`: Whether to return log probabilities
   - `logprob_start_len`: Start position for logprobs (-1 = output only)
   - `top_logprobs_num`: Number of top logprobs per position
   - `token_ids_logprob`: Specific token IDs to track

5. **Advanced Features**:
   - `lora_path`/`lora_id`: LoRA adapter selection
   - `session_params`: Continual prompting sessions
   - `priority`: Request priority for scheduling
   - `bootstrap_*`: Disaggregated inference parameters

6. **Flags**:
   - `stream`: Enable streaming responses
   - `return_hidden_states`: Return model hidden states
   - `return_entropy`: Return token entropy
   - `background`: Background processing (OpenAI responses API)

**Critical Method: `normalize_batch_and_arguments()`**

This method is the key to SGLang's flexible batch processing:

```python
def normalize_batch_and_arguments(self):
    self._validate_inputs()           # Ensure valid input configuration
    self._determine_batch_size()      # Detect single vs batch
    self._handle_parallel_sampling()  # Handle n > 1 in sampling params

    if self.is_single:
        self._normalize_single_inputs()
    else:
        self._normalize_batch_inputs()  # Expands all fields to lists
```

**Normalization Behavior**:
- Single requests: Converts to simple values with defaults
- Batch requests: Expands all parameters to match batch size
- Parallel sampling (n > 1): Automatically replicates inputs n times
- Ensures all fields are list-aligned for batch processing

**Example**:
```python
# Input: Single text with n=3 parallel sampling
req = GenerateReqInput(
    text="Hello",
    sampling_params={"n": 3, "temperature": 0.8}
)
req.normalize_batch_and_arguments()

# After normalization:
# req.text = ["Hello", "Hello", "Hello"]
# req.batch_size = 3
# req.is_single = False (converted to batch)
# req.rid = [uuid1, uuid2, uuid3]
```

### TokenizedGenerateReqInput

**Output from TokenizerManager, input to Scheduler.**

**Key Fields**:
- `input_text`: Original text (for reference)
- `input_ids`: Tokenized input as list of integers
- `mm_inputs`: Processed multimodal data (ready for model)
- `sampling_params`: Fully parsed `SamplingParams` object
- `return_logprob`, `logprob_start_len`, etc.: Logprob configuration
- `session_params`: Parsed session parameters
- `lora_id`: Resolved LoRA adapter ID
- `trace_context`: Distributed tracing metadata

**Key Differences from GenerateReqInput**:
1. Text is tokenized to `input_ids`
2. Sampling params are parsed objects, not dicts
3. Multimodal data is preprocessed
4. LoRA paths are resolved to IDs
5. Single request only (batching happens at next layer)

### BatchTokenizedGenerateReqInput

**Batched version sent to Scheduler.**

```python
@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    batch: List[TokenizedGenerateReqInput]
```

**Features**:
- Implements `__len__`, `__getitem__`, `__iter__`
- Allows scheduler to iterate over individual requests
- Maintains batch-level metadata (rids, http_worker_ipcs)

### EmbeddingReqInput

**Input structure for embedding/cross-encoder requests.**

Similar to `GenerateReqInput` but specialized for embeddings:

**Key Differences**:
- `is_cross_encoder_request`: Flag for cross-encoder mode
- `dimensions`: Matryoshka embeddings dimension control
- `sampling_params`: Set to `max_new_tokens=0` (no generation)
- Supports text, input_ids, or multimodal data

**Usage**: Text embeddings, image embeddings, cross-encoder ranking.

## Output Structures

### BatchTokenIDOutput

**Output from Scheduler containing raw token IDs and metadata.**

**Key Fields**:

1. **Generated Tokens**:
   - `decode_ids`: New token IDs to decode
   - `output_ids`: Complete output sequence (when skip_tokenizer_init is on)
   - `read_offsets`: Position for incremental decoding

2. **Completion Status**:
   - `finished_reasons`: Why each request finished (stop token, length, etc.)
   - `decoded_texts`: Incrementally decoded text (if available)

3. **Token Counts**:
   - `prompt_tokens`: Input token count
   - `completion_tokens`: Generated token count
   - `cached_tokens`: KV cache hits
   - `spec_verify_ct`: Speculative decoding verification count
   - `spec_accepted_tokens`: Accepted speculative tokens

4. **Logprobs** (all parallel arrays):
   - `input_token_logprobs_val/idx`: Input token logprobs
   - `output_token_logprobs_val/idx`: Output token logprobs
   - `input_top_logprobs_val/idx`: Top-k input logprobs
   - `output_top_logprobs_val/idx`: Top-k output logprobs
   - `input_token_ids_logprobs_val/idx`: Specific token ID logprobs
   - `output_token_ids_logprobs_val/idx`: Specific token ID logprobs
   - `output_token_entropy_val`: Entropy values

5. **Advanced Features**:
   - `output_hidden_states`: Model hidden states
   - `placeholder_tokens_idx/val`: Multimodal token placeholders
   - `retraction_counts`: Request retraction tracking
   - `token_steps`: Training step IDs for weight tracking

6. **Detokenization Config**:
   - `skip_special_tokens`: Skip special tokens in output
   - `spaces_between_special_tokens`: Spacing control
   - `no_stop_trim`: Don't trim stop sequences

### BatchStrOutput

**Final output from DetokenizerManager containing human-readable text.**

**Key Fields**:
- `output_strs`: Decoded output strings
- `output_ids`: Optional token IDs
- `finished_reasons`: Finish reason dictionaries (serialized)
- All token counts, logprobs, and metadata from `BatchTokenIDOutput`

**Purpose**: Ready for HTTP response serialization.

### BatchMultimodalOutput

**Specialized output for multimodal generation (e.g., image generation).**

**Key Fields**:
- `outputs`: Can be strings, bytes, or structured dicts
- `decoded_ids`: Token IDs for multimodal outputs
- `return_bytes`: Whether to return raw bytes

**Usage**: Image generation models, structured multimodal outputs.

### BatchEmbeddingOutput

**Output for embedding requests.**

**Key Fields**:
- `embeddings`: List of embedding vectors or sparse dicts
- `prompt_tokens`: Input token count
- `cached_tokens`: Cache hit count
- `finished_reasons`: Completion status
- `placeholder_tokens_idx/val`: Multimodal token info

**Note**: No `completion_tokens` since no generation occurs.

## Control Request Structures

### Weight Management

**UpdateWeightFromDiskReqInput**: Load weights from disk
- `model_path`: Path to new weights
- `load_format`: Format specification
- `abort_all_requests`: Clear in-flight requests
- `weight_version`: Track weight versions
- `is_async`: Asynchronous update
- `keep_pause`: Keep scheduler paused
- `token_step`: Training step tracking

**UpdateWeightsFromDistributedReqInput**: Distributed weight updates
- `names`, `dtypes`, `shapes`: Weight tensor metadata
- `group_name`: Communication group

**UpdateWeightsFromTensorReqInput**: Direct tensor updates
- `serialized_named_tensors`: Serialized weight tensors
- Supports HTTP transmission

**UpdateWeightsFromIPCReqInput**: IPC-based updates (Checkpoint Engine)
- `zmq_handles`: ZMQ socket paths per device

### LoRA Management

**LoadLoRAAdapterReqInput**: Load LoRA adapter
- `lora_name`: Adapter name
- `lora_path`: Path to adapter
- `pinned`: Pin in memory
- `lora_id`: Unique identifier

**UnloadLoRAAdapterReqInput**: Unload LoRA adapter
- `lora_name`: Adapter to unload
- `lora_id`: Unique identifier

### Session Management

**OpenSessionReqInput**: Create continual prompting session
- `capacity_of_str_len`: Max session size
- `session_id`: Optional session ID

**CloseSessionReqInput**: Close session
- `session_id`: Session to close

### Cache Management

**FlushCacheReqInput**: Flush KV cache
**ClearHiCacheReqInput**: Clear high-priority cache

### Request Control

**AbortReq**: Abort running requests
- `abort_all`: Abort all or specific request
- `finished_reason`: Why abort occurred
- `abort_message`: Human-readable message

**BlockReqInput**: Block/unblock scheduler
- `type`: BLOCK or UNBLOCK

### Profiling and Debugging

**ProfileReq**: Control PyTorch profiler
- `type`: START_PROFILE or STOP_PROFILE
- `output_dir`: Where to save traces
- `num_steps`: Steps to profile
- `activities`: What to profile (CPU, CUDA, etc.)
- `profile_by_stage`: Stage-specific profiling
- `merge_profiles`: Merge multi-rank traces

**GetInternalStateReq**: Get scheduler state
**SetInternalStateReq**: Modify scheduler state

**ConfigureLoggingReq**: Runtime logging configuration
- `log_requests`: Enable request logging
- `log_requests_level`: Logging level
- `dump_requests_folder`: Where to dump requests

### Load Balancing

**GetLoadReqInput**: Query scheduler load
**GetLoadReqOutput**: Load statistics
- `dp_rank`: Data parallel rank
- `num_reqs`: Total requests
- `num_waiting_reqs`: Queued requests
- `num_tokens`: Token count

**WatchLoadUpdateReq**: Broadcast load updates
- `loads`: List of load stats from all ranks

## Advanced Features

### Multimodal Support

**Type Definitions**:
```python
ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, Dict]
MultimodalDataInputItem = Union[ImageDataInputItem, VideoDataInputItem, AudioDataInputItem]
MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],  # Multiple images per request
    List[MultimodalDataInputItem],         # One image per request
    MultimodalDataInputItem,               # Single image
]
```

**Flexible Input Formats**:
1. Single image: `image_data=img`
2. Batch of single images: `image_data=[img1, img2, img3]`
3. Batch of multi-images: `image_data=[[img1, img2], [img3, img4]]`

**Automatic Modality Detection**:
- Single image: `modalities=["image"]`
- Multiple images: `modalities=["multi-images"]`
- Video: `modalities=["video"]`

### Session Parameters

**SessionParams**:
```python
@dataclass
class SessionParams:
    id: Optional[str]              # Session identifier
    rid: Optional[str]             # Request ID within session
    offset: Optional[int]          # Token offset in session
    replace: Optional[bool]        # Replace mode
    drop_previous_output: Optional[bool]  # Drop previous outputs
```

**Purpose**: Enable continual prompting where context is maintained across multiple requests.

### Parallel Sampling

When `sampling_params.n > 1`, the request is automatically expanded:

```python
# Input
GenerateReqInput(text="Hello", sampling_params={"n": 3})

# After normalization
text = ["Hello", "Hello", "Hello"]
batch_size = 3
rid = [rid_0, rid_1, rid_2]
```

**Rules**:
- All inputs replicated n times
- Each parallel sample gets unique rid
- Cannot use list parameters with n > 1 (raises ValueError)

### Disaggregated Inference

**Bootstrap Parameters**:
- `bootstrap_host`: Prefill node hostname
- `bootstrap_port`: Prefill node port
- `bootstrap_room`: Room ID for coordination
- `bootstrap_pair_key`: Pairing key for security

**Purpose**: Enable prefill/decode disaggregation for efficiency.

### Data Parallel Routing

**Field**: `data_parallel_rank`

**Purpose**: Route specific requests to specific data parallel replicas for sticky routing or debugging.

## Usage Patterns

### Creating a Basic Request

```python
from sglang.srt.managers.io_struct import GenerateReqInput

req = GenerateReqInput(
    text="What is machine learning?",
    sampling_params={"temperature": 0.7, "max_new_tokens": 100},
    stream=True
)
req.normalize_batch_and_arguments()
# Ready to send to TokenizerManager
```

### Batch Request with Images

```python
req = GenerateReqInput(
    text=["Describe this image.", "What do you see?"],
    image_data=[img1, img2],
    sampling_params={"temperature": 0.8, "max_new_tokens": 50}
)
req.normalize_batch_and_arguments()
# Automatically handles batch normalization
```

### LoRA Request

```python
req = GenerateReqInput(
    text="Write a poem about AI",
    lora_path="/path/to/poetry_lora",
    sampling_params={"max_new_tokens": 200}
)
req.normalize_batch_and_arguments()
# TokenizerManager will resolve lora_path to lora_id
```

### Session-based Request

```python
req = GenerateReqInput(
    text="Continue the story: Once upon a time",
    session_params={"id": "story_session_123"},
    sampling_params={"max_new_tokens": 100}
)
req.normalize_batch_and_arguments()
# Context maintained across multiple requests
```

### Logprobs Request

```python
req = GenerateReqInput(
    text="The capital of France is",
    return_logprob=True,
    logprob_start_len=0,  # Include input logprobs
    top_logprobs_num=5,   # Top 5 at each position
    sampling_params={"max_new_tokens": 10}
)
req.normalize_batch_and_arguments()
```

## Design Principles

### 1. Flexibility

The structures support multiple input formats:
- Single or batch
- Text, token IDs, or embeddings
- With or without multimodal data
- Various sampling configurations

### 2. Explicit Defaults

All optional fields have clear defaults, reducing boilerplate while maintaining clarity.

### 3. Progressive Transformation

Data structures become more specific as they flow through the pipeline:
- API layer: Flexible, user-friendly
- TokenizerManager: Tokenized, normalized
- Scheduler: Batched, ready for inference
- DetokenizerManager: Decoded, ready for response

### 4. Batch-First Design

All processing is designed for batches, with single requests as a special case (batch_size=1). This enables:
- Unified code paths
- Efficient GPU utilization
- Easy parallel sampling

### 5. Extensibility

New features are added as optional fields with defaults, maintaining backward compatibility:
- LoRA support added without breaking existing code
- Multimodal support seamlessly integrated
- Sessions added transparently

## Performance Considerations

### Memory Efficiency

**Shallow Copies**: Use of lists and references rather than deep copies minimizes memory overhead.

**Incremental Decoding**: `read_offsets` enable streaming without storing full outputs.

**Sparse Logprobs**: Only requested logprobs are computed and stored.

### Batch Optimization

**Dynamic Batching**: Requests can be batched dynamically in TokenizerManager.

**Aligned Fields**: All batch fields are aligned, enabling vectorized operations.

### Serialization

**Dataclasses**: Native serialization support via dataclass libraries.

**Flat Structures**: Minimal nesting reduces serialization overhead.

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Normalize

**Problem**: Using `GenerateReqInput` directly without calling `normalize_batch_and_arguments()`.

**Solution**: Always normalize immediately after creation:
```python
req = GenerateReqInput(...)
req.normalize_batch_and_arguments()  # REQUIRED
```

### Pitfall 2: List Parameters with Parallel Sampling

**Problem**: Passing list of sampling params when n > 1.

**Error**: `ValueError: Cannot use list token_ids_logprob with parallel_sample_num > 1`

**Solution**: Use single dict when using parallel sampling:
```python
# Wrong
sampling_params=[{...}, {...}]  # with n > 1

# Correct
sampling_params={..., "n": 3}
```

### Pitfall 3: Multimodal Batch Size Mismatch

**Problem**: `image_data` length doesn't match `text` length.

**Error**: `ValueError: The length of image_data should be equal to the batch size.`

**Solution**: Ensure multimodal data matches batch size or use single input for all:
```python
# Correct
text=["Q1", "Q2"], image_data=[img1, img2]

# Also correct (share image)
text=["Q1", "Q2"], image_data=img1
```

### Pitfall 4: Using Wrong RID in Batch

**Problem**: Using string rid for batch requests.

**Effect**: RIDs are auto-expanded as `{rid}_0`, `{rid}_1`, etc.

**Solution**: Either let it auto-generate or provide list of rids:
```python
# Auto-generate (recommended)
rid=None

# Manual list
rid=["req_1", "req_2", "req_3"]
```

## Validation and Error Handling

### Input Validation

**GenerateReqInput._validate_inputs()**:
- Ensures exactly one of text, input_ids, or input_embeds is specified
- Raises `ValueError` with clear message

**EmbeddingReqInput validation**:
- At least one of text, input_ids, or image_data required
- text and input_ids cannot both be provided

### Batch Size Validation

**Automatic Detection**:
- Detects batch size from first input type
- All other inputs must match or be single (broadcast)

### LoRA Validation

**Type Checking**:
- `lora_path` must be string or list
- Raises `ValueError` for invalid types

## Testing and Debugging

### Check Normalization

```python
req = GenerateReqInput(text="Hello", sampling_params={"n": 2})
req.normalize_batch_and_arguments()
print(f"Batch size: {req.batch_size}")
print(f"Is single: {req.is_single}")
print(f"Parallel samples: {req.parallel_sample_num}")
print(f"Text: {req.text}")
print(f"RIDs: {req.rid}")
```

### Inspect Tokenized Request

```python
tokenized_req = tokenizer_manager.tokenize_generate_req(req, ...)
print(f"Input IDs: {tokenized_req.input_ids}")
print(f"Sampling params: {tokenized_req.sampling_params}")
print(f"MM inputs: {tokenized_req.mm_inputs}")
```

### Trace Request Flow

Set `trace_context` field for distributed tracing across all stages.

## Related Files

- `tokenizer_manager.py`: Converts GenerateReqInput → TokenizedGenerateReqInput
- `scheduler.py`: Processes BatchTokenizedGenerateReqInput → BatchTokenIDOutput
- `detokenizer_manager.py`: Converts BatchTokenIDOutput → BatchStrOutput
- `http_server.py`: Creates GenerateReqInput from HTTP requests
- `openai/serving_*.py`: OpenAI API compatibility layer
- `schedule_batch.py`: Defines `BaseFinishReason` and batch scheduling

## Summary

The io_struct module is the **contract layer** between all major components of SGLang. It provides:

1. **Type Safety**: Clear types for inter-process communication
2. **Flexibility**: Support for diverse input formats and configurations
3. **Performance**: Batch-first design for GPU efficiency
4. **Extensibility**: Easy to add new features without breaking changes
5. **Traceability**: Request IDs flow through entire pipeline

Understanding these structures is essential for:
- Adding new features to SGLang
- Debugging request processing issues
- Optimizing performance
- Integrating with external systems

The normalization system in `GenerateReqInput` is particularly critical, as it handles the complex logic of converting diverse user inputs into a consistent internal format suitable for efficient batch processing.
