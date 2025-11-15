# Weight Loading System in SGLang

## Overview

SGLang's weight loading system is responsible for loading model weights from various sources and formats into memory for inference. The system is designed to be flexible, supporting multiple weight formats, quantization methods, and loading strategies.

**Location**: `python/sglang/srt/model_loader/`

**Main Entry Point**: `get_model()` in `__init__.py`

## Architecture

The weight loading system follows a loader pattern with several key abstractions:

```
get_model() → get_model_loader() → BaseModelLoader → load_model()
                                         ↓
                           [DefaultModelLoader, GGUFModelLoader, etc.]
                                         ↓
                              weight_utils.py (iterators)
```

## Key Components

### 1. Model Loaders (`loader.py`)

All loaders inherit from `BaseModelLoader` with two main responsibilities:
- `download_model()`: Download model if needed
- `load_model()`: Load weights into a PyTorch model

#### Available Loaders

**DefaultModelLoader** (lines 287-617)
- Handles standard model formats: safetensors, PyTorch (.pt, .bin), and ModelScope
- Supports multithread loading via `enable_multithread_load` config
- Main flow:
  1. `_prepare_weights()`: Download or locate weight files (lines 357-435)
  2. `_get_weights_iterator()`: Get iterator based on format (lines 437-484)
  3. `_get_all_weights()`: Combine primary and secondary weights (lines 486-499)
  4. `load_weights_and_postprocess()`: Load and post-process (lines 602-617)

**LayeredModelLoader** (lines 619-688)
- Loads weights layer-by-layer to reduce peak memory
- Used for memory-efficient quantization with TorchAO
- Requires model to implement `load_weights_to_module()` method
- Creates model on meta device first, then materializes layer by layer

**DummyModelLoader** (lines 691-735)
- Sets random weights for testing/benchmarking
- Uses `initialize_dummy_weights()` from `weight_utils.py`

**ShardedStateLoader** (lines 738-915)
- Loads pre-sharded checkpoints per TP rank
- Pattern: `model-rank-{rank}-part-{part}.safetensors`
- Much faster for large tensor-parallel models
- See `examples/runtime/engine/save_sharded_state.py`

**BitsAndBytesModelLoader** (lines 917-1390)
- Loads 4-bit/8-bit quantized models
- Supports pre-quantized and on-the-fly quantization
- Handles LoRA adapter configurations
- Creates `bnb_quant_state` attributes on parameters

**GGUFModelLoader** (lines 1393-1495)
- Loads GGUF format quantized models
- Maps GGUF tensor names to HuggingFace names
- Handles different GGUF architectures via `gguf.MODEL_ARCH_NAMES`

**ModelOptModelLoader** (lines 1774-2040)
- NVIDIA ModelOpt quantization (FP8, FP4)
- Supports calibration-based quantization
- Can restore from checkpoint or quantize on-the-fly
- Export to HuggingFace format via `export_path`

**RemoteModelLoader** (lines 1597-1739)
- Loads from remote storage (S3, KV stores)
- Two connectors:
  - `ConnectorType.KV`: Key-value storage
  - `ConnectorType.FS`: Filesystem-based (S3)

**RemoteInstanceModelLoader** (lines 1498-1594)
- Loads weights from another running sglang instance
- Uses NCCL for weight transfer between instances
- Requires matching TP configuration

### 2. Weight Utilities (`weight_utils.py`)

#### Downloading (lines 402-495)

**download_weights_from_hf()** (lines 402-461)
- Downloads model weights from HuggingFace Hub
- Uses file locks to prevent concurrent downloads
- Supports `allow_patterns` to filter weight files
- Checks local cache first in CI environments via `find_local_hf_snapshot_dir()`

**download_safetensors_index_file_from_hf()** (lines 464-495)
- Downloads the `model.safetensors.index.json` file
- Used to filter duplicate safetensors files

#### Weight Iterators

All iterators return `Generator[Tuple[str, torch.Tensor], None, None]`

**safetensors_weights_iterator()** (lines 610-644)
- Loads safetensors files sequentially
- Supports optional mmap disable via `disable_mmap` parameter
- Uses `safetensors.safe_open()` for memory-efficient loading

**multi_thread_safetensors_weights_iterator()** (lines 647-699)
- Parallel loading with ThreadPoolExecutor
- Configurable `max_workers` (default: 4 from DefaultModelLoader)
- Significantly faster for models with many shards

**pt_weights_iterator()** (lines 702-717)
- Loads PyTorch .bin/.pt files
- Uses `torch.load()` with `weights_only=True` for security

**multi_thread_pt_weights_iterator()** (lines 720-750)
- Parallel version of pt_weights_iterator

**np_cache_weights_iterator()** (lines 550-595)
- Converts torch weights to numpy for faster loading
- Caches numpy arrays in `np/` subdirectory
- Uses file lock for concurrent access safety

**gguf_quant_weights_iterator()** (lines 765-796)
- Loads GGUF quantized weights
- Yields both weight and qweight_type tensors
- Uses `gguf.GGUFReader` from gguf library

**runai_safetensors_weights_iterator()** (lines 881-899)
- Uses RunAI Model Streamer for efficient loading
- Configured via environment variables

#### Quantization Config (lines 156-256)

**get_quant_config()** (lines 156-256)
- Reads quantization config from model's HF config or checkpoint
- Handles multiple formats:
  - Standard HF `quantization_config`
  - Vision models with `text_config.quantization_config`
  - Compressed-tensors `compression_config`
  - BitsAndBytes adapter configs
  - ModelOpt configs (FP8, FP4)
- Returns appropriate `QuantizationConfig` subclass

#### Helper Functions

**convert_bin_to_safetensor_file()** (lines 99-138)
- Converts PyTorch checkpoint to safetensors
- Handles shared pointers and ensures contiguity
- Validates file size and tensor equality

**filter_duplicate_safetensors_files()** (lines 503-521)
- Filters safetensors files using index file
- Prevents loading both sharded and consolidated files

**filter_files_not_needed_for_inference()** (lines 524-540)
- Removes training artifacts (optimizer.bin, scheduler.pt, etc.)

**initialize_dummy_weights()** (lines 926-955)
- Initializes random weights for benchmarking
- Uses per-parameter seeds for consistency across devices
- Handles FP8 and other low-precision dtypes

**maybe_remap_kv_scale_name()** (lines 958-1031)
- Remaps FP8 k/v_scale parameter names
- Handles deprecated `kv_scale` format
- Supports ModelOpt and Quark scale naming conventions

**kv_cache_scales_loader()** (lines 1109-1149)
- Loads KV cache scaling factors from JSON
- Validates schema against TP configuration
- Returns layer-to-scale mapping for current TP rank

### 3. Loader Utilities (`utils.py`)

**get_model_architecture()** (lines 82-102)
- Resolves model class from HF config architectures
- Checks ModelRegistry for native support
- Falls back to Transformers implementation if needed
- Handles special cases (e.g., quantized Mixtral)

**resolve_transformers_arch()** (lines 27-79)
- Handles models not natively supported by SGLang
- Loads custom modules from HF auto_map
- Checks backend compatibility
- May fall back to `TransformersForCausalLM` wrapper

**post_load_weights()** (lines 109-118)
- Calls model's `post_load_weights()` if available
- Handles special cases (DeepseekV3ForCausalLMNextN)

**set_default_torch_dtype()** (lines 18-24)
- Context manager to temporarily set PyTorch default dtype
- Ensures model initialization uses correct precision

### 4. Remote Instance Utilities (`remote_instance_weight_loader_utils.py`)

**trigger_init_weights_send_group_for_remote_instance_request()** (lines 11-43)
- Initializes NCCL group for weight transfer
- Each TP rank pair creates a world size 2 group
- Uses HTTP POST to seed instance

**trigger_transferring_weights_request()** (lines 46-69)
- Triggers actual weight transfer from seed instance
- Called in separate thread by TP rank 0

## Weight Loading Flow

### Standard Flow (DefaultModelLoader)

1. **Model Initialization** (`_initialize_model()` in loader.py:217-262)
   - Get model class from registry via `get_model_architecture()`
   - Create quantization config via `_get_quantization_config()`
   - Instantiate model with config and quant_config

2. **Weight Preparation** (`_prepare_weights()` in loader.py:357-435)
   - Check if model is local or needs download
   - Determine format: AUTO, SAFETENSORS, MISTRAL, PT, NPCACHE
   - Apply `allow_patterns` based on format
   - Download via `download_weights_from_hf()` if not local
   - Filter duplicate safetensors files if needed
   - Filter training artifacts

3. **Weight Iteration** (`_get_weights_iterator()` in loader.py:437-484)
   - Select iterator based on format:
     - NPCACHE → `np_cache_weights_iterator()`
     - Safetensors → `safetensors_weights_iterator()` or `multi_thread_safetensors_weights_iterator()`
     - PT/BIN → `pt_weights_iterator()` or `multi_thread_pt_weights_iterator()`
   - Apply prefix to weight names if needed
   - Yield (name, tensor) pairs

4. **Weight Loading** (`load_weights_and_postprocess()` in loader.py:602-617)
   - Call model's `load_weights(weights_iterator)` method
   - For each module with `quant_method`:
     - Use `device_loading_context()` to temporarily move to GPU if CPU offloaded
     - Call `quant_method.process_weights_after_loading(module)`
   - Clear NPU cache if applicable

5. **Post Processing**
   - Call `post_load_weights()` if model implements it
   - Set model to eval mode

### LayeredModelLoader Flow

1. Create model on **meta device** (no memory allocation)
2. Verify model has `load_weights_to_module()` method
3. Get all weights iterator
4. For each module (recursively):
   - Materialize on target device with `to_empty()`
   - Load weights for that module via `load_weights_to_module()`
   - Apply TorchAO quantization if configured
5. Result: only one layer in memory at a time during loading

### Remote Loading Flows

**RemoteInstanceModelLoader**:
1. Build NCCL group between seed and target instances
2. TP rank 0 triggers transfer request to seed instance
3. All ranks broadcast receive weights via `torch.distributed.broadcast()`
4. Synchronize and destroy process group
5. Call `post_load_weights()` if available

**RemoteModelLoader**:
- **KV Connector**: Load weights directly from key-value store by TP rank
- **FS Connector**: Stream weights from filesystem (S3), iterate and load

## Special Features

### Multithread Loading

Enabled via `load_config.model_loader_extra_config`:
```python
{
    "enable_multithread_load": True,
    "num_threads": 8  # Default: 8
}
```

Benefits:
- Faster loading for sharded models
- Parallel I/O reduces wall-clock time
- Trade-off: higher memory usage during loading

### CPU Quantization

Controlled by `SGL_CPU_QUANTIZATION` environment variable. Flow:
1. Initialize model on CPU (via `load_model_with_cpu_quantization()` at loader.py:1742-1771)
2. Load weights
3. Process quantization on CPU
4. Move to target device
5. Used by DummyModelLoader when flag is set

### Device Loading Context

`device_loading_context()` (loader.py:116-173):
- Temporarily moves CPU parameters to GPU for processing
- Quantization methods expect weights on target device
- Automatically restores to CPU after processing
- Handles pin_memory for faster transfers
- Tracks original device state to avoid corrupting already-moved params

### ModelOpt Quantization

Two modes:
1. **Pre-quantized**: Load existing ModelOpt checkpoint
2. **Calibration**: Quantize on-the-fly with calibration data

Workflow (`_setup_modelopt_quantization()` at loader.py:1783-1879):
- Try restore from `checkpoint_restore_path`
- If not available, run calibration:
  - Load calibration dataset (default: cnn_dailymail)
  - Create forward loop with dataloader
  - Call `mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)`
  - Save to `checkpoint_save_path` if provided
- Export to HuggingFace format if `export_path` provided

### Weight Filtering

**Duplicate Safetensors** (weight_utils.py:503-521):
- Some models have both sharded and consolidated safetensors
- Use `model.safetensors.index.json` to determine which files are needed
- Only load files referenced in the weight_map

**Training Artifacts** (weight_utils.py:524-540):
- Filter out: training_args.bin, optimizer.bin, optimizer.pt, scheduler.pt, scaler.pt
- Reduces download size and loading time

### Sharded Model Validation

`find_local_hf_snapshot_dir()` (weight_utils.py:259-399) checks:
- All shards present for models like `model-00001-of-00009.safetensors`
- No `.incomplete` files in blobs directory
- At least one valid weight file matching patterns
- Broken symlinks are skipped

### Lock-Based Concurrency

All download operations use file locks:
```python
with get_lock(model_name_or_path, cache_dir):
    # Download or convert operations
```

- Locks stored in temp directory (system-level)
- Mode 0o666 for cross-user sharing
- Prevents race conditions in multi-process scenarios

### Format Detection

DefaultModelLoader auto-detects format (loader.py:373-387):
1. Check `load_config.load_format`:
   - AUTO: try both .safetensors and .bin
   - SAFETENSORS: only .safetensors
   - MISTRAL: consolidated*.safetensors with special index
   - PT: only .pt files
   - NPCACHE: .bin with numpy caching
2. Glob for files matching patterns
3. First matching pattern determines format
4. Set `use_safetensors` flag accordingly

### Weight Remapping

**Prefix Remapping**: Replace key prefixes during loading
- Configured via model class's `remap_prefix` attribute
- Applied in quantization config loading

**FP8 Scale Remapping**: `maybe_remap_kv_scale_name()` (weight_utils.py:958-1031)
- Maps deprecated `kv_scale` to separate `k_scale`/`v_scale`
- Handles ModelOpt naming: `.self_attn.k_proj.k_scale` → `.self_attn.attn.k_scale`
- Supports Quark naming: `.q_proj.output_scale` → `.attn.q_scale`

### Model Selection

`get_model_loader()` (loader.py:2043-2095) selects loader based on:
1. ModelOpt quantization flags → `ModelOptModelLoader`
2. `load_format` is a custom class → instantiate it
3. DUMMY → `DummyModelLoader`
4. SHARDED_STATE → `ShardedStateLoader`
5. BITSANDBYTES → `BitsAndBytesModelLoader`
6. GGUF → `GGUFModelLoader`
7. LAYERED → `LayeredModelLoader`
8. REMOTE → `RemoteModelLoader`
9. REMOTE_INSTANCE → `RemoteInstanceModelLoader`
10. Default → `DefaultModelLoader`

## Key Design Patterns

### Iterator Pattern
Weight loading uses generators extensively:
- Memory efficient: one tensor at a time
- Supports streaming from remote sources
- Easy to compose (add prefixes, filter, etc.)

### Strategy Pattern
Different loaders for different formats/sources:
- Each loader encapsulates loading strategy
- Common interface via `BaseModelLoader`
- Selected at runtime based on config

### Factory Pattern
`get_model_loader()` creates appropriate loader:
- Centralizes loader selection logic
- Extensible: add new loaders by extending `BaseModelLoader`

### Context Manager Pattern
Used for resource management:
- `set_default_torch_dtype()`: restore dtype after use
- `device_loading_context()`: restore device after quantization
- File locks: automatically released
- Remote connectors: auto-close connections

## Configuration

Weight loading is configured via:

**LoadConfig**:
- `load_format`: Format to load (AUTO, SAFETENSORS, PT, etc.)
- `download_dir`: Cache directory for downloads
- `model_loader_extra_config`: Format-specific options
- `ignore_patterns`: Files to skip during download

**ModelConfig**:
- `model_path`: Model identifier or path
- `revision`: Git revision for HF models
- `quantization`: Quantization method
- `dtype`: Model data type

**DeviceConfig**:
- `device`: Target device (cuda, cpu, npu)
- `gpu_id`: GPU index for multi-GPU setups

## Performance Considerations

**Multithread Loading**:
- Faster for sharded models with many files
- Higher memory usage during loading
- Good for models with 10+ shard files

**Memory-Mapped Loading**:
- Default for safetensors (unless `disable_mmap=True`)
- Lower memory footprint
- Slightly slower for small files

**Layered Loading**:
- Essential for quantizing large models
- Trades loading time for peak memory
- Required for some quantization methods

**NPCACHE Format**:
- Converts weights to numpy on first load
- Subsequent loads are faster
- Requires extra disk space

**Sharded State Loading**:
- Fastest for large TP models
- Requires pre-sharding with `save_sharded_state.py`
- Each rank reads only its shard

## Error Handling

**Download Failures**:
- File locks prevent concurrent corruption
- `.incomplete` files detected and trigger re-download
- Offline mode via `HF_HUB_OFFLINE` environment variable

**Missing Weights**:
- `load_weights()` methods check for missing keys
- Sharded loaders validate all shards present
- Clear error messages with missing key names

**Format Mismatches**:
- Auto-detection tries multiple patterns
- Falls back to .pt if .safetensors not found
- Warns about incompatible formats

**Quantization Errors**:
- Validates GPU capability vs quantization requirements
- Checks dtype compatibility
- Provides clear error messages for unsupported configs

## Extension Points

To add a new loader:

1. Subclass `BaseModelLoader`
2. Implement `download_model()` and `load_model()`
3. Add to `get_model_loader()` selection logic
4. Add corresponding `LoadFormat` enum value if needed

To add a new weight format:

1. Implement weight iterator in `weight_utils.py`
2. Add format detection logic to `_prepare_weights()`
3. Add iterator call to `_get_weights_iterator()`

To add a new quantization method:

1. Create `QuantizationConfig` subclass
2. Register in quantization registry
3. Implement `process_weights_after_loading()` method
4. Add config loading logic to `get_quant_config()`

## References

- Adapted from vLLM's model loading system (see file headers)
- Uses HuggingFace Hub for model downloads
- Supports GGUF via gguf-py library
- Integrates with ModelOpt for NVIDIA quantization
- Uses safetensors for efficient serialization
