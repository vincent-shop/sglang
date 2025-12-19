# SGLANG_ENABLE_SPEC_V2 Incompatibility with Qwen3-Next Hybrid Models

## Summary

`SGLANG_ENABLE_SPEC_V2=1` is **not compatible** with Qwen3-Next (and other hybrid GDN models). The standard EAGLE/NEXTN speculative decoding works correctly, but the V2 overlap scheduler crashes.

## Test Results

| Configuration | Result |
|---------------|--------|
| Without `SGLANG_ENABLE_SPEC_V2` | ✅ Works |
| With `SGLANG_ENABLE_SPEC_V2=1` | ❌ Crashes |

## Error

```
ValueError: Invalid forward mode: forward_batch.forward_mode=<ForwardMode.DRAFT_EXTEND_V2: 7>
```

Full stack trace:
```
File "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py", line 558, in forward_batch_generation
    self.draft_worker._draft_extend_for_decode(model_worker_batch, batch_output)
File "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py", line 453, in _draft_extend_for_decode
    forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
File "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_info_v2.py", line 188, in prepare_for_extend_to_fill_draft_kvcache
    draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
File "/sgl-workspace/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 544, in init_forward_metadata
    attn_backend.init_forward_metadata(forward_batch)
File "/sgl-workspace/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 92, in _forward_metadata
    raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")
ValueError: Invalid forward mode: forward_batch.forward_mode=<ForwardMode.DRAFT_EXTEND_V2: 7>
```

## Root Cause

The `hybrid_linear_attn_backend.py` (used by Qwen3-Next's GatedDeltaNet/linear attention layers) does not support the `ForwardMode.DRAFT_EXTEND_V2` mode that `EAGLEWorkerV2` uses.

Additionally, `EAGLEWorkerV2` is missing the `hybrid_gdn_config` handling that exists in the standard `EAGLEWorker`:

```python
# eagle_worker.py:698 - EXISTS in standard EAGLEWorker
if self.target_worker.model_runner.hybrid_gdn_config is not None:
    # ... update mamba state after verify
    self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(...)

# eagle_worker_v2.py - MISSING this handling
```

## Reproduction

### Working (without SPEC_V2)
```bash
cd /sgl-workspace/sglang
source .venv/bin/activate
SGLANG_ENABLE_JIT_DEEPGEMM=0 python -m sglang.bench_offline_throughput \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --num-prompts 10 \
  --random-input-len 256 \
  --random-output-len 64 \
  --dataset-name random \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algorithm NEXTN
```

### Crashing (with SPEC_V2)
```bash
cd /sgl-workspace/sglang
source .venv/bin/activate
SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_SPEC_V2=1 python -m sglang.bench_offline_throughput \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --num-prompts 10 \
  --random-input-len 256 \
  --random-output-len 64 \
  --dataset-name random \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algorithm NEXTN
```

## Environment

```
Python: 3.12.3
CUDA available: True
GPU 0: NVIDIA H200
GPU 0 Compute Capability: 9.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 12.9, V12.9.86
CUDA Driver Version: 550.163.01
PyTorch: 2.8.0+cu128
sglang: 0.5.4.post1
sgl_kernel: 0.3.16.post4
flashinfer_python: 0.4.1
triton: 3.4.0
transformers: 4.57.1
torchao: 0.9.0
xgrammar: 0.1.25
```

## Required Fixes

To support `SGLANG_ENABLE_SPEC_V2=1` with hybrid GDN models:

1. **Update `hybrid_linear_attn_backend.py`** to handle V2 forward modes:
   - `ForwardMode.DRAFT_EXTEND_V2`
   - Any other V2-specific modes

2. **Update `eagle_worker_v2.py`** to include `hybrid_gdn_config` handling:
   - Add `update_mamba_state_after_mtp_verify()` call after verification (similar to `eagle_worker.py:698`)

## Related Files

- `python/sglang/srt/speculative/eagle_worker.py` - Standard EAGLE worker with hybrid_gdn support
- `python/sglang/srt/speculative/eagle_worker_v2.py` - V2 worker missing hybrid_gdn support
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` - Missing V2 forward mode support
- `python/sglang/srt/models/qwen3_next.py` - Qwen3-Next model implementation
- `python/sglang/srt/models/qwen3_next_mtp.py` - Qwen3-Next MTP draft model

## Additional Note

There's also a minor warning when running without SPEC_V2:
```
length of new_indices: 8 != length of topk_p: 9, this should not happen
```

This occurs in `eagle_info.py:739` during batch filtering and may indicate a synchronization issue, though inference still completes successfully.
