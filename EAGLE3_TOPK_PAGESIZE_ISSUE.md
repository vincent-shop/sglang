# Unblocking triton + topk>1 + page_size>1 for Beta Spec Overlap

## What is Beta Spec Overlap?

The `--enable-beta-spec` feature (PR #11398) enables **overlap scheduling** for EAGLE speculative decoding, which allows CPU scheduling to overlap with GPU computation for better performance.

**Key Changes**:
- Introduces V2 EAGLE worker that overlaps draft and verify phases
- Adds `FutureMap` to store draft outputs that will be needed in the next iteration
- Stores EAGLE state (topk_p, topk_index, hidden_states, verified_id, new_seq_lens) in a circular buffer
- Allows the scheduler to prepare the next batch while GPU is still processing the current batch
- Controlled by `enable_beta_spec` server arg which sets `disable_overlap_schedule=False`
- **Designed to work with triton backend**, not flashinfer

**Modified Files in PR**:
- `python/sglang/srt/managers/overlap_utils.py` - FutureMap implementation
- `python/sglang/srt/managers/schedule_batch.py` - allocate_for_eagle_v2(), prepare_for_decode()
- `python/sglang/srt/speculative/eagle_info_v2.py` - EagleDraftInputV2Mixin with prepare_for_decode()
- `python/sglang/srt/layers/attention/` - verify buffer management for overlap

## Problem Description
Currently, **topk>1 with page_size>1 is blocked** for the triton backend. When attempting to run:

```bash
python -m sglang.launch_server \
    --attention-backend triton \
    --enable-beta-spec \
    --speculative-eagle-topk 2 \
    --page-size 2 \
    ...
```

**The server immediately fails with**:
```
ValueError: speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend.
```

**Blocking Location**: `python/sglang/srt/server_args.py:1268` in `_handle_speculative_decoding()`

### Secondary Issue (when using flashinfer workaround)
If you bypass the validation by using `--attention-backend flashinfer`, you hit a different bug:

```
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 4 but got size 2 for tensor number 1 in the list.
```

**Error Location**: `python/sglang/srt/models/llama_eagle3.py:87`
```python
hidden_states = torch.cat([embeds, hidden_states], dim=-1)
```

**Full Traceback**:
```
File /root/workspace/sglang/python/sglang/srt/managers/scheduler.py:3056 in run_scheduler_process
File /root/workspace/sglang/python/sglang/srt/managers/scheduler.py:1044 in event_loop_overlap
File /root/workspace/sglang/python/sglang/srt/managers/scheduler.py:2207 in run_batch
File /root/workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py:550 in forward_batch_generation
File /root/workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py:462 in _draft_extend_for_decode
File /root/workspace/sglang/python/sglang/srt/models/llama.py:469 in forward
File /root/workspace/sglang/python/sglang/srt/models/llama_eagle3.py:165 in forward
File /root/workspace/sglang/python/sglang/srt/models/llama_eagle3.py:87 in forward (midlayer)
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 4 but got size 2
```

## Relevant Files

### Core Files
1. **`python/sglang/srt/models/llama_eagle3.py`**
   - Lines 87: Error location (tensor concatenation in midlayer)
   - Lines 144-178: LlamaModel.forward() - prepares embeds and hidden_states
   - Lines 74-100: LlamaDecoderLayer.forward() (midlayer) - performs the concatenation

2. **`python/sglang/srt/speculative/eagle_worker_v2.py`**
   - Lines 435-487: _draft_extend_for_decode() - calls the problematic forward pass
   - Lines 309-386: draft_forward() - uses select_top_k_tokens_tmp to expand batch
   - Lines 451-456: prepare_for_extend_to_fill_draft_kvcache call
   - Line 462: draft_runner.model.forward call that triggers the error

3. **`python/sglang/srt/speculative/eagle_info_v2.py`**
   - Lines 168-190: prepare_for_extend_to_fill_draft_kvcache() - prepares forward batch
   - Lines 345-391: select_top_k_tokens_tmp() - expands batch size based on topk
   - Line 357: `hidden_states = hidden_states.repeat_interleave(topk, dim=0)` - batch expansion
   - Line 356: `input_ids = topk_index.flatten()` - creates expanded input_ids

4. **`python/sglang/srt/speculative/eagle_info.py`**
   - Lines 576-614: EagleDraftInput dataclass definition
   - Line 585: hidden_states shape comment: `# shape: (b, hidden_size)`

### Configuration Files
5. **`python/sglang/srt/server_args.py`**
   - Line 1268: Validation that blocks topk>1 with page_size>1 for non-flashinfer backends

## Root Cause Analysis

### The Problem Flow

1. **Draft Extend Phase** (`_draft_extend_for_decode`):
   - Input: `batch_result.logits_output.hidden_states` from target model (shape: [batch_size, hidden_size])
   - Creates `draft_input` with this hidden_states
   - Calls `prepare_for_extend_to_fill_draft_kvcache(batch, next_token_ids, num_draft_tokens, draft_runner)`

2. **Batch Preparation** (`prepare_for_extend_to_fill_draft_kvcache`):
   - Sets `batch.input_ids = predict` (next_token_ids)
   - For batch_size=1, num_draft_tokens=2: creates input_ids of length 2
   - Sets `batch.spec_info = self` (the draft_input with hidden_states of shape [1, hidden_size])
   - Creates forward_batch from this modified batch

3. **EAGLE3 Forward** (`LlamaModel.forward` in llama_eagle3.py):
   - Line 153: `embeds = self.embed_tokens(input_ids)` → shape [2, hidden_size] (2 tokens)
   - Line 160: `hidden_states = forward_batch.spec_info.hidden_states` → shape [1, hidden_size] (1 sequence)
   - Line 165-171: Passes both to midlayer
   - **Mismatch**: embeds has batch dimension 2, hidden_states has batch dimension 1

4. **Midlayer Concatenation** (line 87):
   - Tries: `torch.cat([embeds, hidden_states], dim=-1)`
   - Fails because dimension 0 (batch) doesn't match: embeds[2, hidden_size] vs hidden_states[1, hidden_size]

### Why topk=1 Works
When topk=1 and num_draft_tokens=4:
- extend_num_tokens = 1 * 4 = 4
- batch.input_ids has 4 tokens
- embeds shape: [4, hidden_size]
- But wait, hidden_states would still be [1, hidden_size]...

**Need to investigate**: How does topk=1 case avoid this mismatch?

## Investigation Steps

1. **Compare topk=1 vs topk>1 batch preparation**
   - Check how input_ids shape differs
   - Check how hidden_states should be expanded or tiled

2. **Check draft_forward vs draft_extend**
   - draft_forward uses select_top_k_tokens_tmp which expands hidden_states
   - draft_extend should also expand hidden_states to match input_ids

3. **Look for hidden_states expansion logic**
   - Search for `repeat_interleave` or `repeat` on hidden_states
   - Check if there's missing logic in prepare_for_extend_to_fill_draft_kvcache

4. **Compare with EAGLE2 implementation**
   - Check `python/sglang/srt/models/llama_eagle.py` for differences
   - See if EAGLE2 has similar issues or different handling

5. **Check attention backend differences**
   - Investigate why flashinfer backend is required for topk>1 + page_size>1
   - Check if there's special handling in flashinfer attention backend

## Hypothesis
The bug is in `prepare_for_extend_to_fill_draft_kvcache` (lines 168-190 in eagle_info_v2.py):
- When setting `batch.input_ids = predict`, it creates input_ids with shape based on num_draft_tokens
- But the hidden_states in `self` (the draft_input) is not expanded to match
- Should expand hidden_states similar to how select_top_k_tokens_tmp does it at line 357

## How to Test

### Working Command (from PR #11398 - EAGLE with topk=1)
```bash
source .venv/bin/activate
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python -m sglang.launch_server \
    --dtype float16 \
    --model-path unsloth/Meta-Llama-3.1-8B-Instruct \
    --attention-backend triton \
    --decode-log-interval 1 \
    --cuda-graph-bs $(seq -s ' ' 1 64) \
    --enable-beta-spec \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --host 127.0.0.1
```
**Result**: ✅ Works - server starts and processes requests successfully with topk=1

### Blocked Command (topk=2, page_size=2 with triton)
```bash
source .venv/bin/activate
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python -m sglang.launch_server \
    --dtype float16 \
    --model-path unsloth/Meta-Llama-3.1-8B-Instruct \
    --attention-backend triton \
    --decode-log-interval 1 \
    --cuda-graph-bs $(seq -s ' ' 1 64) \
    --enable-beta-spec \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 2 \
    --page-size 2
```
**Result**: ❌ Blocked by validation at server_args.py:1268
```
ValueError: speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results 
for paged attention backends. This combination is only supported for the 'flashinfer' backend.
```

### Workaround with flashinfer (also fails)
```bash
source .venv/bin/activate
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python -m sglang.launch_server \
    --model-path unsloth/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 2 \
    --page-size 2 \
    --enable-beta-spec \
    --attention-backend flashinfer \
    --dtype float16
```
**Result**: ❌ Crashes with tensor shape mismatch at llama_eagle3.py:87

**Note**: flashinfer is NOT the intended backend for beta-spec-overlap. The goal is to make triton backend work with topk>1 + page_size>1.

## Investigation Goals

### Primary Goal: Unblock triton + topk>1 + page_size>1
The validation at `python/sglang/srt/server_args.py:1268` states this combination is "unstable and produces incorrect results for paged attention backends". We need to:

1. **Understand why the validation exists** - What specific issue does it prevent?
2. **Fix the underlying issue** - Make it work correctly with triton backend
3. **Remove or update the validation** - Once fixed, allow triton backend with topk>1 + page_size>1

### Secondary Investigation: EAGLE3 tensor shape bug (flashinfer)
If using flashinfer as a workaround, there's a separate tensor shape mismatch bug that needs fixing:

**File**: `python/sglang/srt/speculative/eagle_info_v2.py`
**Function**: `prepare_for_extend_to_fill_draft_kvcache` (lines 168-190)
**Issue**: `self.hidden_states` is not expanded to match the batch size of extended input_ids
**Likely Fix**: Add logic similar to select_top_k_tokens_tmp:357 to expand hidden_states

## Next Steps

1. Research git history of server_args.py:1268 validation - when was it added and why?
2. Look for related issues or PRs about topk>1 + page_size>1 stability
3. Compare triton vs flashinfer attention backend implementations for paged attention
4. Test if the tensor shape fix alone enables triton backend to work
5. Identify what makes it "unstable" for triton and fix the root cause

