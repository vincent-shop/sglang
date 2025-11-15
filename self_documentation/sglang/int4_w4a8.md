# INT4 W4A8 Fused Triton MoE Integration Guide

## Goal
Introduce a production-ready `int4_w4a8` quantization mode (4-bit weights, 8-bit activations) for the fused Triton MoE path. The feature must reuse the existing W4 handling from the GPTQ/AWQ (`int4_w4a16`) flow while inheriting per-token INT8 activation quantization from the `int8_w8a8` path. This document ties each task to the components described in `FUSED_MOE_TRITON_DOCUMENTATION.md` so the implementation stays aligned with the current architecture.

## Baseline References
- **Kernel Layer**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` (`fused_moe_kernel`, `fused_moe_kernel_gptq_awq`).
- **Invoke Wrapper & Quant Bookkeeping**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` (`invoke_fused_moe_kernel`, `fused_experts_impl`).
- **Quant Metadata**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` (`TritonMoeQuantInfo`, `try_get_optimal_moe_config`).
- **Weight/Tensor Loading**: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` + ModelOpt loader helpers.
- **Token Preparation & Reduction**: `moe_align_block_size.py`, `moe_sum_reduce_triton` (unchanged but affects scale tensor shapes and padding).

## Implementation Roadmap

### 1. Runtime Plumbing
1. **Quant mode flagging**
   - Extend `TritonMoeQuantInfo.dtype` (or equivalent enum) with `"int4_w4a8"`.
   - Flow the new flag through `fused_experts` → `invoke_fused_moe_kernel` so both GEMMs can toggle INT4×INT8 execution. Mirror the way `use_fp8_w8a8` and `use_int4_w4a16` are plumbed today.
2. **Autotune integration**
   - Update autotune key construction in `tuning_fused_moe_triton.py` to hash the new dtype so cached configs remain disjoint.
   - Make sure `try_get_optimal_moe_config` can look up JSON configs with `dtype="int4_w4a8"`; fall back to heuristics if no tuned entry exists.
3. **Torch dispatch**
   - Ensure both `torch.ops.sglang.inplace_fused_experts` and `outplace_fused_experts` receive the new quant metadata. The dispatcher already threads `moe_runner_config.quant_info`; add the new flag to that structure.

### 2. Activation Quantization (W4A8)
1. **Per-token INT8 scales**
   - Reuse the `int8_w8a8` path inside `invoke_fused_moe_kernel`: allocate activation scratch buffers, compute per-token scale (`amax -> scale = 127/max_abs`), and quantize activations before launching the Triton kernel. Follow the block described under *FP8/INT8 Channel-wise* in the system doc—`a_scale` must be shaped `[tokens_in_chunk, top_k, cdiv(K, block_k)]` when group quant is enabled.
2. **Padding alignment**
   - Keep the per-token scale tensor padded to the same `BLOCK_SIZE_M` boundary that `moe_align_block_size` enforces (see “Token Alignment”). The sorter may add padding tokens whose scales should be zero-filled to avoid garbage loads.
3. **Wrapper updates**
   - In `invoke_fused_moe_kernel`, gate the existing INT8 quant logic behind `use_int8_w8a8 or use_int4_w4a8`. This avoids duplicating kernels while keeping activation packing identical.

### 3. Kernel Updates
1. **Entry-point branching**
   - Add `use_int4_w4a8` checks alongside `use_int4_w4a16` in `fused_moe_kernel` signature/launch params. You can share most of the GPTQ/AWQ path except for the activation type and integer dot product type.
2. **Weight unpack + dequant**
   - Reuse the existing nibble unpack logic (two 4-bit weights per byte). Ensure the group-wise scale/zero-point fetch uses the same stride math as `int4_w4a16`.
   - Convert unpacked weights to `int8` (sign-extend 4 bits) before the dot product so Triton can perform `tl.dot(int8, int8)` just like the W8A8 path. If keeping weights in `int4_t/uint8_t`, insert explicit widening to `int32` accumulators.
3. **Accumulator arithmetic**
   - Accumulate in FP32 (`accumulator = tl.zeros(..., dtype=tl.float32)`) as described in the documentation’s quantization section. Apply activation and weight scales post-loop: `accumulator *= a_scale * b_scale` (per-channel) or incorporate scale multiplication inside the loop for block-wise quant.
4. **Interface invariants**
   - Maintain identical tile selection semantics (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`). The configuration system already expects these keys; only quantization-specific metadata (scale pointers, zero points) changes.
5. **Second GEMM**
   - Mirror the first GEMM changes for the `w2` pass. Remember that the activation tensor feeding GEMM2 is still INT8, but its scales differ due to the activation function.

### 4. Weight Formatting & Loader Changes
1. **Pack weights**
   - Align with ModelOpt’s GPTQ/AWQ layout: experts stored as `[E, K, N]` with two signed int4 values per byte in the K dimension. Ensure `block_shape[1]` (group size over K) matches the kernel’s expectation so scale lookups stay coalesced.
2. **Scale tensors**
   - For each expert and group, store `w_scale` (`float16/float32`) and optional `w_zp` (`int8`). Kernel code can disable zero-point subtraction by compiling with `has_zp=False` if we commit to symmetric quantization.
3. **Loader plumbing**
   - Update `FusedMoELayer.load_quantized_weights` (or equivalent loader in `layer.py`) to detect `int4_w4a8` checkpoints, convert them to the Triton-friendly layout, and populate `TritonMoeQuantInfo` with weight/activation scale tensors.
4. **Validation helpers**
   - Extend any ModelOpt conversion scripts so they emit the new dtype and the scale metadata required by the Triton backend.

### 5. Autotuning & Configs
1. **Add tuning entries**
   - Generate baseline configs under `configs/triton_*/` with naming `dtype=int4_w4a8`. Start from the W4A16 configs but prefer smaller `BLOCK_SIZE_K` (typically 64 or 128) to balance INT8 activation bandwidth.
2. **Heuristic fallback**
   - Update `get_default_config` to treat `int4_w4a8` like other quantized dtypes (favor `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=32` for throughput). Ensure `BLOCK_SIZE_K` stays divisible by the weight group size used for int4 packing.
3. **Autotune scripts**
   - Teach `tuning_fused_moe_triton.py` to synthesize representative benchmarks by supplying `--dtype int4_w4a8`, per-token activation scales, and group quant parameters. Reuse the chunking strategy described in the documentation when generating workloads.

### 6. Testing Strategy
1. **Kernel correctness**
   - Extend existing Triton kernel unit tests to cover random tensors with shared seeds. Compare against the reference `matmul_ogs` path (`moe_matmul.py`) to validate output within tolerance (`rtol=1e-2`, `atol=1e-2`).
2. **Integration**
   - Add an end-to-end MoE test that executes `fused_experts` with `use_int4_w4a8=True`, `top_k ∈ {1,2}`, and verifies parity against the CUTLASS W4A8 baseline.
3. **Autotune regression**
   - Include the new dtype in autotune CI, ensuring configs exist for key shapes (e.g., H100 8x14336x4096, L40 16x8192x4096).
4. **Performance watch**
   - Track latency vs. the existing `int8_w8a8` and `int4_w4a16` modes; document regressions above agreed thresholds.

### 7. Debugging Notes
- **Scale shape mismatches**: Follow the guidance in the system doc’s “Quantization Scale Shape Errors”; ensure activation scales align with `cdiv(K, block_k)`.
- **Expert padding**: When debugging zero outputs, inspect the EP filtering logic (`fused_moe_kernel:404`) to confirm expert IDs aren’t being set to `-1` by mistake.
- **Chunking side effects**: Last-chunk resizing (`fused_experts_impl:451`) must slice both activation quant buffers and scale tensors—add tests for batches not divisible by `CHUNK_SIZE`.

## Definition of Done
- `int4_w4a8` flag is plumbed through `fused_experts`, quant metadata structs, and Triton launch wrappers.
- Both GEMMs in `fused_moe_kernel` run INT4×INT8 with the correct unpacking, scaling, and FP32 accumulation.
- ModelOpt (or other loaders) produces properly packed INT4 weights and scale tensors recognized by the Triton backend.
- Tuning configs exist (or defaults work) for representative shapes; autotune tooling understands the new dtype.
- Unit and integration tests compare against the CUTLASS W4A8 baseline with acceptable accuracy and latency.
