from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    assign_draft_cache_locs,
    generate_simulated_accept_index,
)
from sglang.srt.utils.common import fast_topk, is_cuda, is_hip, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
    )
    from sgl_kernel.top_k import fast_topk
elif is_hip():
    from sgl_kernel import verify_tree_greedy


@dataclass
class EagleDraftInputV2Mixin:
    """
    V2 mixin for EagleDraftInput that adds overlap-specific methods.

    WHY THIS EXISTS: The overlap scheduler requires different preparation logic than
    the sequential non-overlap version. This mixin adds V2-specific methods without
    modifying the base EagleDraftInput class.

    MIXED INTO: EagleDraftInput (defined in eagle_info.py)

    KEY DIFFERENCE FROM V1: Uses FutureMap to store/retrieve draft outputs between
    iterations, enabling CPU-GPU overlap.
    """

    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        """
        Pre-allocates KV cache slots for the next decode iteration.

        WHY THIS EXISTS: In overlap mode, we need to allocate KV cache BEFORE running
        the draft model because the scheduler might already be preparing the next batch.
        This happens in ScheduleBatch.prepare_for_decode() which is called on the
        CPU while the previous batch is still running on GPU.

        WHEN IT'S CALLED: Called by ScheduleBatch.prepare_for_decode() in the main
        scheduler loop, before the batch is sent to the model worker.

        WHAT IT DOES:
        1. Waits for previous verify to finish (sync point via verify_done event)
        2. Calculates how many new KV cache slots needed (ALLOC_LEN_PER_DECODE)
        3. Allocates slots from tree_cache
        4. Updates req_to_token_pool to include new allocations
        5. Updates allocate_lens to track total allocated space

        OVERLAP BENEFIT: This work happens on CPU while GPU is busy with previous batch,
        reducing idle time.
        """
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool

        bs = batch.batch_size()

        # TODO(lsyin): implement over-allocation
        # Now seq_lens and allocate_lens are correct
        batch.maybe_wait_verify_done()

        page_size = batch.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            new_allocate_lens = batch.seq_lens + self.ALLOC_LEN_PER_DECODE
            num_needed_tokens = (new_allocate_lens - self.allocate_lens).sum().item()
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                self.allocate_lens,
            )
            new_allocate_lens = batch.seq_lens + self.ALLOC_LEN_PER_DECODE
            new_allocate_lens_cpu = new_allocate_lens.cpu()
            allocate_lens_cpu = self.allocate_lens.cpu()
            extend_num_tokens = sum(new_allocate_lens_cpu - allocate_lens_cpu).item()
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                self.allocate_lens,
                allocate_lens_cpu,
                new_allocate_lens,
                new_allocate_lens_cpu,
                last_loc,
                extend_num_tokens,
            )

        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            self.allocate_lens,
            new_allocate_lens,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        self.allocate_lens = new_allocate_lens

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        """
        Prepares ForwardBatch for running the draft model in V2 overlap mode.

        WHY THIS EXISTS: The draft phase needs to set up KV cache locations and
        positions for the draft model's forward pass. In overlap mode, this preparation
        happens while the GPU may still be executing the previous verify batch.

        WHEN IT'S CALLED: Called by EagleDraftWorker.draft() at the start of draft phase.

        WHAT IT DOES:
        1. Allocates out_cache_loc buffer (bs * topk * num_steps slots)
        2. Uses assign_draft_cache_locs kernel to fill cache locations with tree-aware
           layout so topk>1 works correctly even with overlap
        3. Sets positions for each token (seq_lens repeated topk times)
        4. Creates ForwardBatch for draft model
        5. Checks if CUDA graph can be used

        Returns: (forward_batch, can_cuda_graph) tuple
        """
        bs = len(batch.seq_lens)

        # Assign cache locations
        batch.out_cache_loc = torch.empty(
            (bs * topk * num_steps,),
            dtype=torch.int64,
            device=batch.input_ids.device,
        )
        token_allocator = getattr(draft_model_runner, "token_to_kv_pool_allocator", None)
        if token_allocator is None:
            raise RuntimeError(
                "Draft model runner is missing token_to_kv_pool_allocator; "
                "cannot determine page_size for EAGLE overlap draft preparation."
            )

        page_size = getattr(token_allocator, "page_size", 1)
        if page_size != 1 and topk > 1:
            raise NotImplementedError(
                "EAGLE overlap mode currently expects page_size == 1. "
                "Support for page_size > 1 will be added separately."
            )

        extend_lens = torch.full(
            (bs,),
            topk * num_steps,
            dtype=torch.int64,
            device=batch.seq_lens.device,
        )
        num_new_pages_per_topk = torch.ones(
            (bs,), dtype=torch.int64, device=batch.seq_lens.device
        )

        assign_draft_cache_locs[(bs,)](
            batch.req_pool_indices,
            req_to_token_pool.req_to_token,
            batch.seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            batch.out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
            topk,
            num_steps,
            page_size,
            next_power_of_2(bs),
            next_power_of_2(num_steps),
        )

        # Get a forward batch
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ModelWorkerBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
    ):
        """
        Prepares ForwardBatch for draft extend phase after verification completes.

        WHY THIS EXISTS: After target model verifies and accepts some tokens, we need to
        run the draft model in "extend" mode to fill its KV cache with the accepted tokens.
        This is necessary because the draft model's KV cache only had speculative tokens,
        but now we know which ones were actually accepted and need to properly cache them.

        WHEN IT'S CALLED: Called by EagleDraftWorker._draft_extend_for_decode() after
        verification completes. This runs in the plan_stream (separate CUDA stream) to
        prepare while GPU might be busy.

        WHAT IT DOES:
        1. Updates batch.input_ids with all accepted tokens (predict)
        2. Updates seq_lens to include all draft tokens (temporarily, will be corrected)
        3. Sets extend_seq_lens to num_draft_tokens for each sequence
        4. Sets forward_mode to DRAFT_EXTEND_V2 (special mode for this phase)
        5. Creates ForwardBatch and initializes attention backend metadata

        THE EXTEND TRICK: We extend by num_draft_tokens, then later select only the
        last token's hidden state (at accept_length - 1 position) because that's the
        "real" next token that was verified.

        Returns: ForwardBatch ready for draft extend forward pass
        """
        seq_lens_cpu_ = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        batch.extend_seq_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.extend_prefix_lens = seq_lens_cpu_.tolist()
        batch.extend_num_tokens = extend_num_tokens
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    """
    V2 mixin for EagleVerifyInput that adds overlap-specific verify preparation.

    WHY THIS EXISTS: The verify phase in overlap mode needs special preparation
    that can happen in a separate CUDA stream (plan_stream) while the GPU is
    still executing the draft phase. This reduces CPU idle time.

    MIXED INTO: EagleVerifyInput (defined in eagle_info.py)
    """

    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        """
        Prepares ForwardBatch for target model verification in overlap mode.

        WHY THIS EXISTS: The verify phase needs to set up KV cache locations for all
        draft tokens and prepare the attention backend. In overlap mode, this preparation
        can happen in a separate CUDA stream (plan_stream) while the draft model is
        still running, enabling true CPU-GPU overlap.

        WHEN IT'S CALLED: Called by EAGLEWorkerV2.verify() inside plan_stream context.
        This happens BEFORE the draft phase completes, enabling overlap.

        WHAT IT DOES:
        1. Sets batch.input_ids to all draft tokens from tree
        2. Allocates out_cache_loc for all draft tokens (bs * draft_token_num)
        3. Calls assign_extend_cache_locs kernel to get cache indices
        4. Sets forward_mode to TARGET_VERIFY
        5. Creates ForwardBatch for verification
        6. Pre-plans attention backend (init_forward_metadata or replay_prepare)
           ⚠️ This uses FUTURE VALUES - tree_mask/positions aren't ready yet!

        THE OVERLAP TRICK: We plan with wrong tree_mask/positions (from previous batch),
        then correct them later in update_verify_buffers_to_fill_after_draft() after
        draft completes. This lets us start planning early for overlap.

        Returns: (verify_forward_batch, can_run_cuda_graph) tuple
        """
        # Assign cache locations
        bs = len(batch.req_pool_indices)
        batch.input_ids = self.draft_token
        device = batch.input_ids.device
        batch.out_cache_loc = torch.empty(
            (bs * self.draft_token_num,),
            dtype=torch.int64,
            device=device,
        )

        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.draft_token_num,
            batch.out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

        # Get a forward batch
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
    ):
        """
        Verifies draft tokens against target model's predictions and finds accepted tokens.

        WHY THIS EXISTS: This is the core verification logic of EAGLE. After the target
        model processes all draft tokens in parallel, we need to traverse the tree and
        determine which tokens to accept/reject based on agreement between draft and target.

        WHEN IT'S CALLED: Called by EAGLEWorkerV2.verify() after target model completes
        its forward pass.

        WHAT IT DOES:
        1. Reshapes draft_token into tree structure (bs, draft_token_num)
        2. Creates output buffers (predict, accept_index, accept_length)
        3. GREEDY MODE: Uses verify_tree_greedy - accepts if draft == argmax(target)
        4. SAMPLING MODE: Uses tree_speculative_sampling_target_only for rejection sampling
           - Applies temperature/top_k/top_p to target logits
           - Walks tree using retrive_next_token/retrive_next_sibling pointers
           - Accepts tokens probabilistically based on target vs draft distributions
        5. Optionally simulates acceptance for testing (SIMULATE_ACC_LEN > 0)
        6. Adds bonus token (always 1 extra token even if all draft rejected)

        THE TREE WALK: The retrive_* tensors encode the tree structure and guide
        traversal. At each node, we check if target accepts draft token, then follow
        retrive_next_token (accept) or retrive_next_sibling (reject) pointer.

        Returns: (predict, accept_length, accept_index) - the verified tokens
        """
        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict = torch.zeros(
            (bs * (self.spec_steps + 1),), dtype=torch.int32, device=device
        )
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

            verify_tree_greedy(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            # This is currently not used
            draft_probs = torch.empty_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # Include the bonus token
        accept_length.add_(1)
        return predict, accept_length, accept_index


@torch.compile(dynamic=True)
def select_top_k_tokens_tmp(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    """
    Selects top-k tokens at each step of tree construction during draft forward.

    WHY THIS EXISTS: EAGLE draft model runs multiple forward passes (num_steps times),
    and at each step we need to expand the tree by selecting the best topk tokens from
    each current branch. This builds the speculation tree level by level.

    WHEN IT'S CALLED: Called inside EagleDraftWorker.draft_forward() loop for each of
    speculative_num_steps iterations. This is the core tree-building logic.

    WHAT IT DOES:
    i=0 (First step after initial token):
        - Takes topk best tokens: input_ids = topk_index.flatten()
        - Duplicates hidden states topk times (each token needs its own state)
        - Creates parent indices [-1, 0, 1, ...] where -1 is root

    i>0 (Subsequent steps):
        - For each of topk tokens, we predicted topk more → topk² candidates
        - Computes combined scores: parent_score * child_score
        - Selects best topk from topk² candidates
        - Tracks which parent each selected token came from
        - Updates parent indices for tree structure

    THE TREE GROWTH: With topk=2:
        Step 0: 1 → 2 tokens
        Step 1: 2 → 4 candidates → select best 2
        Step 2: 2 → 4 candidates → select best 2

    FIXME NOTE: This is a duplicate of select_top_k_tokens in spec_utils.py and should
    be unified to avoid code duplication.

    Returns: (input_ids, hidden_states, scores, tree_info)
        tree_info contains (scores, tokens, parent_indices) for tree construction
    """
    # FIXME(lsyin): remove this duplicate code
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=hidden_states.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
            0, hidden_states.shape[0], step=topk, device=hidden_states.device
        ).repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


@triton.jit
def fill_new_verified_id(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    """
    Triton kernel that extracts the last accepted token ID for each sequence.

    WHY THIS EXISTS: After verification, we have a flattened array of ALL accepted tokens
    across the batch. We need to extract just the LAST accepted token for each sequence
    because that's the verified token we'll use as input for the next draft iteration.

    WHEN IT'S CALLED: Called in EAGLEWorkerV2.verify() after sample() completes, before
    constructing next_draft_input.

    WHAT IT DOES:
    For each sequence (pid):
        1. Reads how many tokens were accepted (accept_lens[pid])
        2. Calculates index of last accepted token: pid * num_draft_tokens + accept_length - 1
        3. Reads that token from verified_id array
        4. Writes to new_verified_id[pid]

    EXAMPLE: If sequence 0 accepted 3 tokens and num_draft_tokens=6:
        verified_id index = 0 * 6 + 3 - 1 = 2 (the 3rd accepted token)

    NOTE: Cannot fuse accept_lens in-place operations here because kernel reads it.
    """
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    accept_length = tl.load(accept_lens + pid)

    verified_id_idx = num_draft_tokens * pid + accept_length - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    """
    Triton kernel that maps accepted token indices to their KV cache locations.

    WHY THIS EXISTS: After verification, accept_index tells us WHICH tokens in the tree
    were accepted (by their position in draft_token array). We need to map these
    positions to their actual KV cache locations so we can move the KV cache data
    to the target model's cache.

    WHEN IT'S CALLED: Called in EAGLEWorkerV2.move_accepted_tokens_to_target_kvcache()
    when page_size=1 (currently not used in V2, but prepared for future use).

    WHAT IT DOES:
    For each accepted position (pid):
        1. Counts how many tokens before pid were accepted (by checking accept_index)
        2. Uses this count as destination index (dst)
        3. Reads source index from accept_index[pid]
        4. If valid (src > -1), reads cache location from out_cache_loc[src]
        5. Writes to compacted output array accepted_out_cache_loc[dst]

    EXAMPLE: accept_index = [0, -1, 2, 3, -1, 5]
        Position 0: dst=0, src=0 → accepted_out_cache_loc[0] = out_cache_loc[0]
        Position 2: dst=1, src=2 → accepted_out_cache_loc[1] = out_cache_loc[2]
        Position 3: dst=2, src=3 → accepted_out_cache_loc[2] = out_cache_loc[3]

    RESULT: Compacts sparse accepted tokens into dense array for efficient KV cache move.
    """
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """
    Triton kernel that assigns KV cache locations for extend operations.

    WHY THIS EXISTS: During verify phase and draft extend phase, we need to allocate
    KV cache for a range of tokens [start_offset, end_offset). This kernel reads the
    cache indices from req_to_token pool and writes them to out_cache_loc in a
    compacted, batch-aware manner.

    WHEN IT'S CALLED:
    1. In prepare_for_v2_verify() - for all draft tokens that need verification
    2. In move_accepted_tokens_to_target_kvcache() - for accepted tokens

    WHAT IT DOES:
    For each sequence (pid):
        1. Reads start and end positions for this sequence
        2. Calculates output offset by summing lengths of all previous sequences
           (This creates a compacted output without gaps)
        3. Reads cache indices from req_to_token[req_pool_idx][start:end]
        4. Writes to out_cache_loc at compacted position

    EXAMPLE with 2 sequences extending by 3 and 5 tokens:
        Seq 0: reads indices [10,11,12], writes to out_cache_loc[0:3]
        Seq 1: reads indices [20,21,22,23,24], writes to out_cache_loc[3:8]

    COMPACTION BENEFIT: Output is dense array with no gaps, efficient for batched operations.
    """
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE
