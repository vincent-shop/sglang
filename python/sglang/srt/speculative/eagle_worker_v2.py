import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch
from torch.cuda import Stream as CudaStream

from sglang.srt.environ import (
    envs,  # Environment variables manager, provides SGLANG_ENABLE_OVERLAP_PLAN_STREAM, etc.
)
from sglang.srt.managers.schedule_batch import (
    ModelWorkerBatch,  # Contains input_ids, cache locs, seq_lens for forward pass
)
from sglang.srt.managers.scheduler import (
    GenerationBatchResult,  # Contains logits_output, next_token_ids, accept_lens from generation
)
from sglang.srt.managers.tp_worker import (
    TpModelWorker,  # Manages model execution with tensor parallelism across GPUs
)
from sglang.srt.model_executor.forward_batch_info import (  # CaptureHiddenMode: NULL/LAST/FULL, ForwardBatch: complete forward pass data
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.server_args import (
    ServerArgs,  # Server configuration including model paths, device settings, spec params
)
from sglang.srt.speculative.base_spec_worker import (  # Abstract base classes defining draft() and forward_batch_generation() interfaces
    BaseDraftWorker,
    BaseSpecWorker,
)
from sglang.srt.speculative.draft_utils import (
    DraftBackendFactory,  # Creates attention backends for draft model based on backend type (triton/flashinfer/etc)
)
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,  # CUDA graph capture/replay for draft forward passes, reduces kernel launch overhead
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,  # CUDA graph for draft extend phase (filling KV cache after verification)
)
from sglang.srt.speculative.eagle_info import (  # Core data structures: DraftInput has topk_p/index/hidden_states, VerifyInput has tree structure
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,  # Triton kernel: assigns KV cache locations for accepted tokens in extend phase
)
from sglang.srt.speculative.eagle_info_v2 import (
    fill_accepted_out_cache_loc,  # Triton kernel: maps accepted indices to their cache locations
)
from sglang.srt.speculative.eagle_info_v2 import (
    fill_new_verified_id,  # Triton kernel: extracts last accepted token ID for each sequence
)
from sglang.srt.speculative.eagle_info_v2 import (
    select_top_k_tokens_tmp,  # Temporary duplicate of select_top_k_tokens for tree construction (FIXME noted)
)
from sglang.srt.speculative.eagle_utils import (  # TreeMaskMode: FULL_MASK/QLEN_ONLY/QLEN_ONLY_BITPACKING, build_tree: creates tree attention mask
    TreeMaskMode,
    build_tree_kernel_efficient,
)
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,  # Enum: NONE/EAGLE/EAGLE3/STANDALONE/NGRAM, controls spec decoding type
)
from sglang.srt.speculative.spec_utils import (
    detect_nan,  # Checks logits for NaN values and raises error if found (debugging)
)
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,  # Context manager that patches tensor parallel group for draft model
)
from sglang.srt.speculative.spec_utils import (
    load_token_map,  # Loads vocabulary subset mapping for hot tokens from file/HF hub
)
from sglang.srt.utils.common import (
    empty_context,  # No-op context manager that yields immediately
)
from sglang.srt.utils.common import (
    fast_topk,  # Efficient top-k selection using torch.topk with optimizations
)
from sglang.srt.utils.common import (
    get_available_gpu_memory,  # Returns available GPU memory in GB using nvidia-ml-py
)
from sglang.srt.utils.common import (
    next_power_of_2,  # Rounds up to next power of 2 for GPU kernel block sizes
)

logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[Optional[CudaStream], contextlib.AbstractContextManager]:
    """
    Creates a separate CUDA stream for planning attention backend metadata.

    WHY THIS EXISTS: The overlap scheduler can prepare attention metadata (like KV indices,
    masks, positions) for the NEXT batch in a separate stream while the GPU is executing
    the CURRENT batch. This provides additional parallelism beyond just CPU-GPU overlap.

    WHEN ENABLED: Controlled by SGLANG_ENABLE_OVERLAP_PLAN_STREAM environment variable
    (default: False). Can provide additional speedup but adds complexity.

    WHAT IT DOES:
    - If enabled: Creates new CUDA stream + context manager for stream switching
    - If disabled: Returns None and nullcontext (no-op)

    Returns: (plan_stream, plan_stream_ctx) - stream and its context manager
    """
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream: CudaStream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.cuda.stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
    """
    Draft worker for EAGLE V2 overlap mode - manages the small draft model.

    WHY THIS EXISTS: In EAGLE, we have two models working together (draft + target).
    This class manages the DRAFT MODEL which:
    - Is small and fast (often just 1 layer)
    - Shares embeddings/lm_head with target model
    - Generates speculative tokens in a tree structure
    - Uses target model's hidden states for better accuracy

    KEY DIFFERENCE FROM V1 (EAGLEWorker): V2 is designed for overlap scheduling:
    - Integrates with plan_stream for async preparation
    - Uses V2-specific preparation methods (prepare_for_v2_draft)
    - Works with FutureMap for storing/retrieving draft outputs

    ROLE IN OVERLAP: Runs in forward_stream while scheduler prepares next batch on CPU.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        """
        Initializes the draft worker with model, attention backends, and CUDA graphs.

        WHY INITIALIZATION IS COMPLEX:
        1. Must share memory pools with target (req_to_token_pool, token_to_kv_pool)
        2. Must share embeddings and lm_head to save memory
        3. Must create separate attention backends for each speculation step
        4. Must capture CUDA graphs for each batch size
        5. Must handle hot token mapping if provided (vocabulary reduction)
        """
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        with empty_context():
            # Init draft worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner

        self.init_token_map()
        self.init_lm_head()

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(self.draft_runner.tp_group):
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_token_map(self):
        """
        Loads hot token mapping for vocabulary reduction.

        WHY THIS EXISTS: EAGLE can use a reduced vocabulary (e.g., 30K most common tokens
        instead of full 128K) for the draft model. This saves memory and computation.
        The "hot token map" is a list of indices mapping reduced vocab to full vocab.

        WHEN NEEDED: Optional optimization. If provided via --speculative-token-map,
        draft model only predicts hot tokens, then maps back to full vocabulary.

        EAGLE3 NOTE: EAGLE3 models have built-in hot token mapping, so user-provided
        maps are ignored.

        WHAT IT DOES:
        - EAGLE3: Uses model's built-in hot_token_id (set in init_lm_head)
        - EAGLE2: Loads from file if provided, otherwise None (full vocabulary)
        - Sets json_model_override_args to tell model about reduced vocab size
        """
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        """
        Shares embedding and lm_head weights between target and draft models.

        WHY THIS EXISTS: EAGLE draft model MUST share weights with target model for
        two critical reasons:
        1. MEMORY EFFICIENCY: Embeddings and lm_head are huge (billions of parameters),
           sharing saves massive amounts of GPU memory
        2. CONSISTENCY: Draft must use same token representations as target for
           accurate speculation

        THE FUSION ARCHITECTURE: EAGLE draft model architecture is:
            hidden = fc(concat(embed(token), target_hidden_state))
        So it needs the SAME embeddings as target to maintain consistency.

        WHAT IT DOES:
        - EAGLE3: May or may not share lm_head depending on model config
        - EAGLE2: Always shares both embedding and lm_head
        - Hot token handling: If using reduced vocab, slices lm_head to hot tokens only
        """
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        """
        Creates attention backends for draft model's decode and extend phases.

        WHY THIS EXISTS: EAGLE draft model has TWO types of attention operations:
        1. DECODE: Multi-step speculation (num_steps forward passes building the tree)
        2. EXTEND: Single forward pass to fill KV cache with accepted tokens

        Each needs its own attention backend because:
        - Decode: Uses multiple attn backends (one per step) for tree structure
        - Extend: Uses standard extend attention with actual accepted tokens

        WHAT IT DOES:
        - Creates draft_attn_backend via DraftBackendFactory.create_decode_backend()
        - Creates draft_extend_attn_backend via create_draft_extend_backend()
        - Sets tree_mask_mode to FULL_MASK (complete attention mask for tree)

        ATTENTION BACKEND: Could be triton/flashinfer/fa3/etc based on server config.
        """
        # Create multi-step attn backends and cuda graph runners

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """
        Captures CUDA graphs for draft and draft_extend forward passes.

        WHY THIS EXISTS: CUDA graphs are a performance optimization that records a sequence
        of CUDA operations and replays them with minimal CPU overhead. For EAGLE, this is
        critical because we run many small forward passes (one per speculation step) where
        kernel launch overhead would dominate.

        WHEN TO SKIP: If num_steps=1 or disable_cuda_graph=True, graphs aren't captured.

        WHAT IT CAPTURES:
        1. Draft graphs: For each batch size, captures all num_steps forward passes
        2. Draft extend graphs: For filling KV cache with accepted tokens

        MEMORY COST: Can consume several GB of GPU memory, that's why we log before/after.

        SPEEDUP: Reduces kernel launch overhead from ~10μs to ~1μs per forward, critical
        when running 3-5 forward passes per draft iteration.
        """
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture extend
        if self.draft_extend_attn_backend:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        """
        Runs the draft phase: generates tree of speculative tokens.

        WHY THIS EXISTS: This is the core of EAGLE's speedup. The small draft model
        predicts multiple future tokens in a tree structure, which the target model
        will then verify in parallel. This converts sequential generation (N forward
        passes) into parallel verification (1 forward pass).

        THE DRAFT FLOW:
        1. prepare_for_v2_draft() - sets up KV cache and ForwardBatch
        2. draft_forward() - runs num_steps forward passes building tree
        3. build_tree_kernel_efficient() - creates tree mask and navigation metadata
        4. Returns EagleVerifyInput with draft tokens + tree structure

        CUDA GRAPH OPTIMIZATION: If batch size matches a captured graph, uses fast replay
        instead of running forward passes from scratch.

        THE TREE STRUCTURE: With topk=2, num_steps=3:
            Step 0: Predict 2 tokens from root
            Step 1: For each token, predict 2 more (4 total)
            Step 2: For each token, predict 2 more (4 total)
            Select best 6 tokens overall → forms the tree sent to target

        OVERLAP INTEGRATION: tree_mask_buf and position_buf are taken from target's
        attention backend to write directly into CUDA graph buffers, saving copies.

        Returns: EagleVerifyInput ready for target model verification
        """
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            if self.speculative_num_steps > 1:
                # Skip attention backend init for 1-step draft,
                # `draft_forward` only does sample in this case.
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        """
        Executes multi-step forward passes to build the speculation tree.

        WHY THIS EXISTS: This is where the actual tree construction happens. We run the
        draft model multiple times (num_steps), where each iteration expands the tree
        by one level. The tree structure allows EAGLE to explore multiple token paths
        simultaneously rather than just a single linear prediction.

        THE MULTI-STEP LOOP:
        For each step i in range(num_steps):
            1. select_top_k_tokens_tmp() - picks best topk tokens and builds tree level
            2. If not last step: run draft model forward to get next predictions
            3. Extract topk best tokens from predictions
            4. Update hidden states for next iteration

        KV CACHE LAYOUT: out_cache_loc is reshaped from [bs*topk*num_steps] to
        [num_steps, bs*topk] so each step gets its own cache locations.

        HOT TOKEN HANDLING: If using vocabulary reduction, maps reduced vocab indices
        back to full vocabulary using hot_token_id.

        ORGANIZATION AT END:
        - Concatenates all scores/tokens/parents from all steps
        - Selects best (num_draft_tokens - 1) tokens overall
        - Sorts indices to maintain tree order

        WHY num_steps-1 FORWARD PASSES: We get 1 token from initial extend, then
        (num_steps - 1) from iteration, so no need for last forward.

        Returns: (parent_list, top_scores_index, draft_tokens)
            - parent_list: Parent indices for tree structure
            - top_scores_index: Which tokens were selected from full tree
            - draft_tokens: The actual token IDs to verify
        """
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens_tmp(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        """Placeholder for abstract method - not used in V2."""
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Initializes draft model's state after target model completes prefill.

        WHY THIS EXISTS: After target model processes the initial prompt, we need to:
        1. Fill draft model's KV cache with the prompt tokens
        2. Get draft model's hidden states for the last token
        3. Predict topk next tokens as starting point for future speculation

        This sets up the draft model so it's ready to start speculating in decode phase.

        WHEN IT'S CALLED: Called by EAGLEWorkerV2.forward_batch_generation() during
        prefill mode (not decode mode). This is the INITIALIZATION step.

        THE INPUT TRICK: Replaces last token of each sequence with the verified token:
            Original: [token1, token2, token3, ...]
            Modified: [token2, token3, ..., verified_token]
        This is because EAGLE draft model predicts NEXT token, so we shift the sequence.

        WHAT IT RETURNS: EagleDraftInput with:
        - topk_p, topk_index: Top-k predictions for next draft
        - hidden_states: Draft model's hidden states
        - verified_id: The sampled token from target
        - new_seq_lens, allocate_lens: Updated sequence lengths

        This becomes the spec_info for the next decode iteration.
        """
        # Construct input_ids
        pt = 0
        for i, extend_len in enumerate(batch.extend_seq_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], next_token_ids[i].reshape(1))
            )
            pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            allocate_lens=batch.seq_lens,
        )
        batch.spec_info = next_draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        logits_output, _ = self.draft_runner.forward(forward_batch)

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        """
        Updates draft model's KV cache and predictions after verification completes.

        WHY THIS EXISTS: After target verifies draft tokens, we need to:
        1. Fill draft model's KV cache with ALL accepted tokens (not just last one)
        2. Extract hidden state at the LAST accepted position for each sequence
        3. Generate new topk predictions from that position for next draft iteration

        This maintains draft model's state in sync with verified tokens.

        WHEN IT'S CALLED: Called by EAGLEWorkerV2.forward_batch_generation() after
        verify() completes. This is the THIRD step in the decode flow:
            Draft → Verify → Draft Extend

        THE SELECT_INDEX TRICK: We processed all draft_token_num tokens through draft
        model, but only care about hidden state at the last ACCEPTED token:
            select_index = seq_idx * num_draft_tokens + accept_lens - 1
        This gives us the hidden state of the actual next token.

        OVERLAP OPTIMIZATION: Preparation happens in plan_stream (separate CUDA stream)
        while GPU might still be running verification, then we wait and run the actual
        forward in main stream.

        WHAT IT DOES:
        1. prepare_for_extend_to_fill_draft_kvcache() in plan_stream
        2. Wait for plan_stream to finish
        3. Run draft model forward on all accepted tokens
        4. Select hidden states at last accepted position
        5. Generate topk predictions for next iteration
        6. Update batch_result.next_draft_input with new predictions

        SIDE EFFECT: Modifies batch_result.next_draft_input in-place, adding topk_p,
        topk_index, and hidden_states for the next draft phase.
        """
        # Batch 2: Draft extend
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
            )

        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

        # Run draft extend batch in the main compute stream
        draft_logits_output = self.draft_runner.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    """
    Top-level EAGLE worker for V2 overlap mode - orchestrates draft and target models.

    WHY THIS EXISTS: This is the main coordinator for EAGLE speculative decoding in
    overlap mode. It owns both the draft worker and target worker, and orchestrates
    the full EAGLE cycle: Draft → Verify → Draft Extend.

    RELATIONSHIP TO COMPONENTS:
    - Owns: EagleDraftWorker (this class created it)
    - References: TpModelWorker target_worker (passed in, not owned)
    - Called by: Scheduler via model_worker.forward_batch_generation()

    THE EAGLE V2 CYCLE (in decode mode):
    1. draft() - Draft worker generates tree of speculative tokens
    2. verify() - Target worker verifies tokens in parallel
    3. _draft_extend_for_decode() - Draft worker updates its state

    THE EAGLE V2 CYCLE (in prefill mode):
    1. Target worker processes prompt
    2. _draft_extend_for_prefill() - Initializes draft worker

    KEY V2 FEATURES:
    - Uses plan_stream for async preparation (optional)
    - Integrates with FutureMap for overlap scheduling
    - Pre-plans verify attention while draft is running
    - Corrects verify buffers after draft completes (overlap trick)
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        """
        Initializes V2 worker by creating draft worker and setting up overlap infrastructure.

        WHY SIMPLE: Most initialization happens in EagleDraftWorker.__init__. This class
        just wraps it and adds:
        - Dummy tensors for future use (num_new_pages_per_topk, extend_lens)
        - plan_stream for async preparation

        The actual complexity is in draft_worker initialization.
        """
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        """
        Main entry point for EAGLE generation - routes to decode or prefill flow.

        WHY THIS EXISTS: This is called by the scheduler for every generation step.
        It routes to different logic based on whether we're in prefill (first tokens)
        or decode (subsequent tokens) mode.

        WHEN IT'S CALLED: Called by scheduler's run_batch() → model_worker.forward_batch_generation().
        This happens in the forward_stream in overlap mode.

        DECODE MODE FLOW (most iterations):
        1. draft_worker.draft() - Generate speculative tree
        2. verify() - Target model verifies tree in parallel
        3. _draft_extend_for_decode() - Update draft state with accepted tokens
        Returns: GenerationBatchResult with verified tokens

        PREFILL MODE FLOW (first iteration):
        1. target_worker.forward_batch_generation() - Process prompt
        2. _draft_extend_for_prefill() - Initialize draft model
        Returns: GenerationBatchResult with first token + draft predictions

        THE MODE SWITCH: After prefill, all subsequent calls use decode mode until
        sequence finishes or new request arrives.

        OVERLAP BENEFIT: In decode mode, while this runs on GPU (forward_stream),
        the scheduler on CPU is already preparing the next batch.
        """
        if model_worker_batch.forward_mode.is_decode():
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            assert draft_input.is_draft_input()
            verify_input: EagleVerifyInput = self.draft_worker.draft(model_worker_batch)
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch, draft_input.allocate_lens)
            self.draft_worker._draft_extend_for_decode(model_worker_batch, batch_output)
            return batch_output
        else:
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                model_worker_batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output

    def verify(
        self,
        batch: ModelWorkerBatch,
        cur_allocate_lens: torch.Tensor,
    ):
        """
        Runs target model verification and determines which draft tokens to accept.

        WHY THIS EXISTS: This is the verification phase of EAGLE. The target model processes
        all draft tokens in parallel (instead of sequentially) and we compare its predictions
        with the draft's predictions to determine acceptance.

        THE VERIFICATION FLOW:
        1. prepare_for_v2_verify() in plan_stream - Pre-plan attention (WRONG tree_mask!)
        2. Wait for plan_stream and correct buffers with update_verify_buffers_to_fill_after_draft()
        3. Run target model forward on all draft tokens in parallel
        4. sample() - Compare target vs draft predictions, find accepted tokens
        5. Create next_draft_input for next iteration

        THE OVERLAP TRICK EXPLAINED:
        Problem: We want to start planning verify attention early (overlap with draft)
        But: tree_mask and positions depend on draft output which isn't ready yet!
        Solution: Plan with OLD values, then CORRECT them after draft finishes

        This is why we call update_verify_buffers_to_fill_after_draft() - it re-computes
        the tree-dependent buffers (custom_mask, positions) with the CORRECT values
        from the just-completed draft phase.

        MEMORY SAFETY: batch.seq_lens.record_stream() is critical! Since seq_lens was
        allocated in another stream (forward_stream), we must tell PyTorch's memory
        manager not to reuse it until current stream finishes.

        WHAT IT RETURNS: GenerationBatchResult containing:
        - logits_output: Target model's logits
        - next_token_ids: All accepted tokens (flattened)
        - accept_lens: How many tokens each sequence accepted
        - next_draft_input: Draft predictions for next iteration (will be filled by draft_extend)
        - allocate_lens: Current KV cache allocation sizes

        VERIFICATION EVENT: Records verify_done event so next iteration knows when it's
        safe to read seq_lens (synchronization for overlap).
        """
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(torch.cuda.current_stream())

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Run target verify batch in the main compute stream
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Sample
        if self.enable_nan_detection:
            detect_nan(logits_output)
        (
            predict,
            accept_length,
            accept_index,
        ) = verify_input.sample(batch, logits_output)
        new_seq_lens = batch.seq_lens + accept_length
        verify_done = torch.cuda.Event()
        verify_done.record()

        all_verified_id = predict[accept_index]
        verified_id = torch.empty_like(accept_length, dtype=torch.int32)
        fill_new_verified_id[(bs,)](
            all_verified_id,
            accept_length,
            verified_id,
            self.speculative_num_draft_tokens,
        )

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=cur_allocate_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            allocate_lens=cur_allocate_lens,
        )

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Moves accepted tokens' KV cache from draft pool to target pool.

        WHY THIS EXISTS: Draft and target models have SEPARATE KV cache pools (both use
        token_to_kv_pool_allocator but for different layers). When tokens are verified,
        we need to copy their KV cache values from draft's pool to target's pool so
        target model can use them in future forward passes.

        WHEN IT'S CALLED: This method is prepared but NOT currently used in V2 overlap mode.
        It's here for future optimization when we want to preserve draft KV cache instead
        of recomputing it.

        THE TWO-POOL DESIGN:
        - Draft pool: Holds KV cache for draft model (typically 1 layer)
        - Target pool: Holds KV cache for target model (e.g., 32 layers)
        - They're logically separate even though they share the allocator

        WHAT IT DOES:
        1. assign_extend_cache_locs() - Gets target KV cache locations for accepted tokens
        2. fill_accepted_out_cache_loc() - Maps accept_index to actual cache locations
        3. move_kv_cache() - Physical copy of KV data from draft to target pool

        COMPACTION: accept_index is sparse (has -1 for rejected tokens), so we compact
        it to only include accepted tokens before moving KV cache.

        WHY NOT USED YET: Current implementation re-extends through target model instead
        of moving KV cache. This method is for future optimization.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )
