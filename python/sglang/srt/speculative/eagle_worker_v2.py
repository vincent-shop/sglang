import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch
from torch.cuda import Stream as CudaStream

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
    select_top_k_tokens_tmp,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    detect_nan,
    draft_tp_context,
    load_token_map,
)
from sglang.srt.utils.common import (
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    next_power_of_2,
)

logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[Optional[CudaStream], contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream: CudaStream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.cuda.stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
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
        """Capture cuda graphs."""
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
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
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
        extend_lens = batch_result.accept_lens
        if extend_lens is None:
            extend_lens = torch.zeros(0, device=self.device, dtype=torch.int32)
        if extend_lens.numel() == 0 or batch_result.accepted_token_ids.numel() == 0:
            next_draft_input = batch_result.next_draft_input
            hidden_states = batch_result.logits_output.hidden_states
            hidden_size = (
                hidden_states.shape[-1]
                if hidden_states is not None and hidden_states.numel() > 0
                else 0
            )
            device = hidden_states.device if hidden_states is not None else self.device
            next_draft_input.topk_p = torch.empty(
                (0, self.topk), dtype=torch.float32, device=device
            )
            next_draft_input.topk_index = torch.empty(
                (0, self.topk), dtype=torch.int64, device=device
            )
            next_draft_input.hidden_states = torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype if hidden_states is not None else torch.float32,
                device=device,
            )
            next_draft_input.num_tokens_per_batch = 0
            return

        accept_indices = batch_result.accept_indices.to(torch.long)
        accept_index_flat = accept_indices[accept_indices != -1]
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states[accept_index_flat],
        )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.accepted_token_ids,
                extend_lens,
                self.draft_runner,
            )

        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

        # Run draft extend batch in the main compute stream
        draft_logits_output = self.draft_runner.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

        select_index = torch.cumsum(extend_lens, dim=0) - 1

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
        next_draft_input.num_tokens_per_batch = (
            int(extend_lens.max().item()) if extend_lens.numel() > 0 else 0
        )


class EAGLEWorkerV2(BaseSpecWorker):
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
        stride = self.speculative_num_draft_tokens
        if bs > 0 and stride > 0 and predict.shape[0] != bs * stride:
            tokens_per_seq = predict.shape[0] // max(bs, 1)
            predict = predict.reshape(bs, tokens_per_seq)
            if tokens_per_seq >= stride:
                predict = predict[:, :stride]
            else:
                pad = predict[:, -1:].expand(-1, stride - tokens_per_seq)
                predict = torch.cat([predict, pad], dim=1)
            predict = predict.reshape(-1)
        new_seq_lens = batch.seq_lens + accept_length
        verify_done = torch.cuda.Event()
        verify_done.record()

        if bs == 0 or stride == 0:
            accepted_tokens_flat = torch.empty(
                0, dtype=torch.int32, device=self.device
            )
            accepted_tokens_strided = accepted_tokens_flat
        else:
            lengths = accept_length.tolist()
            accepted_tokens_list = []
            accepted_tokens_strided = torch.empty(
                (bs, stride), dtype=torch.int32, device=self.device
            )
            for i in range(bs):
                valid = min(int(lengths[i]), stride)
                seq_indices = accept_index[i, :valid].to(torch.long)
                seq_tokens = predict[seq_indices]
                if valid > 0:
                    accepted_tokens_list.append(seq_tokens)
                    accepted_tokens_strided[i, :valid] = seq_tokens
                    pad_value = seq_tokens[-1]
                    if valid < stride:
                        accepted_tokens_strided[i, valid:] = pad_value
                else:
                    accepted_tokens_strided[i].fill_(0)
            accepted_tokens_flat = (
                torch.cat(accepted_tokens_list)
                if accepted_tokens_list
                else torch.empty(0, dtype=torch.int32, device=self.device)
            )
            accepted_tokens_strided = accepted_tokens_strided.reshape(-1)

        verified_id = torch.empty_like(accept_length, dtype=torch.int32)
        if bs > 0 and stride > 0:
            fill_new_verified_id[(bs,)](
                accepted_tokens_strided,
                accept_length,
                verified_id,
                stride,
            )

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=cur_allocate_lens,
            verify_done=verify_done,
        )
        next_draft_input.num_tokens_per_batch = self.speculative_num_draft_tokens

        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accepted_tokens_strided,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            allocate_lens=cur_allocate_lens,
        )
        result.accept_indices = accept_index
        result.accepted_token_ids = accepted_tokens_flat
        return result

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            accept_length: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = accept_index.numel()

        if size == 0:
            return

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
