# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Molmo2 model compatible with HuggingFace weights."""

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from functools import partial

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers


# ==============================================================================
# Vision Backbone Components
# ==============================================================================


class Molmo2ViTMLP(nn.Module):
    """MLP for Vision Transformer blocks."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.w2(x)
        return x


class Molmo2ViTAttention(nn.Module):
    """Multi-head attention for Vision Transformer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        input_dim: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        input_dim = input_dim or hidden_size

        self.wq = nn.Linear(input_dim, num_heads * head_dim, bias=True)
        self.wk = nn.Linear(input_dim, num_kv_heads * head_dim, bias=True)
        self.wv = nn.Linear(input_dim, num_kv_heads * head_dim, bias=True)
        self.wo = nn.Linear(num_heads * head_dim, hidden_size, bias=True)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_kv is None:
            inputs_kv = inputs_q

        batch_size, seq_len, _ = inputs_q.shape

        q = self.wq(inputs_q)
        k = self.wk(inputs_kv)
        v = self.wv(inputs_kv)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.wo(attn_output)

        return attn_output


class Molmo2VisionBlock(nn.Module):
    """Single block in the Vision Transformer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        layer_norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention = Molmo2ViTAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
        )
        self.feed_forward = Molmo2ViTMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Molmo2VisionTransformer(nn.Module):
    """Vision Transformer for Molmo2."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_hidden_layers: int,
        image_num_pos: int,
        image_patch_size: int,
        layer_norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_patch_size = image_patch_size
        self.image_num_pos = image_num_pos

        # Patch embedding
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            hidden_size,
            bias=True,
        )

        # Positional embedding
        self.positional_embedding = nn.Parameter(
            torch.zeros(image_num_pos, hidden_size)
        )

        # Transformer blocks
        self.resblocks = nn.ModuleList(
            [
                Molmo2VisionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    layer_norm_eps=layer_norm_eps,
                    quant_config=quant_config,
                    prefix=add_prefix(f"resblocks.{i}", prefix),
                )
                for i in range(num_hidden_layers)
            ]
        )

    def add_pos_emb(self, x: torch.Tensor, patch_num: Tuple[int, int]) -> torch.Tensor:
        pos_emb = self.positional_embedding
        pos_emb_size = int(math.sqrt(pos_emb.shape[0]))
        pos_emb = pos_emb.reshape(pos_emb_size, pos_emb_size, pos_emb.shape[1])

        patch_num_0, patch_num_1 = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x

    def forward(
        self,
        x: torch.Tensor,
        patch_num: Optional[Tuple[int, int]] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            x: (batch_size, num_patch, n_pixels)
            patch_num: (patch_h, patch_w) for positional embedding interpolation
        Returns:
            List of hidden states from each transformer block
        """
        if patch_num is None:
            patch_num_side = int(math.sqrt(self.image_num_pos))
            patch_num = (patch_num_side, patch_num_side)

        x = self.patch_embedding(x)
        x = self.add_pos_emb(x, patch_num)

        hidden_states = []
        for block in self.resblocks:
            x = block(x)
            hidden_states.append(x)

        return hidden_states


class Molmo2ImageProjectorMLP(nn.Module):
    """MLP projector from vision features to text hidden size."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Molmo2VisionBackbone(nn.Module):
    """Complete vision backbone including ViT, pooling, and projector."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        vit_config = config.vit_config
        adapter_config = config.adapter_config

        # Determine which layers to use
        self.vit_layers = []
        for layer in adapter_config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(layer + vit_config.num_hidden_layers)

        # Maybe reduce ViT layers if we don't need all
        last_layer_needed = max(self.vit_layers) + 1
        num_hidden_layers = min(last_layer_needed, vit_config.num_hidden_layers)

        # Vision transformer
        self.image_vit = Molmo2VisionTransformer(
            hidden_size=vit_config.hidden_size,
            num_heads=vit_config.num_attention_heads,
            num_kv_heads=vit_config.num_key_value_heads,
            head_dim=vit_config.head_dim,
            intermediate_size=vit_config.intermediate_size,
            num_hidden_layers=num_hidden_layers,
            image_num_pos=vit_config.image_num_pos,
            image_patch_size=vit_config.image_patch_size,
            layer_norm_eps=vit_config.layer_norm_eps,
            quant_config=quant_config,
            prefix=add_prefix("image_vit", prefix),
        )

        # Pooling attention
        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = Molmo2ViTAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_kv_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            quant_config=quant_config,
            prefix=add_prefix("image_pooling_2d", prefix),
        )

        # Image projector
        self.image_projector = Molmo2ImageProjectorMLP(
            input_dim=adapter_config.hidden_size,
            hidden_dim=adapter_config.intermediate_size,
            output_dim=adapter_config.text_hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("image_projector", prefix),
        )

        self.pooling_attention_mask = adapter_config.pooling_attention_mask

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, num_crops, num_patch, n_pixels)
        Returns:
            (batch_size, num_crops, num_patch, concat_hidden_size)
        """
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        # Concatenate selected layer features
        features = []
        for layer in self.vit_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        image_features = image_features.view(B, T, N, -1)
        return image_features

    def forward(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (batch_size, num_crops, num_patch, n_pixels)
            pooled_patches_idx: (batch_size, num_pooled_patches, pool_dim)
        Returns:
            Pooled and projected features for valid tokens
        """
        batch_size, num_image = images.shape[:2]
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.encode_image(images)

        dim = image_features.shape[-1]
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        # Use pooled_patches_idx to arrange features for pooling
        batch_idx = torch.arange(
            pooled_patches_idx.shape[0],
            dtype=torch.long,
            device=pooled_patches_idx.device,
        )
        batch_idx = torch.tile(
            batch_idx.view(batch_size, 1, 1),
            [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]],
        )

        # Reshape and gather
        to_pool = image_features.reshape(batch_size, -1, dim)[
            batch_idx, torch.clamp(pooled_patches_idx, min=0)
        ]
        to_pool = to_pool * valid.to(self.dtype)[:, :, :, None]
        to_pool = to_pool.reshape(-1, pooled_patches_idx.shape[-1], dim)

        # Attention pooling
        if self.pooling_attention_mask:
            attn_mask = valid.reshape(-1, 1, 1, valid.shape[-1])
            # attn_mask needs to be float with -inf for masked positions
            attn_mask = attn_mask.to(self.dtype)
            attn_mask = torch.where(attn_mask == 0, float("-inf"), 0.0)
            denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
            denom = torch.where(denom == 0, 1, denom)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(
                to_pool.dtype
            )
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)

        pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled_features = pooled_features.reshape(
            batch_size, -1, pooled_features.shape[-1]
        )

        # Project to text hidden size
        pooled_features = self.image_projector(pooled_features)

        # Return only valid tokens
        return pooled_features.view(-1, pooled_features.shape[-1])[
            valid_token.flatten()
        ]


# ==============================================================================
# Text Backbone Components (adapted from OLMo2)
# ==============================================================================


class Molmo2Attention(nn.Module):
    """Attention block for Molmo2 text backbone."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads

        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)

        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # QKV projection (att_proj in checkpoint)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.tp_rank = get_tensor_model_parallel_rank()

        # QK normalization - OLMo2 style per-layer normalization
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim,
            eps=config.layer_norm_eps,
        )
        self.q_norm = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        # Determine rope scaling for this layer
        rope_scaling_layers = getattr(config, "rope_scaling_layers", None)
        if rope_scaling_layers is not None and layer_id not in rope_scaling_layers:
            # This layer uses default rope (no scaling)
            rope_scaling = {"rope_type": "default"}
        else:
            rope_scaling = config.rope_scaling

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        # Output projection (attn_out in checkpoint)
        self.o_proj = RowParallelLinear(
            self.head_dim * self.total_num_heads,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply QK normalization (OLMo2 style with tensor parallelism support)."""
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm.forward_native(q)
        k = self.k_norm.forward_native(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Molmo2MLP(nn.Module):
    """MLP block for Molmo2 text backbone.

    Note: Molmo2 uses `x * silu(gate)` ordering (gate first), which is the
    opposite of the standard SiluAndMul. We need to swap the chunks.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # ff_proj in checkpoint (projects to 2 * intermediate for gating)
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )

        # ff_out in checkpoint
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        # Molmo2 uses: silu(gate) * x, so we need to swap chunks
        # The checkpoint has [x, gate] ordering but SiluAndMul expects [gate, x]
        gate, up = gate_up.chunk(2, dim=-1)
        gate_up_swapped = torch.cat([up, gate], dim=-1)
        x = self.act_fn(gate_up_swapped)
        x, _ = self.down_proj(x)
        return x


class Molmo2DecoderLayer(nn.Module):
    """Decoder layer for Molmo2 (post-norm style)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Molmo2Attention(
            config, layer_id, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Molmo2MLP(config, quant_config, prefix=add_prefix("mlp", prefix))

        # Post-attention and post-FFN norms (norm_after=True in config)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ff_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Post-norm: norm is applied after the residual connection output
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Molmo2TextModel(nn.Module):
    """Text transformer model for Molmo2."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Combined embedding (embedding + new_embedding)
        # org_num_embeddings is the base vocab size (for weight loading)
        # num_embeddings is total vocab including additional tokens
        self.additional_vocab_size = getattr(config, "additional_vocab_size", 0)
        total_vocab_size = config.vocab_size + self.additional_vocab_size
        self.wte = VocabParallelEmbedding(
            total_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("wte", prefix),
        )

        self.blocks = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Molmo2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("blocks", prefix),
        )
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.wte

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = input_embeds

        for layer_id, decoder_layer in enumerate(self.blocks):
            hidden_states = decoder_layer(positions, hidden_states, forward_batch)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


# ==============================================================================
# Main Model Class
# ==============================================================================


class Molmo2ForConditionalGeneration(nn.Module):
    """Molmo2 multimodal model for conditional generation."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Text config
        text_config = config.text_config if hasattr(config, "text_config") else config

        # Text model (transformer)
        self.transformer = Molmo2TextModel(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("model.transformer", prefix),
        )

        # Vision backbone
        if (
            hasattr(config, "vit_config")
            and config.vit_config is not None
            and hasattr(config, "adapter_config")
            and config.adapter_config is not None
        ):
            self.vision_backbone = Molmo2VisionBackbone(
                config,
                quant_config=quant_config,
                prefix=add_prefix("model.vision_backbone", prefix),
            )
        else:
            self.vision_backbone = None

        # LM head
        self.unpadded_vocab_size = text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            text_config.hidden_size,
            org_num_embeddings=text_config.vocab_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

        # Store token IDs for multimodal processing
        self.image_patch_id = getattr(config, "image_patch_id", None)
        self.image_start_token_id = getattr(config, "image_start_token_id", None)
        self.image_end_token_id = getattr(config, "image_end_token_id", None)

    def get_input_embeddings(self):
        return self.transformer.wte

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """Replace image patch tokens with pad values for radix attention."""
        if not mm_inputs or not mm_inputs.mm_items:
            return input_ids

        # Use the multimodal token padding pattern
        # Only replace image_patch_id tokens with pad_value
        if self.image_patch_id is None:
            return input_ids

        input_ids_tensor = torch.as_tensor(input_ids)

        # Get pad values for each item
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        if not pad_values:
            return input_ids

        # Replace all image_patch_id tokens with the pad_value
        # For simplicity, use the first pad_value (all items should have same pad_value in a request)
        pad_value = pad_values[0]
        input_ids_tensor[input_ids_tensor == self.image_patch_id] = pad_value

        return input_ids_tensor.tolist()

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract features from image inputs."""
        all_features = []
        for item in items:
            pixel_values = item.feature  # (num_crops, num_patches, pixels)
            pooling_idx = item.model_specific_data.get("image_token_pooling")

            if pixel_values.dim() == 3:
                # Add batch dimension
                pixel_values = pixel_values.unsqueeze(0)
            if pooling_idx.dim() == 2:
                pooling_idx = pooling_idx.unsqueeze(0)

            features = self.vision_backbone(pixel_values, pooling_idx)
            all_features.append(features)

        return torch.cat(all_features, dim=0)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract features from video inputs."""
        all_features = []
        for item in items:
            pixel_values = item.feature  # (num_frames, num_patches, pixels)
            pooling_idx = item.model_specific_data.get("video_token_pooling")

            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            if pooling_idx.dim() == 2:
                pooling_idx = pooling_idx.unsqueeze(0)

            features = self.vision_backbone(pixel_values, pooling_idx)
            all_features.append(features)

        return torch.cat(all_features, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self,
            multimodal_model=self,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_video_feature,
            },
            placeholder_tokens=None,  # Using mm_item.pad_value
            positions=positions,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def __call__(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        forward_batch: ForwardBatch = None,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # This is called by general_mm_embed_routine as the language_model
        if input_embeds is not None:
            # Called with embeddings (after mm processing)
            hidden_states = self.transformer(
                input_ids=None,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
            return hidden_states
        else:
            # Standard forward
            return self.forward(input_ids, positions, forward_batch, input_embeds)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Weight name mappings from checkpoint to our module names
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # For QKV projection
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # For MLP gate/up projection
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params = set()

        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Remap checkpoint names to our module names
            # Text model: model.transformer.blocks.*.self_attn.att_proj -> transformer.blocks.*.self_attn.qkv_proj
            name = name.replace("model.transformer.", "transformer.")
            name = name.replace("self_attn.att_proj", "self_attn.qkv_proj")
            name = name.replace("self_attn.attn_out", "self_attn.o_proj")
            name = name.replace("mlp.ff_proj", "mlp.gate_up_proj")
            name = name.replace("mlp.ff_out", "mlp.down_proj")
            name = name.replace("attn_norm", "attn_norm")
            name = name.replace("ff_norm", "ff_norm")

            # Handle split embedding (wte.embedding + wte.new_embedding)
            if "wte.embedding" in name and "new_embedding" not in name:
                # This is the base embedding
                name = name.replace("wte.embedding", "wte.weight")
                if name in params_dict:
                    param = params_dict[name]
                    # Store for later concatenation or load directly if vocab size matches
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    # Load only the first vocab_size rows
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
            elif "wte.new_embedding" in name:
                # This is the additional embedding, skip (handled above or separately)
                # For VocabParallelEmbedding, we'd need to handle this differently
                # For now, skip as the processor should not generate tokens beyond vocab_size
                continue

            # Vision backbone
            name = name.replace("model.vision_backbone.", "vision_backbone.")
            name = name.replace(
                "image_vit.transformer.resblocks", "image_vit.resblocks"
            )

            # Handle stacked params (QKV, gate/up)
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_stacked = True
                break

            if is_stacked:
                continue

            # Direct loading
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                # Try without model. prefix
                alt_name = name.replace("model.", "", 1)
                if alt_name in params_dict:
                    name = alt_name
                else:
                    continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)


EntryClass = Molmo2ForConditionalGeneration
