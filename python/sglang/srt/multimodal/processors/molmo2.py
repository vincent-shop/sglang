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
"""Multimodal processor for Molmo2."""

from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.molmo2 import Molmo2ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


# Special tokens for Molmo2
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"
IMAGE_PATCH_TOKEN = "<im_patch>"
IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"
IM_COL_TOKEN = "<im_col>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
FRAME_START_TOKEN = "<frame_start>"
FRAME_END_TOKEN = "<frame_end>"


class Molmo2MultimodalProcessor(BaseMultimodalProcessor):
    """Multimodal processor for Molmo2 models."""

    models = [Molmo2ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.hf_config = hf_config

        self.image_patch_id = getattr(hf_config, "image_patch_id", None)
        self.image_start_token_id = getattr(hf_config, "image_start_token_id", None)
        self.image_end_token_id = getattr(hf_config, "image_end_token_id", None)
        self.image_col_id = getattr(hf_config, "image_col_id", None)
        self.low_res_image_start_token_id = getattr(
            hf_config, "low_res_image_start_token_id", None
        )
        self.frame_start_token_id = getattr(hf_config, "frame_start_token_id", None)
        self.frame_end_token_id = getattr(hf_config, "frame_end_token_id", None)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=IMAGE_PROMPT,
            video_token=VIDEO_PROMPT,
            image_token_id=self.image_patch_id,
            video_token_id=self.image_patch_id,  # Videos also use image_patch tokens
        ).build(_processor)

    def _get_image_patch_offsets(
        self,
        input_ids: torch.Tensor,
        image_patch_id: int,
    ) -> List[Tuple[int, int]]:
        """
        Get offsets of contiguous image patch token regions.

        Returns list of (start, end) tuples for each contiguous region.
        """
        mask = input_ids == image_patch_id
        if not mask.any():
            return []

        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]

        return list(zip(start_positions.tolist(), end_positions.tolist()))

    def _split_offsets_by_item(
        self,
        all_offsets: List[Tuple[int, int]],
        expected_counts: List[int],
    ) -> List[List[Tuple[int, int]]]:
        """
        Split offsets into groups based on expected token counts per item.

        Args:
            all_offsets: All (start, end) offset pairs
            expected_counts: Number of patch tokens expected per item

        Returns:
            List of offset lists, one per item
        """
        result = []
        offset_idx = 0

        for count in expected_counts:
            item_offsets = []
            tokens_collected = 0

            while tokens_collected < count and offset_idx < len(all_offsets):
                start, end = all_offsets[offset_idx]
                tokens_in_range = end - start + 1
                tokens_collected += tokens_in_range
                item_offsets.append((start, end))
                offset_idx += 1

            result.append(item_offsets)

        return result

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Image.Image]]],
        audio_data,
        input_text: Union[str, List[int]],
        request_obj,
        max_req_input_len: int = None,
        **kwargs,
    ):
        """Process multimodal data for Molmo2."""
        video_data = getattr(request_obj, "video_data", None)

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )

        if not base_output.images and not base_output.videos:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return {
                "input_ids": input_ids.tolist(),
                "mm_items": [],
            }

        processor_kwargs = {}
        if base_output.images:
            processor_kwargs["images"] = base_output.images
        if base_output.videos:
            processor_kwargs["videos"] = base_output.videos

        result = self._processor(
            text=base_output.input_text,
            return_tensors="pt",
            **processor_kwargs,
        )

        input_ids = result["input_ids"].flatten()

        mm_items = []

        if "pixel_values" in result:
            pixel_values = result["pixel_values"]
            image_token_pooling = result.get("image_token_pooling")
            image_grids = result.get("image_grids")
            image_num_crops = result.get("image_num_crops")

            all_offsets = self._get_image_patch_offsets(input_ids, self.image_patch_id)

            # Each image contributes: resized_h * resized_w (global) + height * width (local crops)
            if image_grids is not None:
                expected_counts = []
                for grid in image_grids:
                    resized_h, resized_w, height, width = grid.tolist()
                    global_patches = resized_h * resized_w
                    local_patches = height * width
                    expected_counts.append(global_patches + local_patches)

                offsets_per_image = self._split_offsets_by_item(
                    all_offsets, expected_counts
                )
            else:
                offsets_per_image = [all_offsets]

            num_images = len(image_grids) if image_grids is not None else 1

            pooling_offset = 0
            crops_offset = 0

            for i in range(num_images):
                if image_num_crops is not None:
                    num_crops = image_num_crops[i].item()
                else:
                    num_crops = pixel_values.shape[0] // num_images

                image_pixels = pixel_values[crops_offset : crops_offset + num_crops]
                crops_offset += num_crops

                if image_token_pooling is not None and image_grids is not None:
                    grid = image_grids[i]
                    resized_h, resized_w, height, width = grid.tolist()
                    num_pooled = resized_h * resized_w + height * width
                    pooling = image_token_pooling[
                        pooling_offset : pooling_offset + num_pooled
                    ]
                    pooling_offset += num_pooled
                else:
                    pooling = image_token_pooling

                item = MultimodalDataItem(modality=Modality.IMAGE)
                item.feature = image_pixels
                item.offsets = (
                    offsets_per_image[i] if i < len(offsets_per_image) else []
                )

                item.model_specific_data = {
                    "image_token_pooling": pooling,
                }
                if image_grids is not None:
                    item.model_specific_data["image_grid"] = image_grids[i]

                mm_items.append(item)

        if "pixel_values_videos" in result:
            pixel_values_videos = result["pixel_values_videos"]
            video_token_pooling = result.get("video_token_pooling")
            video_grids = result.get("video_grids")

            all_offsets = self._get_image_patch_offsets(input_ids, self.image_patch_id)

            if video_grids is not None:
                expected_counts = []
                for grid in video_grids:
                    num_frames, h, w = grid.tolist()
                    expected_counts.append(num_frames * h * w)

                offsets_per_video = self._split_offsets_by_item(
                    all_offsets, expected_counts
                )
            else:
                offsets_per_video = [all_offsets]

            num_videos = len(video_grids) if video_grids is not None else 1

            pooling_offset = 0
            frame_offset = 0

            for i in range(num_videos):
                if video_grids is not None:
                    num_frames = video_grids[i][0].item()
                else:
                    num_frames = pixel_values_videos.shape[0] // num_videos

                video_pixels = pixel_values_videos[
                    frame_offset : frame_offset + num_frames
                ]
                frame_offset += num_frames

                if video_token_pooling is not None and video_grids is not None:
                    grid = video_grids[i]
                    num_frames_grid, h, w = grid.tolist()
                    num_pooled = num_frames_grid * h * w
                    pooling = video_token_pooling[
                        pooling_offset : pooling_offset + num_pooled
                    ]
                    pooling_offset += num_pooled
                else:
                    pooling = video_token_pooling

                item = MultimodalDataItem(modality=Modality.VIDEO)
                item.feature = video_pixels
                item.offsets = (
                    offsets_per_video[i] if i < len(offsets_per_video) else []
                )

                item.model_specific_data = {
                    "video_token_pooling": pooling,
                }
                if video_grids is not None:
                    item.model_specific_data["video_grid"] = video_grids[i]

                mm_items.append(item)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.image_patch_id,
            "im_start_id": self.image_start_token_id,
            "im_end_id": self.image_end_token_id,
        }
