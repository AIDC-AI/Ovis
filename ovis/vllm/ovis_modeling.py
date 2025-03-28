# SPDX-License-Identifier: Apache-2.0

# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/ovis/modeling_ovis.py
# Copyright 2023 The vLLM team.
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
#
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
""" PyTorch Ovis model."""
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor
from transformers import BatchFeature, AutoTokenizer

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import maybe_prefix, flatten_bn, AutoWeightsLoader, init_vllm_registered_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs, NestedTensors,
                                    )
from vllm.multimodal.parse import (ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement   )
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from .aimv2.visual_tokenizer_aimv2 import Aimv2VisualTokenizer
from .processing_ovis import OvisProcessor
from .ovis_config import OvisConfig

from torch.nn import init

# Cannot find the following 2 numbers from hf config.
IGNORE_ID = -100


MAX_SEGMENTS = 10  # default value in the ovis modeling
NUMBER_OF_TOKEN_TO_RESERVE_FOR_SEGMENT = 256

class OvisImagePatchInputs(TypedDict):
    type: Literal["image_patches"]
    flat_data: torch.Tensor
    """
    Shape: 
    `(batch_size * num_patches, patch_size_x * patch_size_y * num_channels)`
    """

    patches_per_image: List[int]
    """
    List of number of total patches for each image in the batch.
    This is used to restore the first two dimensions of `flat_data`.
    """

class VisualEmbedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

class OvisProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(OvisConfig)

    def get_hf_processor(self):
        return self.ctx.get_hf_processor(OvisProcessor)

    def get_image_processor(self) -> OvisProcessor:
        return self.get_hf_processor().image_processor # type: ignore

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}# totest hte case where a single image ios passed self.get_hf_config().multimodal_max_length // (MAX_SEGMENTS * NUMBER_OF_TOKEN_TO_RESERVE_FOR_SEGMENT)} # 32k is model token limit at the moment

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:

        return {"image":  (mm_counts['image'] * MAX_SEGMENTS * 256) + 11} # 6 image pos token, don't ask why

    def get_image_size(self) -> ImageSize:
        image_processor = self.get_image_processor()
        return ImageSize(width=image_processor.size['shortest_edge'] * 9 * 2,
                         height=image_processor.size['shortest_edge'] * 9 * 2)


class OvisDummyInputsBuilder(BaseDummyInputsBuilder[OvisProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int]
    ) -> ProcessorInputs:
        target_width, target_height = \
            self.info.get_image_size()
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
        }

        return ProcessorInputs(
            prompt_text='''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
Describe the image.<|im_end|>
<|im_start|>assistant''',
            mm_data=mm_data,

        )


class OvisMultiModalProcessor(BaseMultiModalProcessor[OvisProcessingInfo]):

    def _get_token_value(self, tok):
        return self.info.get_tokenizer()(self.info.get_tokenizer().extra_special_tokens[tok])["input_ids"]


    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
        #    # Avoid warning from HF logger for text-only input
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            #prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids) nope
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        return processed_outputs

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:

        return prompt_tokens

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"), grids=MultiModalFieldConfig.batched("image"))

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:

        def get_replacement_tokens_ovis(grid):
            """
            Calculates the placeholder for the sequence, starting from the grid

            Args:
                grid: the grid tuple for the image
            Returns:
                list: Placeholder sequence for the image with padding
            """
            hf_processor = self.info.get_hf_processor()
            # Get the base placeholder tokens
            placeholder_tokens = hf_processor.construct_image_placeholders(grid)
            image_atom_token_id = \
            self.info.get_tokenizer()(self.info.get_tokenizer().extra_special_tokens['image_atom'])['input_ids'][0]

            # Extract the padding token ID from tokenizer
            image_padding_token_id = \
                self.info.get_tokenizer()(self.info.get_tokenizer().extra_special_tokens['image_pad'])['input_ids'][0]

            # Create a new list with padding tokens inserted
            padded_placeholder_tokens = []
            for token in placeholder_tokens:
                padded_placeholder_tokens.append(token)
                if token == image_atom_token_id:
                    # Add 255 padding tokens after each image atom token
                    padded_placeholder_tokens.extend([image_padding_token_id] * 255)

            return padded_placeholder_tokens

        return [
            PromptReplacement(
                modality="image",
                target= self.info.get_tokenizer()(
                    self.info.get_tokenizer()
                    .extra_special_tokens['image_token']
                )['input_ids'],
                replacement=get_replacement_tokens_ovis(grid),
            )
        for grid in out_mm_kwargs["grids"]]


#useful for comparison of numerical identity between implementations
'''import torch
import tempfile
import os
import numpy as np


def compare_with_saved_tensor(tensor, saved_tensor_path):
    """
    Loads a tensor from disk and compares it with an existing tensor.

    Args:
        tensor: torch.Tensor - The tensor to compare against
        saved_tensor_path: str - Path to the saved tensor file

    Returns:
        dict: Comparison metrics and information
    """
    # Load the saved tensor properly using a file handle
    try:
        with open(saved_tensor_path, 'rb') as f:
            loaded_tensor = torch.load(f)
    except TypeError as e:
        if 'BFloat16' in str(e):
            # Handle BFloat16 tensors by loading with a different approach
            with open(saved_tensor_path, 'rb') as f:
                loaded_tensor = torch.load(f, map_location=torch.device('cpu'))
                # Convert to a supported dtype if needed
                loaded_tensor = loaded_tensor.to(dtype=torch.float32)
            # Also convert the comparison tensor to float32
            tensor = tensor.to(dtype=torch.float32)
        else:
            raise e

    # Ensure both tensors are on the same device
    if tensor.device != loaded_tensor.device:
        loaded_tensor = loaded_tensor.to(tensor.device)

    # Basic shape comparison
    shapes_match = tensor.shape == loaded_tensor.shape

    if not shapes_match:
        return {
            "shapes_match": False,
            "tensor_shape": tensor.shape,
            "loaded_tensor_shape": loaded_tensor.shape,
            "error": "Shapes don't match, cannot compute element-wise metrics"
        }

    # Element-wise comparison
    diff = tensor - loaded_tensor
    abs_diff = torch.abs(diff)

    # Compute metrics
    metrics = {
        "shapes_match": True,
        "tensor_shape": tensor.shape,
        "tensor_dtype": str(tensor.dtype),
        "loaded_tensor_dtype": str(loaded_tensor.dtype),
        "exact_match": torch.equal(tensor, loaded_tensor),
        "mean_diff": diff.mean().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_abs_diff": abs_diff.max().item(),
        "min_abs_diff": abs_diff.min().item(),
        "std_diff": diff.std().item(),
        "l2_norm_diff": torch.norm(diff).item(),
        "percent_exact_match": (tensor == loaded_tensor).float().mean().item() * 100,
        "nonzero_count_original": torch.count_nonzero(tensor).item(),
        "nonzero_count_loaded": torch.count_nonzero(loaded_tensor).item()
    }

    # Generate histogram data for the differences
    diff_np = diff.flatten().cpu().float().numpy()
    hist, bin_edges = np.histogram(diff_np, bins=10)
    metrics["diff_histogram"] = {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist()
    }

    # Find positions with largest differences
    if tensor.numel() > 0:
        top_k = min(10, tensor.numel())
        flat_indices = torch.topk(abs_diff.flatten(), k=top_k)[1]

        # Convert flat indices to multi-dimensional indices
        top_diff_positions = []
        for idx in flat_indices:
            idx = idx.item()
            # Convert flat index to multi-dimensional index
            multi_idx = np.unravel_index(idx, tensor.shape)
            # Get values at this position
            original_val = tensor[multi_idx].item()
            loaded_val = loaded_tensor[multi_idx].item()
            diff_val = diff[multi_idx].item()

            top_diff_positions.append({
                "position": multi_idx,
                "original_value": original_val,
                "loaded_value": loaded_val,
                "difference": diff_val
            })

        metrics["top_differences"] = top_diff_positions

    return metrics'''

@MULTIMODAL_REGISTRY.register_processor(OvisMultiModalProcessor,
                                        info=OvisProcessingInfo,
                                        dummy_inputs=OvisDummyInputsBuilder)
class OvisForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.padding_idx = config.pad_token_id
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config.with_hf_config(config.llm_config),
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.visual_tokenizer = Aimv2VisualTokenizer(
            config = config.visual_tokenizer_config,
            quant_config=quant_config,
            prefix=f"{prefix}.visual_tokenizer",
            image_processor_name_or_path=config.visual_tokenizer_config.backbone_config.name_or_path,
        ).to(self.config.torch_dtype)

        self.vte = VisualEmbedding(
            self.config.visual_tokenizer_config.vocab_size,
            self.config.hidden_size,
            device='cuda',
            dtype=self.visual_tokenizer.dtype
        )

        # we'll instantiate a tokenizer and keep just the external mapping
        tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)

        self.extra_token_mapping = {
            k: tokenizer(v)['input_ids'][0] for k, v in tokenizer.extra_special_tokens.items()
        }

        self.extra_token_mapping_for_substitution = {
            k: tokenizer(v)['input_ids'][0] for k, v in tokenizer.extra_special_tokens.items() if k in
                                                                                                  {'image_atom',
                                                                                                   'image_pad'}
        }


        self.visual_indicators_embeds_dict = None
        #VocabParallelEmbedding( if enabled leads to numerical diff
        #    self.config.visual_tokenizer_config.vocab_size,
        #    self.config.hidden_size,
        #    params_dtype=self.visual_tokenizer.dtype,
        #    quant_config=quant_config,
        #    prefix=f"{prefix}.vte"
        #)


        #self.make_empty_intermediate_tensors = (
        #    self.language_model.make_empty_intermediate_tensors) ?


    def _init_embed_representation(self):
        if not self.visual_indicators_embeds_dict:
            # we precalcualte the embeddings for the image tokens
            visual_vocab_size = self.visual_tokenizer.config.vocab_size
            visual_indicator_embeds = self.vte(
                torch.tensor(
                    list(range(visual_vocab_size - 5, visual_vocab_size)),
                    dtype=torch.long,
                    device=self.vte.device
                )
            )

            self.visual_indicators_embeds_dict = {
                'image_start': visual_indicator_embeds[0],
                'image_prefix': visual_indicator_embeds[1],
                'image_col_sep': visual_indicator_embeds[2],
                'image_row_sep': visual_indicator_embeds[3],
                'image_end': visual_indicator_embeds[4],
             }

    @property
    def sampler(self):
        return self.llm.sampler

    def merge_multimodal(
            self,
            text_input_ids: Union[List[torch.Tensor], torch.Tensor],
            pixel_values: Optional[Union[List[torch.Tensor], torch.Tensor, object]],
            left_padding: bool = True # must be true during inference
    ): #  todo check when different sized  inputs are batched
        # todo the tokenizer do not uses /n
        # we need to decompose the pixel_value_tensor
        # vllm batches it fi it is ccompatible otherwise it will pass it as  list
        self._init_embed_representation()
        if pixel_values is not None and not isinstance(pixel_values, list):
            if pixel_values.dim() == 6:
                # if is [tensor_batch, 1, num_segments, ch, w, h] we need -> [tensor_batch, num_segments, ch, w, h]
                pixel_values = pixel_values.squeeze(1)
                pixel_values = [pixel_value.to(self.config.torch_dtype) for pixel_value in pixel_values]
            else:
                pixel_values = [pixel_values]

        # When inference, sample can include only text with `None` pixel_value
        num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
        if sum(num_images) > 0:
            visual_tokens = self.visual_tokenizer(
                torch.cat(
                [x for x in pixel_values if x is not None],
                dim=0).to(self.visual_tokenizer.dtype)
            )

            visual_embeds = self.vte(visual_tokens) # 1:1 numeric eq.


        else:
            # just placeholders
            visual_embeds = [None] * len(num_images)

        input_embeds = []

        for text_input_id, visual_embed in zip(text_input_ids, visual_embeds):

            placeholder_token_mask = torch.zeros_like(text_input_id, dtype=torch.bool)
            for value in self.extra_token_mapping_for_substitution.values():
                placeholder_token_mask |= torch.eq(text_input_id, value)

            text_embed = torch.zeros((text_input_id.shape[0],self.llm.model.norm.hidden_size),
                                     device=text_input_id.device, dtype=self.visual_tokenizer.dtype)
            text_embed[~placeholder_token_mask] = self.llm.model.embed_tokens(text_input_id[~placeholder_token_mask]) # 1:1

            for key, indicator_id in self.extra_token_mapping.items():
                if key in self.visual_indicators_embeds_dict:
                    text_embed[text_input_id == indicator_id] = self.visual_indicators_embeds_dict[key].to(text_embed.device)
            #image_atom_positions = torch.where(torch.eq(text_input_id, self.extra_token_mapping['image_atom']))[0].tolist()
            #if len(image_atom_positions) > 0:
                #if not is_testing:
                #    input_embed_parts = []
                #    prev_image_atom_position = -1
                #    for index, image_atom_position in enumerate(image_atom_positions):
                #        input_embed_parts.append(
                #            text_embed[prev_image_atom_position + 1:image_atom_position, :])
#
                #        input_embed_parts.append(visual_embeds[index])
#
                #        prev_image_atom_position = image_atom_position
                #    if prev_image_atom_position + 1 < text_input_id.shape[0]:
                #        input_embed_parts.append(
                #            text_embed[prev_image_atom_position + 1:, :])
#
                #    input_embed = torch.cat(input_embed_parts, dim=0)
                #else:

                    # here we have already preallocated the multimodal tokens (in the testing phase) se the logic should be different
                    # we should check consider that each atom token should replace 256 text tokens embeddings

            # It just needs this unified verison, since if no  images aare present it should just skip this
            text_embed[placeholder_token_mask] = visual_embeds.view(-1, text_embed.shape[-1])


            #else:
            #    input_embed = text_embed

            input_embeds.append(text_embed)

        
        batch_input_embeds = self.pad_truncate_sequence(input_embeds, batch_first=True, padding_value=0.0,
                                                        left_padding=left_padding)

        return batch_input_embeds

    def pad_truncate_sequence(self, sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0, left_padding: bool = False) -> torch.Tensor:
        if not left_padding:
            pad_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
            return pad_sequence[:,:self.config.multimodal_max_length]
        else:
            pad_sequence = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in sequences],batch_first=True, padding_value=padding_value).flip(dims=[1])
            return pad_sequence[:,-self.config.multimodal_max_length:]


    def get_tensor_formatted(self, input: Union[torch.Tensor, List]) -> List[torch.Tensor]:
        '''
        if thhe input is list check if its input arte 1d if so usueeze() them in 0
        if it is a tensor it needs to be splittend in a list
        :param input:
        :return:
        '''
        if isinstance(input, list):
            output_list = []
            for element in input:
                if element.dim() == 1:
                    output_list.append(element.unsqueeze(0))
                else:
                    output_list.append(element)
            return output_list
        else:
            return [tensor for tensor in input] if input.dim() > 1 else [input]


    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None and 'pixel_values' in kwargs: # vllm batches the input or make it a list but does not have a attn mask
            inputs_embeds = self.merge_multimodal(text_input_ids=self.get_tensor_formatted(input_ids) ,
                                                  pixel_values=kwargs['pixel_values'],)
                                                  #is_testing = kv_caches[0].numel() == 0) valid approach but probably not needed
            #input_ids = None
        # up until here we have a inputs_embeds 100% numerical identity between the OG HF Transformers implementation and ours
        hidden_states = self.llm(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.llm.logits_processor(
            self.llm.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.llm.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)