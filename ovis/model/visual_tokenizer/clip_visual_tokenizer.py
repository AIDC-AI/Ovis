from datetime import datetime
from typing import Dict

import deepspeed
import torch
from torch import Tensor
from transformers import AutoConfig, AutoModel
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers.integrations import is_deepspeed_zero3_enabled

from ovis.util.constants import BEGIN_LINE, END_LINE
from ovis.util.utils import rank0_print
from .base_visual_tokenizer import BaseVisualTokenizerConfig, BaseVisualTokenizer

MODEL_TYPE = "clip_visual_tokenizer"


class ClipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = MODEL_TYPE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


class ClipVisualTokenizer(BaseVisualTokenizer):
    config_class = ClipVisualTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["CLIPEncoderLayer"]
    _image_processor_class = CLIPImageProcessor
    _image_processor_kwargs = dict(do_center_crop=False)
    _backbone_class = CLIPVisionModel
    _backbone_name_or_path = "openai/clip-vit-large-patch14-336"

    def __init__(self, config: ClipVisualTokenizerConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        head_dim = self.config.vocab_size
        if self.config.use_indicators:
            head_dim -= 2  # reserved for two image indicator tokens
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.config.hidden_size, head_dim, bias=False),
            torch.nn.LayerNorm(head_dim)
        )

    def re_init_layers(self, re_init_layer_begin):
        layer_dict = self.get_re_init_layer_dict(re_init_layer_begin)
        for name, layer in layer_dict.items():
            rank0_print(BEGIN_LINE)
            rank0_print(f'[{datetime.now()}] Before layer re-initialization of {name}: ')
            for k, v in layer.named_parameters():
                with deepspeed.zero.GatheredParameters([v]):
                    rank0_print(f'{k}: {v}')
            with deepspeed.zero.GatheredParameters(list(layer.parameters(recurse=True)), modifier_rank=0):
                if not is_deepspeed_zero3_enabled() or deepspeed.comm.get_rank() == 0:
                    layer.apply(self.backbone._init_weights)
            rank0_print(f'[{datetime.now()}] After layer re-initialization of {name}:')
            for k, v in layer.named_parameters():
                with deepspeed.zero.GatheredParameters([v]):
                    rank0_print(f'{k}: {v}')
            rank0_print(END_LINE)

    def get_re_init_layer_dict(self, re_init_layer_begin: int) -> Dict[str, torch.nn.Module]:
        assert re_init_layer_begin >= 0, "negative index is prohibited"
        layer_dict = dict()
        for i in range(re_init_layer_begin, self.backbone.config.num_hidden_layers):
            layer_dict[f'backbone.vision_model.encoder.layers.{i}'] = self.backbone.vision_model.encoder.layers[i]
        return layer_dict

    def get_monitor_tensors(self):
        return dict(
            backbone_bottom=self.backbone.vision_model.encoder.layers[0].self_attn.k_proj.weight,
            backbone_top=self.backbone.vision_model.encoder.layers[-1].self_attn.out_proj.weight,
            head=self.head[0].weight
        )

    def get_image_size(self):
        height = self.image_processor.crop_size["height"]
        width = self.image_processor.crop_size["width"]
        return height, width

    def forward(self, pixel_values) -> Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        output = self.backbone(
            pixel_values, output_hidden_states=True, return_dict=True)
        features = output.last_hidden_state
        if self.config.drop_cls_token:
            features = features[:, 1:, :]
        logits = self.head(features)
        tokens = self.tokenize(logits)
        if self.config.use_indicators:
            # tokens' shape is [BatchSize, #Token, VocabSize-2], so padding with [BatchSize, #Token, 2], after
            # which, tokens' shape should become [BatchSize, #Token, VocabSize]
            batch_size, token_len, _ = tokens.shape
            padding_tensor = torch.zeros(size=(batch_size, token_len, 2),
                                         dtype=tokens.dtype,
                                         device=tokens.device,
                                         layout=tokens.layout,
                                         requires_grad=False)
            tokens = torch.cat((tokens, padding_tensor), dim=2)

            # adding indicator tokens, after which tokens' shape should become [BatchSize, 1+#Token+1, VocabSize]
            begin_indicator = torch.zeros(size=(batch_size, 1),
                                          dtype=torch.long,
                                          device=tokens.device,
                                          requires_grad=False) + self.config.vocab_size - 2
            begin_indicator_token = torch.nn.functional.one_hot(begin_indicator,
                                                                num_classes=self.config.vocab_size).to(
                dtype=tokens.dtype)
            end_indicator = torch.zeros(size=(batch_size, 1),
                                        dtype=torch.long,
                                        device=tokens.device,
                                        requires_grad=False) + self.config.vocab_size - 1
            end_indicator_token = torch.nn.functional.one_hot(end_indicator,
                                                              num_classes=self.config.vocab_size).to(dtype=tokens.dtype)
            tokens = torch.cat((begin_indicator_token, tokens, end_indicator_token), dim=1)
        return tokens


AutoConfig.register(MODEL_TYPE, ClipVisualTokenizerConfig)
AutoModel.register(ClipVisualTokenizerConfig, ClipVisualTokenizer)
