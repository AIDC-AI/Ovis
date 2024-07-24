import logging
from datetime import datetime
from typing import Dict

import deepspeed
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel
from transformers import SiglipVisionModel, SiglipImageProcessor
from transformers.integrations import is_deepspeed_zero3_enabled

from ovis.util.constants import BEGIN_LINE, END_LINE
from ovis.util.utils import rank0_print
from .base_visual_tokenizer import BaseVisualTokenizerConfig, BaseVisualTokenizer

MODEL_TYPE = "siglip_visual_tokenizer"


class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = MODEL_TYPE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            logging.warning(
                f'SiglipVisionModel has no cls token, so `drop_cls_token=True` is ignored and reset to `False`')
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


class SiglipVisualTokenizer(BaseVisualTokenizer):
    config_class = SiglipVisualTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["SiglipVisionTransformer"]
    _image_processor_class = SiglipImageProcessor
    _image_processor_kwargs = {}
    _backbone_class = SiglipVisionModel
    _backbone_name_or_path = "google/siglip-so400m-patch14-384"

    def __init__(self, config: SiglipVisualTokenizerConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        head_dim = self.config.vocab_size
        if self.config.use_indicators:
            head_dim -= 2  # reserved for two image indicator tokens
        if self.config.hd_booster is None:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(
                    self.backbone.config.hidden_size * self.config.hidden_stride * self.config.hidden_stride, head_dim,
                    bias=False),
                torch.nn.LayerNorm(head_dim)
            )
        elif self.config.hd_booster in ['s2wrapper', 's2wrapper-adaptive']:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(
                    self.backbone.config.hidden_size * self.config.hidden_stride * self.config.hidden_stride * 2,
                    head_dim, bias=False),
                torch.nn.LayerNorm(head_dim)
            )
        else:
            raise ValueError(f'Unsupported hd_booster {self.config.hd_booster}')

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
        height = self.image_processor.size["height"]
        width = self.image_processor.size["width"]
        return height, width

    def encode(self, pixel_values):
        if self.config.hd_booster is None:
            output = self.backbone(
                pixel_values, output_hidden_states=True, return_dict=True)
            features = output.hidden_states[-1]
            if self.config.drop_cls_token:
                features = features[:, 1:, :]
        elif self.config.hd_booster in ['s2wrapper', 's2wrapper-adaptive']:
            n, c, side, _ = pixel_values.shape
            if self.config.hd_booster == 's2wrapper-adaptive':
                pixel_values_mask = torch.isinf(pixel_values)  # [n, c, side, side]
                pixel_values = torch.masked_fill(pixel_values, pixel_values_mask, 0.0)
            pixel_values = pixel_values.reshape(n * 5, c // 5, side, side)
            output = self.backbone(pixel_values, output_hidden_states=True, return_dict=True)
            features = output.hidden_states[-1]
            if self.config.drop_cls_token:
                features = features[:, 1:, :]
            _, l, d = features.shape
            features = features.reshape(n, 5, l, d)
            features_overall = features[:, 0, :, :]  # [n, l, d]
            features_parts = features[:, 1:, :, :]  # [n, 4, l, d]
            sqrt_l = int(l ** 0.5)
            assert sqrt_l ** 2 == l, "The token sequence length should be a perfect square."
            features_parts = features_parts.reshape(n, 4, sqrt_l, sqrt_l, d)  # [n, 4, sqrt(l), sqrt(l), d]
            features_top = torch.concat([features_parts[:, 0, :, :, :], features_parts[:, 1, :, :, :]],
                                       dim=-2)  # [n, sqrt(l), sqrt(l)*2, d]
            features_bottom = torch.concat([features_parts[:, 2, :, :, :], features_parts[:, 3, :, :, :]],
                                          dim=-2)  # [n, sqrt(l), sqrt(l)*2, d]
            features_merge = torch.concat([features_top, features_bottom], dim=-3)  # [n, sqrt(l)*2, sqrt(l)*2, d]
            features_pool = F.interpolate(features_merge.permute(0, 3, 1, 2).to(torch.float32), size=sqrt_l,
                                          mode='area')  # [n, d, sqrt_l, sqrt_l]
            features_pool = features_pool.flatten(2).permute(0, 2, 1).to(features.dtype)  # [n, l, d]
            if self.config.hd_booster == 's2wrapper-adaptive':
                features_pool_mask = torch.unsqueeze(torch.unsqueeze(pixel_values_mask[:, -1, -1, -1], dim=-1), dim=-1)  # [n, 1, 1]
                features_pool = torch.masked_fill(features_pool, features_pool_mask, 0.0)
            features = torch.cat([features_overall, features_pool], dim=-1)  # [n, l, 2*d]
        else:
            raise ValueError(f'Unsupported hd_booster {self.config.hd_booster}')

        # merge number of `hidden_stride * hidden_stride` hidden states together to reduce token sequence length
        # e.g., for hidden_stride=3, this leads to a token length reduction: 729 -> 81
        if self.config.hidden_stride > 1:
            n, l, d = features.shape  # this `d` maybe different from the above `d
            sqrt_l = int(l ** 0.5)
            assert sqrt_l ** 2 == l, "The token sequence length should be a perfect square."
            assert l % (
                        self.config.hidden_stride ** 2) == 0, "The token sequence length should be divisible by `hidden_stride**2`."
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            features = features.reshape(n, sqrt_l // self.config.hidden_stride, self.config.hidden_stride,
                                        sqrt_l // self.config.hidden_stride, self.config.hidden_stride, d)
            features = features.permute(0, 1, 3, 2, 4, 5)  # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.flatten(3)  # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.reshape(n, l // (self.config.hidden_stride * self.config.hidden_stride),
                                        self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(self, pixel_values) -> Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values)
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


AutoConfig.register(MODEL_TYPE, SiglipVisualTokenizerConfig)
AutoModel.register(SiglipVisualTokenizerConfig, SiglipVisualTokenizer)
