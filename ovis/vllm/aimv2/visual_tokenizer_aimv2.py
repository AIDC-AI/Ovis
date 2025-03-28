import PIL
import torch
from torch import softmax
from torch.nn.functional import gumbel_softmax, pad
from transformers import AutoImageProcessor
from vllm.config import QuantizationConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from .aimv2.modeling_aimv2 import AIMv2Model
from ..ovis_config import BaseVisualTokenizerConfig
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305] # kept for vocab prefixed tokens

class Aimv2VisualTokenizer(torch.nn.Module):

    def __init__(self, config: BaseVisualTokenizerConfig, quant_config: QuantizationConfig, prefix: str = "", **kwargs):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(kwargs['image_processor_name_or_path'])
        self.config = config
        self.image_processor.do_center_crop = False
        self.backbone = AIMv2Model(
            config = config.backbone_config, # noqa
            quant_config=quant_config,
            prefix=f"{prefix}.visual_tokenizer"
        )
        head_dim = config.vocab_size - len(IMAGE_INDICATOR_IDS)  # reserved tokens for IMAGE_INDICATORS
        self.head = torch.nn.Sequential(
            ColumnParallelLinear(
                config.backbone_config.hidden_size * config.hidden_stride * config.hidden_stride, head_dim,
                bias=False,
            ),
            torch.nn.LayerNorm(head_dim)
        )

        assert all((self.image_processor.do_resize,
                    not getattr(self.image_processor, 'do_center_crop', False),
                    self.image_processor.do_rescale,
                    self.image_processor.do_normalize
                    )), f"image_processor `{self.image_processor}` is not supported currently"

    @property
    def dtype(self):
        return self.backbone.dtype

    @property
    def device(self):
        return self.backbone.device

    def get_backbone(self):
        return self.backbone

    def get_image_processor(self):
        return self.image_processor

    def get_head(self):
        return self.head

    def get_image_size(self):
        raise NotImplementedError


    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.config.tokenize_function == 'softmax':
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == 'gumbel_argmax':
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, but got {self.config.tokenize_function}')
        return tokens

    def encode(self, pixel_values):
        features = self.backbone(pixel_values)
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together to reduce token sequence length
        # e.g., for hidden_stride=2, this leads to a token length reduction: 1024 -> 256 for aimv2
        if self.config.hidden_stride > 1:
            n, l, d = features.shape  # this `d` maybe different from the above `d
            sqrt_l = int(l ** 0.5)
            assert sqrt_l ** 2 == l, "The token sequence length should be a perfect square."
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (self.config.hidden_stride - (sqrt_l % self.config.hidden_stride)) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(n, sqrt_l // self.config.hidden_stride, self.config.hidden_stride,
                                        sqrt_l // self.config.hidden_stride, self.config.hidden_stride, d)
            features = features.permute(0, 1, 3, 2, 4, 5)  # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.flatten(3)  # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1, self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(self, pixel_values) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values)
        logits, _ = self.head[0](features) # we spllit the sequncial here for not throwing an error
        logits = self.head[1](logits)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with [BatchSize, #Token, 5], after
        # which, tokens' shape should become [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(batch_size, token_len, len(IMAGE_INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens




