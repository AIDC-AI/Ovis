from typing import Optional, Tuple, Union

import torch
from vllm.attention import Attention, AttentionType
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear, ColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

from .configuration_aimv2 import AIMv2Config
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)#ColumnParallelLinear(in_features,
                   #              hidden_features,
                   #              bias=bias,
                   #              quant_config=quant_config,
                   #              prefix=f"{prefix}.fc1")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)#ColumnParallelLinear(hidden_features,
                   #              in_features,
                   #              bias=bias,
                   #              quant_config=quant_config,
                   #              prefix=f"{prefix}.fc2")
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)#RowParallelLinear(in_features,
                   #           hidden_features,
                   #           bias=bias,
                   #           quant_config=quant_config,
                   #           prefix=f"{prefix}.fc3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel= self.fc1(x)#, _ = self.fc1(x)
        gate= self.fc3(x)#, _ = self.fc3(x)
        x_parallel = F.silu(x_parallel) * gate
        out =self.fc2(x_parallel)#, _ = self.fc2(x_parallel)
        return out



class AIMv2PatchEmbed(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AIMv2ViTPreprocessor(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = AIMv2PatchEmbed(config)
        self.pos_embed = nn.Parameter(torch.zeros((1, num_patches, config.hidden_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(x)
        _, N, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)
        tokens = tokens + pos_embed[:, :N]
        return tokens


class AIMv2Attention(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        dim = config.hidden_size

        self.num_heads = config.num_attention_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)#QKVParallelLinear(
                   # hidden_size=dim,
                   # head_size=dim // config.num_attention_heads,
                   # total_num_heads=config.num_attention_heads,
                   # bias=config.qkv_bias,
                   # quant_config=quant_config,
                   # prefix=f"{prefix}.qkv")
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj = nn.Linear(dim, dim, bias=config.use_bias)#RowParallelLinear(input_size=dim,
                    #                  output_size=dim,
                    #                  bias = config.use_bias,
                    #                  quant_config=quant_config,
                    #                  prefix=f"{prefix}.proj")

        self.proj_drop = nn.Dropout(config.projection_dropout)

    def forward( # todo might implement multiple attn implementations
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x) #, _ = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x= self.proj(x)#, _ = self.proj(x)
        x = self.proj_drop(x)
        return x


class AIMv2Block(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        self.attn = AIMv2Attention(config, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class AIMv2Transformer(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()

        self.blocks = nn.ModuleList(
            [AIMv2Block(config, quant_config, prefix=f"{prefix}.blocks.{i}") for i in range(config.num_hidden_layers)]
        )
        self.post_trunk_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        #outputs = []
        for block in self.blocks: # they take the -1 as the ref embeddings, like a clip skip
            tokens = block(tokens, mask)
            #outputs.append(tokens)
        #tokens = self.post_trunk_norm(tokens) NO NORM IN THE OG IMPLEMENTATION
        return tokens


class AIMv2Model(torch.nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str = ""):
        super().__init__()
        self.preprocessor = AIMv2ViTPreprocessor(config)
        self.trunk = AIMv2Transformer(config, quant_config=quant_config, prefix=f"{prefix}.trunk")

    @property
    def dtype(self):
        return self.trunk.blocks[0].attn.qkv.weight.dtype

    @property
    def device(self):
        return self.trunk.blocks[0].attn.qkv.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:

        x = self.preprocessor(pixel_values)
        x = self.trunk(
            x, mask
        )

        return x

