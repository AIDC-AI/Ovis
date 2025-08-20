from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig


class Siglip2NavitConfig(PretrainedConfig):
    """This is the configuration class to store the configuration of an [`Siglip2Navit`].

    Args:
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the SwiGLU representations.
        num_hidden_layers: Number of hidden layers in the Transformer.
        num_attention_heads: Number of attention heads for each attention layer
            in the Transformer.
        num_channels: Number of input channels.
        image_size: Image size.
        patch_size: Patch size.
        rms_norm_eps: Epsilon value used for the RMS normalization layer.
        attention_dropout: Dropout ratio for attention probabilities.
        projection_dropout: Dropout ratio for the projection layer after the attention.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        use_bias: Whether to add a bias in the feed-forward and projection layers.
        kwargs: Keyword arguments for the [`PretrainedConfig`].
    """

    model_type: str = "siglip2_navit"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        num_patches: int = -1,
        image_size: int = 512,
        patch_size: int = 16,
        hidden_act: str="gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_stride: int = 2,
        window_size: int = 112,
        fullatt_block_indexes: Optional[list] = None,
        temporal_patch_size: int = 1,
        preserve_original_pe: bool = True,
        use_rope: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_stride = hidden_stride
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.temporal_patch_size = temporal_patch_size
        self.preserve_original_pe = preserve_original_pe
        self.use_rope = use_rope

__all__ = ["Siglip2NavitConfig"]