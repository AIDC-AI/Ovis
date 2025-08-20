from typing import Union, Optional

from transformers import PretrainedConfig, Qwen3Config

from . import Siglip2NavitConfig


class OvisConfig(PretrainedConfig):
    model_type = "ovis"
    sub_configs = dict(llm_config=Qwen3Config, vit_config=Siglip2NavitConfig)

    def __init__(self,
        llm_config: Optional[Union[Qwen3Config, dict]] = None,
        vit_config: Optional[Union[Siglip2NavitConfig, dict]] = None,
        visual_vocab_size=65536,
        hidden_size=None,
        conversation_formatter_class=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(llm_config, dict):
            llm_config = Qwen3Config(**llm_config)
        self.llm_config = llm_config
        if isinstance(vit_config, dict):
            vit_config = Siglip2NavitConfig(**vit_config)
        self.vit_config = vit_config
        self.visual_vocab_size = visual_vocab_size
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        if kwargs.get('attn_implementation'):
            self.llm_config._attn_implementation = kwargs['attn_implementation']
            self.vit_config._attn_implementation = kwargs['attn_implementation']
