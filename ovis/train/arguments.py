from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    llm_name_or_path: Optional[str] = field(default=None)
    visual_tokenizer_type: str = field(default=None)
    visual_vocab_size: int = field(default=8192)
    visual_use_indicators: bool = field(default=False)
    visual_drop_cls_token: bool = field(default=False)
    visual_tokenize_function: str = field(default='softmax')
    visual_tau: float = field(default=1.0)
    visual_depths: Optional[str] = field(default=None)
    visual_hidden_stride: int = field(default=1)
    visual_hd_booster: Optional[str] = field(default=None)
    multimodal_max_length: int = field(default=2048)
    conversation_formatter_class: str = field(default=None)
    pad_token_id: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_names: Optional[str] = field(default=None)  # a|b|c
    dataset_info: Optional[str] = field(default='dataset_info_v1_5')
    ovis_pretrained_path: Optional[str] = field(default=None)
    visual_tokenizer_pretrained_path: Optional[str] = field(default=None)
    caption_template: Optional[str] = field(default=None)
    stage: Optional[int] = field(default=None)
    train_modules: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    visual_max_tau: float = field(default=5.0)
    visual_min_tau: float = field(default=0.05)
    save_safetensors: bool = field(default=True)
    monitor_step: int = field(default=100)
    visual_re_init_layer_begin: Optional[int] = field(default=None)
    vte_re_init: bool = field(default=False)
    text_max_length: int = field(default=1024)
    train_attn_implementation: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.gradient_checkpointing:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
        super().__post_init__()
