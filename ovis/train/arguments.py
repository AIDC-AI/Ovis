from dataclasses import dataclass, field
from typing import Optional

import transformers

from ovis.util.utils import rankN_print


@dataclass
class ModelArguments:
    llm_name_or_path: Optional[str] = field(default=None)
    vit_name_or_path: Optional[str] = field(default=None)
    visual_vocab_size: int = field(default=65536)
    conversation_formatter_class: str = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    accepts_loss_kwargs: bool = field(default=True)
    vit_hidden_stride: int = field(default=2)
    vit_window_size: int = field(default=112)
    vit_temporal_patch_size: int = field(default=1)
    vit_fullatt_block_indexes: Optional[str] = field(default=None)
    vit_preserve_original_pe: Optional[bool] = field(default=True)
    vit_use_rope: Optional[bool] = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    data_info_version: Optional[str] = field(default=None)
    data_name: Optional[str] = field(default=None)  # a|b|c
    data_type: Optional[str] = field(default=None)  # caption, conversation
    ovis_pretrained_path: Optional[str] = field(default=None)
    stage: Optional[int] = field(default=None)
    train_modules: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    save_safetensors: bool = field(default=True)
    monitor_step: int = field(default=100)
    model_init_seed: int = field(default=0)
    multimodal_max_length: int = field(default=4096)
    text_max_length: Optional[int] = field(default=4096)
    min_frames: int = field(default=8)
    max_frames: int = field(default=8)
    overall_ratio: Optional[str] = field(default=None)
    mix_data_name: Optional[str] = field(default=None)
    mix_ratio: Optional[float] = field(default=None)
    min_lr_rate: Optional[float] = field(default=None)
    single_image_min_pixels: int = field(default=448*448)
    single_image_max_pixels: int = field(default=1792*1344)
    multiple_image_min_pixels: int = field(default=448*448)
    multiple_image_max_pixels: int = field(default=448*448)
    video_min_pixels: int = field(default=448*448)
    video_max_pixels: int = field(default=448*448)

    def __post_init__(self):
        if self.min_lr_rate is not None:
            self.lr_scheduler_kwargs = {
                "min_lr_rate": self.min_lr_rate
            }
        if self.gradient_checkpointing:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
        if self.stage < 3:
            self.save_safetensors = False
        super().__post_init__()
        assert self.model_init_seed != self.seed, "`model_init_seed` should be different from `seed`"