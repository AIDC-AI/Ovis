import json
import logging
import os
import traceback
from typing import Dict, Sequence, Union, List

import numpy as np
import torch
import moviepy.editor as mp
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ovis.model.modeling_ovis import Ovis
from ovis.train.arguments import TrainingArguments
from ovis.util.constants import IGNORE_ID, BEGIN_LINE, END_LINE, VISUAL_ATOM_ID, INDICATOR_IDS


class MultimodalDataset(Dataset):
    def __init__(self, name: str, info: Dict, model: Ovis, training_args: TrainingArguments):
        self.name = name
        self.model = model
        self.training_args = training_args

        self.meta_file = info['meta_file']
        self.image_dir = info['image_dir']
        self.caption_template = info.get('caption_template', None)
        self.text_tokenizer = self.model.text_tokenizer
        self.visual_tokenizer = self.model.visual_tokenizer
        self.text_max_length = training_args.text_max_length
        self.min_frames = training_args.min_frames
        self.max_frames = training_args.max_frames

        self.samples = self.load()

    def load(self):
        raise NotImplementedError

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def read_image(self, path):
        try:
            full_path = os.path.join(self.image_dir, path)
            image = Image.open(full_path).convert('RGB')
            return image, None
        except Exception as e:
            return None, e

    def read_video(self, sample, min_frames, max_frames):
        def _sampling_idx(_len, _min, _max):
            if _len < _min or _len > _max:
                tgt_len = _min if _len < _min else _max
                stride = _len / tgt_len
                sampled_ids = []
                for i in range(tgt_len):
                    start = int(np.round(stride * i))
                    end = int(np.round(stride * (i + 1)))
                    sampled_ids.append(min(_len - 1, (start + end) // 2))
                return sampled_ids
            else:
                return list(range(_len))

        if "video_frames" in sample:
            frames = []
            frames_paths = sample['video_frames']
            sampled_ids = _sampling_idx(len(frames_paths), min_frames, max_frames)
            for idx in sampled_ids:
                frame, last_e = self.read_image(os.path.join(self.image_dir, frames_paths[idx]))
                if frame is None:
                    return None, last_e
                frames.append(frame)
            return frames, None
        elif "video" in sample:
            video_path = os.path.join(self.image_dir, sample['video'])

            max_tries = 2
            last_e = None
            for _ in range(max_tries):
                try:
                    with mp.VideoFileClip(video_path) as clip:
                        total_frames = int(clip.fps * clip.duration)
                        sampled_ids = _sampling_idx(total_frames, min_frames, max_frames)
                        frames = [clip.get_frame(idx / clip.fps) for idx in sampled_ids]
                        frames = [Image.fromarray(frame, mode='RGB') for frame in frames]

                    if len(frames) == 0 or any(frame.size[0] < 5 or frame.size[1] < 5 for frame in frames):
                        raise ValueError("frames are empty or there exists very small frame")
                    return frames, None
                except Exception as e:
                    last_e = f"read video error: {e}\n detailed info: {traceback.format_exc()}"
            return None, last_e
        else:
            return None, RuntimeError(f"missing `video_frames` and `video` in sample: {json.dumps(sample)}")
        
    def truncate_inputs(
        self, input_ids, pixel_values, grid_thws, labels, max_length
    ):
        input_ids = input_ids[0, :max_length]
        labels = labels[0, :max_length]
        if input_ids[-1] in (VISUAL_ATOM_ID, INDICATOR_IDS[0], INDICATOR_IDS[2]):  # incomplete visual input
            last_text_id_pos = (input_ids >= 0).nonzero()[-1].item() + 1
            input_ids = input_ids[:last_text_id_pos]
            labels = labels[:last_text_id_pos]
        num_visual_atom = torch.eq(input_ids, VISUAL_ATOM_ID).sum().item()
        if num_visual_atom > 0:
            vit = self.model.visual_tokenizer.vit
            ratio = vit.config.temporal_patch_size * vit.config.hidden_stride ** 2
            cumsum_patches = grid_thws.prod(dim=1).cumsum(dim=0)
            last_grid_index = (cumsum_patches // ratio == num_visual_atom).nonzero().item()
            pixel_values = pixel_values[:cumsum_patches[last_grid_index]]
            grid_thws = grid_thws[:last_grid_index+1]
        else:
            pixel_values, grid_thws = None, None

        return input_ids, pixel_values, grid_thws, labels


class DataCollatorForMultimodalDataset:
    def __init__(self, text_tokenizer: PreTrainedTokenizer):
        self.text_tokenizer = text_tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        keys = ("input_ids", "pixel_values", "grid_thws", "attention_mask", "labels")
        input_ids, pixel_values, grid_thws, attention_mask, labels = (
            tuple(instance[key] for instance in instances) for key in keys
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id
        )
        pixel_values = [x for x in pixel_values if x is not None]
        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        grid_thws = [x for x in grid_thws if x is not None]
        grid_thws = torch.cat(grid_thws, dim=0) if len(grid_thws) > 0 else None
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=False
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_ID
        )
        if 0 not in attention_mask:
            input_ids = F.pad(input_ids, (0, 1), value=self.text_tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, 1), value=False)
            labels = F.pad(labels, (0, 1), value=IGNORE_ID)
            
        if torch.all(labels == IGNORE_ID):
            logging.warning(f'[DataCollatorForMultimodalDataset] All samples in the current batch are ignored.')
        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=attention_mask,
            labels=labels
        )

