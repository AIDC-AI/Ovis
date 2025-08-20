import copy
import json
import logging
from datetime import datetime
from typing import Dict

import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import VIDEO_TOKEN, IMAGE_TOKEN, IGNORE_ID
from ovis.util.utils import rank0_print


class ConversationDataset(MultimodalDataset):
    def load(self):
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} from {self.meta_file} begin")
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        rank0_print(f'#samples: {len(samples)}')
        rank0_print(f'sample: {samples[0]}')
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} end")
        return samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        conversations = sample["conversations"]

        # try:
        images = None
        videos = None
        n_image_or_frame = 0
        if 'image' in sample:
            images = []
            image_paths = sample['image']
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for image_path in image_paths:
                image, last_e = self.read_image(image_path)
                assert image is not None, f"Failed to read image from {image_path}"
                images.append(image)
            n_image_or_frame = len(images)
        elif 'video' in sample or 'video_frames' in sample:
            video, last_e = self.read_video(sample, min_frames=self.min_frames, max_frames=self.max_frames)
            video_path = sample.get('video') or sample.get('video_frames')
            assert video is not None, f"Failed to read video from {video_path}"
            videos = [video]
            n_image_or_frame = len(video)

        if images is None and videos is None:
            min_pixels = 0
            max_pixels = 0
        elif videos is not None:
            min_pixels = self.training_args.video_min_pixels
            max_pixels = self.training_args.video_max_pixels
        elif len(images) == 1:
            min_pixels = self.training_args.single_image_min_pixels
            max_pixels = self.training_args.single_image_max_pixels
        else:
            min_pixels = self.training_args.multiple_image_min_pixels
            max_pixels = self.training_args.multiple_image_max_pixels

        if min_pixels < 0:
            min_pixels = self.training_args.single_image_min_pixels
        if max_pixels < 0:
            max_pixels = max(min_pixels, self.training_args.single_image_max_pixels // n_image_or_frame)

        prompt, input_ids, pixel_values, grid_thws, labels = self.model.preprocess_inputs(
            conversations,
            images=images,
            videos=videos,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            generation_preface=None,
            return_labels=True,
        )
        if pixel_values is None:
            input_ids, pixel_values, grid_thws, labels = self.truncate_inputs(
                input_ids, pixel_values, grid_thws, labels, max_length=self.training_args.text_max_length
            )
        else:
            input_ids, pixel_values, grid_thws, labels = self.truncate_inputs(
                input_ids, pixel_values, grid_thws, labels, max_length=self.training_args.multimodal_max_length
            )
        assert self.text_tokenizer.pad_token_id not in input_ids, \
            "The sample's text contains a padding token: `{self.text_tokenizer.pad_token}`"

        del sample
        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=torch.full_like(input_ids, fill_value=True, dtype=torch.bool),
            labels=labels
        )
