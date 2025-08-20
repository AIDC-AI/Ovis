import logging
from datetime import datetime
from typing import Dict

import pandas
import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import IMAGE_TOKEN, IGNORE_ID
from ovis.util.utils import rank0_print


class CaptionDataset(MultimodalDataset):

    def load(self):
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} from {self.meta_file} begin")
        samples = pandas.read_parquet(self.meta_file, engine='pyarrow')
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} end")
        return samples


    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        image_path = sample['image']
        if isinstance(image_path, list):
            assert len(image_path) == 1
            image_path = image_path[0]
        text = sample['caption'].replace(IMAGE_TOKEN, '').strip()
        caption_template = sample['caption_template']

        # process text
        head, tail = caption_template.split(IMAGE_TOKEN)
        head_ids = self.text_tokenizer(head, add_special_tokens=False).input_ids
        tail_ids = self.text_tokenizer(tail, add_special_tokens=False).input_ids
        text_ids = self.text_tokenizer(text, add_special_tokens=False).input_ids

        # process image
        try:
            image, last_e = self.read_image(image_path)
            pixel_values, grid_thws = self.visual_tokenizer.preprocess(
                image=image,
                min_pixels=self.training_args.single_image_min_pixels,
                max_pixels=self.training_args.single_image_max_pixels
            )
            num_image_atoms = grid_thws[0].prod().item()
            num_image_atoms //= self.visual_tokenizer.vit.config.hidden_stride ** 2
            num_image_atoms //= self.visual_tokenizer.vit.config.temporal_patch_size
            image_placeholders = [INDICATOR_IDS[0]] + [VISUAL_ATOM_ID] * num_image_atoms + [INDICATOR_IDS[1]]
            input_ids = head_ids + image_placeholders + tail_ids + text_ids
            labels = [IGNORE_ID] * (len(input_ids) - len(text_ids)) + text_ids
            assert self.text_tokenizer.pad_token_id not in input_ids, \
                "The sample's text contains a padding token: `{self.text_tokenizer.pad_token}`"
        except Exception as e:
            logging.exception(f'processing smaple failed with i: {i}, idx: {idx}, image_path: {image_path}')
            pixel_values, grid_thws = None, None
            input_ids = [0]
            labels = [IGNORE_ID]

        input_ids = input_ids[:self.training_args.multimodal_max_length]
        labels = labels[:self.training_args.multimodal_max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.full_like(input_ids, fill_value=True, dtype=torch.bool)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=attention_mask,
            labels=labels
        )
