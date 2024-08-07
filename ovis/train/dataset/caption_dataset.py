import logging
from datetime import datetime
from typing import Dict

import pandas
import torch

from ovis.model.modeling_ovis import Ovis
from ovis.train.arguments import TrainingArguments
from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from ovis.util.utils import rank0_print


class CaptionDataset(MultimodalDataset):
    def __init__(self, name: str, info: Dict, model: Ovis, training_args: TrainingArguments):
        super().__init__(name, info, model, training_args)
        self.caption_template = info.get('caption_template', training_args.caption_template)

    def load(self):
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} from {self.meta_file} begin")
        samples = pandas.read_parquet(self.meta_file, engine='pyarrow')
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} end")
        return samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples.iloc[i]
        text = sample['caption']
        image_path = sample['image_path']

        # read and preprocess image
        pixel_values = self.visual_tokenizer.get_zero_pixel_values(n=1)
        valid_image = False
        image, e = self.read_image(image_path)
        if image is None:
            logging.warning(
                f'reading image failed with index: {i}, image path: {image_path}, and exception: {e}')
        else:
            try:
                pixel_values = self.visual_tokenizer.preprocess_image(image)
                valid_image = True
            except Exception as e:
                logging.warning(
                    f'preprocessing image failed with index: {i}, image path: {image_path}, and exception: {e}')

        # preprocess text
        if text is None:
            logging.warning(f'text is `None`, index: {i}')
            text = ""
        if not valid_image:
            logging.warning(f'image is not valid, so set text as empty, index: {i}, image path: {image_path}')
            text = ""
        text = text.replace(IMAGE_TOKEN, '')
        if self.caption_template is None:
            token_ids = self.text_tokenizer(text, add_special_tokens=False).input_ids
            input_ids = [IMAGE_TOKEN_INDEX] + token_ids
            labels = [IGNORE_INDEX] + token_ids
        else:
            head, tail = self.caption_template.split(IMAGE_TOKEN)
            head_ids = self.text_tokenizer(head, add_special_tokens=False).input_ids
            tail_ids = self.text_tokenizer(tail, add_special_tokens=False).input_ids
            text_ids = self.text_tokenizer(text, add_special_tokens=False).input_ids
            input_ids = head_ids + [IMAGE_TOKEN_INDEX] + tail_ids + text_ids
            labels = [IGNORE_INDEX] * (len(input_ids) - len(text_ids)) + text_ids

        input_ids = input_ids[:self.text_max_length]
        labels = labels[:self.text_max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
