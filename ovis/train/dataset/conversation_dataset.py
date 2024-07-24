import copy
import json
import logging
from datetime import datetime
from typing import Dict

import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import IGNORE_INDEX
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
        conversations = copy.deepcopy(sample["conversations"])
        prompt, input_ids, labels = self.conversation_formatter.format(conversations)
        input_ids = input_ids[:self.text_max_length]
        labels = labels[:self.text_max_length]

        if 'image' in sample:
            valid_images = False
            pixel_values = None

            # read images
            image_paths = sample['image']
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            images = []
            for image_path in image_paths:
                image, e = self.read_image(image_path)
                if image is None:
                    logging.warning(
                        f'reading image failed with index: {i}, image path: {image_path}, and exception: {e}')
                    images = None
                    break
                else:
                    images.append(image)

            # process images
            if images is not None:
                try:
                    pixel_values = torch.cat([self.visual_tokenizer.preprocess_image(image) for image in images], dim=0)
                    valid_images = True
                except Exception as e:
                    logging.warning(
                        f'processing image failed with index: {i}, image paths: {image_paths}, and exception: {e}')

            if not valid_images:
                logging.warning(f'image is not valid, so set labels ignored for sample:\n{sample}')
                pixel_values = self.visual_tokenizer.get_zero_pixel_values(n=len(image_paths))
                labels = torch.tensor([IGNORE_INDEX] * len(labels), dtype=torch.long)
        else:
            pixel_values = self.visual_tokenizer.get_zero_pixel_values(n=1)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
        )
