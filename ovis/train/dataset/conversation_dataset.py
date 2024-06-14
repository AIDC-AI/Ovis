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

        pixel_values = torch.zeros(1, 3, self.image_height, self.image_width)
        is_multimodal = 'image' in sample
        valid_image = False
        if is_multimodal:
            # read and process image
            image_path = sample['image'][0]
            image, last_e = self.read_image(image_path)
            if image is None:
                logging.warning(
                    f'reading image failed with index: {i}, image path: {image_path}, and exception: {last_e}')
            else:
                try:
                    pixel_values = self.visual_tokenizer.preprocess_image(image)
                    valid_image = True
                except Exception as e:
                    logging.warning(
                        f'processing image failed with index: {i}, image path: {image_path}, and exception: {last_e}')

        prompt, input_ids, labels = self.conversation_formatter.format(conversations)

        if is_multimodal and not valid_image:
            logging.warning(f'image is not valid, so set labels ignored for sample:\n{sample}')
            labels = torch.tensor([IGNORE_INDEX] * len(labels), dtype=torch.long)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
        )
