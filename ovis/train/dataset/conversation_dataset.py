import copy
import json
import logging
from datetime import datetime
from typing import Dict

import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
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

        images = None
        max_partition = None
        if 'image' in sample:
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
                images.append(image)
        elif 'video' in sample:
            raise RuntimeError('video is to be supported')

        if images:
            max_partition = self.max_partitions[0] if len(images) == 1 else self.max_partitions[1]

        prompt, input_ids, pixel_values, labels = self.model.preprocess_inputs(
            conversations,
            images,
            max_partition=max_partition,
            generation_preface=None,
            return_labels=True,
            propagate_exception=False
        )

        if pixel_values is None:
            pixel_values, _ = self.visual_tokenizer.mock_input()

        input_ids = input_ids[:self.text_max_length]
        labels = labels[:self.text_max_length]

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
