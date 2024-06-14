import logging
import os.path
from typing import Dict, Sequence, Union, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ovis.model.modeling_ovis import Ovis
from ovis.train.arguments import TrainingArguments
from ovis.util.constants import IGNORE_INDEX


class MultimodalDataset(Dataset):
    def __init__(self, name: str, meta_file: str, image_dir: str, model: Ovis, training_args: TrainingArguments):
        self.name = name
        self.meta_file = meta_file
        self.image_dir = image_dir
        self.text_tokenizer = model.get_text_tokenizer()
        self.visual_tokenizer = model.get_visual_tokenizer()
        self.image_height, self.image_width = self.visual_tokenizer.get_image_size()
        self.conversation_formatter = model.get_conversation_formatter()
        self.training_args = training_args
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


class DataCollatorForMultimodalDataset:
    def __init__(self, text_tokenizer: PreTrainedTokenizer):
        self.text_tokenizer = text_tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        pixel_values, input_ids, labels = tuple([instance[key] for instance in instances]
                                                for key in ("pixel_values", "input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        num_valid_label = torch.not_equal(labels, IGNORE_INDEX).sum().item()
        if num_valid_label == 0:
            logging.warning(
                f'[DataCollatorForMultimodalDataset] All labels in a batch are ignored, which may lead to training instability\n{input_ids=}\n{attention_mask=}\n{labels=}')
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values
        )
