import json
import os
import pathlib

import deepspeed
import flash_attn
import torch
import transformers
from deepspeed import get_accelerator
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, Trainer
from transformers.trainer_utils import set_seed

from ovis.model.configuration_ovis import OvisConfig
from ovis.model.modeling_ovis import Ovis, VisualTokenizer
from ovis.train.arguments import ModelArguments, TrainingArguments
from ovis.train.callback import MonitorCallback
from ovis.train.dataset.caption_dataset import CaptionDataset
from ovis.train.dataset.conversation_dataset import ConversationDataset
from ovis.train.dataset.multimodal_dataset import DataCollatorForMultimodalDataset
from ovis.util.constants import BEGIN_LINE, END_LINE
from ovis.util.utils import smart_unit, rank0_print, rankN_print, replace_torch_load_with_weights_only_false


def load_model(model_args: ModelArguments, training_args: TrainingArguments):
    model, loading_info = Ovis.from_pretrained(
        training_args.ovis_pretrained_path,
        output_loading_info=True,
        trust_remote_code=True
    )
    rankN_print(BEGIN_LINE)
    rankN_print(f'Loading info of Ovis:\n{loading_info}')
    rankN_print(END_LINE)

    model.accepts_loss_kwargs = model_args.accepts_loss_kwargs
    if model_args.attn_implementation:
        model.llm.config._attn_implementation = model_args.attn_implementation
        model.visual_tokenizer.vit.config._attn_implementation = model_args.attn_implementation
    model.llm.config.use_cache = False
    model.config.use_cache = False
    rank0_print(BEGIN_LINE)
    rank0_print(f'model.config:\n{model.config}')
    rank0_print(END_LINE)

    return model


def load_data(model: Ovis, training_args: TrainingArguments):
    # construct data module
    if training_args.data_type == 'caption':
        train_dataset = CaptionDataset(model, training_args)
    elif training_args.data_type == 'conversation':
        train_dataset = ConversationDataset(model, training_args)
    else:
        raise ValueError(f'Invalid data type: {training_args.data_type}')
    data_module = dict(
        train_dataset=train_dataset,
        data_collator=DataCollatorForMultimodalDataset(model.text_tokenizer, training_args)
    )
    return data_module


def train(model_args: ModelArguments, training_args: TrainingArguments):
    # save args to checkpoint dir
    with training_args.main_process_first(local=False):
        if training_args.process_index == 0:
            def args2dict(args):
                return {k: str(v) for k, v in args.__dict__.items()}

            args_log = json.dumps(dict(
                model_args=args2dict(model_args),
                training_args=args2dict(training_args)
            ), ensure_ascii=False, indent=2)
            print(args_log)
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(training_args.output_dir, 'model_training_args.json'), 'w',
                      encoding='utf-8') as f:
                f.write(args_log + '\n')

    # load model & data
    model = load_model(model_args, training_args)

    # select train modules, support different learning rate for different modules
    model.requires_grad_(False)
    parameters = []
    for module_name_lr in training_args.train_modules.split('|'):
        module_name_lr = module_name_lr.replace(' ', '').split(':')
        module_lr = training_args.learning_rate
        if len(module_name_lr) == 2:
            module_name, module_lr = module_name_lr[0], float(module_name_lr[1])
        elif len(module_name_lr) == 1:
            module_name = module_name_lr[0]
        else:
            raise ValueError
        match module_name:
            case 'all':
                module = model
            case 'llm':
                module = model.llm
            case 'visual_tokenizer':
                module = model.visual_tokenizer
            case 'visual_tokenizer.head':
                module = model.visual_tokenizer.head
            case 'visual_tokenizer.vit':
                module = model.visual_tokenizer.vit
            case 'visual_tokenizer.vit.last_block':
                module = model.visual_tokenizer._get_last_block()
            case 'vte':
                module = model.vte
            case _:
                raise ValueError(f'Invalid train module name: {module_name}')
        module.requires_grad_(True)
        parameters.append({'params': module.parameters(), 'lr': module_lr})

    optimizer = torch.optim.AdamW(parameters, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    rank0_print(BEGIN_LINE)
    rank0_print('Parameters to train:')
    param_lr_mapping = {}
    for group in optimizer.param_groups:
        lr = group['lr']
        for param in group['params']:
            param_lr_mapping[param] = lr
    rank0_print(f'LLM\'s attn implementation: {model.llm.config._attn_implementation}')
    rank0_print(
        f'ViT\'s attn implementation: {model.visual_tokenizer.vit.config._attn_implementation}'
    )
    rank0_print(END_LINE)

    # construct data module
    datasets = []
    dataset_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f'dataset/{training_args.data_info_version}.json')
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    for name in training_args.data_name.split('|'):
        info = dataset_info[name]
        data_format = info['data_format']
        if data_format == 'caption':
            dataset = CaptionDataset(name, info, model, training_args)
        elif data_format == 'conversation':
            dataset = ConversationDataset(name, info, model, training_args)
        else:
            raise ValueError(f'Invalid data format `{data_format}` for dataset `{name}`')
        datasets.append(dataset)
    data_module = dict(
        train_dataset=ConcatDataset(datasets),
        data_collator=DataCollatorForMultimodalDataset(model.text_tokenizer)
    )

    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[MonitorCallback],
        **data_module
    )
    rankN_print(BEGIN_LINE)
    rankN_print(f'model_accepts_loss_kwargs: {trainer.model_accepts_loss_kwargs}')
    rankN_print(END_LINE)
    rankN_print(BEGIN_LINE)
    rankN_print('Dataset sample tensor:')
    rankN_print(data_module['train_dataset'][0])
    rankN_print(END_LINE)
    rankN_print(BEGIN_LINE)
    rankN_print('Dataset sample input_ids decoding:')
    rankN_print(model.text_tokenizer.decode([x for x in data_module['train_dataset'][0]['input_ids'] if x >= 0]))
    rankN_print(END_LINE)
    rankN_print(BEGIN_LINE)
    rankN_print('Dataset sample labels decoding:')
    rankN_print(model.text_tokenizer.decode([x for x in data_module['train_dataset'][0]['labels'] if x >= 0]))
    rankN_print(END_LINE)
    rankN_print(BEGIN_LINE)
    rankN_print(f'#param of model: {smart_unit(model.num_parameters())}')
    rankN_print(f'#param of llm: {smart_unit(model.llm.num_parameters())}')
    rankN_print(f'#param of vit: {smart_unit(model.visual_tokenizer.vit.num_parameters())}')
    rankN_print(f'#param of vte: {smart_unit(model.vte.weight.numel())}')
    rankN_print(f'#dtype of model: {model.dtype}')
    rankN_print(f'#dtype of llm: {model.llm.dtype}')
    rankN_print(f'#dtype of vit: {model.visual_tokenizer.vit.dtype}')
    rankN_print(f'#dtype of vte: {model.vte.weight.dtype}')
    rankN_print(END_LINE)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # save model
    model.llm.config.use_cache = True
    model.config.use_cache = True
    trainer.save_model()


if __name__ == '__main__':
    replace_torch_load_with_weights_only_false()
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, training_args)
