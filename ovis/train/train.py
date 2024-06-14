import json
import os
import pathlib

import deepspeed
import transformers
from torch.utils.data import ConcatDataset
from transformers import Trainer
from transformers.integrations.deepspeed import unset_hf_deepspeed_config, set_hf_deepspeed_config

from callback import TuneTauCallback, MonitorCallback
from ovis.model.configuration_ovis import OvisConfig
from ovis.model.modeling_ovis import Ovis
from ovis.train.arguments import ModelArguments, TrainingArguments
from ovis.train.dataset.caption_dataset import CaptionDataset
from ovis.train.dataset.conversation_dataset import ConversationDataset
from ovis.train.dataset.multimodal_dataset import DataCollatorForMultimodalDataset
from ovis.util.constants import BEGIN_LINE, END_LINE
from ovis.util.utils import smart_unit, rank0_print


def train():
    # parse args
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

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

    # load model
    if not training_args.ovis_pretrained_path:
        config = OvisConfig(
            visual_tokenizer_config=dict(
                model_type=model_args.visual_tokenizer_type + "_visual_tokenizer",
                vocab_size=model_args.visual_vocab_size,
                tokenize_function=model_args.visual_tokenize_function,
                tau=model_args.visual_tau,
                depths=model_args.visual_depths,
                use_indicators=model_args.visual_use_indicators,
                drop_cls_token=model_args.visual_drop_cls_token,
                hidden_stride=model_args.visual_hidden_stride
            ) if not training_args.visual_tokenizer_pretrained_path else None,
            multimodal_max_length=model_args.multimodal_max_length,
            conversation_formatter_class=model_args.conversation_formatter_class
        )
        # deepspeed zero.Init with bfloat16 sometimes fail, so temporarily disable zero.Init here
        unset_hf_deepspeed_config()
        model = Ovis(config,
                     train_from_scratch=True,
                     llm_name_or_path=model_args.llm_name_or_path,
                     visual_tokenizer_pretrained_path=training_args.visual_tokenizer_pretrained_path,
                     pad_token_id=model_args.pad_token_id)
        if getattr(training_args, 'hf_deepspeed_config', None) is not None:
            set_hf_deepspeed_config(training_args.hf_deepspeed_config)
    else:
        model, loading_info = Ovis.from_pretrained(training_args.ovis_pretrained_path,
                                                   multimodal_max_length=model_args.multimodal_max_length,
                                                   output_loading_info=True)
        rank0_print(BEGIN_LINE)
        rank0_print(f'Loading info of Ovis:\n{loading_info}')
        rank0_print(END_LINE)
        training_args.visual_re_init_layer_begin = None  # re_init a pretrained Ovis is harmful, so disabled
        training_args.vte_re_init = False

    model.config.use_cache = False
    text_tokenizer = model.get_text_tokenizer()

    rank0_print(BEGIN_LINE)
    rank0_print(f'model.config:\n{model.config}')
    rank0_print(END_LINE)

    # maybe re-init some visual backbone layers
    if training_args.visual_re_init_layer_begin is not None and training_args.visual_tokenizer_pretrained_path is None:
        model.get_visual_tokenizer().re_init_layers(training_args.visual_re_init_layer_begin)

    # maybe re-init vte
    if training_args.vte_re_init:
        with deepspeed.zero.GatheredParameters([model.get_wte().weight]):
            mean = model.get_wte().weight.mean().item()
            std = model.get_wte().weight.std().item()
        rank0_print(f'Statistics of embedding table of LLM: {mean=}, {std=}')
        model.re_init_vte(mean, std)

    # select train modules
    model.requires_grad_(False)
    for module in training_args.train_modules.split('|'):
        if module == 'all':
            model.requires_grad_(True)
        elif module == 'llm':
            model.get_llm().requires_grad_(True)
        elif module == 'visual_tokenizer':
            model.get_visual_tokenizer().requires_grad_(True)
        elif module == 'visual_tokenizer.backbone':
            model.get_visual_tokenizer().get_backbone().requires_grad_(True)
        elif module == 'visual_tokenizer.re_init_layers':
            assert training_args.visual_re_init_layer_begin is not None, \
                "To train `visual_tokenizer.re_init_layers`, you need set `visual_re_init_layer_begin` in training args"
            for name, layer in model.get_visual_tokenizer().get_re_init_layer_dict(
                    training_args.visual_re_init_layer_begin).items():
                layer.requires_grad_(True)
        elif module == 'visual_tokenizer.head':
            model.get_visual_tokenizer().get_head().requires_grad_(True)
        elif module == 'vte':
            model.get_vte().requires_grad_(True)
        else:
            raise ValueError(f'Invalid train module name: {module}')

    rank0_print(BEGIN_LINE)
    rank0_print('Parameters to train:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)
    rank0_print(END_LINE)

    # construct data module
    datasets = []
    dataset_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_info.json')
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    for name in training_args.dataset_names.split('|'):
        info = dataset_info[name]
        meta_file = info['meta_file']
        image_dir = info['image_dir']
        data_format = info['data_format']
        if data_format == 'caption':
            dataset = CaptionDataset(name, meta_file, image_dir, model, training_args)
        elif data_format == 'conversation':
            dataset = ConversationDataset(name, meta_file, image_dir, model, training_args)
        else:
            raise ValueError(f'Invalid data format `{data_format}` for dataset `{name}`')
        datasets.append(dataset)
    data_module = dict(
        train_dataset=ConcatDataset(datasets),
        data_collator=DataCollatorForMultimodalDataset(text_tokenizer)
    )

    # train
    train_callbacks = [MonitorCallback]
    if model_args.visual_tokenize_function == 'gumbel_argmax':
        train_callbacks.append(TuneTauCallback)
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=train_callbacks,
        **data_module)
    rank0_print(BEGIN_LINE)
    rank0_print('Dataset sample tensor:')
    rank0_print(data_module['train_dataset'][0])
    rank0_print(END_LINE)
    rank0_print(BEGIN_LINE)
    rank0_print('Dataset sample input_ids decoding:')
    rank0_print(text_tokenizer.decode([x for x in data_module['train_dataset'][0]['input_ids'] if x >= 0]))
    rank0_print(END_LINE)
    rank0_print(BEGIN_LINE)
    rank0_print('Dataset sample labels decoding:')
    rank0_print(text_tokenizer.decode([x for x in data_module['train_dataset'][0]['labels'] if x >= 0]))
    rank0_print(END_LINE)
    rank0_print(BEGIN_LINE)
    rank0_print(f'#param of model: {smart_unit(model.num_parameters())}')
    rank0_print(f'#param of llm: {smart_unit(model.get_llm().num_parameters())}')
    rank0_print(f'#param of visual_tokenizer: {smart_unit(model.get_visual_tokenizer().num_parameters())}')
    rank0_print(f'#param of vte: {smart_unit(model.get_vte().weight.numel())}')
    rank0_print(END_LINE)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # save model
    model.config.use_cache = True
    trainer.save_model()


if __name__ == '__main__':
    train()
