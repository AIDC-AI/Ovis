#!/bin/bash

ROOT=/playground
EXPNAME="${0##*/}"; EXPNAME="${EXPNAME%.sh}"

deepspeed ovis/train/train.py \
        --deepspeed scripts/zero2.json \
        --stage 1 \
        --dataset_info dataset_info_v1 \
        --dataset_names coyo-10m \
        --caption_template "<image>'s caption: " \
        --llm_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --conversation_formatter_class Llama3ConversationFormatter \
        --pad_token_id 128001 \
        --visual_vocab_size 131072 \
        --visual_use_indicators True \
        --visual_drop_cls_token True \
        --visual_tokenize_function softmax \
        --visual_tokenizer_type clip \
        --visual_re_init_layer_begin 23 \
        --train_modules 'visual_tokenizer.re_init_layers|visual_tokenizer.head|vte' \
        --multimodal_max_length 1024 \
        --output_dir $ROOT/checkpoints/ovis/$EXPNAME \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 128 \
        --num_train_epochs 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 0.1 \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --max_grad_norm 1.0 \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --bf16 True \
        --dataloader_num_workers 8 \
        --gradient_checkpointing True \
        --run_name $EXPNAME \
        --report_to tensorboard
