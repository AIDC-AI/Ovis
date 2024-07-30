#!/bin/bash

ROOT=/playground
EXPNAME="${0##*/}"; EXPNAME="${EXPNAME%.sh}"

deepspeed ovis/train/train.py \
        --deepspeed scripts/zero2.json \
        --stage 1 \
        --dataset_names 'pixelprose-14m|wikipedia-348k|ocr-469k' \
        --llm_name_or_path google/gemma-2-9b-it \
        --conversation_formatter_class GemmaConversationFormatter \
        --visual_vocab_size 131072 \
        --visual_use_indicators True \
        --visual_drop_cls_token True \
        --visual_tokenize_function softmax \
        --visual_tokenizer_type siglip \
        --visual_hd_booster 's2wrapper' \
        --train_modules 'visual_tokenizer.backbone.layer.-1|visual_tokenizer.head|vte' \
        --multimodal_max_length 1152 \
        --text_max_length 421 \
        --output_dir $ROOT/checkpoints/ovis/$EXPNAME \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 128 \
        --num_train_epochs 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 0.1 \
        --save_total_limit 1 \
        --learning_rate 3e-4 \
        --max_grad_norm 1.0 \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --bf16 True \
        --dataloader_num_workers 8 \
        --dataloader_drop_last True \
        --dataloader_persistent_workers True \
        --gradient_checkpointing True \
        --run_name $EXPNAME \
        --report_to tensorboard
