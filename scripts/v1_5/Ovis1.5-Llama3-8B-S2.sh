#!/bin/bash

ROOT=/playground
EXPNAME="${0##*/}"; EXPNAME="${EXPNAME%.sh}"

deepspeed ovis/train/train.py \
        --deepspeed scripts/zero3.json \
        --stage 2 \
        --dataset_names 'allava-caption-laion-4v-469k|allava-caption-vflan-4v-195k|cc12m-description-387k' \
        --ovis_pretrained_path $ROOT/checkpoints/ovis/Ovis1.5-Llama3-8B-S1 \
        --train_modules 'visual_tokenizer|vte' \
        --multimodal_max_length 1755 \
        --text_max_length 1024 \
        --output_dir $ROOT/checkpoints/ovis/$EXPNAME \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 32 \
        --num_train_epochs 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 0.2 \
        --save_total_limit 1 \
        --learning_rate 5e-5 \
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
