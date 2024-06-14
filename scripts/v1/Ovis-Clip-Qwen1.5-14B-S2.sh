#!/bin/bash

ROOT=/playground
EXPNAME="${0##*/}"; EXPNAME="${EXPNAME%.sh}"

deepspeed ovis/train/train.py \
        --deepspeed scripts/zero3.json \
        --stage 2 \
        --dataset_names 'llava-pretrain-558k|sharegpt4v-pretrain-82k|allava-caption-laion-4v-485k|allava-caption-vflan-4v-203k|laion-description-11k|cc12m-description-1m' \
        --ovis_pretrained_path $ROOT/checkpoints/ovis/Ovis-Clip-Qwen1.5-14B-S1 \
        --train_modules 'visual_tokenizer|vte' \
        --multimodal_max_length 2048 \
        --output_dir $ROOT/checkpoints/ovis/$EXPNAME \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --num_train_epochs 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 0.2 \
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
