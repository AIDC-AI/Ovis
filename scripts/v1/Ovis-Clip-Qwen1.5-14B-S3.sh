#!/bin/bash

ROOT=/playground
EXPNAME="${0##*/}"; EXPNAME="${EXPNAME%.sh}"

deepspeed ovis/train/train.py \
        --deepspeed scripts/zero3.json \
        --stage 3 \
        --dataset_names 'scienceqa-train-val-17k|textvqa-train-35k|allava-instruct-laion-4v-485k|allava-instruct-vflan-4v-203k|arxivqa-100k|q-instruct-198k|llava-finetune-665k|geo-177k|lrv-and-chart-instruction-343k|synthdog-en-ocr-200k|allava-evol-instruct-143k|cc12m-qa-387k' \
        --ovis_pretrained_path $ROOT/checkpoints/ovis/Ovis-Clip-Qwen1.5-14B-S2 \
        --train_modules 'all' \
        --multimodal_max_length 2048 \
        --output_dir $ROOT/checkpoints/ovis/$EXPNAME \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --num_train_epochs 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --weight_decay 0. \
        --warmup_ratio 0.05 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --bf16 True \
        --dataloader_num_workers 8 \
        --gradient_checkpointing True \
        --run_name $EXPNAME \
        --report_to tensorboard
