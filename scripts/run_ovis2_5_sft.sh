#!/bin/bash

set -e

# Experiment name is taken from script filename
EXPNAME="${0##*/}"
EXPNAME="${EXPNAME%.sh}"

OVIS_CKPT_DIR="AIDC-AI/Ovis2.5-9B"
data_name="geometry3k_local"

CMDARG="--deepspeed scripts/zero_configs/zero1_cp.json \
  --stage 3 \
  --data_info_version ovis2_5_sft_datainfo \
  --data_name ${data_name} \
  --data_type conversation \
  --data_seed 5171 \
  --accepts_loss_kwargs True \
  --ovis_pretrained_path ${OVIS_CKPT_DIR} \
  --attn_implementation flash_attention_2 \
  --single_image_min_pixels 200704 \
  --single_image_max_pixels 3211264 \
  --min_frames 10 \
  --max_frames 10 \
  --train_modules all \
  --multimodal_max_length 6000 \
  --text_max_length 6000 \
  --output_dir path/to/checkpoints/$EXPNAME \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 0.4 \
  --save_total_limit 10 \
  --learning_rate 2e-5 \
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
  --report_to none \
  --run_name $EXPNAME"

echo "Training arguments:"
echo "$CMDARG"

# Run with torchrun
torchrun --nproc_per_node=2 ovis/train/train.py $CMDARG
