#!/bin/bash

ROOT=/playground
EXPNAME="${0##*/}"; EXPNAME="${EXPNAME%.sh}"

DATAS="
A-OKVQA-18k|\
CLEVR-MATH-85k|\
FigureQA-100k|\
Geometry-2k|\
IAM-7k|\
InfographicVQA-24k|\
MathQA-395k|\
MathV-360k|\
OpenCQA-5k|\
PlotQA-157k|\
Super-CLEVR-30k|\
Symbolic-Reasoning-TabMW-31k|\
ViQuAE-2k|\
ai2d-mc-15k|\
c7s-ai2d-15k|\
c7s-alfworldgpt-45k|\
c7s-allava-laion-500k|\
c7s-allava-vflan-200k|\
c7s-arxivqa-100k|\
c7s-chartqa-28k|\
c7s-clean-llava-instruct-150k-llavar-20k|\
c7s-clevr-700k|\
c7s-code-feedback-66k|\
c7s-docvqa-39k|\
c7s-filtered-data-engine-161k|\
c7s-geo170k|\
c7s-gpt77k|\
c7s-idefics375k|\
c7s-idk-11k|\
c7s-laion-gpt4v-11k|\
c7s-lnqa-302k|\
c7s-lvis-instruct4v-220k|\
c7s-mathinstruct-262k|\
c7s-oodvqa-8k|\
c7s-orca-math-200k|\
c7s-pathvqa-32k|\
c7s-q-instruct-200k|\
c7s-qalign-200k|\
c7s-random-3rd-dvqa-2325k|\
c7s-scienceqa-12k|\
c7s-screenqa-79k|\
c7s-sharegpt4v-mix665k-cap23k-coco-ap9k-lcs3k-sam9k-div2k|\
c7s-sketchyvqa-8k|\
c7s-synthdog-500k-modified|\
c7s-tallyqa-250k|\
c7s-vizwiz-20k|\
c7s-wizardlm-143k|\
cc12m-qa-387k|\
doclaynet-65k|\
doclie-real-100k|\
dtvqa-27k|\
funsd-1k|\
hme-74k|\
hwl-eng-10k|\
icqa-train-val-40k|\
infovqa-multi-conversation-5k|\
kvqa-25k|\
lrv-instruct-and-chart-343k|\
mmc-base-410k|\
mmmath-6k|\
ocr-vqa-multi-conversation-207k|\
okvqa-14k|\
orandCAR-5k|\
poie-9k|\
sroie-3k|\
stvqa-78k|\
tqa-train-34k|\
tqa-train-val-20k|\
visualdialog-125k|\
vqa-v2-multi-conversation-184k|\
vsr-train-dev-12k
"

deepspeed ovis/train/train.py \
        --deepspeed scripts/zero3.json \
        --stage 3 \
        --dataset_names ${DATAS} \
        --ovis_pretrained_path $ROOT/checkpoints/ovis/Ovis1.5-Llama3-8B-S2 \
        --train_modules 'all' \
        --multimodal_max_length 1755 \
        --text_max_length 1024 \
        --output_dir $ROOT/checkpoints/ovis/$EXPNAME \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 32 \
        --num_train_epochs 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 0.2 \
        --save_total_limit 1 \
        --learning_rate 1e-5 \
        --max_grad_norm 1.0 \
        --weight_decay 0. \
        --warmup_ratio 0.05 \
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
