#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home/sbjeon/.conda/envs/smraboothwan/bin/python

MODEL_PATH='/home/sbjeon/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'
LORA_DIR='train/output/train_lora/head_wise_dog_skateboarding'
PROMPT="A sks dog wearing a knitted sweater is vvt skateboarding through an open-air market, dodging fruit stands and weaving between busy shoppers."
# 'A sks dog is playing guitar on stage under spotlights'

$PYTHON inference/inference_lora_head.py \
    --model_path    "$MODEL_PATH" \
    --lora_dir      "$LORA_DIR" \
    --prompt        "$PROMPT" \
    --results_dir   inference/results \
    --id_scale      1.0 \
    --motion_scale  1.0 \
    --guidance_scale        6.0 \
    --num_inference_steps   50 \
    --num_frames            49 \
    --height                480 \
    --width                 720 \
    --fps                   8 \
    --seed                  42
