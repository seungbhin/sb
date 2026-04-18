#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home/sbjeon/.conda/envs/smraboothwan/bin/python
MODEL_PATH='/home/sbjeon/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'

$PYTHON analysis/analyze_heads.py \
    --model_path $MODEL_PATH \
    --prompt "A dog playing guitar outdoors" \
    --num_frames 9 \
    --num_inference_steps 10 \
    --output_dir analysis/results
