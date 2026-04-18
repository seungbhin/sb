#!/bin/bash
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

export LORA_DIR='train/output/train_lora/dog_guitar'
export PROMPT='A sks dog is playing guitar on stage under spotlights'

python inference/lora_inference.py \
  --lora_dir     $LORA_DIR \
  --prompt       "$PROMPT" \
  --results_dir  inference/results \
  --id_scale     1 \
  --motion_scale 0.2 \
  --seed 42