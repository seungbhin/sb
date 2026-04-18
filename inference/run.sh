#!/bin/bash

set -e  # 에러나면 바로 종료

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LORA_PATH="/home/sbjeon/workspace2/sb/train/output/dog_guitar_timestep_7"
export MODEL_PATH="/home/sbjeon/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01"

PROMPT="A dog wearing a white shirt is playing the guitar on the beach"

ID_SCALE=2.0
MOTION_SCALE=0.0
SHARED_SCALE=2.0

RESULTS_DIR="/home/sbjeon/workspace2/sb/inference/results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

cat > "${RESULTS_DIR}/args.json" << EOF
{
    "lora_path": "${LORA_PATH}",
    "prompt": "${PROMPT}",
    "id_scale": ${ID_SCALE},
    "motion_scale": ${MOTION_SCALE},
    "shared_scale": ${SHARED_SCALE},
    "timestamp": "${TIMESTAMP}"
}

EOF

python lora_inference.py \
  --model_path "${MODEL_PATH}" \
  --lora_path "${LORA_PATH}" \
  --prompt "${PROMPT}" \
  --output_path "${RESULTS_DIR}/video.mp4" \
  --id_scale ${ID_SCALE} \
  --motion_scale ${MOTION_SCALE} \
  --shared_scale ${SHARED_SCALE}