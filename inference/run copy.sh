#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LORA_PATH="/home/sbjeon/workspace/sb/train/output/playingGuitar"
PROMPT="A dog is playing Guitar"
# "A dog playing a flute, wearing a light linen shirt and a straw hat, gentle ocean waves rolling in the background"
# "a dog is playing flute on the beach"
ID_SCALE=1.0
MOTION_SCALE=1.0
SHARED_SCALE=1.0

RESULTS_DIR="/home/sbjeon/workspace/sb/inference/results/${TIMESTAMP}"
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
  --model_path "/home/sbjeon/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01" \
  --lora_path "${LORA_PATH}" \
  --prompt "${PROMPT}" \
  --output_path "${RESULTS_DIR}/video.mp4" \
  --id_scale ${ID_SCALE} --motion_scale ${MOTION_SCALE} --shared_scale ${SHARED_SCALE}