#!/bin/bash
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
export TRAIN_STEP=1000
export MODEL_PATH='/home/sbjeon/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'

# export MODEL_PATH='THUDM/CogVideoX-2b'
# '/home/sbjeon/workspace/vc/DualReal/CogVideoX-2b'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL 안정성 설정 (단일 GPU DeepSpeed 환경)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=0

export ID_PATH='/home/sbjeon/workspace2/sb/train/test_data/identity/dog'
export REF_PATH='/home/sbjeon/workspace2/sb/train/test_data/identity/dog/images/00.png'
export MOTION_PATH='/home/sbjeon/workspace2/sb/train/test_data/motion/playingGuitar'
BASE_OUTPUT_PATH='/home/sbjeon/workspace2/sb/train/output/train_lora/dog_guitar'
export OUTPUT_PATH="$BASE_OUTPUT_PATH"
if [ -d "$OUTPUT_PATH" ]; then
  i=1
  while [ -d "${BASE_OUTPUT_PATH}_${i}" ]; do
    i=$((i + 1))
  done
  export OUTPUT_PATH="${BASE_OUTPUT_PATH}_${i}"
fi
echo "출력 경로: $OUTPUT_PATH"

accelerate launch --config_file train/config/finetune_adapter_single.yaml --multi_gpu \
  train/train_lora.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir ".cache" \
  --enable_tiling \
  --enable_slicing \
  --caption_column_id prompts.txt \
  --caption_column_motion prompts.txt \
  --video_column_id videos.txt \
  --video_column_motion videos.txt \
  --seed 42 \
  --mixed_precision bf16 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --train_batch_size_other 1 \
  --max_train_steps $TRAIN_STEP \
  --checkpointing_steps 100000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --allow_tf32 \
  --use_8bit_adam \
  --instance_data_root_id $ID_PATH \
  --instance_data_root_motion $MOTION_PATH \
  --output_dir $OUTPUT_PATH \
  --motion_ratio 0.5 \
  --id_rank 32 \
  --id_lora_alpha 32 \
  --motion_rank 64 \
  --motion_lora_alpha 64 \
