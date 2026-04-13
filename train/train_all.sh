#!/bin/bash
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
export TRAIN_STEP=1000
export MODEL_PATH='/home/sbjeon/workspace/vc/DualReal/CogVideoX-5b'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IDENTITY_ROOT='train/test_data/identity'
MOTION_ROOT='train/test_data/motion'
OUTPUT_ROOT='train/output'

IDENTITIES=(
    backpack
    bear_plushie
    book
    car
    cat
    clock
    dog
    instrument
    monster_toy
    pink_plushie
    sloth_plushie
    terracotta_warrior
    tortoise_plushie
    unicorn_toy
    wolf_plushie
)

MOTIONS=(
    bear_walking
    boat_sailing
    bus_traveling
    dog_walking
    mallard_flying
    person_dancing
    person_lifting_barbell
    person_playing_cello
    person_playing_flute
    person_twirling
    person_walking
    train_turning
)

TOTAL=$(( ${#IDENTITIES[@]} * ${#MOTIONS[@]} ))
COUNT=0

echo "======================================"
echo " Total training jobs: $TOTAL"
echo " Identity: ${#IDENTITIES[@]}, Motion: ${#MOTIONS[@]}"
echo "======================================"

for IDENTITY in "${IDENTITIES[@]}"; do
    for MOTION in "${MOTIONS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        OUTPUT_PATH="${OUTPUT_ROOT}/${IDENTITY}_${MOTION}"

        # 이미 완료된 경우 스킵
        if [ -d "$OUTPUT_PATH" ] && [ "$(ls -A $OUTPUT_PATH 2>/dev/null)" ]; then
            echo "[${COUNT}/${TOTAL}] SKIP (already exists): ${IDENTITY} + ${MOTION}"
            continue
        fi

        echo ""
        echo "======================================"
        echo "[${COUNT}/${TOTAL}] Training: ${IDENTITY} + ${MOTION}"
        echo "======================================"

        export ID_PATH="${IDENTITY_ROOT}/${IDENTITY}"
        export REF_PATH="${IDENTITY_ROOT}/${IDENTITY}/images/00.png"
        export MOTION_PATH="${MOTION_ROOT}/${MOTION}"

        accelerate launch --config_file train/config/finetune_adapter_single.yaml --multi_gpu \
          train/train_lora_timestep.py \
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
          --train_batch_size 3 \
          --train_batch_size_other 1 \
          --max_train_steps $TRAIN_STEP \
          --checkpointing_steps 100000 \
          --gradient_accumulation_steps 1 \
          --learning_rate 1e-3 \
          --allow_tf32 \
          --use_8bit_adam \
          --instance_data_root_id $ID_PATH \
          --instance_data_root_motion $MOTION_PATH \
          --output_dir $OUTPUT_PATH

        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed: ${IDENTITY} + ${MOTION}" >&2
        else
            echo "[DONE] ${IDENTITY} + ${MOTION} → ${OUTPUT_PATH}"
        fi

    done
done

echo ""
echo "======================================"
echo " All $TOTAL jobs completed."
echo "======================================"
