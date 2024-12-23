#!/bin/bash

# 환경 변수 설정
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

# Python 스크립트 파일명

PYTHON_SCRIPT="mmg_training/train_MMG_Model_1223_MMG.py"

# 학습에 필요한 인자 설정
PROCESSED_DATA_DIR="latents_data_32s_40frames_vggsound_sparse_new_normalization"
#CSV_FILE="path/to/your/dataset_info.csv"  # 실제 CSV 파일 경로로 변경
AUDIO_MODEL_NAME="auffusion/auffusion-full"
VIDEOCRAFTER_CKPT="scripts/evaluation/model.ckpt"
VIDEOCRAFTER_CONFIG="configs/inference_t2v_512_v2.0.yaml"
NUM_EPOCHS=1000
NUM_GPUS=1
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=1e-5
VIDEO_FPS=12.5
DATE="1223_MMG"
AUDIO_LOSS_WEIGHT=1.0
VIDEO_LOSS_WEIGHT=4.0
DATASET_NAME="vggsound_sparse"
NUM_WORKERS=0
WANDB_PROJECT="MMG_auffusion_videocrafter"
CHECKPOINT_DIR="MMG_CHECKPOINTS_1223"

# accelerate launch 명령어 실행
accelerate launch $PYTHON_SCRIPT \
    --processed_data_dir "$PROCESSED_DATA_DIR" \
    --csv_file "$CSV_FILE" \
    --audio_model_name "$AUDIO_MODEL_NAME" \
    --videocrafter_ckpt "$VIDEOCRAFTER_CKPT" \
    --videocrafter_config "$VIDEOCRAFTER_CONFIG" \
    --num_epochs "$NUM_EPOCHS" \
    --num_gpus "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --video_fps "$VIDEO_FPS" \
    --date "$DATE" \
    --audio_loss_weight "$AUDIO_LOSS_WEIGHT" \
    --video_loss_weight "$VIDEO_LOSS_WEIGHT" \
    --dataset_name "$DATASET_NAME" \
    --num_workers "$NUM_WORKERS" \
    --wandb_project "$WANDB_PROJECT" \
    --checkpoint_dir "$CHECKPOINT_DIR"
