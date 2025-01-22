#!/bin/bash

# 기본값 설정
csv_path="/workspace/data/MMG_TA_dataset_audiocaps_wavcaps/MMG_TA_dataset_preprocessed.csv"  # 실제 CSV 파일 경로
audio_dir="/workspace/data/MMG_TA_dataset_audiocaps_wavcaps/preprocessed_audio"  # 실제 오디오 파일 폴더 경로
OUTPUT_DIR="audio_teacher_LoRA_checkpoint" # checkpoint 저장 폴더 경로
WANDB_PROJECT="audio_teacher_lora_training"
TRAIN_BATCH_SIZE=1
GRAD_ACC_STEPS=1
LR=1e-5
NUM_EPOCHS=10
MIXED_PRECISION="bf16"
PRETRAINED_MODEL="auffusion/auffusion-full"
NUM_WORKERS=4
SAVE_CHECKPOINT=1

# Evaluation 관련
EVAL_EVERY=1  # N 에폭마다 평가
INFERENCE_BATCH_SIZE=1
INFERENCE_SAVE_PATH="audio_lora_training_inference_data" # inference 저장 경로
ETA_AUDIO=0.0
GUIDANCE_SCALE=7.5
NUM_INFERENCE_STEPS=50
TARGET_FOLDER="/workspace/data/MMG_TA_dataset_audiocaps_wavcaps/preprocessed_audio_random_3_2s" # 비교한 gt test 데이터

# 기타 dataset 파라미터
SAMPLE_RATE=16000
SLICE_DURATION=3.2  # 초 단위
HOP_SIZE=160
N_MELS=256
SEED=42

# 디렉토리 확인 및 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$INFERENCE_SAVE_PATH"

if [ ! -f "$csv_path" ]; then
    echo "Error: CSV 파일을 찾을 수 없습니다 at $csv_path"
    exit 1
fi

if [ ! -d "$audio_dir" ]; then
    echo "Error: audio_dir 디렉토리를 찾을 수 없습니다 at $audio_dir"
    exit 1
fi

if [ ! -d "$TARGET_FOLDER" ]; then
    echo "Error: target_folder 디렉토리를 찾을 수 없습니다 at $TARGET_FOLDER"
    exit 1
fi

# 환경 변수 체크
if [ -z "$WANDB_PROJECT" ]; then
    echo "Warning: WANDB_PROJECT가 설정되지 않았습니다. 기본값을 사용합니다: $WANDB_PROJECT"
fi

# 로깅
echo "======================= Training Configuration ======================="
echo "CSV Path: $csv_path"
echo "Audio Directory: $audio_dir"
echo "Output Directory: $OUTPUT_DIR"
echo "WandB Project: $WANDB_PROJECT"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Gradient Accumulation Steps: $GRAD_ACC_STEPS"
echo "Evaluate Every (epochs): $EVAL_EVERY"
echo "Mixed Precision: $MIXED_PRECISION"
echo "Pretrained Model: $PRETRAINED_MODEL"
echo "Number of Workers: $NUM_WORKERS"
echo "Save Checkpoint Every: $SAVE_CHECKPOINT steps"
echo "Sample Rate: $SAMPLE_RATE"
echo "Slice Duration: $SLICE_DURATION seconds"
echo "Hop Size: $HOP_SIZE"
echo "Number of Mel Bands: $N_MELS"
echo "Random Seed: $SEED"
echo ""
echo "======================= Evaluation Configuration ======================="
echo "Inference Batch Size: $INFERENCE_BATCH_SIZE"
echo "Inference Save Path: $INFERENCE_SAVE_PATH"
echo "ETA Audio: $ETA_AUDIO"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "Number of Inference Steps: $NUM_INFERENCE_STEPS"
echo "Target Folder: $TARGET_FOLDER"
echo "=========================================================================="

# train.py 실행
accelerate launch audio_lora_training/train.py \
    --csv_path "$csv_path" \
    --audio_dir "$audio_dir" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --mixed_precision "$MIXED_PRECISION" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --num_workers "$NUM_WORKERS" \
    --save_checkpoint "$SAVE_CHECKPOINT" \
    --inference_batch_size "$INFERENCE_BATCH_SIZE" \
    --inference_save_path "$INFERENCE_SAVE_PATH" \
    --eta_audio "$ETA_AUDIO" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --target_folder "$TARGET_FOLDER" \
    --sample_rate "$SAMPLE_RATE" \
    --slice_duration "$SLICE_DURATION" \
    --hop_size "$HOP_SIZE" \
    --n_mels "$N_MELS" \
    --seed "$SEED"

# 종료 메시지
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed. Please check the logs for more details."
    exit 1
fi
