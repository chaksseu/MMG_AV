#!/bin/bash

# 기본값 설정
csv_path="/home/work/kby_hgh/again_mmg_TA_dataset_zip_0326/MMG_TA_dataset_filtered_0321.csv"  # 실제 CSV 파일 경로
audio_dir="/home/work/kby_hgh/workspace/data/MMG_TA_dataset_audiocaps_wavcaps_spec_0320"  # spec path
WANDB_PROJECT="audio_teacher_full_training_0416"
TRAIN_BATCH_SIZE=32
GRAD_ACC_STEPS=1
LR=1e-6 ## 1e-6
NUM_EPOCHS=128
MIXED_PRECISION="no"
PRETRAINED_MODEL="auffusion/auffusion-full"
NUM_WORKERS=8

RESUME_CHECKPOINT=""

DATE="0709"


# Evaluation 관련
EVAL_EVERY=1460 # 730  # N step
INFERENCE_BATCH_SIZE=16

OUTPUT_DIR="/home/work/kby_hgh/AUDIO_FULL_CHECKPOINT_${DATE}/${LR}" # checkpoint 저장 폴더 경로
INFERENCE_SAVE_PATH="/home/work/kby_hgh/MMG_Inferencce_folder/audio_full_inference_${DATE}_${LR}" # inference 저장 경로

ETA_AUDIO=0.0
GUIDANCE_SCALE=7.5
NUM_INFERENCE_STEPS=25
TARGET_FOLDER="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320" # 비교한 gt test 데이터

VGG_CSV_PATH="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/vggsound_sparse_curated_292.csv"
VGG_TARGET_FOLDER="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/audio"
VGG_INFERENCE_PATH="/home/work/kby_hgh/audio_full_vggsound_sparse_inference_${DATE}_${LR}"

AC_CSV_PATH="/home/work/kby_hgh/MMG_AC_test_dataset/0407_one_cap_AC_test.csv"
AC_TARGET_FOLDER="/home/work/kby_hgh/MMG_AC_test_dataset/0408_AC_test_trimmed_wavs"
AC_INFERENCE_PATH="/home/work/kby_hgh/audio_full_audiocaps_inference_${DATE}_${LR}"

#TARGET_FOLDER="/home/jupyter/MMG_01/"
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
echo "VGG_CSV_PATH: $VGG_CSV_PATH"
echo "VGG_TARGET_FOLDER: $VGG_TARGET_FOLDER"
echo "VGG_INFERENCE_PATH: $VGG_INFERENCE_PATH"
echo "AC_CSV_PATH: $AC_CSV_PATH"
echo "AC_TARGET_FOLDER: $AC_TARGET_FOLDER"
echo "AC_INFERENCE_PATH: $AC_INFERENCE_PATH"


echo "=========================================================================="

# train.py 실행
accelerate launch audio_lora_training/train_full_0603.py \
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
    --seed "$SEED" \
    --vgg_csv_path "$VGG_CSV_PATH" \
    --vgg_target_folder "$VGG_TARGET_FOLDER" \
    --vgg_inference_path "$VGG_INFERENCE_PATH" \
    --ac_csv_path "$AC_CSV_PATH" \
    --ac_target_folder "$AC_TARGET_FOLDER" \
    --ac_inference_path "$AC_INFERENCE_PATH" \
    --resume_checkpoint "$RESUME_CHECKPOINT"

# 종료 메시지
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed. Please check the logs for more details."
    exit 1
fi
