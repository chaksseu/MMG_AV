#!/bin/bash

# ========================= 기본값 설정 =========================
VIDEO_CSV_PATH="/home/user/video_dataset.csv"       # 실제 CSV 파일 경로
VIDEO_DIR="/home/user/preprocessed_video"           # 비디오 파일 폴더 경로
OUTPUT_DIR="/home/user/video_LoRA_checkpoint"       # 체크포인트 저장 폴더 경로
WANDB_PROJECT="video_teacher_lora_training"         # WandB 프로젝트 이름
TRAIN_BATCH_SIZE=1
GRAD_ACC_STEPS=1
LR=1e-5
NUM_EPOCHS=16
MIXED_PRECISION="no"                              # ["no", "fp16", "bf16"] 중 선택
NUM_WORKERS=4
SAVE_CHECKPOINT=2                                   # 에폭마다 저장 (예: 2 에폭마다)
VIDEOCRAFTER_CKPT="scripts/evaluation/model.ckpt"   # 미리 학습된 VideoCrafter 모델 ckpt 경로
VIDEOCRAFTER_CONFIG="configs/inference_t2v_512_v2.0.yaml"  # VideoCrafter config 경로
VIDEO_FPS=12.5
TARGET_FRAMES=40

# ========================= 평가 관련 설정 =========================
EVAL_EVERY=1                # N 에폭마다 평가
INFERENCE_BATCH_SIZE=1
INFERENCE_SAVE_PATH="/home/user/video_lora_inference"
GUIDANCE_SCALE=12.0
NUM_INFERENCE_STEPS=25
TARGET_FOLDER="/home/user/video_eval_gt"  # 평가 시 사용될 GT 폴더

SEED=42
SLICE_DURATION=3.2   # 예: 5초짜리 비디오로 가정 (inference에서 사용 예시)

# (필요 시) VGG 관련 설정
VGG_CSV_PATH=""                        # VGG eval 용 CSV 파일
VGG_INFERENCE_SAVE_PATH=""             # VGG eval 시 inference 저장 폴더
VGG_TARGET_FOLDER=""                   # VGG eval 시 GT 폴더 (비워두면 실행 안 됨)

# ========================= 디렉토리 확인 및 생성 =========================
mkdir -p "$OUTPUT_DIR"
mkdir -p "$INFERENCE_SAVE_PATH"

# CSV 파일 체크
if [ ! -f "$VIDEO_CSV_PATH" ]; then
    echo "Error: CSV 파일을 찾을 수 없습니다: $VIDEO_CSV_PATH"
    exit 1
fi

# 비디오 폴더 체크
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: 비디오 폴더를 찾을 수 없습니다: $VIDEO_DIR"
    exit 1
fi

# 평가용 GT 폴더 체크
if [ ! -d "$TARGET_FOLDER" ]; then
    echo "Error: 평가용 GT 폴더를 찾을 수 없습니다: $TARGET_FOLDER"
    exit 1
fi

# (필요 시) VGG 관련 폴더 체크
if [ -n "$VGG_CSV_PATH" ] && [ ! -f "$VGG_CSV_PATH" ]; then
    echo "Warning: VGG CSV 파일을 찾을 수 없습니다: $VGG_CSV_PATH"
fi
if [ -n "$VGG_TARGET_FOLDER" ] && [ ! -d "$VGG_TARGET_FOLDER" ]; then
    echo "Warning: VGG GT 폴더를 찾을 수 없습니다: $VGG_TARGET_FOLDER"
fi

# ========================= 로깅(파라미터 확인) =========================
echo "======================= Training Configuration ======================="
echo "CSV Path: $VIDEO_CSV_PATH"
echo "Video Directory: $VIDEO_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "WandB Project: $WANDB_PROJECT"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Gradient Accumulation Steps: $GRAD_ACC_STEPS"
echo "Evaluate Every (epochs): $EVAL_EVERY"
echo "Mixed Precision: $MIXED_PRECISION"
echo "Number of Workers: $NUM_WORKERS"
echo "Save Checkpoint Every (epochs): $SAVE_CHECKPOINT"
echo "VideoCrafter CKPT: $VIDEOCRAFTER_CKPT"
echo "VideoCrafter Config: $VIDEOCRAFTER_CONFIG"
echo "Video FPS: $VIDEO_FPS"
echo "Target Frames: $TARGET_FRAMES"
echo "Random Seed: $SEED"
echo "Slice Duration (for inference): $SLICE_DURATION"
echo ""
echo "======================= Evaluation Configuration ======================="
echo "Inference Batch Size: $INFERENCE_BATCH_SIZE"
echo "Inference Save Path: $INFERENCE_SAVE_PATH"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "Number of Inference Steps: $NUM_INFERENCE_STEPS"
echo "Target Folder (GT): $TARGET_FOLDER"
echo "========================================================================"

# ========================= 실행 명령어 =========================
# 실제로는 train_video.py 또는 main(args)를 담은 파이썬 파일 이름을 맞춰 주세요.
accelerate launch train_video.py \
    --csv_path "$VIDEO_CSV_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --mixed_precision "$MIXED_PRECISION" \
    --num_workers "$NUM_WORKERS" \
    --save_checkpoint "$SAVE_CHECKPOINT" \
    --videocrafter_ckpt "$VIDEOCRAFTER_CKPT" \
    --videocrafter_config "$VIDEOCRAFTER_CONFIG" \
    --video_fps "$VIDEO_FPS" \
    --target_frames "$TARGET_FRAMES" \
    --inference_batch_size "$INFERENCE_BATCH_SIZE" \
    --inference_save_path "$INFERENCE_SAVE_PATH" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --target_folder "$TARGET_FOLDER" \
    --seed "$SEED" \
    --slice_duration "$SLICE_DURATION" \
    --vgg_csv_path "$VGG_CSV_PATH" \
    --vgg_inference_save_path "$VGG_INFERENCE_SAVE_PATH" \
    --vgg_target_folder "$VGG_TARGET_FOLDER" \

# ========================= 종료 메시지 =========================
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed. Please check the logs for more details."
    exit 1
fi
