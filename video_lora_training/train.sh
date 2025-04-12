#!/bin/bash

# ========================= 기본값 설정 =========================
VIDEO_CSV_PATH="/home/work/kby_hgh/0411_processed_Openvid_train.csv"               # 실제 CSV 파일 경로
VIDEO_DIR="/home/work/kby_hgh/processed_OpenVid_1M_videos"                   # 비디오 파일 폴더 경로

WANDB_PROJECT="video_teacher_lora_training_0213"                 # WandB 프로젝트 이름
TRAIN_BATCH_SIZE=2 # 1
GRAD_ACC_STEPS=64 # 128
LR=1e-4
NUM_EPOCHS=16
MIXED_PRECISION="bf16"                                        # ["no", "fp16", "bf16"] 중 선택
NUM_WORKERS=4
VIDEOCRAFTER_CKPT="scripts/evaluation/model.ckpt"           # 미리 학습된 VideoCrafter 모델 ckpt 경로
VIDEOCRAFTER_CONFIG="configs/inference_t2v_512_v2.0.yaml"   # VideoCrafter config 경로
VIDEO_FPS=12.5
TARGET_FRAMES=40
HEIGHT=320
WIDTH=512
VIDEO_LOSS_WEIGHT=4.0

# RESUME_CHECKPOINT="/home/jupyter/video_lora_training_checkpoints_0211/checkpoint-step-12288"

# ========================= 평가 관련 설정 =========================
EVAL_EVERY=4096                # N step마다 평가
INFERENCE_BATCH_SIZE=2
INFERENCE_SAVE_PATH="/home/jupyter/video_lora_inference_0213"
GUIDANCE_SCALE=12.0
NUM_INFERENCE_STEPS=25
TARGET_FOLDER="/home/jupyter/preprocessed_WebVid_10M_gt_test_videos_1k_random_crop_0210"  # 평가 시 사용될 GT 폴더
SEED=42
DDIM_ETA=0.0

OUTPUT_DIR="/home/work/kby_hgh/VIDEO_LORA_CHECKPOINT_0410/${LR}" # checkpoint 저장 폴더 경로

# (필요 시) VGG 관련 설정
VGG_CSV_PATH="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/vggsound_sparse_curated_292.csv"                        # VGG eval 용 CSV 파일
VGG_INFERENCE_SAVE_PATH="/home/work/kby_hgh/video_lora_vggsound_sparse_inference_0410_${LR}"             # VGG eval 시 inference 저장 폴더
VGG_TARGET_FOLDER="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/video"                   # VGG eval 시 GT 폴더 (비워두면 실행 안 됨)



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
# if [ ! -d "$TARGET_FOLDER" ]; then
#     echo "Error: 평가용 GT 폴더를 찾을 수 없습니다: $TARGET_FOLDER"
#     exit 1
# fi

# # (필요 시) VGG 관련 폴더 체크
# if [ -n "$VGG_CSV_PATH" ] && [ ! -f "$VGG_CSV_PATH" ]; then
#     echo "Warning: VGG CSV 파일을 찾을 수 없습니다: $VGG_CSV_PATH"
# fi
# if [ -n "$VGG_TARGET_FOLDER" ] && [ ! -d "$VGG_TARGET_FOLDER" ]; then
#     echo "Warning: VGG GT 폴더를 찾을 수 없습니다: $VGG_TARGET_FOLDER"
# fi

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
echo "Evaluate Every (steps): $EVAL_EVERY"
echo "Mixed Precision: $MIXED_PRECISION"
echo "Number of Workers: $NUM_WORKERS"
echo "VideoCrafter CKPT: $VIDEOCRAFTER_CKPT"
echo "VideoCrafter Config: $VIDEOCRAFTER_CONFIG"
echo "Video FPS: $VIDEO_FPS"
echo "Target Frames: $TARGET_FRAMES"
echo "Random Seed: $SEED"
echo "VIDEO_LOSS_WEIGHT: $VIDEO_LOSS_WEIGHT"
# echo "RESUME_CHECKPOINT: $RESUME_CHECKPOINT"
echo ""
echo "======================= Additional Arguments =========================="
echo "Height: $HEIGHT"
echo "Width: $WIDTH"
echo "DDIM Eta: $DDIM_ETA"
echo ""
echo "======================= Evaluation Configuration ======================"
echo "Inference Batch Size: $INFERENCE_BATCH_SIZE"
echo "Inference Save Path: $INFERENCE_SAVE_PATH"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "Number of Inference Steps: $NUM_INFERENCE_STEPS"
echo "Target Folder (GT): $TARGET_FOLDER"
echo "========================================================================"

# ========================= 실행 명령어 =========================
# 아래에서 "train.py"는 실제 질문에 첨부된 Python 스크립트 파일 이름에 맞춰 사용합니다.
accelerate launch video_lora_training/train.py \
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
    --videocrafter_ckpt "$VIDEOCRAFTER_CKPT" \
    --videocrafter_config "$VIDEOCRAFTER_CONFIG" \
    --video_fps "$VIDEO_FPS" \
    --target_frames "$TARGET_FRAMES" \
    --inference_batch_size "$INFERENCE_BATCH_SIZE" \
    --inference_save_path "$INFERENCE_SAVE_PATH" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --target_folder "$TARGET_FOLDER" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --ddim_eta "$DDIM_ETA" \
    --seed "$SEED" \
    --vgg_csv_path "$VGG_CSV_PATH" \
    --vgg_inference_save_path "$VGG_INFERENCE_SAVE_PATH" \
    --vgg_target_folder "$VGG_TARGET_FOLDER" \
    --video_loss_weight "$VIDEO_LOSS_WEIGHT"
    # --resume_checkpoint "$RESUME_CHECKPOINT"

# ========================= 종료 메시지 =========================
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed. Please check the logs for more details."
    exit 1
fi
