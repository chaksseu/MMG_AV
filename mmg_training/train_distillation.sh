#!/bin/bash

# ========================= 기본값 설정 =========================
DATE="0322"
LEARNING_RATE=1e-4
NUM_EPOCHS=100
NUM_GPU=1
TRAIN_BATCH_SIZE=2 # 2
GRADIENT_ACCUMULATION=1 # 256
INFERENCE_BATCH_SIZE=4
EVAL_EVERY=16000

NUM_INFERENCE_STEPS=25
DTYPE="bf16"

SEED=42
DURATION=3.2
VIDEOCRAFTER_CONFIG="configs/inference_t2v_512_v2.0.yaml"
VIDEOCRAFTER_CKPT_PATH="scripts/evaluation/model.ckpt"
AUDIO_MODEL_NAME="auffusion/auffusion-full"
HEIGHT=320
WIDTH=512
FRAMES=40
FPS=12.5
AUDIO_LOSS_WEIGHT=1.0
VIDEO_LOSS_WEIGHT=1.0
TA_AUDIO_LOSS_WEIGHT=1.0
TV_VIDEO_LOSS_WEIGHT=1.0

CSV_PATH="/workspace/vggsound_processing/New_VGGSound_0311.csv" #"/workspace/processed_vggsound_sparse_0218/processed_vggsound_sparse_mmg.csv"
SPECTROGRAM_DIR="/workspace/data/preprocessed_VGGSound_train_spec_0310" #"/workspace/processed_vggsound_sparse_0218/spec"
VIDEO_DIR="/workspace/data/preprocessed_VGGSound_train_videos_0313" #"/workspace/processed_vggsound_sparse_0218/video"

TA_CSV_PATH="/workspace/data/MMG_TA_dataset_audiocaps_wavcaps/MMG_TA_dataset_filtered_0321.csv" #"/workspace/processed_vggsound_sparse_0218/processed_vggsound_sparse_mmg.csv"
TA_SPECTROGRAM_DIR="/workspace/data/MMG_TA_dataset_audiocaps_wavcaps_spec_0320" #"/workspace/processed_vggsound_sparse_0218/spec"
TV_CSV_PATH="/workspace/processed_OpenVid_0321.csv"   #"/workspace/processed_vggsound_sparse_0218/processed_vggsound_sparse_mmg.csv"
TV_VIDEO_DIR="/workspace/data/preprocessed_openvid_videos_train_0318" #"/workspace/processed_vggsound_sparse_0218/video"


SAMPLING_RATE=16000
HOP_SIZE=160
NUM_WORKERS=4

CROSS_MODAL_CHECKPOINT_PATH="/workspace/MMG_CHECKPOINT/checkpint_0313/checkpoint-step-79999"
VIDEO_LORA_CKPT_PATH="/workspace/video_lora_training_checkpoints_0213/checkpoint-step-16384/model.safetensors"
AUDIO_LORA_CKPT_PATH="/workspace/GCP_BACKUP_0213/checkpoint-step-6400/model.safetensors"
INFERENCE_SAVE_PATH="/workspace/MMG_Inferencce_folder"
CKPT_SAVE_PATH="/workspace/MMG_CHECKPOINT"
AUDIO_DDIM_ETA=0.0
VIDEO_DDIM_ETA=0.0
AUDIO_GUIDANCE_SCALE=7.5
VIDEO_UNCONDITIONAL_GUIDANCE_SCALE=12.0
VGG_CSV_PATH="/workspace/vggsound_sparse_curated_292.csv"
VGG_GT_TEST_PATH="/workspace/vggsound_sparse_test_curated_final"
AVSYNC_CSV_PATH="/workspace/processed_vggsound_sparse_0218/avsync_test"
AVSYNC_GT_TEST_PATH="/workspace/processed_vggsound_sparse_0218/avsync_gt_test.csv"

# ========================= 디렉토리 확인 및 생성 =========================
# CSV 파일 체크
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV 파일을 찾을 수 없습니다: $CSV_PATH"
    exit 1
fi

# spectrogram 폴더 체크
if [ ! -d "$SPECTROGRAM_DIR" ]; then
    echo "Error: Spectrogram 폴더를 찾을 수 없습니다: $SPECTROGRAM_DIR"
    exit 1
fi

# 비디오 폴더 체크
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: 비디오 폴더를 찾을 수 없습니다: $VIDEO_DIR"
    exit 1
fi

# 체크포인트 및 Inference 저장 폴더 생성
mkdir -p "$CKPT_SAVE_PATH"
mkdir -p "$INFERENCE_SAVE_PATH"

# ========================= 로깅(파라미터 확인) =========================
echo "======================= Training Configuration ======================="
echo "Seed: $SEED"
echo "Duration: $DURATION"
echo "VideoCrafter Config: $VIDEOCRAFTER_CONFIG"
echo "VideoCrafter Checkpoint: $VIDEOCRAFTER_CKPT_PATH"
echo "Audio Model Name: $AUDIO_MODEL_NAME"
echo "Height: $HEIGHT"
echo "Width: $WIDTH"
echo "Frames: $FRAMES"
echo "FPS: $FPS"
echo "Audio Loss Weight: $AUDIO_LOSS_WEIGHT"
echo "Video Loss Weight: $VIDEO_LOSS_WEIGHT"
echo "TA_Audio Loss Weight: $TA_AUDIO_LOSS_WEIGHT"
echo "TV_Video Loss Weight: $TV_VIDEO_LOSS_WEIGHT"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Learning Rate: $LEARNING_RATE"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "CSV Path: $CSV_PATH"
echo "Spectrogram Dir: $SPECTROGRAM_DIR"
echo "Video Dir: $VIDEO_DIR"
echo "TA_CSV Path: $TA_CSV_PATH"
echo "TA_Spectrogram Dir: $TA_SPECTROGRAM_DIR"
echo "TV_CSV Path: $TV_CSV_PATH"
echo "TV_Video Dir: $TV_VIDEO_DIR"
echo "Sampling Rate: $SAMPLING_RATE"
echo "Hop Size: $HOP_SIZE"
echo "Number of Workers: $NUM_WORKERS"
echo "Date: $DATE"
echo "Number of GPU: $NUM_GPU"
echo "Data Type: $DTYPE"
echo "Cross Modal Checkpoint Path: $CROSS_MODAL_CHECKPOINT_PATH"
echo "Video LORA Checkpoint Path: $VIDEO_LORA_CKPT_PATH"
echo "Audio LORA Checkpoint Path: $AUDIO_LORA_CKPT_PATH"
echo "Inference Save Path: $INFERENCE_SAVE_PATH"
echo "Checkpoint Save Path: $CKPT_SAVE_PATH"
echo "Inference Batch Size: $INFERENCE_BATCH_SIZE"
echo "Audio DDIM Eta: $AUDIO_DDIM_ETA"
echo "Video DDIM Eta: $VIDEO_DDIM_ETA"
echo "Number of Inference Steps: $NUM_INFERENCE_STEPS"
echo "Audio Guidance Scale: $AUDIO_GUIDANCE_SCALE"
echo "Video Unconditional Guidance Scale: $VIDEO_UNCONDITIONAL_GUIDANCE_SCALE"
echo "Eval Every: $EVAL_EVERY"
echo "VGG CSV Path: $VGG_CSV_PATH"
echo "VGG GT Test Path: $VGG_GT_TEST_PATH"
echo "AVSync CSV Path: $AVSYNC_CSV_PATH"
echo "AVSync GT Test Path: $AVSYNC_GT_TEST_PATH"
echo "======================================================================="

# ========================= 실행 명령어 =========================
accelerate launch mmg_training/train_MMG_Model_0322_MMG_LoRA_distillation.py \
    --seed "$SEED" \
    --duration "$DURATION" \
    --videocrafter_config "$VIDEOCRAFTER_CONFIG" \
    --videocrafter_ckpt_path "$VIDEOCRAFTER_CKPT_PATH" \
    --audio_model_name "$AUDIO_MODEL_NAME" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --frames "$FRAMES" \
    --fps "$FPS" \
    --audio_loss_weight "$AUDIO_LOSS_WEIGHT" \
    --video_loss_weight "$VIDEO_LOSS_WEIGHT" \
    --ta_audio_loss_weight "$TA_AUDIO_LOSS_WEIGHT" \
    --tv_video_loss_weight "$TV_VIDEO_LOSS_WEIGHT" \
    --gradient_accumulation "$GRADIENT_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --csv_path "$CSV_PATH" \
    --spectrogram_dir "$SPECTROGRAM_DIR" \
    --video_dir "$VIDEO_DIR" \
    --ta_csv_path "$TA_CSV_PATH" \
    --ta_spectrogram_dir "$TA_SPECTROGRAM_DIR" \
    --tv_csv_path "$TV_CSV_PATH" \
    --tv_video_dir "$TV_VIDEO_DIR" \
    --sampling_rate "$SAMPLING_RATE" \
    --hop_size "$HOP_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --date "$DATE" \
    --num_gpu "$NUM_GPU" \
    --dtype "$DTYPE" \
    --video_lora_ckpt_path "$VIDEO_LORA_CKPT_PATH" \
    --audio_lora_ckpt_path "$AUDIO_LORA_CKPT_PATH" \
    --inference_save_path "$INFERENCE_SAVE_PATH" \
    --ckpt_save_path "$CKPT_SAVE_PATH" \
    --inference_batch_size "$INFERENCE_BATCH_SIZE" \
    --audio_ddim_eta "$AUDIO_DDIM_ETA" \
    --video_ddim_eta "$VIDEO_DDIM_ETA" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --audio_guidance_scale "$AUDIO_GUIDANCE_SCALE" \
    --video_unconditional_guidance_scale "$VIDEO_UNCONDITIONAL_GUIDANCE_SCALE" \
    --eval_every "$EVAL_EVERY" \
    --vgg_csv_path "$VGG_CSV_PATH" \
    --vgg_gt_test_path "$VGG_GT_TEST_PATH" \
    --avsync_csv_path "$AVSYNC_CSV_PATH" \
    --avsync_gt_test_path "$AVSYNC_GT_TEST_PATH" \
    --cross_modal_checkpoint_path "$CROSS_MODAL_CHECKPOINT_PATH"

# ========================= 종료 메시지 =========================
if [ $? -eq 0 ]; then
    echo "실행이 성공적으로 완료되었습니다."
else
    echo "실행 중 오류가 발생했습니다. 로그를 확인하세요."
    exit 1
fi
