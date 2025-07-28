#!/bin/bash
#!/bin/bash

# ========================= 기본값 설정 =========================
DATE="0503"
LOG_NAME="MMG_OURS"


LEARNING_RATE=1e-4
NUM_EPOCHS=32
NUM_GPU=8
TRAIN_BATCH_SIZE=1 # 2
GRADIENT_ACCUMULATION=32 # 32 # 256
INFERENCE_BATCH_SIZE=1

EVAL_EVERY=633 # 633


TENSORBOARD_LOG_DIR="tensorboard/${DATE}_${LEARNING_RATE}_${LOG_NAME}"
INFER_NAME="NEW_MMG"

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
CSV_PATH="/home/work/kby_hgh/MMG_01/vggsound_processing/0403_combined_split_csvs/llm_combined_vgg_csv_0404.csv" #"/workspace/processed_vggsound_sparse_0218/processed_vggsound_sparse_mmg.csv"
SPECTROGRAM_DIR="/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_dataset_0318/preprocessed_VGGSound_train_spec_0310" #"/workspace/processed_vggsound_sparse_0218/spec"
VIDEO_DIR="/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_no_crop_videos_0329" #"/home/work/kby_hgh/workspace/data/preprocessed_VGGSound_train_no_crop_videos_0329" #"/workspace/processed_vggsound_sparse_0218/video"
SAMPLING_RATE=16000
HOP_SIZE=160
NUM_WORKERS=8

CROSS_MODAL_CHECKPOINT_PATH="/home/work/kby_hgh/MMG_CHECKPOINT/checkpint_tensorboard/0503_1e-4_MMG_OURS/checkpoint-step-20255"

VIDEO_LORA_CKPT_PATH="/home/work/kby_hgh/VIDEO_LORA_CHECKPOINT_0413/1e-5/checkpoint-step-40960/model.safetensors"
AUDIO_LORA_CKPT_PATH="/home/work/kby_hgh/AUDIO_LORA_CHECKPOINT_0416/1e-6/checkpoint-step-43800/model.safetensors"

INFERENCE_SAVE_PATH="/home/work/kby_hgh/"
CKPT_SAVE_PATH="/home/work/kby_hgh/MMG_CHECKPOINT"

AUDIO_DDIM_ETA=0.0
VIDEO_DDIM_ETA=0.0
AUDIO_GUIDANCE_SCALE=7.5
VIDEO_UNCONDITIONAL_GUIDANCE_SCALE=12.0

VGG_CSV_PATH="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320/vggsound_sparse_curated_292.csv"
VGG_GT_TEST_PATH="/home/work/kby_hgh/vggsound_sparse_test_curated_final_0320"

VBENCH_CSV_PATH="/home/work/kby_hgh/vbench_all_captions.csv"                 
VBENCH_GT_TEST_PATH="/home/work/kby_hgh/video_lora_vbench_inference_0413_1e-5/step_0"

AC_CSV_PATH="/home/work/kby_hgh/MMG_AC_test_dataset/0407_one_cap_AC_test.csv"
AC_GT_TEST_PATH="/home/work/kby_hgh/MMG_AC_test_dataset/0408_AC_test_trimmed_wavs"

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
echo "TENSORBOARD_LOG_DIR: $TENSORBOARD_LOG_DIR" 
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
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Learning Rate: $LEARNING_RATE"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "CSV Path: $CSV_PATH"
echo "Spectrogram Dir: $SPECTROGRAM_DIR"
echo "Video Dir: $VIDEO_DIR"
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
echo "VBENCH_CSV_PATH: $VBENCH_CSV_PATH"
echo "VBENCH_GT_TEST_PATH: $VBENCH_GT_TEST_PATH"
echo "AC_CSV_PATH: $AC_CSV_PATH"
echo "AC_GT_TEST_PATH: $AC_GT_TEST_PATH"
echo "INFER_NAME: $INFER_NAME"
echo "======================================================================="

# ========================= 실행 명령어 =========================
# 아래에서 "train.py"는 실제 실행할 Python 스크립트 파일 이름입니다.
accelerate launch mmg_training/train_MMG_Model_0503_MMG_LoRA.py \
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
    --gradient_accumulation "$GRADIENT_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --csv_path "$CSV_PATH" \
    --spectrogram_dir "$SPECTROGRAM_DIR" \
    --video_dir "$VIDEO_DIR" \
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
    --ac_csv_path "$AC_CSV_PATH" \
    --ac_gt_test_path "$AC_GT_TEST_PATH" \
    --vbench_csv_path "$VBENCH_CSV_PATH" \
    --vbench_gt_test_path "$VBENCH_GT_TEST_PATH" \
    --tensorboard_log_dir "$TENSORBOARD_LOG_DIR" \
    --cross_modal_checkpoint_path "$CROSS_MODAL_CHECKPOINT_PATH" \
    --infer_name "$INFER_NAME"

# ========================= 종료 메시지 =========================
if [ $? -eq 0 ]; then
    echo "실행이 성공적으로 완료되었습니다."
else
    echo "실행 중 오류가 발생했습니다. 로그를 확인하세요."
    exit 1
fi
