#!/bin/bash

# 기본값 설정
OUTPUT_DIR="checkpoints"
WANDB_PROJECT="audio_teacher_lora"
TRAIN_BATCH_SIZE=1
GRAD_ACC_STEPS=1
VAL_BATCH_SIZE=1
LR=1e-5
NUM_EPOCHS=10
EVAL_EVERY=100
SAMPLE_RATE=16000
SLICE_DURATION=3.2
HOP_SIZE=160
N_MELS=256
MIXED_PRECISION="bf16"


# 사용법 출력 함수
usage() {
    echo "Usage: $0 --csv_path PATH --audio_dir PATH [options]"
    echo "Options:"
    echo "  --output_dir PATH         Directory to save checkpoints (default: checkpoints)"
    echo "  --wandb_project NAME      Weights & Biases project name (default: audio_teacher_lora)"
    echo "  --train_batch_size INT    Training batch size (default: 2)"
    echo "  --val_batch_size INT      Validation batch size (default: 2)"
    echo "  --lr FLOAT                Learning rate (default: 1e-5)"
    echo "  --num_epochs INT          Number of training epochs (default: 10)"
    echo "  --gradient_accumulation_steps INT (default: 32)"
    echo "  --eval_every INT          Evaluate every N epochs (default: 2)"
    echo "  --sample_rate INT         Audio sample rate (default: 16000)"
    echo "  --slice_duration FLOAT    Audio slice duration in seconds (default: 3.2)"
    echo "  --hop_size INT            Hop size for spectrogram (default: 160)"
    echo "  --n_mels INT              Number of mel bands (default: 256)"
    echo "  --device DEVICE           Device to use (cuda or cpu) (default: cuda)"
    exit 1
}

# 명령줄 인자 파싱
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --csv_path) csv_path="$2"; shift ;;
        --audio_dir) audio_dir="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift ;;
        --val_batch_size) VAL_BATCH_SIZE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        --gradient_accumulation_steps) GRAD_ACC_STEPS="$2"; shift ;;
        --eval_every) EVAL_EVERY="$2"; shift ;;
        --sample_rate) SAMPLE_RATE="$2"; shift ;;
        --slice_duration) SLICE_DURATION="$2"; shift ;;
        --hop_size) HOP_SIZE="$2"; shift ;;
        --n_mels) N_MELS="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# 필수 인자 체크
if [ -z "$csv_path" ] || [ -z "$audio_dir" ]; then
    echo "Error: --csv_path and --audio_dir are required."
    usage
fi

# train.py 실행
python train.py \
    --csv_path "$csv_path" \
    --audio_dir "$audio_dir" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --sample_rate "$SAMPLE_RATE" \
    --slice_duration "$SLICE_DURATION" \
    --hop_size "$HOP_SIZE" \
    --n_mels "$N_MELS" \
    --mixed_precision "$MIXED_PRECISION"
