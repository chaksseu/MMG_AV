#!/usr/bin/env bash


# 설정해야하는 값들
BATCH_SIZE=4
NUM_GPUS=3
INFER_SCRIPT="MMG_multi_gpu_inference_mmg_1217.py"
PROMPT_FILE="prompts/test_prompts_293_24.txt"
MMG_CHECKPOINT="/workspace/workspace/MMG_01/MMG_CHECKPOINTS_1217/1217_MMG_lr_1e-05_batch_512_epoch_90_vggsound_sparse/model.safetensors"
SAVEDIR_FOLDER="0106_output_MMG_epoch90_50steps"

# 공통 옵션 정의 (고정)
CKPT_PATH="scripts/evaluation/model.ckpt"
CONFIG_PATH="configs/inference_t2v_512_v2.0.yaml"
PRETRAINED_MODEL="auffusion/auffusion-full"


echo "================================================================================"
echo "[INFO] Inference 스크립트     : $INFER_SCRIPT"
echo "[INFO] 입력 Prompt(TEXT) 파일       : $PROMPT_FILE"
echo "================================================================================"

echo "[INFO] MMG Inference를 실행합니다."

accelerate launch $INFER_SCRIPT \
  --num_gpus $NUM_GPUS \
  --ckpt_path $CKPT_PATH \
  --config $CONFIG_PATH \
  --prompt_file $PROMPT_FILE \
  --pretrained_model_name_or_path $PRETRAINED_MODEL \
  --cross_modal_checkpoint_path $MMG_CHECKPOINT \
  --savedir $SAVEDIR_FOLDER \
  --bs $BATCH_SIZE