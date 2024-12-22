#!/bin/bash

# Script configuration
NAME="base_512_v2"
CKPT_PATH='scripts/evaluation/model.ckpt'
CONFIG_PATH="configs/inference_t2v_512_v2.0.yaml"
PROMPT_FILE="prompts/test_prompts.txt"
RESULTS_DIR="results"
SEED=123
MODE="base"
N_SAMPLES=1
BATCH_SIZE=2
HEIGHT=256
WIDTH=256
GUIDANCE_SCALE=12.0
DDIM_STEPS=100
DDIM_ETA=1.0
FPS=12.5
FRAMES=40
SAVEFPS=12.5

# Run inference
python3 scripts/evaluation/inference.py \
    --seed "$SEED" \
    --mode "$MODE" \
    --ckpt_path "$CKPT_PATH" \
    --config "$CONFIG_PATH" \
    --savedir "$RESULTS_DIR/$NAME" \
    --n_samples "$N_SAMPLES" \
    --bs "$BATCH_SIZE" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --unconditional_guidance_scale "$GUIDANCE_SCALE" \
    --ddim_steps "$DDIM_STEPS" \
    --ddim_eta "$DDIM_ETA" \
    --prompt_file "$PROMPT_FILE" \
    --fps "$FPS" \
    --frames "$FRAMES" \
    --savefps "$SAVEFPS" \