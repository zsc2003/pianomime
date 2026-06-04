#!/bin/bash

WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python pianomime/single_task/train_ppo.py \
    --root-dir /tmp/robopianist/rl/ \
    --n-steps-lookahead 10 \
    --residual-action \
    --frame-stack 4 \
    --num-envs 32 \
    --lr-decay-rate 0.999 \
    --n-steps 512 \
    --mimic-task "Petrunko_3" \
    --environment-name "Petrunko_3" \
    --use-note-trajectory \
    --total-iters 2000 \
    --residual-factor 0.03 \
    --deepmimic \
    --tqdm-bar