#!/bin/bash

WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python pianomime/single_task/train_ppo.py \
    --root-dir ~/shared-nvme/pianomime_logs \
    --n-steps-lookahead 10 \
    --residual-action \
    --frame-stack 4 \
    --num-envs 8 \
    --lr-decay-rate 0.999 \
    --n-steps 512 \
    --mimic-task "NeverGonnaGiveYouUp_1" \
    --environment-name "NeverGonnaGiveYouUp_1_smooth" \
    --use-note-trajectory \
    --enable-smoothness-reward \
    --smoothness-weight 0.5 \
    --beta-action 3.0 \
    --beta-accel 30.0 \
    --total-iters 2000 \
    --residual-factor 0.03 \
    --deepmimic \
    --tqdm-bar