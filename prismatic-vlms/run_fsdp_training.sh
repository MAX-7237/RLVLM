#!/bin/bash

# DataParallel Multi-GPU Training Script
# Usage: ./run_dataparallel_training.sh

set -e

echo "Starting DataParallel training with all available GPUs..."

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0

# Run with Python (DataParallel handles multi-GPU automatically)
python scripts/stage1/train_stage1.py \
    --model-id reproduction-llava-v15+7b \
    --data-path /home/ubuntu/vlm_prune/RLVLM/ai2d/training_data_1024.pt \
    --epochs 1 \
    --batch-size 1 \
    --actor-lr 1e-4 \
    --num-workers 0

echo "DataParallel training completed!"
