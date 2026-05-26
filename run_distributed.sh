#!/bin/bash

echo "Checking GPU count..."

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

echo "GPUs detected: $GPU_COUNT"

if [ "$GPU_COUNT" -ge 2 ]; then

    echo "Launching distributed training..."

    NCCL_P2P_DISABLE=1 torchrun \
        --standalone \
        --nproc_per_node=$GPU_COUNT \
        train.py \
        --epochs 5 \
        --batch-size 64 \
        --lr 0.001

else

    echo "Launching single GPU training..."

    python train.py \
        --epochs 5 \
        --batch-size 64 \
        --lr 0.001

fi
