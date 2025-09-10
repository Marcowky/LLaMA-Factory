#!/bin/bash

set -x

export TMPDIR=~/tmp_0909
export WANDB_BASE_URL="https://api.bandw.top"
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export CUDA_VISIBLE_DEVICES=4,5,6,8

DATASETS=(
    "weatherrft_500hpa_situation"
    "weatherrft_850hpa_situation"
    "weatherrft_land_situation"
    "weatherrft_max_temp"
    "weatherrft_min_temp"
    "weatherrft_phenomena"
    "weatherrft_rain"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Training with dataset: $DATASET"
    numactl --cpunodebind=1 --membind=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml dataset=${DATASET} output_dir=saves/qwen2_5vl-7b-${DATASET}/full/sft
done