#!/bin/bash

set -x

export WANDB_BASE_URL="https://api.bandw.top"
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export CUDA_VISIBLE_DEVICES=1,2,3,4

llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml