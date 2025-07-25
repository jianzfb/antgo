#!/usr/bin/env bash

MAINPY=$1
GPU_IDS=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
PORT=${PORT:-8990}
CRTDIR=$(pwd)

# gpu visible in env
export CUDA_VISIBLE_DEVICES=$GPU_IDS
# gpu number
GPU_NUM=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# show command
echo python3 -m torch.distributed.launch --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --nproc_per_node $GPU_NUM --master_port $PORT $CRTDIR/$MAINPY --distributed ${@:6}

# execute
python3 -m torch.distributed.launch \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --nproc_per_node $GPU_NUM \
    --master_port $PORT \
    $CRTDIR/$MAINPY --distributed ${@:6}
