#!/usr/bin/env bash

MAINPY=$1
GPUS=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
PORT=${PORT:-8990}
CRTDIR=$(pwd)

python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $CRTDIR/$MAINPY --distributed ${@:6}