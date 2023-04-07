#!/usr/bin/env bash

MAINPY=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-8990}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CRTDIR=$(pwd)

python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $CRTDIR/$MAINPY --distributed ${@:3}