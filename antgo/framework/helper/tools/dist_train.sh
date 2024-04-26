#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
PORT=${PORT:-8990}
CRTDIR=$(pwd)

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --launcher pytorch --distributed --config=$CONFIG ${@:6}