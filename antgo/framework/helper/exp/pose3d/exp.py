# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
sys.path.append('/workspace/antgo')
import argparse
import torch
import torch.backends.cudnn as cudnn
from antgo.framework.helper.reader import *
from antgo.framework.helper.trainer import *
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.tools import args
from antgo.framework.helper.models.pose3d.backbone import *
from antgo.framework.helper.models.pose3d.head import *
from antgo.framework.helper.models.pose3d.model import *


from pico_hand_dataset_kv import *
from antgo.utils import logger
import json
import logging


cfg_dict = dict(
    optimizer = dict(type='Adam', lr=1e-3),
    optimizer_config = dict(grad_clip=None),
    lr_config = dict(
        policy='step',
        by_epoch=True,
        warmup=None,
        gamma=0.1,
        step=[25, 35, 38]),
    log_config = dict(
        interval=5,    
        hooks=[
            dict(type='TextLoggerHook'),
        ]),
    model=dict(
        type='DDRTwoHandPose3DModel',
        backbone=dict(
            type='DDRMobileNetV2'),
        pose_head=dict(
            type='TwoHand3DPoseLatent')
        ),    
    checkpoint_config = dict(interval=1,out_dir='./'),        
    seed=0,
    data = dict(
        train=dict(
            type='KVPicoHandDataset',
            path='/workspace/handtt/dataset/temp',            
            pipeline=[dict(type='TorchVisionCompose', processes=[['ToTensor', {}]])]
        ),
        train_dataloader=dict(samples_per_gpu=4, workers_per_gpu=1, shuffle=True, drop_last=True),
    ),
    gpu_ids=[0],
    log_level=logging.INFO,
    evaluation=dict(out_dir='./out')
)


def main():
    # argument parse and create log
    exp_args = args.nn_args()

    print(f'distributed {exp_args.distributed}')
    trainer = Trainer(cfg_dict, './', device='cpu', distributed=exp_args.distributed, diff_seed=exp_args.diff_seed, deterministic=exp_args.deterministic)
    trainer.make_dataloader(with_validate=False)
    trainer.make_model()
    
    with trainer.train_context(exp_args.max_epochs) as runner:
        for epoch in range(runner._epoch, runner._max_epochs):
            # 运行一个epoch
            trainer.run_on_train(mode='train')

if __name__ == "__main__":
    main()
