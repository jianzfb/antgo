# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
sys.path.append('/workspace/antgo')
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from antgo.framework.helper.reader import *
from antgo.framework.helper.trainer import *
from antgo.framework.helper.tester import *
from antgo.framework.helper.tools import args
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.cnn.backbone import *
from antgo.framework.helper.models.detectors.backbone import *
from antgo.framework.helper.models.detectors.head import *
from antgo.framework.helper.models.detectors.model import *
from antgo.antgo.framework.helper.exp.detectors.transformer import *

import logging

cfg_dict = dict(
    # optimizer
    optimizer = dict(type='Adam', lr=0.001,  weight_decay=0.0005),
    optimizer_config = dict(grad_clip=None),    
    # learning policy
    lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=5,
        warmup_ratio=0.001,
        step=[40,45]),             # 25, 35; 15, 25 ; 5,10
    log_config = dict(
        interval=1,    
        hooks=[
            dict(type='TextLoggerHook'),
        ]),
    model = dict(
        type='TTFNet',
        backbone=dict(type='DDRMobileNetV2'),
        neck=None,
        bbox_head=dict(
            type='FcosHeadBayesian',
            in_channel=32,
            feat_channel=32,
            num_classes=3,
            down_stride=4,
            img_width=320,
            img_height=256,
            rescale=(640/320, 480/256),
            score_thresh=0.0,
            enable_dropout=False,
            train_cfg=None,
            test_cfg=dict(topk=5, local_maximum_kernel=3, max_per_img=5)                
        ),
    ),
    checkpoint_config = dict(interval=5, out_dir='./checkpoint/det'),        
    seed=0,
    data=dict(
        train=[
            dict( 
                type="KVLabelHandDetection",
                path="/home/byte_pico_zhangjian52_hdfs/data/activelearning/label/xx",
                pipeline=[    
                    dict(type='ResizeP', target_dim=[320,256]),                                                         
                    dict(type='RandomPasteObject', prob=0.5, obj_num=3),                                                           # 随机在目标框周围画线                
                    dict(type='RandomNoiseAround', prob=0.5, max_noise=15),                
                    dict(type='RandomDistortP', brightness_lower=0.5, brightness_upper=1.5, hue_prob = 0.0),
                    dict(type='RandomRotationInPico', degree=15),                                                       # 随机旋转
                    dict(type='Normalize', mean=[0.0], std=[255.0],to_rgb=False),
                    dict(type='ImageToTensor', keys=['image']),
                ],
            ),          
        ],
        train_dataloader=dict(
            samples_per_gpu=2, 
            workers_per_gpu=1,
            drop_last=False,
            shuffle=True
        )
    ),
    gpu_ids=[0],
    log_level=logging.INFO,
    evaluation=dict(out_dir='./out', interval=1, metric=dict(type='PicoBoxEval'))
)

args.DEFINE_nn_args()

def main():
    # argument parse and create log
    nn_args = args.parse_args()
    # cudnn.benchmark = True    

    print(f'distributed {nn_args.distributed}')
    trainer = Trainer(cfg_dict, './', device='cpu', distributed=nn_args.distributed, diff_seed=nn_args.diff_seed, deterministic=nn_args.deterministic)
    trainer.make_dataloader(with_validate=False)
    trainer.make_model()

    with trainer.train_context(nn_args.max_epochs) as runner:
        for epoch in range(runner._epoch, runner._max_epochs):
            # 运行一个epoch
            trainer.run_on_train()
    
if __name__ == "__main__":
    main()