# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               MultiImageMixDataset, RepeatDataset)
from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from .pipelines import *
from antgo.dataflow.dataset.interhand26M import InterHand26M
from antgo.dataflow.dataset.imagenet import ImageNet
from antgo.framework.helper.reader import *

@DATASETS.register_module()
class InterHand26MReader(Reader):
    def __init__(self, 
                train_or_test='train', 
                dir='', 
                trans_test='rootnet', 
                output_hm_shape=(64, 64, 64), 
                input_img_shape= (256, 256), 
                bbox_3d_size= 400, 
                output_root_hm_shape=64, 
                bbox_3d_size_root=400,
                pipeline=None,
                inputs_def=None, **kwargs):
        
        super().__init__(
            InterHand26M(
                train_or_test=train_or_test, 
                dir=dir, 
                params={
                    'trans_test': trans_test,
                    'output_hm_shape': output_hm_shape,
                    'input_img_shape':input_img_shape,
                    'bbox_3d_size': bbox_3d_size,      
                    'output_root_hm_shape': output_root_hm_shape, 
                    'bbox_3d_size_root': bbox_3d_size_root
                }
            ), 
            pipeline=pipeline, 
            inputs_def=inputs_def)


@DATASETS.register_module()
class ImageNetReader(Reader):
    def __init__(self, 
                    train_or_test, 
                    dir='',
                    data_prefix= '', 
                    ann_file = None, 
                    classes = None, 
                    extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'),
                    pipeline=None,
                    inputs_def=None, **kwargs):

        super().__init__(        
            ImageNet(
                train_or_test, 
                dir , 
                params={
                    'data_prefix': data_prefix,
                    'ann_file': ann_file,
                    'classes': classes,
                    'extensions': extensions
                }
            ), 
            pipeline=pipeline, 
            inputs_def=inputs_def)


__all__ = [
    'DATASETS','build_dataloader','build_dataset','ClassBalancedDataset','ConcatDataset','MultiImageMixDataset','RepeatDataset',
    'DistributedGroupSampler','DistributedSampler','GroupSampler'
]
