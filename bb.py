import enum
import sys

from antgo.pipeline import *
from antgo.pipeline.extent import op
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *

import cv2
import numpy as np

'''
deploy.preprocess 
mean, std, channel permute, need NCHW
'''

# ss = glob['file_path']('/workspace/dataset/test/*.png').stream(). \
#     image_decode['file_path', 'image'](). \
#     resize_op['image', 'resized_image'](size=(512,384)). \
#     deploy.preprocess_func['resized_image', 'preprocessed_image'](meanv=np.array([128,128,128], dtype=np.float32), stdv=np.array([128,128,128], dtype=np.float32), permute=np.array([2,0,1], dtype=np.int32), needed_expand_batch_dim=True, needed_chw=True). \
#     inference_onnx_op['preprocessed_image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')](onnx_path='/workspace/models/bb-epoch_16-model.onnx', input_fields=["image"]). \
#     runas_op[('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3'), ('box', 'label')](func=post_process_func).\
#     plot_bbox[("resized_image", "box", 'label'),"out"](thres=0.2, color=[[0,0,255]], category_map={'0': 'person'}).image_save['out', 'save'](folder='./CC/').run()


# platform=android,windows,linux,service
# with SDK[('image', 'mask'), 'preprocessed_image'](
#     (np.ones((384,512,3), dtype=np.uint8))
# ) as sdk:
#     # 配置SDK输入输出上下文信息
#     sdk.configInput('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')
#     sdk.configOutput('preprocessed_image',  'EAGLEEYE_SIGNAL_RGB_IMAGE')

# 构建计算流图并编译
placeholder['image'](np.ones((384,512,3), dtype=np.uint8)). \
    resize_op['image', 'resized_image'](out_size=(512,384)). \
    deploy.preprocess_func['resized_image', 'preprocessed_image'](meanv=np.array([0.5,0.5,0.5], dtype=np.float32), stdv=np.array([1,1,1], dtype=np.float32), permute=np.array([2,0,1], dtype=np.int32), needed_expand_batch_dim=True). \
    build(
        platform='android/arm64-v8a', 
        eagleeye_path='/workspace/eagleeye/install',
        project_config={
            'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
            'output': [('preprocessed_image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
            'name': 'demo',
            'git': ''
        }
    )

