from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper import reader
from antgo.dataflow.dataset.dataset import Dataset
import numpy as np
import torch


# @reader.register
# class YourDataset(Dataset):
#     def __init__(self, train_or_test="train", dir=None, ext_params=None):
#         super().__init__(train_or_test, dir, ext_params)
#         pass
    
#     @property
#     def size(self):
#         # 返回数据集大小
#         return 0
    
#     def sample(self, id):
#         # 根据id，返回对应样本
#         return {}
