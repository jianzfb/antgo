import bisect
import collections
import copy
import math
from collections import defaultdict
from xml.etree.ElementTree import iselement

import numpy as np
from antgo.framework.helper.utils import build_from_cfg, print_log
import torch
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from .builder import DATASETS, PIPELINES
import json


@DATASETS.register_module()
class IterableDatasetFilter(torch.utils.data.IterableDataset):
    def __init__(self, dataset, not_in_anns=None) -> None:
        self.dataset = dataset
        self.not_in_map = {}
        for ann_file in not_in_anns:
            with open(ann_file, 'r') as fp:
                content = json.load(fp)
            for sample_info in content:
                sample_id = f'{sample_info["image_file"].split("/")[1]}'
                if sample_id.endswith('.png') or sample_id.endswith('.jpg'):
                    sample_id = sample_id[:-4]
                elif sample_id.endswith('.jpeg'):
                    sample_id = sample_id[:-5]
                self.not_in_map[sample_id] = True

    def __iter__(self):
        dataset_iter = iter(self.dataset)
        while True:
            try:
                sample = next(dataset_iter)
                if len(self.not_in_map) > 0:
                    sample_tag = sample["image_meta"]["tag"]                   
                    sample_id = f"{sample_tag}_{sample['image_meta']['image_file'].replace('/','-')}"
                    if sample_id in self.not_in_map:
                        continue
                    
                yield sample
            except StopIteration:
                break

    def __len__(self):
        # 这里返回的是最大数，实际数量可能少于此（部分样本会被过滤）
        return len(self.dataset)
