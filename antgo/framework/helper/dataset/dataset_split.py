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


@DATASETS.register_module()
class DatasetSamplingByClass(object):
    def __init__(self, dataset, sampling_num=4000, class_num=10) -> None:
        self.dataset = build_from_cfg(dataset, DATASETS, None)
        sample_labels = [self.dataset.get_cat_ids(i) for i in range(len(self.dataset))]
        label_per_class = sampling_num // class_num
        sample_labels = np.array(sample_labels)

        labeled_idx = []
        for i in range(class_num):
            idx = np.where(sample_labels == i)[0]
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        labeled_idx = np.array(labeled_idx)
        assert len(labeled_idx) == sampling_num
        self.sampling_idx = labeled_idx

    def __getitem__(self, idx):
        return self.dataset[self.sampling_idx[idx]]

    def __len__(self):
        return len(self.sampling_idx)
