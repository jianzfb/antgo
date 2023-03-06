from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from typing import List
from functools import partial
import pickle
import os
import tfrecord
from tfrecord.tools.tfrecord2idx import *

numpy_dtype_map = {
    0: np.float32,
    1: np.int32,
    2: np.int64,
    3: np.int8,
    4: np.uint8
}

class TFDataWriter(object):
    def __init__(self, prefix, output_path, size_in_shard=100000, num_shards=1) -> None:
        self.num_shards = num_shards
        self.prefix = prefix
        self.output_path = output_path
        self.size_in_shard = size_in_shard

    def write(self, data_iterator):
        data_num = len(data_iterator)     
        size_in_shard = self.size_in_shard
        if size_in_shard < 0:
            size_in_shard = (data_num+self.num_shards-1) // self.num_shards
        else:
            num_shards = data_num // size_in_shard
            if num_shards * size_in_shard != data_num:
                num_shards += 1
            self.num_shards = num_shards

        tfwriter = None
        shard_i = -1
        count = 0

        # step 1: write tfrecord
        for data in data_iterator:
            data_and_description = {}
            for k,v in data.items():
                if isinstance(data, str):
                    data_and_description[k] = (v.encode('utf-8'), 'byte')
                elif isinstance(v, bytes):
                    data_and_description[k] = (v, 'byte')
                elif isinstance(v, np.ndarray):
                    if v.dtype not in [np.float32, np.int32, np.int64, np.int8, np.uint8]:
                        continue

                    data_and_description[k] = (v.tobytes(), 'byte')
                    data_and_description[f'__{k}_shape'] = (list(v.shape), 'int')
                    if v.dtype == np.float32:
                        data_and_description[f'__{k}_type'] = (0, 'int')
                    elif v.dtype == np.int32:
                        data_and_description[f'__{k}_type'] = (1, 'int')
                    elif v.dtype == np.int64:
                        data_and_description[f'__{k}_type'] = (2, 'int')
                    elif v.dtype == np.int8:
                        data_and_description[f'__{k}_type'] = (3, 'int')
                    elif v.dtype == np.uint8:
                        data_and_description[f'__{k}_type'] = (4, 'int')
                elif isinstance(v, list):
                    if isinstance(v[0], float):
                        data_and_description[k] = (v, 'float')
                    elif isinstance(v[0], int):
                        data_and_description[k] = (v, 'int')
                elif isinstance(v, float):
                    data_and_description[k] = (v, 'float')
                elif isinstance(v, int):
                    data_and_description[k] = (v, 'int')

                if k not in data_and_description:
                    print(f'ignore {k} in data')

            if len(data_and_description) == 0:
                continue

            if (count // size_in_shard) != shard_i:
                if tfwriter is not None:
                    tfwriter.close()
                shard_i = count // size_in_shard
                filename = os.path.join(self.output_path, "./%s-%.5d-of-%.5d-tfrecord"%(self.prefix, shard_i, self.num_shards))
                tfwriter = tfrecord.TFRecordWriter(filename)

            tfwriter.write(data_and_description)            
            count += 1

        if tfwriter is not None:
            tfwriter.close()

        # step 2: create index
        for shard_i in range(self.num_shards):
            tfrecord_file = os.path.join(self.output_path, "./%s-%.5d-of-%.5d-tfrecord"%(self.prefix, shard_i, self.num_shards))
            index_file = os.path.join(self.output_path, "./%s-%.5d-of-%.5d-index"%(self.prefix, shard_i, self.num_shards))
            create_index(tfrecord_file, index_file)