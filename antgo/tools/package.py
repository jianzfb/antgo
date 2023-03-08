from antgo.ant import environment
from antgo.dataflow.dataset.tfrecord_dataset import *
import os
import json

class __DataIter(object):
    def __init__(self, image_folder, annotation_file) -> None:
        pass
    
# support lmdb, tfrecord
# independent: json file
def make_package(image_folder, annotation_file, num_shards):
    tfrecord_data_writer = TFRecordDataWriter()
    with open(annotation_file, 'r') as fp:
        content = json.load(fp)
    
    # for info in content:
    #     image_file = info['image_file']
        
    tfrecord_data_writer.write()