import sys
sys.path.append('/root/workspace/antgo')
import os
import json
import logging
from antgo.utils import args
from antgo.dataflow.dataset import Cifar10
from antgo.framework.helper.dataset.tfdataset import *

class __CifarDataGenerator(object):
    def __init__(self, flag) -> None:
        self.dataset = Cifar10(flag, './dataset/')
        
        pass
    def __len__(self):
        return self.dataset.size
    
    def __iter__(self):
        for i in range(self.dataset.size):
            sample = self.dataset.sample(i)
            sample.update({
                'tag': f'{i}',
                'image_file': f'{i}.png'
            })
            yield sample
    
def main():
    tgt_folder = './dataset/tfrecord'
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    tfw = TFDataWriter('cifar_10_train', tgt_folder, 100000)
    tfw.write(__CifarDataGenerator('train'))

    tfw = TFDataWriter('cifar_10_test', tgt_folder, 100000)
    tfw.write(__CifarDataGenerator('test'))


if __name__ == "__main__":
    main()