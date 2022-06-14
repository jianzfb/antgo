from antgo.antgo.dataflow.imgaug.operators import DecodeImage
from ..builder import PIPELINES
import torchvision.transforms as transforms
from antgo.dataflow.imgaug.operators import *


@PIPELINES.register_module()
class DecodeImageP(object):
    def __init__(self, to_rgb, with_mixup=False, with_cutmix=False):
        self.decode_image = DecodeImage(to_rgb, with_mixup, with_cutmix)

    def __call__(self, sample):
        return self.decode_image(sample)
