from .compose import Compose
from .formatting import ToTensor, ImageToTensor
from .transforms import Normalize, RandomErasing, RandomFlip, RandomResizedCrop, Resize, CenterCrop
from antgo.dataflow import imgaug as local_imgaug
from ..builder import PIPELINES

def register_antgo_pipeline():
    for imgaug_module_name in local_imgaug.__all__:
        PIPELINES.register_module()(getattr(local_imgaug, imgaug_module_name))

register_antgo_pipeline()
__all__ = [
    'Compose', 'ToTensor', 'ImageToTensor', 'Normalize',
    'RandomErasing', 'RandomFlip', 'RandomResizedCrop','Resize','CenterCrop'
]