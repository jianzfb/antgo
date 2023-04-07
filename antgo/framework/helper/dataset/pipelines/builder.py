import functools
from ..builder import PIPELINES
import torchvision
import inspect

import torchvision.transforms as transforms
from collections.abc import Sequence
from antgo.framework.helper.utils.registry import build_from_cfg
from antgo.dataflow import imgaug as local_imgaug

def __transform_init__(self, keys, **kwargs):
    self.keys = keys
    self.parent_cls.__init__(self, **kwargs)


def __transform_call__(self, samples):
    for key in self.keys:
        samples[key] = self.parent_cls.__call__(self, samples[key])
    return samples
    

def register_torchvision_transforms():
    torch_transfoms = []
    for module_name in dir(torchvision.transforms):
        if module_name.startswith('__'):
            continue
        
        __transform = getattr(torchvision.transforms, module_name)
        if inspect.isclass(__transform) and module_name not in ['AutoAugmentPolicy', 'InterpolationMode']:
            __transform_proxy = \
                type(
                    module_name, 
                    (__transform,), 
                    {   
                        'name': module_name,
                        '__doc__': f'{module_name}',
                        '__init__':  __transform_init__,
                        '__call__': __transform_call__,
                        'parent_cls': __transform
                    }
                )
            PIPELINES.register_module()(__transform_proxy)
            torch_transfoms.append(module_name)


TORCH_VISION_TRANSFORMS = register_torchvision_transforms()


def register_antgo_pipeline():
    for imgaug_module_name in local_imgaug.__all__:
        PIPELINES.register_module()(getattr(local_imgaug, imgaug_module_name))

ANTGO_TRANSFORMS = register_antgo_pipeline()


# @PIPELINES.register_module()
# class TorchVisionCompose(transforms.Compose):
#     def __init__(self, processes):
#         process_t = []
#         for pn, pp in processes:
#             process_t.append(getattr(transforms, pn)(**pp))

#         super().__init__(process_t)
    
#     def __call__(self, sample):
#         image = sample['image']
#         sample['image'] = super(TorchVisionCompose, self).__call__(image)
#         return sample


# @PIPELINES.register_module()
# class Compose(object):
#     """Compose a data pipeline with a sequence of transforms.

#     Args:
#         transforms (list[dict | callable]):
#             Either config dicts of transforms or transform objects.
#     """

#     def __init__(self, transforms):
#         assert isinstance(transforms, Sequence)
#         self.transforms = []
#         for transform in transforms:
#             if isinstance(transform, dict):
#                 transform = build_from_cfg(transform, PIPELINES)
#                 self.transforms.append(transform)
#             elif callable(transform):
#                 self.transforms.append(transform)
#             else:
#                 raise TypeError('transform must be callable or a dict, but got'
#                                 f' {type(transform)}')

#     def __call__(self, data):
#         for t in self.transforms:
#             data = t(data)
#             if data is None:
#                 return None
#         return data

#     def __repr__(self):
#         format_string = self.__class__.__name__ + '('
#         for t in self.transforms:
#             format_string += f'\n    {t}'
#         format_string += '\n)'
#         return format_string