from ..builder import PIPELINES
import torchvision.transforms as transforms
from collections.abc import Sequence
from antgo.framework.helper.utils.registry import build_from_cfg


@PIPELINES.register_module()
class TorchVisionCompose(transforms.Compose):
    def __init__(self, processes):
        process_t = []
        for pn, pp in processes:
            process_t.append(getattr(transforms, pn)(**pp))

        super().__init__(process_t)
    
    def __call__(self, sample):
        image = sample['image']
        sample['image'] = super(Compose, self).__call__(image)
        return sample


@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
