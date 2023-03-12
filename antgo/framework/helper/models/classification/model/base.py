import torch
from antgo.framework.helper.runner import BaseModule
from abc import ABCMeta, abstractmethod

class BaseClassifier(BaseModule, metaclass=ABCMeta):
    """Base class for classifiers."""

    def __init__(self, init_cfg=None):
        super(BaseClassifier, self).__init__(init_cfg)

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def forward_test(self, image, **kwargs):
        """
        Args:
            image (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        if isinstance(image, torch.Tensor):
            image = [image]
        for var, name in [(image, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(image) == 1:
            return self.simple_test(image[0], **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')
