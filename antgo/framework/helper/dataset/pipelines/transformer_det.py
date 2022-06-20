from ..builder import PIPELINES
from antgo.dataflow.imgaug.operators import *


@PIPELINES.register_module()
class ResizeP(Resize):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@PIPELINES.register_module()
class RandomFlipImageP(RandomFlipImage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

@PIPELINES.register_module()
class RandomDistortP(RandomDistort):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

@PIPELINES.register_module()
class RandomCropP(RandomCrop):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
