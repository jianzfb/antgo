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

@PIPELINES.register_module()
class CropImageP(CropImage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@PIPELINES.register_module()
class RandomRotationP(Rotation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@PIPELINES.register_module()
class CutmixImageP(CutmixImage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@PIPELINES.register_module()
class MixupImageP(MixupImage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
