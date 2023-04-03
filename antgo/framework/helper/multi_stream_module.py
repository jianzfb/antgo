from typing import Dict
from antgo.framework.helper.models.detectors.base import BaseDetector
from antgo.framework.helper.base_module import BaseModule


class MultiSteamModule(BaseModule):
    def __init__(
        self, model=dict(), train_cfg=None, test_cfg=None, init_cfg=None):
        super(MultiSteamModule, self).__init__()
        self.submodules = list(model.keys())
        for k, v in model.items():
            setattr(self, k, v)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.inference_on = self.test_cfg.get("inference_on", None)

    def model(self, **kwargs):
        if "submodule" in kwargs:
            assert (
                kwargs["submodule"] in self.submodules
            ), "Detector does not contain submodule {}".format(kwargs["submodule"])
            model = getattr(self, kwargs["submodule"])
        else:
            model = getattr(self, self.inference_on)
        return model

    def freeze(self, model_ref: str):
        assert model_ref in self.submodules
        model = getattr(self, model_ref)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward_test(self, image, image_meta=None, **kwargs):
        kwargs.update({
            'image_meta': image_meta
        })
        return self.model(**kwargs).forward_test(image, **kwargs)

    def extract_feat(self, image):
        return self.model().extract_feat(image)

    def simple_test(self, image, image_meta=None, **kwargs):
        kwargs.update({
            'image_meta': image_meta
        })        
        return self.model(**kwargs).simple_test(image, **kwargs)
