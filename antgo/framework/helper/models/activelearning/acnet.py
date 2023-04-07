import torch
import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import MODELS
from antgo.framework.helper.runner import BaseModule
from antgo.framework.helper.utils import Registry, build_from_cfg

@MODELS.register_module()
class ACModule(BaseModule):
    def __init__(self, model, test_cfg, train_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        self.model = build_from_cfg(model, MODELS)

        # feature_from, uncertainty_from, bayesian_from
        self.test_cfg = test_cfg
 
    def forward(self, *args, **kwargs):
        out = {}

        feature_extract_proxy = None
        if self.test_cfg.get('feature_config', None) is not None:
            feature_extract_proxy = getattr(getattr(self.model, self.test_cfg.feature_config.from_name), 'forward', None)
            from_index = self.test_cfg.feature_config.from_index
            def feature_extract_func(x):
                x = feature_extract_proxy(x)
                feature = x
                if isinstance(x, tuple):
                    feature = x[from_index]
                batch_size = feature.shape[0]
                out.update({
                    'feature': feature.view(batch_size, -1)
                })
                return x

            setattr(getattr(self.model, self.test_cfg.feature_config.from_name), 'forward', feature_extract_func)

        uncertainty_extract_proxy = None
        if self.test_cfg.get('uncertainty_config', None) is not None:
            uncertainty_extract_proxy = getattr(self.model, self.test_cfg.uncertainty_config.from_name, None)
            with_sigmoid = self.test_cfg.uncertainty_config.with_sigmoid
            from_index = self.test_cfg.feature_config.from_index
            def uncertainty_extract_func(x):
                x = uncertainty_extract_proxy(x)
                probability = x
                if isinstance(x, tuple):
                    probability = x[from_index]
                                
                if with_sigmoid:
                    probability = probability.sigmoid()
                                    
                out.update({
                    'uncertainty': probability
                })
                return x
            setattr(self.model, self.test_cfg.uncertainty_config.from_name, uncertainty_extract_func)

        bayesian_proxy = None
        if self.test_cfg.get('bayesian_config', None) is not None:
            bayesian_proxy = getattr(self.model, self.test_cfg.bayesian_config.from_name, None)
            p = self.test_cfg.bayesian_config.p
            def bayesian_func(x):
                x = bayesian_proxy(x)
                x = F.dropout(x, p,training=True)
                return x
            setattr(self.model, self.test_cfg.bayesian_config.from_name, bayesian_func)

        result = self.model(*args, **kwargs)

        # 恢复
        if feature_extract_proxy is not None:
            setattr(getattr(self.model, self.test_cfg.feature_config.from_name), 'forward', feature_extract_proxy)
        if uncertainty_extract_proxy is not None:
            setattr(self.model, self.test_cfg.uncertainty_config.from_name, uncertainty_extract_proxy)
        if bayesian_proxy is not None:
            setattr(self.model, self.test_cfg.bayesian_config.from_name, bayesian_proxy)

        out.update(result)
        return out
