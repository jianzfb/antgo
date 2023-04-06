"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import MODELS
from antgo.framework.helper.runner import BaseModule
from antgo.framework.helper.utils import Registry, build_from_cfg
from antgo.framework.helper.runner.optimizer.builder import build_optimizer
from antgo.framework.helper.models.builder import build_model
from antgo.framework.helper.runner.hooks.hook import *


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim=1024, latent_dim=512):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.latent_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.latent_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z).view(-1)
        return validity


@MODELS.register_module()
class Adda(BaseModule):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None, init_cfg=None):
        super(Adda, self).__init__(init_cfg=init_cfg)

        # model build (具体任务模型)
        self.model = build_model(model)

        # discriminator (判别器模型)
        assert(train_cfg.get('latent_dim'))
        self.discriminator = Discriminator(train_cfg.input_dim, train_cfg.latent_dim)
        self.dis_loss_func = torch.nn.BCELoss()

        self.src_domain_batch_size = train_cfg.get('src_domain_batch_size', 5)      # 目标域数据量 在一个batch里
        self.gen_loss_weight = train_cfg.get('gen_loss_weight', 1.0)                # 生成器损失权重
        self.dis_loss_weight = train_cfg.get('dis_loss_weight', 1.0)                # 判别器损失权重

        self.encoder_config = train_cfg.encoder_config  # from_name='', from_index=-1       

    def train_step(self, data, optimizer, **kwargs):
        # -----------------
        #  STEP 1: Train Generator
        # -----------------        
        # 下游任务处于train状态 + 判别器处于eval状态
        self.model.train()
        self.discriminator.eval()
        optimizer['model'].zero_grad()

        num_samples = 1
        if type(data) == list or type(data) == tuple:
            num_samples = len(data[0]['image'])
        else:
            num_samples = len(data['image'])

        encoder_feature_proxy = getattr(getattr(self.model, self.encoder_config.from_name), 'forward', None)
        from_index = self.encoder_config.from_index
        out = {}
        def feature_extract_func(x):
            x = encoder_feature_proxy(x)
            feature = x
            if isinstance(x, tuple):
                feature = x[from_index]
            batch_size = feature.shape[0]
            out.update({
                'feature': feature.view(batch_size, -1)
            })
            return x
        setattr(getattr(self.model, self.encoder_config.from_name), 'forward', feature_extract_func)

        # 下游任务模型损失
        step_1_losses = self.model(**data, **kwargs)

        src_domain_batch_size = self.src_domain_batch_size
        tgt_domain_batch_size = num_samples - self.src_domain_batch_size
        src_domain_encoder_feature = out['feature'][:self.src_domain_batch_size]       # 例如，仿真数据 (0)
        tgt_domain_encoder_feature = out['feature'][self.src_domain_batch_size:]       # 例如，真实数据 (1)

        # 
        gen_loss = self.dis_loss_func(
            self.discriminator(src_domain_encoder_feature), 
            torch.Tensor(src_domain_encoder_feature.shape[0]).fill_(1.0).to(src_domain_encoder_feature.device))
        step_1_losses.update(
            {
                'adv_loss': self.gen_loss_weight*gen_loss
            }
        )

        step_1_total_loss, step_1_log_vars = self._parse_losses(step_1_losses)
        step_1_total_loss.backward()
        optimizer['model'].step()    # 仅影响生成器

        # ---------------------
        #  STEP 2: Train Discriminator
        # ---------------------
        self.model.eval()
        self.discriminator.train()
        optimizer['discriminator'].zero_grad()

        d_src_loss = self.dis_loss_func(
            self.discriminator(src_domain_encoder_feature.detach()), 
            torch.Tensor(src_domain_encoder_feature.shape[0]).fill_(0.0).to(src_domain_encoder_feature.device))
        d_tgt_loss = self.dis_loss_func(
            self.discriminator(tgt_domain_encoder_feature.detach()), 
            torch.Tensor(tgt_domain_encoder_feature.shape[0]).fill_(1.0).to(tgt_domain_encoder_feature.device))

        # TODO，训练判别器，保证真实样本损失和仿真样本损失 平衡
        src_weight = tgt_domain_batch_size / src_domain_batch_size
        tgt_weight = 1.0
        step_2_losses = {
            'd_src_loss': self.dis_loss_weight * d_src_loss*src_weight,
            'd_tgt_loss': self.dis_loss_weight * d_tgt_loss*tgt_weight
        }
        step_2_total_loss, step_2_log_vars = self._parse_losses(step_2_losses)

        step_2_total_loss.backward()
        optimizer['discriminator'].step()   # 仅影响判别器
        
        # 恢复原有设置
        setattr(getattr(self.model, self.encoder_config.from_name), 'forward', encoder_feature_proxy)

        # 结束
        log_vars = step_1_log_vars
        log_vars.update(step_2_log_vars)
        outputs = dict(
            loss=0.0, log_vars=log_vars, num_samples=num_samples)
        return outputs