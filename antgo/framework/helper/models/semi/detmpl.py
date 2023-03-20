from pydoc import classname
import torch
import torch.nn.functional as F
from antgo.framework.helper.models.builder import MODELS, build_model
from antgo.framework.helper.multi_stream_module import MultiSteamModule
from antgo.framework.helper.models.ema_module import *
from antgo.framework.helper.models.builder import HEADS, build_loss
from antgo.framework.helper.models.detectors.utils.gaussian_target import gaussian2D
import cv2
import numpy as np


@MODELS.register_module()
class DETMPL(MultiSteamModule):
    def __init__(self, model: dict, ema=0.0, train_cfg=None, test_cfg=None) -> None:
        if test_cfg is None:
            test_cfg = dict()
        test_cfg.update(
            {
                'inference_on': 'student'
            }
        )
        super(DETMPL, self).__init__(
            dict(
                teacher=build_model(model['teacher']),
                student=build_model(model['student']),
            ),
            train_cfg=train_cfg if train_cfg is not None else dict(),
            test_cfg=test_cfg if test_cfg is not None else dict(),
        )

        self.avg_student = None
        if ema > 0:
            self.avg_student = ModelEMA(self.student, ema)
            self.inference_on = self.avg_student

        self.temperature = train_cfg.get('temperature', 1.0)
        self.lambda_u = train_cfg.get('lambda_u', 1.0)
        self.uda_steps = train_cfg.get('uda_steps', 1)
        self.grad_clip = train_cfg.get('grad_clip', 1e9)
        self.ema = train_cfg.get('ema', 0)
        self.threshold = train_cfg.get('threshold', 0.95)
        self.label_batch_size = train_cfg.get('label_batch_size', 5)
        self.unlabel_batch_size = train_cfg.get('unlabel_batch_size', 3)    # 无标签数据量 在一个batch里
        
        self.use_sigmoid = train_cfg.get('use_sigmoid', True)               # 是否将挑选出来的featuremap 使用sigmoid 
        self.positive_threshold = train_cfg.get('positive_threshold', 0.8)
        self.negative_threshold = train_cfg.get('negative_threshold', 0.1)
        loss_ch = train_cfg.get('loss_ch', dict(type='GaussianFocalLoss', loss_weight=1.0))
        self.loss_center_heatmap = build_loss(loss_ch)
        
        # 生成gaussian 伪标签的高斯核
        gaussian_kernel = gaussian2D(2, sigma=1)
        gaussian_kernel_h, gaussian_kernel_w = gaussian_kernel.shape[:2]
        self.gaussian_kernel = torch.reshape(gaussian_kernel, (1,1,gaussian_kernel_h, gaussian_kernel_w))
        
    def uda_loss(self, logits_uw, logits_us):
        # logits_uw: 弱增强 结果
        # logits_us: 强增强 结果
        mask, soft_pseudo_label, hard_pseudo_label = self.make_pseudo_label(logits_uw)

        # 损失
        logits_us = logits_us.sigmoid()
        loss_u =  F.binary_cross_entropy(logits_us, soft_pseudo_label, reduction="none") * mask
        loss_u = logits_us.sum()/(mask.sum()+1e-4)
        
        return mask, hard_pseudo_label, loss_u

    def train_step(self, data, optimizer, **kwargs):
        image = data.pop('image')
        other = data
        label_batch_size = self.label_batch_size
        batch_size = image.shape[0]
        unlabel_batch_size = (batch_size - label_batch_size)//2

        label_images, unlabel_ws_images = \
            torch.split(image, [label_batch_size, batch_size-label_batch_size],dim=0) 

        unlabel_w_images=torch.index_select(unlabel_ws_images,dim=0,index=torch.tensor([i for i in range(0,2*unlabel_batch_size,2)]))
        unlabel_s_images=torch.index_select(unlabel_ws_images,dim=0,index=torch.tensor([i for i in range(1,2*unlabel_batch_size,2)]))

        label_kwargs = {}
        unlabel_w_kwargs = {}
        unlabel_s_kwargs = {}
        for k, v in other.items():
            if isinstance(v, list):
                v_label_kwargs = v[:label_batch_size]
                v_unlabel_ws_kwargs = v[label_batch_size:]
                v_unlabel_w_kwargs = [v_unlabel_ws_kwargs[i] for i in range(0, 2*unlabel_batch_size, 2)]
                v_unlabel_s_kwargs = [v_unlabel_ws_kwargs[i] for i in range(1, 2*unlabel_batch_size, 2)]
                
                label_kwargs[k] = v_label_kwargs
                unlabel_w_kwargs[k] = v_unlabel_w_kwargs
                unlabel_s_kwargs[k] = v_unlabel_s_kwargs
                continue

            v_label_kwargs, v_unlabel_ws_kwargs = \
                torch.split(v, [label_batch_size, unlabel_batch_size * 2],dim=0) 
            label_kwargs[k] = v_label_kwargs

            v_unlabel_w_kwargs=torch.index_select(v_unlabel_ws_kwargs,dim=0,index=torch.tensor([i for i in range(0,2*unlabel_batch_size,2)]))
            v_unlabel_s_kwargs=torch.index_select(v_unlabel_ws_kwargs,dim=0,index=torch.tensor([i for i in range(1,2*unlabel_batch_size,2)]))

            unlabel_w_kwargs[k] = v_unlabel_w_kwargs
            unlabel_s_kwargs[k] = v_unlabel_s_kwargs

        t_images = torch.cat((label_images, unlabel_w_images, unlabel_s_images))
        t_kwargs = {k: None for k in label_kwargs.keys()}
        # image_meta 标配信息
        t_kwargs['image_meta'] = []
        t_kwargs['image_meta'].extend(label_kwargs['image_meta'])
        t_kwargs['image_meta'].extend(unlabel_w_kwargs['image_meta'])
        t_kwargs['image_meta'].extend(unlabel_s_kwargs['image_meta'])        

        # 每个模型一般由两个模块组成 forward_train -> loss
        t_logits, t_reg = self.teacher.forward_train(t_images, **t_kwargs) # tuple
        
        t_logits_l = t_logits[:label_batch_size]
        t_logits_uw, t_logits_us = t_logits[label_batch_size:].chunk(2)
        
        t_reg_l = t_reg[:label_batch_size]
        t_reg_uw, t_reg_us = t_reg[label_batch_size:].chunk(2)
        del t_logits, t_reg

        # teacher loss in label data
        t_loss_dict = self.teacher.loss(t_logits_l, t_reg_l, **label_kwargs)
        t_loss_l = sum(_value for _, _value in t_loss_dict.items())

        # teacher loss in unlabel data
        mask, hard_pseudo_label, t_loss_u = self.uda_loss(t_logits_uw.detach() / self.temperature, t_logits_us)

        weight_u = self.lambda_u * min(1., (kwargs['iter'] + 1) / self.uda_steps)
        t_loss_uda = t_loss_l + weight_u * t_loss_u

        s_images = torch.cat((label_images, unlabel_s_images))
        s_kwargs = {k: None for k in label_kwargs.keys()}
        s_kwargs['image_meta'] = []
        s_kwargs['image_meta'].extend(label_kwargs['image_meta'])
        s_kwargs['image_meta'].extend(unlabel_s_kwargs['image_meta'])
        s_logits, s_reg = self.student.forward_train(s_images, **s_kwargs)
        s_logits_l = s_logits[:label_batch_size]
        s_logits_us = s_logits[label_batch_size:]
        
        s_reg_l = s_reg[:label_batch_size]
        s_reg_us = s_reg[label_batch_size:]
        del s_logits, s_reg

        # 查看student未使用伪标签更新前，在有标签数据上的损失
        s_loss_dict_old = self.student.loss(s_logits_l.detach(), s_reg_l.detach(), **label_kwargs)
        s_loss_l_old = sum(_value for _, _value in s_loss_dict_old.items())
        
        # student使用伪标签计算损失并更新 
        # TODO，加入mask，消除不确定区域影响
        s_loss = self.loss_center_heatmap(s_logits_us.sigmoid(), hard_pseudo_label)        
        s_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        
        optimizer['student'].step()
        if self.ema > 0:
            self.avg_student.update_parameters(self.student)     

        ####
        # 查看student在使用伪标签更新后，在有标签数据上的损失
        with torch.no_grad():
            s_loss_dict_new = self.student.forward_train(label_images, **label_kwargs)
            s_loss_l_new = sum(_value for _, _value in s_loss_dict_new.items())
 
        # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
        # dot_product = s_loss_l_old - s_loss_l_new

        # author's code formula
        dot_product = s_loss_l_new - s_loss_l_old
        # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
        # dot_product = dot_product - moving_dot_product

        # TODO，这里感觉应该改成使用弱增强来获得伪标签
        # _, _, hard_pseudo_label = self.make_pseudo_label(t_logits_us.detach())
        # 使用来自于student的反馈信号+teacher的硬标签损失
        t_loss_mpl = dot_product * self.loss_center_heatmap(t_logits_us.sigmoid(), hard_pseudo_label)
        # test
        # t_loss_mpl = torch.tensor(0.).to(args.device)
        t_loss = t_loss_uda + t_loss_mpl
        t_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.teacher.parameters(), self.grad_clip)
        
        optimizer['teacher'].step()
        self.teacher.zero_grad()
        self.student.zero_grad()

        losses = {
            's_loss': s_loss,
            't_loss': t_loss,
            't_loss_l': t_loss_l,
            't_loss_u': t_loss_u,
            't_loss_mpl': t_loss_mpl,            
        }

        losses, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=losses, log_vars=log_vars, num_samples=batch_size)

        outputs['log_vars'].update(
            {'mask': mask.mean().cpu().item()}
        )
        return outputs

    def make_pseudo_label(self, logits_u):
        # 返回 mask, soft_pseudo_label, hard_pseudo_label
        # NxCxHxW
        # logits_u (已经是概率 语义)
        # 计算软标签
        soft_pseudo_label = logits_u.sigmoid()
            
        # 计算有效区域
        # 小于低阈值区域为绝对背景
        # 高于高阈值区域为绝对前景
        positive_mask = soft_pseudo_label.ge(self.positive_threshold).float()
        negative_mask = soft_pseudo_label.le(self.negative_threshold).float()
        mask = ((positive_mask + negative_mask) > 0.5).float()
        
        # 计算硬标签
        hard_pseudo_label = torch.max_pool2d(soft_pseudo_label, 3, 1, padding=1)
        hard_pseudo_label = (hard_pseudo_label == soft_pseudo_label).float()        # 最大值位置为1，其余位置为0

        batch_size, class_num = logits_u.shape[:2]
        with torch.no_grad():
            weight = self.gaussian_kernel.repeat(class_num, 1,1,1)
            hard_pseudo_label = F.conv2d(hard_pseudo_label, weight, bias=None, stride=1, padding=5//2, groups=class_num)

            # 调整最大值为1
            max_val= torch.max(hard_pseudo_label.view(batch_size, class_num, -1), dim=-1, keepdim=True)[0]
            hard_pseudo_label = hard_pseudo_label / (max_val.view(batch_size, class_num, 1, 1)+1e-10)

        return mask, soft_pseudo_label, hard_pseudo_label

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs):
        # 同时兼容从base模型和dense模型加载
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )