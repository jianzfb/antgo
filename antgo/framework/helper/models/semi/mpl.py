import torch
from antgo.framework.helper.models.builder import MODELS, build_model
from antgo.framework.helper.multi_stream_module import MultiSteamModule
from antgo.framework.helper.models.ema_module import *


@MODELS.register_module()
class MPL(MultiSteamModule):
    def __init__(self, model: dict, ema=0.0, train_cfg=None, test_cfg=None, init_cfg=None) -> None:
        if test_cfg is None:
            test_cfg = dict()
        test_cfg.update(
            {
                'inference_on': 'student'
            }
        )
        super(MPL, self).__init__(
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

        self.temperature = train_cfg.get('temperature', 0.7)
        self.lambda_u = train_cfg.get('lambda_u', 8.0)
        self.uda_steps = train_cfg.get('uda_steps', 5000)
        self.grad_clip = train_cfg.get('grad_clip', 1e9)
        self.ema = train_cfg.get('ema', 0)
        self.threshold = train_cfg.get('threshold', 0.6)
        self.label_batch_size = train_cfg.get('label_batch_size', 5)
        self.label_smoothing = train_cfg.get('label_smoothing', 0.15)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def uda_loss(self, logits_uw, logits_us):
        # logits_uw: 弱增强获得的结果
        # logits_us: 强增强获得的结果
        mask, soft_pseudo_label, hard_pseudo_label = self.make_pseudo_label(logits_uw)

        loss_u = 0
        if len(logits_us.shape) == 2:
            loss_u = torch.mean(
                    -(soft_pseudo_label * torch.log_softmax(logits_us, dim=-1)).sum(dim=-1) * mask
                )
        else:
            print('incoming')
            pass
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
                v_unlabel_ws_kwargs = v[v_label_kwargs:]
                v_unlabel_w_kwargs = [v_unlabel_ws_kwargs[i] for i in range(0, 2*unlabel_batch_size, 2)]
                v_unlabel_s_kwargs = [v_unlabel_ws_kwargs[i] for i in range(1, 2*unlabel_batch_size, 2)]
                
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
        t_logits = self.teacher.forward_train(t_images)
        t_logits_l = t_logits[:label_batch_size]
        t_logits_uw, t_logits_us = t_logits[label_batch_size:].chunk(2)
        del t_logits
        
        t_loss_dict = self.teacher.loss(t_logits_l, **label_kwargs)
        t_loss_l = sum(_value for _, _value in t_loss_dict.items())

        # 具体如何基于weak sample和strong sample构建损失，交给MPL自定义
        # soft_pseudo_label = torch.softmax(t_logits_uw.detach() / self.model.temperature, dim=-1)
        # max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        # mask = max_probs.ge(self.model.threshold).float()
        # t_loss_u = torch.mean(
        #         -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
        #     )
        mask, hard_pseudo_label, t_loss_u = self.uda_loss(t_logits_uw.detach() / self.temperature, t_logits_us)

        weight_u = self.lambda_u * min(1., (kwargs['iter'] + 1) / self.uda_steps)
        t_loss_uda = t_loss_l + weight_u * t_loss_u

        s_images = torch.cat((label_images, unlabel_s_images))
        s_logits = self.student.forward_train(s_images)
        s_logits_l = s_logits[:label_batch_size]
        s_logits_us = s_logits[label_batch_size:]
        del s_logits

        s_loss_dict_old = self.student.loss(s_logits_l.detach(), **label_kwargs)
        s_loss_l_old = sum(_value for _, _value in s_loss_dict_old.items())
        s_loss = self.criterion(s_logits_us, hard_pseudo_label)
        
        s_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        
        optimizer['student'].step()
        if self.ema > 0:
            self.avg_student.update_parameters(self.student)     

        ####
        with torch.no_grad():
            s_logits_l = self.student.forward_train(label_images)
            s_loss_dict_new = self.student.loss(s_logits_l.detach(), **label_kwargs)
            s_loss_l_new = sum(_value for _, _value in s_loss_dict_new.items())

        # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
        # dot_product = s_loss_l_old - s_loss_l_new

        # author's code formula
        dot_product = s_loss_l_new - s_loss_l_old
        # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
        # dot_product = dot_product - moving_dot_product

        # TODO，这里感觉应该改成使用弱增强来获得伪标签
        _, _, hard_pseudo_label = self.make_pseudo_label(t_logits_us.detach())
        t_loss_mpl = dot_product * self.criterion(t_logits_us, hard_pseudo_label)
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
        # NxC
        soft_pseudo_label = torch.softmax(logits_u, dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
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