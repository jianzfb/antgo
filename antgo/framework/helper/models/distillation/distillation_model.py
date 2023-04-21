import torch.nn.functional as F
from torch import nn
import torch

from antgo.framework.helper.models.builder import MODELS, build_model
from antgo.framework.helper.multi_stream_module import MultiSteamModule
from antgo.framework.helper.models.builder import build_loss


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x


@MODELS.register_module()
class ReviewKD(MultiSteamModule):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None, init_cfg=None):
        # 默认使用teacher模型作为最佳模型
        if test_cfg is None:
            test_cfg = dict()
        test_cfg.update(
            {
                'inference_on': 'student'
            }
        )
        super(ReviewKD, self).__init__(
            dict(
                teacher=build_model(model.teacher), 
                student=build_model(model.student)
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")

        self.student_cfg= train_cfg.get('student')
        self.teacher_cfg = train_cfg.get('teacher')

        self.student_in_channels = self.student_cfg.get('channels', [96, 128, 160])      # KeyNetF的后三个stage channels
        self.student_shapes = self.student_cfg.get('shapes', [16,8,4])[::-1]                # KetNetF的后三个stage shape

        # resnet50
        self.teacher_out_channels = self.teacher_cfg.get('channels', [512, 1024, 2048]) # resnet50的后三个stage channels
        self.teacher_shapes = self.teacher_cfg.get('shapes', [16,8,4])[::-1]                # resnet50的后三个stage shape

        # align module
        self.align_module = train_cfg.get('align', 'backbone')

        # 蒸馏损失
        self.distill_loss = build_loss(dict(type='HCL'))

        self.kd_loss_weight = train_cfg.get('kd_loss_weight', 1.0)
        self.kd_warm_up = train_cfg.get('kd_warm_up', 20.0)

        abfs = nn.ModuleList()
        mid_channel = min(512, self.student_in_channels[-1])
        for idx, in_channel in enumerate(self.student_in_channels):
            abfs.append(ABF(in_channel, mid_channel, self.teacher_out_channels[idx], idx < len(self.student_in_channels)-1))
        self.abfs = abfs[::-1]

    def forward_train(self, image, image_meta, **kwargs):
        align_module_proxy = getattr(getattr(self.student, self.align_module), 'forward', None)
        out = {}
        def align_student_module_func(x):
            x = align_module_proxy(x)
            out.update({
                'feature': x
            })
            return x
        # 设置新的forward函数
        setattr(getattr(self.student, self.align_module), 'forward', align_student_module_func)
        student_loss = self.student.forward_train(image, image_meta, **kwargs)
        # 恢复原来forward函数
        setattr(getattr(self.student, self.align_module), 'forward', align_module_proxy)        

        x = out['feature'][::-1]
        s_features = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.teacher_shapes[0])
        s_features.append(out_features)
        for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.student_shapes[1:], self.teacher_shapes[1:]):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            s_features.insert(0, out_features)

        # 仅计算特征，忽略head计算
        align_module_proxy = getattr(getattr(self.teacher, self.align_module), 'forward', None)
        out = {}
        def align_teacher_module_func(x):
            x = align_module_proxy(x)
            out.update({
                'feature': x
            })
            return x
        # 设置新的forward函数
        setattr(getattr(self.teacher, self.align_module), 'forward', align_teacher_module_func)
        self.teacher.forward_test(image, image_meta)
        # 恢复原来forward函数
        setattr(getattr(self.teacher, self.align_module), 'forward', align_module_proxy)   
        t_features = out['feature']

        # 计算蒸馏损失
        feature_kd_loss = self.distill_loss(s_features, t_features)

        # 任务损失
        losses = student_loss
        # 蒸馏损失
        losses['review_kd_loss'] = feature_kd_loss * min(1, kwargs.get('epoch')/self.kd_warm_up) * self.kd_loss_weight
        return losses

    def simple_test(self, image, image_meta, rescale=True, **kwargs):
        return self.student.simple_test(image, image_meta, rescale, **kwargs)
