import torch
from antgo.framework.helper.models.builder import MODELS, build_model
from antgo.framework.helper.multi_stream_module import MultiSteamModule
from .losses.quality_focal_loss import QualityFocalLoss


@MODELS.register_module()
class DenseTeacher(MultiSteamModule):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None, init_cfg=None):
        # 默认使用teacher模型作为最佳模型
        if test_cfg is None:
            test_cfg = dict()
        test_cfg.update(
            {
                'inference_on': 'teacher'
            }
        )
        super(DenseTeacher, self).__init__(
            dict(
                teacher=build_model(model), 
                student=build_model(model)
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.freeze("teacher")
        self.key = train_cfg.get('key', 0)                                  # 挑选model进入loss前的featuremap list中的哪个作为半监督信号
        self.use_sigmoid = train_cfg.get('use_sigmoid', True)               # 是否将挑选出来的featuremap 使用sigmoid 

        self.label_batch_size = train_cfg.get('label_batch_size', 5)        # 有标签数据量 在一个batch里
        self.unlabel_batch_size = train_cfg.get('unlabel_batch_size', 3)    # 无标签数据量 在一个batch里

        self.semi_ratio = train_cfg.get('semi_ratio',0.5)                   # 挑选出绝对负样本的最大数量
        self.heatmap_n_thr = train_cfg.get('heatmap_n_thr', 0.25)           # heatmap中，小于此值视为绝对负样本，其余位置通过半监督信号监督
        self.semi_loss_w = train_cfg.get('semi_loss_w', 1.0)                # 半监督损失的权重

    def _get_unsup_dense_loss(self, student_heatmap_, teacher_heatmap_):
        # student_heatmap_: N,C,H,W
        # teacher_heatmap_: N,C,H,W
        num_classes = student_heatmap_.shape[1]
        student_heatmap = student_heatmap_.permute(0, 2, 3, 1).reshape(-1, num_classes)
        teacher_heatmap = teacher_heatmap_.permute(0, 2, 3, 1).reshape(-1, num_classes)
        if self.use_sigmoid:
            student_heatmap = student_heatmap.sigmoid()
            teacher_heatmap = teacher_heatmap.sigmoid()

        with torch.no_grad():
            max_vals = torch.max(teacher_heatmap, 1)[0]
            count_num = int(teacher_heatmap.size(0) * self.semi_ratio)
            sorted_vals, sorted_inds = torch.topk(max_vals, teacher_heatmap.size(0))

            for sorted_n in range(count_num):
                if sorted_vals[sorted_n] < self.heatmap_n_thr:
                    count_num = max(1, sorted_n)
                    break

            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.0
            fg_num = sorted_vals[:count_num].sum()

        loss_heatmap = (
            QualityFocalLoss(
                student_heatmap,
                teacher_heatmap,
                weight=mask,
                reduction="mean",
            )
        )

        return {"loss_heatmap": loss_heatmap}

    def forward_train(self, image, image_meta, **kwargs):
        label_images, unlabel_weak_strong_images = \
            torch.split(image, [self.label_batch_size, self.unlabel_batch_size+self.unlabel_batch_size],dim=0) 
        label_metas = image_meta[:self.label_batch_size]

        unlabel_weak_images=torch.index_select(unlabel_weak_strong_images,dim=0,index=torch.tensor([i for i in range(0,2*self.unlabel_batch_size,2)]))
        unlabel_strong_images=torch.index_select(unlabel_weak_strong_images,dim=0,index=torch.tensor([i for i in range(1,2*self.unlabel_batch_size,2)]))

        unlabel_weak_strong_metas = image_meta[self.label_batch_size:]
        unlabel_weak_metas = [unlabel_weak_strong_metas[i] for i in range(0, 2*self.unlabel_batch_size, 2)]
        unlabel_strong_metas = [unlabel_weak_strong_metas[i] for i in range(1, 2*self.unlabel_batch_size, 2)]
        
        # ignore epoch,iter key in kwargs
        if 'epoch' in kwargs:
            kwargs.pop('epoch')
        if 'iter' in kwargs:
            kwargs.pop('iter')
        
        label_kwargs = {}
        unlabel_weak_kwargs = {}
        unlabel_strong_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v_label_kwargs, v_unlabel_weak_strong_kwargs = \
                    torch.split(v, [self.label_batch_size, self.unlabel_batch_size+self.unlabel_batch_size],dim=0) 
            elif isinstance(v, list):
                v_label_kwargs = v[:self.label_batch_size]
                v_unlabel_weak_strong_kwargs = v[self.label_batch_size:]
            else:
                # ignore other
                continue
            
            label_kwargs[k] = v_label_kwargs

            if isinstance(v, torch.Tensor):
                v_unlabel_weak_kwargs=torch.index_select(v_unlabel_weak_strong_kwargs,dim=0,index=torch.tensor([i for i in range(0,2*self.unlabel_batch_size,2)]))
                v_unlabel_strong_kwargs=torch.index_select(v_unlabel_weak_strong_kwargs,dim=0,index=torch.tensor([i for i in range(1,2*self.unlabel_batch_size,2)]))
            else:
                v_unlabel_weak_kwargs = v_unlabel_weak_strong_kwargs[:self.unlabel_batch_size]
                v_unlabel_strong_kwargs = v_unlabel_weak_strong_kwargs[self.unlabel_batch_size:]
            
            unlabel_weak_kwargs[k] = v_unlabel_weak_kwargs
            unlabel_strong_kwargs[k] = v_unlabel_strong_kwargs

        losses = {}
        # 有监督损失
        output_dict = self.student.forward_train(label_images, label_metas, **label_kwargs)
        assert(isinstance(output_dict, dict))
        output_dict = {"labeled_" + k: v for k, v in output_dict.items()}
        losses.update(**output_dict)

        # 无监督损失
        # 应该返回 heatmap
        for k in unlabel_strong_kwargs.keys():
            unlabel_strong_kwargs[k] = None
            unlabel_weak_kwargs[k] = None

        output_unsup_strong = self.student.forward_train(unlabel_strong_images, unlabel_strong_metas, **unlabel_strong_kwargs)
        strong_heatmap = None
        if isinstance(output_unsup_strong, dict):
            strong_heatmap = output_unsup_strong[self.key]
        else:
            strong_heatmap = output_unsup_strong[int(self.key)]

        # 应该返回 heatmap
        self.teacher.eval()
        output_unsup_weak = self.teacher.forward_train(unlabel_weak_images, unlabel_weak_metas, **unlabel_weak_kwargs)
        weak_heatmap = None
        if isinstance(output_unsup_weak, dict):
            weak_heatmap = output_unsup_weak[self.key]
        else:
            weak_heatmap = output_unsup_weak[int(self.key)]

        unsup_loss = self._get_unsup_dense_loss(
            strong_heatmap,
            weak_heatmap
        )

        unsup_loss = {"unlabeled_" + k: v*self.semi_loss_w for k, v in unsup_loss.items()}
        losses.update(**unsup_loss)

        return losses
    
    def simple_test(self, image, image_meta, rescale=True, **kwargs):
        return self.teacher(image, image_meta, rescale, **kwargs)

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