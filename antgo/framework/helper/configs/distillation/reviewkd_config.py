# 优化器配置
optimizer = dict(type='SGD', lr=0.01,  weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=20))

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
)

# 日志配置
log_config = dict(
    interval=1,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 模型配置
# model 字段根据具体模型进行设置
model = dict(
    type='ReviewKD',
    teacher=None
    student=None,
    train_cfg=dict(
        student=dict(
            in_channels=[96,128,160],
            shapes=[4,8,16]
        ),
        teacher=dict(
            out_channels=[512, 1024, 2048],
            shapes=[4,8,16]
        ),
        kd_loss_weight=1.0,
        kd_warm_up=20.0,
        align_module = 'backbone'
    ),
    test_cfg=None,
    init_cfg=None,
)

# 数据配置
data=dict(
    train=dict(
        type='COCO2017',
        train_or_test='train',
        dir='./coco_dataset',
        ext_params={'task_type': 'OBJECT-DETECTION'},
        pipeline=[
            dict(type="Rotation", degree=15),
            dict(type='ResizeS', target_dim=[128,128]),
            dict(type='ColorDistort', hue=[-5,5,0.3], saturation=[0.8,1.2,0.3], contrast=[0.8,1.2,0.3], brightness=[0.8,1.2,0.3]),
            dict(type='RandomFlipImage', swap_labels=[]),
            dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
            dict(type='IImageToTensor', keys=['image'])
        ],
        inputs_def=dict(
            fields = ["image", 'bboxes', 'labels', 'image_meta']
        )
    ),
    train_dataloader=dict(
        samples_per_gpu=1, 
        workers_per_gpu=1,
        drop_last=True,
        shuffle=True,
        ignore_stack=['bboxes', 'labels', 'image_meta']
    )
)
