# 优化器配置
optimizer = dict(type='SGD', lr=0.05,  weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
)

# 日志配置
log_config = dict(
    interval=5,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 自定义HOOKS
custom_hooks = [
    dict(
        type='EMAHook'
    )
]

# 模型配置
model = dict(
    type='KeypointNet',
    backbone=dict(
        type='KetNetF',
        architecture='resnet34',
        in_channels=3
    ),
    head=dict(
        type='SimpleQNetLite0PoseHeatmap2D',
        num_deconv_filters=[128, 128, 128],
        num_joints=14
    )
)

# 描述模型基本信息
info = dict(
    backbone=dict(
        channels=[96,128,160],
        shapes=[16,8,4]
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict(
        type='LSP',
        train_or_test='train',
        dir='./lsp_dataset',
        pipeline=[
            dict(type="ConvertRandomObjJointsAndOffset", input_size=(128,128), heatmap_size=(16,16), num_joints=14),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'bboxes'],
        )
    ),
    train_dataloader=dict(
        samples_per_gpu=32, 
        workers_per_gpu=2,
        drop_last=True,
        shuffle=True,
    ),
    val=dict(
        type='LSP',
        train_or_test='val',
        dir='./lsp_dataset',
        pipeline=[
            dict(type="ConvertRandomObjJointsAndOffset", input_size=(128,128), heatmap_size=(16,16), num_joints=14, with_random=False),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'bboxes'],
        )
    ),
    val_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    ),
    test=dict(
        type='LSP',
        train_or_test='test',
        dir='./lsp_dataset',
        pipeline=[
            dict(type="ConvertRandomObjJointsAndOffset", input_size=(128,128), heatmap_size=(16,16), num_joints=14, with_random=False),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def=dict(
            fields=['image', 'heatmap', 'offset_x', 'offset_y', 'heatmap_weight', 'joints_vis', 'joints2d', 'bboxes'],
        )
    ),
    test_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    )
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='OKS'), save_best='oks', rule='greater')

# 导出配置
export=dict(
    input_shape_list = [[1,3,128,128]],
    input_name_list=["image"],
    output_name_list=["heatmap", "offset"]
)

max_epochs = 600