# 优化器配置
optimizer = dict(type='SGD', lr=0.01,  weight_decay=5e-4, momentum=0.01, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=10))

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
    type='DenseTeacher',
    train_cfg=dict(
        key=0,
        use_sigmoid=True,
        label_batch_size=128,
        unlabel_batch_size=16,
        semi_ratio=0.5,
        heatmap_n_thr=0.25,
        semi_loss_w=1.0
    ),
    test_cfg=None,
    init_cfg=None,
)

# 自定义 hooks配置
custom_hooks = dict(
    type='MeanTeacher'
)

# 数据配置
data=dict(
    train=[
        dict(
            type="TFDataset",
            data_path_list=[],
            pipeline=[
                dict(type='DecodeImage', to_rgb=True),
                dict(type="KeepRatio", aspect_ratio=1.77),
                dict(type="Rotation", degree=15, border_value=128),            
                dict(type='ResizeS', target_dim=[448,256]),            
                dict(type='ColorDistort'),       
                dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
                dict(type='IImageToTensor', keys=['image']),            
            ],
            inputs_def={
                'fields': ["image", 'bboxes', 'labels', 'image_meta']
            },
            shuffle_queue_size=2048
        ),
        dict(
            type="TFDataset",
            data_path_list=[],
            weak_pipeline=[
                dict(type='DecodeImage', to_rgb=True),
                dict(type="KeepRatio", aspect_ratio=1.77),
                dict(type='ResizeS', target_dim=[448,256]),     
            ],
            strong_pipeline=[
                dict(type='ColorDistort'),       
            ],
            pipeline=[
                dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
                dict(type='IImageToTensor', keys=['image']),            
            ],
            inputs_def={
                'fields': ["image", 'bboxes', 'labels', 'image_meta']
            },
            shuffle_queue_size=2048
        ),            
    ],
    train_dataloader=dict(
        samples_per_gpu=[128,16], 
        workers_per_gpu=4,
        drop_last=True,
        shuffle=True,
        ignore_stack=['bboxes', 'labels', 'image_meta']
    )
)