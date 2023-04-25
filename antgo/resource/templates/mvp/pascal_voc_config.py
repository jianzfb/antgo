# 优化器配置
optimizer = dict(type='SGD', lr=0.05,  weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup_by_epoch=False,
    warmup_iters=2000,
    warmup='linear'    
)

# 日志配置
log_config = dict(
    interval=10,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 模型配置
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DDRKetNetF',
        in_channels=3
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=32,
        channels=32,
        out_channels=21,
        ignore_index=255
    )
)

# 描述模型基本信息
info = dict(
    backbone=dict(
        channels=[96,128,160],
        shapes=[32,16,8]
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict(
        type='Pascal2012',
        train_or_test='train',
        dir='./pascal2012_dataset',
        ext_params=dict(task_type='SEGMENTATION', aug=True),
        pipeline=[
            dict(type='Rotation'),
            dict(type='ResizeS', target_dim=(256,256)),
            dict(type='ColorDistort'),
            dict(type='RandomFlipImage', swap_ids=[], swap_labels=[]),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            dict(type='UnSqueeze', axis=0, keys=['segments'])
        ],
        inputs_def=dict(
            fields=['image', 'segments'],
        )
    ),
    train_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=True,
        shuffle=True,
    ),
    val=dict(
        type='Pascal2012',
        train_or_test='val',
        dir='./pascal2012_dataset',
        ext_params=dict(task_type='SEGMENTATION', aug=True),
        pipeline=[
            dict(type='ResizeS', target_dim=(256,256)),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            dict(type='UnSqueeze', axis=0, keys=['segments'])
        ],        
        inputs_def=dict(
            fields=['image', 'segments'],
        )
    ),
    val_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=False,
        shuffle=False,
    ),
    test=dict(
        type='Pascal2012',
        train_or_test='val',
        dir='./pascal2012_dataset',
        ext_params=dict(task_type='SEGMENTATION', aug=True),
        pipeline=[
            dict(type='ResizeS', target_dim=(256,256)),
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            dict(type='UnSqueeze', axis=0, keys=['segments'])
        ],                
        inputs_def=dict(
            fields=['image', 'segments'],
        )
    ),
    test_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=False,
        shuffle=False,
    )
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='SegMIOU', class_num=21), save_best='miou', rule='greater')

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,256]],
    input_name_list=["image"],
    output_name_list=["heatmap"]
)

max_epochs = 600