# 优化器配置
optimizer = dict(
    teacher=dict(type='SGD', lr=5e-2,  weight_decay=5e-4, momentum=0.01, nesterov=True),
    student=dict(type='SGD', lr=5e-2,  weight_decay=5e-4, momentum=0.01, nesterov=True)
)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
)

# 模型配置
model = dict(
    type='MPL',
    model=dict(
        teacher=None,
        student=None
    ),
    train_cfg=dict(
        use_sigmoid=True,
        label_batch_size=2,     # 128
        temperature=0.7,
        lambda_u=8.0,
        uda_steps=5000,
        grad_clip=1e9,
        ema=0,                  # 0.995
        threshold=0.6,
        label_smoothing=0.15
    ),
    test_cfg=None,
    init_cfg=None
)

# 数据配置
data=dict(
    train=[
        dict(
            type="TFDataset",
            data_path_list=[],
            pipeline=[    
                dict(type='INumpyToPIL', keys=['image'], mode='RGB'),
                dict(type='RandomHorizontalFlip', keys=['image']),
                dict(type='RandomCrop', size=(32,32), padding=int(32*0.125), fill=128, padding_mode='constant', keys=['image']), 
                dict(type='ToTensor', keys=['image']),
                dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            ],
            inputs_def={'fields': ['image', 'label']},
            description={'image': 'numpy', 'label': 'int'}
        ),
        dict(
            type="TFDataset",
            data_path_list=[],
            weak_pipeline=[
                dict(type='INumpyToPIL', keys=['image'], mode='RGB'),
                dict(type='RandomHorizontalFlip', keys=['image']),
                dict(type='RandomCrop', size=(32,32), padding=int(32*0.125), fill=128, padding_mode='constant', keys=['image']),                 
            ],
            strong_pipeline=[
                dict(type='RandAugmentCIFAR', n=2, m=6)
            ],
            pipeline=[    
                dict(type='ToTensor', keys=['image']),
                dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
            ],
            inputs_def={'fields': ['image', 'label']},
            description={'image': 'numpy', 'label': 'int'}
        )
    ],
    train_dataloader=dict(
        samples_per_gpu=[2,2*7],    # 按照开源代码参数谁 [128,128*7]
        workers_per_gpu=1,
        drop_last=True,
        shuffle=True,
    ), 
)

max_epochs = 2