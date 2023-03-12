# 优化器配置
optimizer = dict(type='Adam', lr=0.001,  weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    step=[40,45])             # 25, 35; 15, 25 ; 5,10

# 日志配置
log_config = dict(
    interval=1,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='WideResNet',
        num_classes=10,
        depth=28,
        widen_factor=2,
        dropout=0,
        dense_dropout=0          
    ),
    head=dict(
        type='ClsHead'
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict( 
        type="Cifar10",
        train_or_test='train',
        dir='./dataset',
        pipeline=[    
            dict(type='Normalize', mean=[0.0], std=[255.0],to_rgb=False),
            dict(type='ImageToTensor', keys=['image']),
        ],
        inputs_def={'fields': ['image', 'label']}     
    ),
    train_dataloader=dict(
        samples_per_gpu=2, 
        workers_per_gpu=1,
        drop_last=True,
        shuffle=True,
    ),
    val=dict(
        type="Cifar10",
        train_or_test='test',
        dir='./dataset',
        pipeline=[    
            dict(type='Normalize', mean=[0.0], std=[255.0],to_rgb=False),
            dict(type='ImageToTensor', keys=['image']),
        ],
        inputs_def={'fields': ['image', 'label']}  
    ),
    val_dataloader=dict(
        samples_per_gpu=2, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    ),
    test=dict(
        type="Cifar10",
        train_or_test='test',
        dir='./dataset',
        pipeline=[    
            dict(type='Normalize', mean=[0.0], std=[255.0],to_rgb=False),
            dict(type='ImageToTensor', keys=['image']),
        ],
        inputs_def={'fields': ['image', 'label']}  
    ),
    test_dataloader=dict(
        samples_per_gpu=2, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
    ),    
)

# 评估方案配置
evaluation=dict(out_dir='./output/', interval=1, metric=dict(type='AccuracyEval'), save_best='top_1')

# 导出配置
export=dict(
    input_shape_list = [[1,3,32,32]],
    input_name_list=["image"],
    output_name_list=["pred"]
)

max_epochs = 2