# 优化器配置
optimizer = dict(type='SGD', lr=0.05,  weight_decay=5e-4, momentum=0.01, nesterov=True)
optimizer_config = dict(grad_clip=None)

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
)

# 日志配置
log_config = dict(
    interval=50,    
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='WideResNet',
        num_classes=10,     # TODO 28
        depth=28,
        widen_factor=2,     # TODO 8
        dropout=0,
        dense_dropout=0.2          
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
        dir='./cifar10-dataset',
        pipeline=[    
            dict(type='INumpyToPIL', keys=['image'], mode='RGB'),
            dict(type='RandomHorizontalFlip', keys=['image']),
            dict(type='RandomCrop', size=(32,32), padding=int(32*0.125), fill=128, padding_mode='constant', keys=['image']), 
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def={'fields': ['image', 'label']}     
    ),
    train_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=True,
        shuffle=True,
    ),
    val=dict(
        type="Cifar10",
        train_or_test='test',
        dir='./cifar10-dataset',
        pipeline=[    
            dict(type='INumpyToPIL', keys=['image'], mode='RGB'),                  
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def={'fields': ['image', 'label']}  
    ),
    val_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
        drop_last=False,
        shuffle=False,
    ),
    test=dict(
        type="Cifar10",
        train_or_test='test',
        dir='./cifar10-dataset',
        pipeline=[    
            dict(type='INumpyToPIL', keys=['image'], mode='RGB'),                  
            dict(type='ToTensor', keys=['image']),
            dict(type='Normalize', mean=(0.491400, 0.482158, 0.4465231), std=(0.247032, 0.243485, 0.2615877), keys=['image']),
        ],
        inputs_def={'fields': ['image', 'label']}  
    ),
    test_dataloader=dict(
        samples_per_gpu=128, 
        workers_per_gpu=2,
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

max_epochs = 1500