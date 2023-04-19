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
model = dict(
    type='TTFNet',
    backbone=dict(
        type='ResNet',
            depth=50, 
            in_channels=3, 
            out_indices=[3,2,1]),      
    neck=dict(type="FPN", in_channels=[512, 1024, 2048], out_channels=32, num_outs=1),
    bbox_head=dict(
        type='FcosHead',
        in_channel=32,
        feat_channel=32,
        num_classes=4,
        down_stride=4,
        score_thresh=0.2,
        train_cfg=None,
        test_cfg=dict(topk=100, local_maximum_kernel=3),
        loss_ch=dict(type='GaussianFocalLoss', loss_weight=1.0)
    ),  
)

# 描述模型基本信息
info = dict(
    backbone=dict(
        channels=[512, 1024, 2048],
        shapes=[32,16,8]
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='./output/')       

# 数据配置
data=dict(
    train=dict(
        type='VisalSO',
        train_or_test='train',
        dir='./visalso_dataset',
        pipeline=[
            dict(type="ResizeByShort", short_size=256),
            dict(type="RandomScaledCrop", target_size=(256,256)),
            dict(type="Rotation", degree=15),
            dict(type='ResizeS', target_dim=[256,256]),
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
        samples_per_gpu=4, 
        workers_per_gpu=1,
        drop_last=True,
        shuffle=True,
        ignore_stack=['bboxes', 'labels', 'image_meta']
    ),
    val=dict(
        type='VisalSO',
        train_or_test='val',
        dir='./visalso_dataset',
        pipeline=[
            dict(type="ResizeByShort", short_size=256),
            dict(type='FixedCrop', target_size=(256,256), align='center'),
            dict(type='ResizeS', target_dim=[256,256]),
            dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
            dict(type='IImageToTensor', keys=['image'])
        ],
        inputs_def=dict(
            fields= ["image", 'bboxes', 'labels', 'image_meta']
        )
    ),
    val_dataloader=dict(
        samples_per_gpu=4, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
        ignore_stack=['bboxes', 'labels', 'image_meta']
    ),
    test=dict(
        type='VisalSO',
        train_or_test='test',
        dir='./visalso_dataset',
        pipeline=[
            dict(type="ResizeByShort", short_size=256),
            dict(type='FixedCrop', target_size=(256,256), align='center'),
            dict(type='ResizeS', target_dim=[256,256]),
            dict(type='INormalize', mean=[128.0,128.0,128.0], std=[128.0,128.0,128.0],to_rgb=False, keys=['image']),
            dict(type='IImageToTensor', keys=['image'])
        ],
        inputs_def=dict(
            fields= ["image", 'bboxes', 'labels', 'image_meta']
        )
    ),
    test_dataloader=dict(
        samples_per_gpu=4, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
        ignore_stack=['bboxes', 'labels', 'image_meta']
    )
)

# 评估方案配置
evaluation=dict(
    out_dir='./output/', 
    interval=1, 
    metric=dict(
        type='COCOCompatibleEval', 
        categories=[{'name': f'{label}', 'id': label} for label in range(4)]
    ), 
    save_best='AP@[ IoU=0.50:0.95 | area= all | maxDets=100 ]',
    rule='greater'
)

# 导出配置
export=dict(
    input_shape_list = [[1,3,256,256]],
    input_name_list=["image"],
    output_name_list=["heatmap", "offset"]
)

max_epochs = 60