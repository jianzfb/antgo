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
    type='TTFNet',
    backbone=dict(type='ResNet', depth=50, in_channels=1, out_indices=[3,2,1]),      
    neck=dict(type="FPN", in_channels=[512, 1024, 2048], out_channels=32, num_outs=1),
    bbox_head=dict(
        type='FcosHead',
        in_channel=32,
        feat_channel=32,
        num_classes=3,
        down_stride=4,
        img_width=320,
        img_height=256,
        rescale=(640/320, 480/256),
        score_thresh=0.0,
        train_cfg=None,
        test_cfg=dict(topk=100, local_maximum_kernel=3)                
    ),    
)

# checkpoint配置
checkpoint_config = dict(interval=1, out_dir='hdfs://haruna/home/byte_pico_zhangjian52_hdfs/data/AABBCCDD')       

# 数据配置
data=dict(
    train=dict( 
            type="TFDataset",
            data_path_list=[
                '/root/workspace/A/dettest-00000-of-00003-tfrecord',
                '/root/workspace/A/dettest-00001-of-00003-tfrecord',
                '/root/workspace/A/dettest-00002-of-00003-tfrecord'
            ],
            description={'image': 'byte', 'gt_bbox': 'numpy', 'gt_class': 'int'},
            shuffle_queue_size=2,
            pipeline=[    
                dict(type='DecodeImage', to_rgb=False, ),
                dict(type='ResizeExt', target_dim=[320,256]),                                                         
                dict(type='Normalize', mean=[0.0], std=[255.0],to_rgb=False),
                dict(type='ImageToTensor', keys=['image']),
            ],        
    ),
    train_dataloader=dict(
        samples_per_gpu=3, 
        workers_per_gpu=2,
        drop_last=True,
        shuffle=False,
    )
)

# 评估方案配置
evaluation=dict(out_dir='./out', interval=1, metric=dict(type='PicoBoxEval'))

# 导出配置
export=dict(
    input_shape_list = [],
    input_name_list=[],
    output_name_list=[]
)

max_epochs = 2