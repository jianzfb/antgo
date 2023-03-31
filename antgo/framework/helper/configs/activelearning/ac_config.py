# 模型配置
model = dict(
    type='ACModule',
    test_cfg=dict(
        feature_config=dict(
            from_name='backbone',
            from_index=-1
        ),
    ),
    train_cfg=None,
    init_cfg=dict(
        sampling_num=1000,        # 采样总数
        sampling_count=1,
        sampling_fn=dict(
            type='SamplingByComposer',
            sampling_cfgs=[
                dict(
                    type='SamplingByUniform',
                    sampling_num=500                # 随机采样总数
                ),
                dict(
                    type='SamplingByKCenter',
                    sampling_num=50,                # 聚类采样总数
                    key='feature'
                )
            ]
        ),             
    ),
)


# 数据配置
data=dict(
    test=dict(
        type="TFDataset",
        data_path_list=[],
        pipeline=[
            dict(type='DecodeImage', to_rgb=True),
            dict(type='Meta', keys=['image_file', 'tag']),
            dict(type='ResizeS', target_dim=[448,256]),                                                         
            dict(type='INormalize', mean=[0.0], std=[255.0],to_rgb=True, keys=['image']),
            dict(type='IImageToTensor', keys=['image']),            
        ],
        inputs_def={
            'fields': ["image", 'bboxes', 'labels', 'image_meta']
        },
        shuffle_queue_size=0,           # 取消shuffle
        sample_num_equalizer=False      # 取消样本数多卡均衡
    ),
    test_dataloader=dict(
        samples_per_gpu=2, 
        workers_per_gpu=1,
        drop_last=False,
        shuffle=False,
        ignore_stack=['bboxes', 'labels', 'image_meta']
    )
)