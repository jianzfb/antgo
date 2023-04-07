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
        sampling_num=5000,                          # 采样总数
        sampling_count=5,
        sampling_fn=dict(
            type='SamplingByComposer',
            sampling_cfgs=[
                dict(
                    type='SamplingByUniform',
                    sampling_num=10000              # 随机采样总数 (第一步，随机采样10000)
                ),
                dict(
                    type='SamplingByKCenter',
                    sampling_num=5000,              # 聚类采样总数 (在第二步，从上一步子集中采样5000)
                    key='feature'
                )
            ]
        ),             
    ),
)
