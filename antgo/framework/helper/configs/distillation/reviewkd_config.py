# 优化器配置
optimizer = dict(type='SGD', lr=0.01,  weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=20))

# 学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
)

# 模型配置文件标准格式
# info = dict(
#     backbone=dict(
#         channels=[96,128,160],
#         shapes=[16,8,4]
#     ),
#     neck=dict(
#         channels=[],
#         shapes=[]
#     )
# )

# 模型配置
# model 字段根据具体模型进行设置
model = dict(
    type='ReviewKD',
    model=dict(
        teacher=None,   # placeholder
        student=None,   # placeholder  
    ),    
    train_cfg=dict(
        student=dict(
            channels=[96,128,160],
            shapes=[16,8,4]
        ),
        teacher=dict(
            channels=[512, 1024, 2048],
            shapes=[16,8,4]
        ),
        kd_loss_weight=1.0,
        kd_warm_up=20.0,
        align = 'backbone'
    ),
    test_cfg=None,
    init_cfg=None,
)
