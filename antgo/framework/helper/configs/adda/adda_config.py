# 优化器
optimizer = dict(
    model=dict(),                   # 由具体下游任务定义
    discriminator=dict(
        type='SGD', lr=0.001, weight_decay=1e-4, momentum=0.9, nesterov=True
    )
)

# 学习率
lr_config = dict(
    model=dict(),                   # 由具体下游任务定义
    discriminator=dict(
        policy='CosineAnnealing',
        min_lr=1e-5,
    )
)

# 模型配置
model = dict(
    type='Adda',
    model=dict(),              # 由具体下游任务定义    
    train_cfg=dict(
        input_dim=1024,
        latent_dim=512,
        src_domain_batch_size=5,
        gen_loss_weight=1.0,
        dis_loss_weight=1.0,
        encoder_config=dict(
            from_name='backbone',
            from_index=-1
        )
    )
)