# 简明教程

## 一句话运行模型

## 运行自己的模型

## 辅助工具
### 模型DEMO创建
```
    import sys
    import numpy as np

    from antgo.interactcontext import InteractContext
    ctx = InteractContext()

    ctx.demo.start("b_exp", config={
            'support_user_upload': True,
            'support_user_input': False,
    })
    for data, info in ctx.demo.running_dataset.iterator_value():
        # 在此处接受传入的数据，并使用模型处理

        print(data)

        # 将模型运行结果返回
        ctx.recorder.record({
            'id': info['id'],
            'score': {
                'data': np.random.randint(0,255,(255,255), dtype=np.uint8),
                'type': 'IMAGE'
            },
            'description': {
                'data': 'hello the world',
                'type': 'STRING'
            }
        })
        print('finish')
    ctx.demo.exit()

```
### 数据查验

```
    import sys
    import numpy as np

    from antgo.interactcontext import InteractContext
    ctx = InteractContext()

    # 创建浏览服务，默认会开启本地http服务
    # tags: 设置查验标签
    # white_users: 设置允许参与查验的用户信息
    ctx.browser.start("b_exp", config = {
            'tags': ['hello', 'world'],
            'white_users': {'jian@baidu.com': {'password': '112233'}},
        })

    # 导入数据
    for id in range(10):
        ctx.recorder.record({
            'id': id,
            'score': {
                'data': np.random.randint(0,255,(255,255), dtype=np.uint8),
                'type': 'IMAGE'
            },
            'description': {
                'data': f'hello the world {id}',
                'type': 'STRING'
            }
        })
    
    # 只有在web页面上，对所有数据检查完备后返回
    ctx.browser.waiting()

    # 下载检查结果
    content = ctx.browser.download()
    print(content)
    ctx.browser.exit()
```

### 数据标注工具

这是主动学习的标注模块。
```
    import sys
    import numpy as np

    from antgo.interactcontext import InteractContext
    ctx = InteractContext()

    # 需要在这里设置标注配置信息，如
    # category: 设置类别信息
    # white_users: 设置允许参与的标注人员信息
    # label_type: 设置标注类型，目前仅支持'RECT','POINT','POLYGON'
    ctx.activelearning.start("b_exp", config={
            'metas':{
                'category': [
                    {
                        'class_name': 'A',
                        'class_index': 0,
                    },
                    {
                        'class_name': 'B',
                        'class_index': 1,           
                    }
                ]
            },
            'white_users': {'jian@baidu.com': {'password': '112233'}},
            'label_type': 'RECT',   # 设置标注类型，'RECT','POINT','POLYGON'
        }
    )

    # 切换到标注状态
    ctx.activelearning.labeling()

    # 动态添加需要标注的样本
    for i in range(10):
        # 添加等待标注样本
        ctx.recorder.record(
            {
                'id': i,
                'image': np.random.randint(0,255,(255,255), dtype=np.uint8)
            }
        )

    # 下载当前标注结果（后台等待标注完成后，再下载）
    result = ctx.activelearning.download()

    # 切换到等待状态
    ctx.activelearning.waiting()

    # 全局结束
    ctx.activelearning.exit()
```