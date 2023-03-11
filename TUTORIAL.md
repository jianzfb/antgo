# 简明教程
## 快速开始


## 常用工具
### 转换视频数据到标准训练格式
```
antgo tool extract/videos --src=video-folder-path --tgt=target-folder --frame-rate=15 
```
### 转换松散数据到标准训练格式
```
antgo tool extract/images --src=image-folder-path --tgt=target-folder

// 如果同时想要指定文件名前缀，后缀，扩展名进行过滤，可以如下，
antgo tool extract/images --src=image-folder-path --tgt=target-folder --prefix=prefix --suffix=suffix --ext=ext
```
### 从第三方提供的json/txt标注文件中随机采样
```
// 用于查看GT格式
antgo tool extract/samples --src=json-path --num=1 --feedback
```
### 从baidu/bing/vcg下载图像/视频
```
// --tags 用于指定下载关键词； type:image,keyword:aa/bb 标识下载图像，关键词是aa,bb
antgo tool download/baidu --tags=type:image,keyword:dog/cat
antgo tool download/bing --tags=type:image,keyword:dog/cat
antgo tool download/vcg --tags=type:image,keyword:dog/cat
```

### 浏览样本数据 (仅支持标准GT格式)
```
antgo tool browser/images --src=json-path 
// 如果想要在浏览数据时，给样本快速打标签（一般用于审核数据时使用）
// 使用--tags来设置标签
antgo tool browser/images --src=json-path --tags=valid,invalid --feedback
```

### 使用标签过滤样本（仅支持标准GT格式）
```
// 使用来自于数据浏览服务获得的样本标签记录，进行过滤
// --tags 指定需要含有的标签
// --no-tags 指定不可以包含的标签
antgo tool filter/tags --src=json-path --tgt=from-browser-json --tags=tags --no-tags=no-tags
```

### 标注工具集成（仅支持标准GT格式）
```
// 转换label-studio标注工具结果到标准GT格式
// 如果有必要，加入--prefix来将路径加上子目录
antgo tool label/studio --src=json-path --tgt=target-folder --tags=lefthand:0,righthand:1 --from
// 将标准GT格式转换到label-studio标注工具结果
antgo tool label/studio --src=json-path --tgt=target-folder --to

// 启动标注服务
// --type 支持RECT,POLYGON
// --tags 设置类型标签，例如Car:0,Train:1
antgo tool label/start --src=json-path --tgt=target-folder --tags=xxx --type=RECT 
```

### 打包tfrecord/kv数据（仅支持标准GT格式）
```
// tfrecord
// --src  json文件地址， --tgt 打包后存放地址，--prefix 打包后文件前缀设置，--num 打包后每个文件样本数
antgo tool package/tfrecord --src=json-path --tgt=target-folder --prefix=xxx --num=50000

// kv
// --src  json文件地址， --tgt 打包后存放地址，--prefix 打包后文件前缀设置，--num 打包后每个文件样本数
antgo tool package/kv --src=json-path --tgt=target-folder --prefix=xxx --num=50000
```

## 项目管理
### 创建项目
### 创建项目实验
### 项目模型自优化流水线
#### 创建Baseline模型
#### 添加Teacher模型
#### 创建数据自动标注服务
#### 创建模型半监督服务
#### 创建模型蒸馏服务


## 进阶功能
### 创建模型DEMO服务
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
### 创建数据查看

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

### 创建数据标注服务

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