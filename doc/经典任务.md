# 经典任务入手指南

## 分类任务实验
```
# 第一步 创建mvp代码（Cifar10分类任务训练）
antgo create mvp --name=cifar10

# 第二步 开始训练（使用GPU 0）
# 训练完成后，top-1指标～0.95
python3 ./cifar10/main.py --exp=cifar10 --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/cifar10/output/checkpoint下你将获得checkpoint epoch_1500.pth

# 第四步 导出onnx模型
python3 ./cifar10/main.py --exp=cifar10 --checkpoint=./output/cifar10/output/checkpoint/epoch_1500.pth --process=export
```

## LSP关键点任务实验

```
# 数据准备
# 推荐直接使用本框架共享常规数据集（基于阿里云盘提供支持）
# 运行如下命令后，将在控制台显示阿里云盘授权二维码，授权后，对应数据集将分享至开发者的阿里云盘内
# 在模型训练时，自动从阿里云盘下载对应数据。
antgo share data --name=lsp

# 第一步 创建mvp代码
antgo create mvp --name=lsp

# 第二步 开始训练（使用GPU 0）
# 训练完成后，OKS指标～0.918
python3 ./lsp/main.py --exp=lsp --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/lsp/output/checkpoint下你将获得checkpoint epoch_600.pth

# 第四步 导出onnx模型
python3 ./lsp/main.py --exp=lsp --checkpoint=./output/lsp/output/checkpoint/epoch_600.pth --process=export
```

## COCO检测任务实验

```
# 数据准备
# 推荐直接使用本框架共享常规数据集（基于阿里云盘提供支持）
# 运行如下命令后，将在控制台显示阿里云盘授权二维码，授权后，对应数据集将分享至开发者的阿里云盘内
# 在模型训练时，自动从阿里云盘下载对应数据。
antgo share data --name=coco

# 第一步 创建mvp代码
antgo create mvp --name=coco

# 第二步 开始训练（使用GPU 0）
# 训练完成后，AP(IoU=0.50:0.95)~0.31
python3 ./coco/main.py --exp=coco --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/coco/output/checkpoint下你将获得checkpoint epoch_60.pth

# 第四步 导出onnx模型
python3 ./coco/main.py --exp=coco --checkpoint=./output/coco/output/checkpoint/epoch_60.pth --process=export
```

## VOC分割任务实验

```
# 数据准备
# 推荐直接使用本框架共享常规数据集（基于阿里云盘提供支持）
# 运行如下命令后，将在控制台显示阿里云盘授权二维码，授权后，对应数据集将分享至开发者的阿里云盘内
# 在模型训练时，自动从阿里云盘下载对应数据。
antgo share data --name=voc

# 第一步 创建mvp代码
antgo create mvp --name=pascal_voc

# 第二步 开始训练（使用GPU 0）
# 训练完成后，MIOU指标～0.543
python3 ./pascal_voc/main.py --exp=pascal_voc --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/pascal_voc/output/checkpoint下你将获得checkpoint epoch_600.pth

# 第四步 导出onnx模型
python3 ./pascal_voc/main.py --exp=pascal_voc --checkpoint=./output/pascal_voc/output/checkpoint/epoch_600.pth --process=export
```