# 经典任务入手指南

## 分类任务实验
```
# 第一步 创建mvp代码（Cifar10分类任务训练）
antgo create mvp --name=cifar10

# 第二步 开始训练（使用GPU 0）
python3 ./cifar10/main.py --exp=cifar10 --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/cifar10/output/checkpoint下你将获得checkpoint epoch_1500.pth
# 在测试集上的top-1指标约为0.95

# 第四步 导出onnx模型
python3 ./cifar10/main.py --exp=cifar10 --checkpoint=./output/cifar10/output/checkpoint/epoch_1500.pth --process=export
```

## 2D关键点任务实验

```
# 第一步 创建mvp代码
antgo create mvp --name=lsp

# 第二步 开始训练（使用GPU 0）
python3 ./lsp/main.py --exp=lsp --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/lsp/output/checkpoint下你将获得checkpoint epoch_60.pth

# 第四步 导出onnx模型
python3 ./lsp/main.py --exp=lsp --checkpoint=./output/lsp/output/checkpoint/epoch_60.pth --process=export
```

## 2D检测任务实验

```
# 第一步 创建mvp代码
antgo create mvp --name=visalso

# 第二步 开始训练（使用GPU 0）
python3 ./visalso/main.py --exp=visalso --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/visalso/output/checkpoint下你将获得checkpoint epoch_60.pth

# 第四步 导出onnx模型
python3 ./visalso/main.py --exp=visalso --checkpoint=./output/visalso/output/checkpoint/epoch_60.pth --process=export
```

## 分割任务实验

```
# 第一步 创建mvp代码
antgo create mvp --name=pascal_voc

# 第二步 开始训练（使用GPU 0）
python3 ./pascal_voc/main.py --exp=pascal_voc --gpu-id=0 --process=train

# 第三步 查看训练日志
# 在./output/pascal_voc/output/checkpoint下你将获得checkpoint epoch_60.pth

# 第四步 导出onnx模型
python3 ./pascal_voc/main.py --exp=pascal_voc --checkpoint=./output/pascal_voc/output/checkpoint/epoch_60.pth --process=export
```