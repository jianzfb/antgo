```
# 将当前路径挂接到/workspace，作为转换任务的默认工作目录
# 浮点模型转换任务
docker run -v $(pwd):/workspace snpeconvert bash convert.sh --i=xxx.onnx --o=yyy

# 量化模型转换任务
docker run -v $(pwd):/workspace snpeconvert bash convert.sh --i=xxx.onnx --o=yyy --quantize --data-folder=data_folder 
```