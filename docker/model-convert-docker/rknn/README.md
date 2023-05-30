```
# 将当前路径挂接到/workspace，作为转换任务的默认工作目录
# 浮点模型转换任务
docker run -v $(pwd):/workspace rknnconvert bash convert.sh --i=xxx.onnx --o=yyy --device=rk3568 --mean-values=0,0,0 --std-values=255,255,255

# 量化模型转换任务
docker run -v $(pwd):/workspace rknnconvert bash convert.sh --i=xxx.onnx --o=yyy --image-folder=image_folder --quantize --device=rk3568 --mean-values=0,0,0 --std-values=255,255,255
```