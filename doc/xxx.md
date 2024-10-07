## 简洁发布
### 发布模型DEMO
#### 基于DAG引擎创建数据处理管线
```
from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
import numpy as np

# 模型输出解码函数
def decode_process_func(
    heatmap_level_1, heatmap_level_2, heatmap_level_3, 
    offset_level_1, offset_level_2, offset_level_3):
    level_stride_list = [8,16,32]
    level_heatmap_list = [heatmap_level_1, heatmap_level_2, heatmap_level_3]
    level_offset_list = [offset_level_1, offset_level_2, offset_level_3]

    all_bboxes = []
    all_labels = []
    for level_i, level_stride in enumerate(level_stride_list):
        level_local_cls = level_heatmap_list[level_i][0,0]  # 仅抽取person cls channel
        level_offset = level_offset_list[level_i]
        
        height, width = level_local_cls.shape
        flatten_local_cls = level_local_cls.flatten()
        topk_inds = np.argsort(flatten_local_cls)[::-1][:100]
        topk_scores = flatten_local_cls[topk_inds]
        pos = np.where(topk_scores > 0.45)
        if pos[0].size == 0:
            continue
        
        topk_inds = topk_inds[pos]
        topk_scores = flatten_local_cls[topk_inds]
        
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).astype(np.float32)
        topk_xs = (topk_inds % width).astype(np.float32)
        
        local_reg = np.transpose(level_offset, [0,2,3,1])   # BxHxWx4
        local_reg = np.reshape(local_reg, [-1,4])
        topk_ltrb_off = local_reg[topk_inds]

        tl_x = (topk_xs * level_stride + level_stride//2 - topk_ltrb_off[:,0] * level_stride)
        tl_y = (topk_ys * level_stride + level_stride//2 - topk_ltrb_off[:,1] * level_stride)
        br_x = (topk_xs * level_stride + level_stride//2 + topk_ltrb_off[:,2] * level_stride)
        br_y = (topk_ys * level_stride + level_stride//2 + topk_ltrb_off[:,3] * level_stride)

        bboxes = np.stack([tl_x,tl_y,br_x,br_y, topk_scores], -1)
        labels =  np.array([0]*bboxes.shape[0])
        all_bboxes.append(bboxes)
        all_labels.append(labels)

    all_bboxes = np.concatenate(all_bboxes, 0)
    all_labels = np.concatenate(all_labels)

    return all_bboxes, all_labels


# 场景1：批量读取本地文件夹图像，处理结果保存到./output/文件夹中
glob['file_path']('./test/*.png').stream(). \
    image_decode['file_path', 'image'](). \
    resize_op['image', 'resized_image'](out_size=(512,384)). \
    preprocess_op['resized_image', 'preprocessed_image'](mean=(128,128,128), std=(128,128,128), permute=[2,0,1], expand_dim=True). \
    inference_onnx_op['preprocessed_image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')](onnx_path='coco-epoch_60-model.onnx', input_fields=["image"]). \
    runas_op[('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3'), ('box', 'label')](func=decode_process_func). \
    nms[('box', 'label'), ('filter_box','filter_label')](iou_thres=0.2). \
    plot_bbox[("resized_image", "filter_box", 'filter_label'), "out"](thres=0.2, color=[[0,0,255]], category_map={'0': 'person'}). \
    image_save['out', 'save'](folder='./output/'). \
    run()


# 场景2：逐帧读取视频，并将结果保存成视频
video_dc['image']('./data.mp4'). \
    resize_op['image', 'resized_image'](out_size=(512,384)). \
    preprocess_op['resized_image', 'preprocessed_image'](mean=(128,128,128), std=(128,128,128), permute=[2,0,1], expand_dim=True). \
    inference_onnx_op['preprocessed_image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')](onnx_path='coco-epoch_60-model.onnx', input_fields=["image"]). \
    runas_op[('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3'), ('box', 'label')](func=decode_process_func). \
    nms[('box', 'label'), ('filter_box','filter_label')](iou_thres=0.2). \
    plot_bbox[("resized_image", "filter_box", 'filter_label'), "out"](thres=0.2, color=[[0,0,255]], category_map={'0': 'person'}). \
    to_video("./output.mp4", width=512, height=384)


```

#### 基于DAG引擎创建WEB DEMO
```
# 创建web上下文（main.py）
# 设置web服务输入占位标识，file_path是输入字段并以此构建处理管线，最后通过select算子搜集需要返回的管线中的信息
# 通过管线叶节点demo的参数input, output分别设置输入和输出的占位标识的类型，前端网页将根据此创建HTML元素。
# 目前支持的前端元素类型包括，['image', 'video', 'text', 'slider', 'checkbox', 'select']
# 目前支持的交互绘制元素包括：point,line,rect,polygon
# point: interactive_points
# line: interactive_lines
# rect: interactive_rects
# polygon: interactive_polygon
def debug_image_and_polygons(image, polygons):
	if polygons is None:
		return image

	image = image.copy()
	for poly_points in polygons:
		poly_points = np.array(poly_points).astype(np.int32)
		cv2.fillPoly(image, [poly_points], (255,0,0))
	return image


with web['file_path'](name='demo') as handler:
	app = handler.image_decode['file_path', 'hello'](). \
	interactive_polygon('hello', 'polygons', num=2). \
	runas_op[('hello', 'polygons'), 'out'](func=debug_image_and_polygons). \
	select['hello', 'out'](). \
	demo(
		title="我的web页面",
		description="这是一个测试", 
		input=[
			{'data': 'file_path', 'type': 'image'},
		], 
		output=[
			{'data': 'hello', 'type': 'image'},
			{'data': 'out', 'type': 'image'}
		])
# 注意：antgo web --main=main:app --port=8080 开启服务
```

#### 基于DAG引擎创建API SERVER
```
# 创建api服务上下文（main.py）
# 设置api服务输入占位标识，input是输入字段并以此构建处理管线，最后通过select算子搜集需要返回的管线中的信息
# 由于这里通过image_base64_decode来转换接收数据并将其转换成图像，所以这个api服务接受的数据是二进制图像的base64编码
def out_func(image):
    return 'hello the world'

with api['input'](name='serve') as handler:
    app=handler.image_base64_decode['input', 'image'](). \
        runas_op['image', 'out_str'](func=out_func). \
        select['out_str'](). \
        serve()

# 注意：antgo web --main=main:app --port=8080 开启服务

# 现在可以通过post请求，发起服务调用，注意上传的参数
# with open('xxx.png', 'rb') as fp:
#    image_content = fp.read()
#    image_content_base64 = base64.b64encode(image_content)
#
# request_json={
#    'input': image_content_base64.decode('UTF-8')
# }
# request_json_str = json.dumps(request_json)
# result = requests.post('http://127.0.0.1:8000/serve',request_json_str)
```

### 发布模型部署
#### 基于DAG引擎创建算子流水线编译部署包（使用eagleeye扩展包）

部署包编译后，将在当前文件夹下生成deploy/{project}_plugin 部署包
```
# 创建管线并编译
# 注意：计算管线搭建须知
# 支持的管线算子，包括四种类型，
# (1) 用户自定义的C++算子，通过deploy.xxx 引入（xxx为用户实现的c++函数名）
# (2) eagleeye核心算子(来自于eagleeye库)，通过eagleeye.xxx 引入 (xxx为算子名称)
# (3) xxx_op python端算子，通过独立实现c++算子实现一致性
# (4) inference_onnx_op，这是模型预测算子，在c++端已经实现完整的绑定，在编译时，将自动转换成设置的推断引擎

# 注意：编译参数说明 .build(...)函数
# 重要参数，包括
# platform='android/arm64-v8a' or platform='linux/x86-64'
# project_config = {
#            'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')], # 指定模块输入信息
#            'output': [
#                ('heatmap_level_1', 'EAGLEEYE_SIGNAL_TENSOR')  # 指定模块输出信息
#            ],
#            'name': 'panda',                                   # 项目名称
#            'git': ''                                          # git地址,指定后将在直接拉去此代码仓库
# }

# 使用snpe预测引擎在高通芯片上部署
import sys
import numpy as np
from antgo.pipeline import *
from antgo.pipeline.extent import op
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *

placeholder['image'](np.ones((384,512,3), dtype=np.uint8)). \
    inference_onnx_op[
        'image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')
    ](
        onnx_path='/workspace/models/bb-epoch_16-model.onnx', 
        input_fields=["image"], 
        engine='snpe',  # snpe/rknn/tensorrt
        engine_args={
            'alias_output_names':{
                'heatmap_level_1': '/MaxPool',
                'heatmap_level_2': '/MaxPool_1',
                'heatmap_level_3': '/MaxPool_2',
                'offset_level_1': '/bbox_head/reg_head_list.0/reg_head_list.0.5/Relu',
                'offset_level_2': '/bbox_head/reg_head_list.1/reg_head_list.1.5/Relu',
                'offset_level_3': '/bbox_head/reg_head_list.2/reg_head_list.2.5/Relu',
            },
        }). \
    build(
        platform='android/arm64-v8a', 
        project_config={
            'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
            'output': [
                ('heatmap_level_1', 'EAGLEEYE_SIGNAL_TENSOR')
            ],
            'name': 'panda',
            'git': ''
        }
    )

# 使用rknn预测引擎在rk芯片上运行
placeholder['image'](np.ones((384,512,3), dtype=np.uint8)). \
    inference_onnx_op[
        'image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')
    ](
        onnx_path='/workspace/models/bb-epoch_16-model.onnx', 
        mean= [0.5,0.5,0.5],
        std= [1,1,1],
        engine='rknn',
        engine_args={
            'device':'rk3588',
        #   'quantize': True,                                           是否量化
        #   'calibration-images': '/workspace/calibration-images/'      校准文件路径
        }). \
    build(
        platform='android/arm64-v8a',
        project_config={
            'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
            'output': [
                ('heatmap_level_1', 'EAGLEEYE_SIGNAL_TENSOR')
            ],
            'name': 'panda',
            'git': ''
        }
    )

# 使用tnn预测引擎在移动端gpu运行
placeholder['image'](np.ones((384,512,3), dtype=np.uint8)). \
    inference_onnx_op[
        'image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')
    ](
        onnx_path='/workspace/models/bb-epoch_16-model.onnx', 
        mean= [0.5,0.5,0.5],
        std= [1,1,1],
        engine='tnn',
        engine_args={
            'device': 'GPU'
        }). \
    build(
        platform='android/arm64-v8a',
        project_config={
            'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
            'output': [
                ('heatmap_level_1', 'EAGLEEYE_SIGNAL_TENSOR')
            ],
            'name': 'panda',
            'git': ''
        }
    )

# 使用tensorrt预测引擎在nvidia gpu上运行
placeholder['image'](np.ones((384,512,3), dtype=np.uint8)). \
    inference_onnx_op[
        'image', ('heatmap_level_1', 'heatmap_level_2', 'heatmap_level_3', 'offset_level_1', 'offset_level_2', 'offset_level_3')
    ](
        onnx_path='/workspace/models/bb-epoch_16-model.onnx', 
        mean= [0.5,0.5,0.5],
        std= [1,1,1],
        engine='tensorrt'). \
    build(
        platform='linux/x86-64',
        project_config={
            'input': [('image', 'EAGLEEYE_SIGNAL_RGB_IMAGE')],
            'output': [
                ('heatmap_level_1', 'EAGLEEYE_SIGNAL_TENSOR')
            ],
            'name': 'panda',
            'git': ''
        }
    )

```

#### 部署包运行测试
运行如下脚本，执行部署包
```
from antgo.pipeline import *
project_kwargs = {
    'image': np.ones((384,512,3), dtype=np.uint8)
}
run(project='panda', **project_kwargs)
```

## 相关文档
![]()

## 常见错误信息汇总
### 使用C++扩展时报/miniconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found

这是由于conda环境的libstd库与编译c++库不符导致，可以通过如下方式解决
```
rm /miniconda3/bin/../lib/libstdc++.so.6
ln /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /miniconda3/bin/../lib/libstdc++.so.6
```