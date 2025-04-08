import torch
import antvis.client.mlogger as mlogger
from antgo.utils.config_dashboard import *
from antgo.framework.helper.utils.config import Config
import zlib
import subprocess
import shutil
import json
import yaml
import os
import sys


logger_canvas = None
logger_names = []
def logger_training_info(trainer):
    # 记录实验过程信息（损失函数，迭代步骤）
    global logger_canvas
    global logger_names

    if logger_canvas is None:
        return

    if len(trainer.loss_names) > 1:
        for log_key, log_value in zip(trainer.loss_names, trainer.loss_items):
            log_value = (float)(log_value.cpu().numpy())
            if log_key not in logger_names:
                setattr(logger_canvas, log_key, mlogger.complex.Line(plot_title=log_key, is_series=True))
                logger_names.append(log_key)
            getattr(logger_canvas, log_key).update(log_value)

    if 'loss' not in logger_names:
        setattr(logger_canvas, 'loss', mlogger.complex.Line(plot_title='loss', is_series=True))
        logger_names.append('loss')
    log_value = float(trainer.loss.cpu().detach().numpy())
    getattr(logger_canvas, 'loss').update(log_value)


def yolo_model_train(exp_name, cfg, root, gpu_id, pretrained_model=None, **kwargs):
    # 配置信息转换
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    elif isinstance(cfg, dict):
        cfg = Config.fromstring(json.dumps(cfg), '.json')

    # 检查是否安装ultralytics
    p = subprocess.Popen("pip3 show ultralytics", shell=True, encoding="utf-8", stdout=subprocess.PIPE)
    if p.stdout.read() == '':
        print('Install ultralytics')
        os.system('pip3 install ultralytics')
        os.system('pip3 uninstall -y opencv-python-headless')
        os.system('pip3 install opencv-python-headless')
    from ultralytics import YOLO

    project = cfg.get('project', os.path.abspath(os.path.curdir).split('/')[-1])
    if exp_name is None or exp_name == '':
        exp_name = 'yolo'
    if not kwargs.get('no_manage', False):
        # 创建dashboard        
        create_project_in_dashboard(project, exp_name, False)

    if mlogger.is_ready():
        global logger_canvas
        logger_canvas = mlogger.Container()

    # 构建模型
    model_name = cfg.get('model', None)
    data = cfg.get('data', None)
    if data is None:
        print('Must set data config.')
        return

    if model_name is None:
        model_name = pretrained_model
    if model_name is None:
        print('Must set yolo model name')
        return

    # 解析模型类别（det,cls,seg,pose,obb）
    model_category = 'det'
    if 'cls' in model_name:
        model_category = 'cls'
    elif 'seg' in model_name:
        model_category = 'seg'
    elif 'pose' in model_name:
        model_category = 'pose'
    elif 'obb' in model_name:
        model_category = 'obb'
    else:
        model_category = 'det'

    # 构建模型
    model = None
    if model_name.endswith('.pt') or model_name.endswith('.yaml'):
        model = YOLO(model_name)
    else:
        model = YOLO(f'{model_name}.yaml')

    # 加入实验消息回调
    model.add_callback('on_batch_end', func=logger_training_info)

    # 启动训练
    # 对于分类模型，不需要数据集配置文件
    if model_category != 'cls':
        data_path = data.get('path', None)
        with open(data_path, 'r') as fp:
            data_info = yaml.safe_load(fp)
        data_info['path'] = os.path.dirname(os.path.abspath(data_path))
        with open('./data.yaml', 'w') as fp:
            yaml.safe_dump(data_info, fp)

    # 解析模型训练必须参数，图像大小，批处理大小
    data_imgsz = data.get('imgsz', 640)
    batch_size = data.get('batch_size', 32)
    workers = data.get('workers', 1)

    device = [int(k) for k in gpu_id.split(',')]
    results = model.train(
        data='./data.yaml' if model_category != 'cls' else data.get('path', None), 
        epochs=cfg.get('max_epochs', 100), 
        imgsz=data_imgsz, 
        device=device,
        batch=batch_size
    )
    save_dir = str(model.trainer.save_dir)
    if not root.startswith('ali'):
        root = save_dir

    # 记录指标
    if results is not None and mlogger.is_ready():
        report = {
            'best': {
                'measure': []
            }
        }

        for k,v in results.results_dict.items():
            metric_name = str(k)
            metric_value = float(v)

            report['best']['measure'].append({
                'statistic': {
                    'value': [{
                        'interval': [0,0],
                        'value': metric_value,
                        'type': 'SCALAR',
                        'name': metric_name
                    }]
                }
            })
        mlogger.info.experiment.patch(
            experiment_data=zlib.compress(
                json.dumps(
                    {
                        'REPORT': report,
                        'APP_STAGE': 'TEST'
                    }
                ).encode()
            )
        )

    # 记录checkpoint
    if results is not None and mlogger.is_ready():
        backend = 'disk'
        if root.startswith('ali'):
            backend = 'aliyun'
        elif root.startswith('qiniu'):
            backend = 'qiniu'
        elif root.startswith('htfs'):
            backend = 'hdfs'
        mlogger.FileLogger.root_folder = root   # 远程地址
        file_logger = mlogger.Container()
        file_logger.file = mlogger.FileLogger('file', backend)

        kwargs = {}
        if file_logger.file.backend == 'disk':
            # 需要记录当前机器地址 user@ip
            disk_address = 'root@127.0.0.1'
            if os.path.exists('./address'):
                with open('./address', 'r') as fp:
                    disk_address = fp.read()

            kwargs.update({
                'address': f'{disk_address}'
            })

        # 打包checkpoint
        os.system(f'tar -cf {exp_name}.tar {save_dir}')
        file_logger.file.update(f'{exp_name}.tar', **kwargs)


def yolo_model_export(exp_name, pretrained_model, **kwargs):
    # 检查是否安装ultralytics
    p = subprocess.Popen("pip3 show ultralytics", shell=True, encoding="utf-8", stdout=subprocess.PIPE)
    if p.stdout.read() == '':
        print('Install ultralytics')
        os.system('pip3 install ultralytics')
        os.system('pip3 uninstall -y opencv-python-headless')
        os.system('pip3 install opencv-python-headless')

    print('Fix export onnx to multi-platform')
    fix_file_path = os.path.dirname(os.path.dirname(__file__))
    fix_file_path = os.path.join(fix_file_path, 'resource/fix/yolo/head.py')

    # analyze prefix
    prefix_index = 0
    for index, info in enumerate(fix_file_path.split('/')):
        if info.startswith('python'):
            prefix_index = index
            break
    prefix = '/'.join(fix_file_path.split("/")[:prefix_index+1])

    package_path = ''
    for check_sys_path in sys.path:
        if check_sys_path.endswith('site-packages') and check_sys_path.startswith(prefix):
            package_path = check_sys_path
            break

    print(f'Copy {fix_file_path} {package_path}/ultralytics/nn/modules')
    shutil.copy(fix_file_path, f'{package_path}/ultralytics/nn/modules')

    from ultralytics import YOLO

    model = YOLO(pretrained_model)
    model.export(format="onnx", simplify=True)


def yolo_model_eval(exp_name, cfg, root, gpu_id, pretrained_model, **kwargs):
    # 配置信息转换
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    elif isinstance(cfg, dict):
        cfg = Config.fromstring(json.dumps(cfg), '.json')

    # 检查是否安装ultralytics
    p = subprocess.Popen("pip3 show ultralytics", shell=True, encoding="utf-8", stdout=subprocess.PIPE)
    if p.stdout.read() == '':
        print('Install ultralytics')
        os.system('pip3 install ultralytics')
        os.system('pip3 uninstall -y opencv-python-headless')
        os.system('pip3 install opencv-python-headless')
    from ultralytics import YOLO

    # 激活dashboard
    project = cfg.get('project', os.path.abspath(os.path.curdir).split('/')[-1])
    if not kwargs.get('no_manage', False):
        activate_project_in_dashboard(project, exp_name)

    if mlogger.is_ready() and (pretrained_model is None or pretrained_model == ''):
        # 下载checkpoint
        pass

    # 构建模型
    model_name = cfg.get('model', None)
    assert(model_name is not None)
    data = cfg.get('data', None)
    assert(data is not None)
    model = None
    if pretrained_model is not None and pretrained_model != '':
        model = YOLO(pretrained_model)

    if model is None:
        print('no pretrined model')
        return

    data_path = data.get('path', None)
    with open(data_path, 'r') as fp:
        data_info = yaml.safe_load(fp)

    data_info['path'] = os.path.dirname(data_path)
    data_name = os.path.basename(data_info['path'])
    with open('./data.yaml', 'w') as fp:
        yaml.safe_dump(data_info, fp)

    data_imgsz = data.get('imgsz', 640)
    batch_size = data.get('batch_size', 32)
    workers = data.get('workers', 1)

    device = [int(k) for k in gpu_id.split(',')]
    results = model.val(
            data = './data.yaml',
            imgsz=data_imgsz, 
            device=device,
            batch=batch_size,
    )

    # 记录指标
    if results is not None and mlogger.is_ready():
        report_name = data_name
        report = {
            report_name: {
                'measure': []
            }
        }

        for k,v in results.results_dict.items():
            metric_name = str(k)
            metric_value = float(v)

            report[report_name]['measure'].append({
                'statistic': {
                    'value': [{
                        'interval': [0,0],
                        'value': metric_value,
                        'type': 'SCALAR',
                        'name': metric_name
                    }]
                }
            })
        mlogger.info.experiment.patch(
            experiment_data=zlib.compress(
                json.dumps(
                    {
                        'REPORT': report,
                        'APP_STAGE': 'TEST'
                    }
                ).encode()
            )
        )
        return

    print('report')
    print(results.results_dict)
