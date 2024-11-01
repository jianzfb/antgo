import torch
import antvis.client.mlogger as mlogger
from antgo.utils.config_dashboard import *
from antgo.framework.helper.utils.config import Config
import zlib
import subprocess
import json
import os


logger_canvas = None
logger_names = []
def logger_training_info(trainer):
    # 记录实验过程信息（损失函数，迭代步骤）
    global logger_canvas
    global logger_names

    if logger_canvas is None:
        return

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


def yolo_model_train(exp_name, cfg, root, gpu_id, pretrained_model=None):
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

    # 创建dashboard
    project = cfg.get('project', os.path.abspath(os.path.curdir).split('/')[-1])
    experiment = cfg.filename.split('/')[-1].split('.')[0]
    token = create_project_in_dashboard(project, experiment)

    if mlogger.is_ready():
        # dashboard 环境准备好
        print(f'Show Experiment Dashboard http://ai.vibstring.com/#/ExperimentDashboard?token={token}')
        global logger_canvas
        logger_canvas = mlogger.Container()

    # 构建模型
    model_name = cfg.get('model', None)
    assert(model_name is not None)
    data = cfg.get('data', None)
    assert(data is not None)
    model = None
    if pretrained_model is not None and pretrained_model != '':
        model = YOLO(pretrained_model)
    else:
        model = YOLO(f'{model_name}.yaml')

    # 加入实验消息回调
    model.add_callback('on_batch_end', func=logger_training_info)

    # 启动训练
    data_path = data.get('path', None)
    data_imgsz = data.get('imgsz', 640)
    batch_size = data.get('batch_size', 32)
    workers = data.get('workers', 1)

    device = [int(k) for k in gpu_id.split(',')]
    results = model.train(
        data=data_path, 
        epochs=cfg.get('max_epochs', 100), 
        imgsz=data_imgsz, 
        device=device,
        batch_size=batch_size
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
    if mlogger.is_ready():
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


def yolo_model_export(exp_name, pretrained_model):
    # 检查是否安装ultralytics
    p = subprocess.Popen("pip3 show ultralytics", shell=True, encoding="utf-8", stdout=subprocess.PIPE)
    if p.stdout.read() == '':
        print('Install ultralytics')
        os.system('pip3 install ultralytics')
        os.system('pip3 uninstall -y opencv-python-headless')
        os.system('pip3 install opencv-python-headless')
    from ultralytics import YOLO

    model = YOLO(pretrained_model)
    model.export(format="onnx", simplify=True)


def yolo_model_eval(exp_name, cfg, root, gpu_id, pretrained_model):
    yolo_model_train(exp_name, cfg, root, gpu_id, pretrained_model)
