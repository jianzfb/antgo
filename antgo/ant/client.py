# encoding=utf-8
# @Time    : 23-03-22
# @File    : client.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
from antvis.client.httprpc import *
from antgo.ant import environment
from antgo import config
import time


'''
后台常住,需要定时获取项目最新进展(from server)

# 添加数据后的触发逻辑
添加标注数据，触发->监督训练
添加无标签数据，触发->半监督训练->触发->主动学习流程->触发标注服务->触发(标注完成后)->半监督训练->... (直到无标签数据量不满足需求或测试集指标存在退化)
更新测试数据，触发->baseline模型计算指标->product模型计算指标->teacher模型计算指标->聚合模型计算指标

# 添加expert,product,baseline后的触发逻辑
添加expert后,
    对于已经完成训练的, 触发->teacher模型聚合->触发(测试集指标存在提升)->模型蒸馏/半监督训练 (仅保留最佳指标下的checkpoint) -> product模型 蒸馏/半监督训练
    对于未完成训练的, 触发->teacher模型监督训练->触发teacher模型蒸馏/半监督训练 (仅保留最佳指标下的checkpoint) -> product模型 蒸馏/半监督训练

添加product后,
    触发->监督训练->蒸馏/半监督训练

添加baseline后, 
    触发->监督训练


'''
class ClientBase(object):
    def __init__(self) -> None:
        pass
    
    def submit(self):
        raise NotImplementedError

    def watch(self):
        raise NotImplementedError
    
    def schedule(self, project_info, exp_info):
        exp_name = exp_info['name']
        exp_id = exp_info['id']
        raise NotImplementedError


'''
默认项目全部信息存储在hdfs上
自动化触发, 依靠定时读取hdfs地址实现
'''
class LocalClient(ClientBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # antgo 监控root地址，定时从此目录下获得项目更新信息
        self.root = kwargs['root']
        # 目录结构如下
        # root 
        #   - project
        #        - exp
        #           - id
        #               FINISH/TRAINING/STOP
        #               - output
        #                   - metric
        #                       best.json
        #                   - checkpoint
        #                       ***.pth
        #           - id 
        #               FINISH/TRAINING/STOP
        #               - output
        #                   - metric
        #                       best.json
        #                   - checkpoint
        #                       ***.pth        
        #   - project
        #        - exp
        self.timer = 60*10 # 10分钟一次监控
        self.waiting_queue = []

    def watch(self):
        # 获得所有项目配置信息
        for project_file in os.listdir(config.AntConfig.task_factory):
            if not project_file.endswith('.json'):
                continue
            
            with open(os.path.join(config.AntConfig.task_factory, project_file), 'r') as fp:
                project_info = json.load(fp)
            
            # 项目名称    
            project_name = project_file.replace('.json', '')
            
            # 更新项目最新信息
            for exp_name, exp_list_info in project_info['exp'].items():
                for exp_info in exp_list_info:
                    exp_id = exp_info['id']
                    if exp_info['state'] == 'training':
                        project_state_folder = os.path.join(self.root, project_name, exp_name, exp_id)
                        checkpoint_folder = os.path.join(self.root, project_name, exp_name, exp_id,'output', 'checkpoint')
                        is_exist = environment.hdfs_client.exists(os.path.join(project_state_folder, 'FINISH'))
                        if is_exist:
                            exp_info['state'] = 'finish'
                            exp_info['finish_time'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                            
                            if environment.hdfs_client.exists(os.path.join(checkpoint_folder, 'best.pth')):
                                # 记录最佳指标模型（开启评估过程）
                                exp_info['checkpoint'] = os.path.join(checkpoint_folder, 'best.pth')
                            else:
                                # 记录最后的一个模型（未开启评估过程）
                                exp_info['checkpoint'] = os.path.join(checkpoint_folder, 'latest.pth')

                            best_metric_record = os.path.join(self.root, project_name, exp_name, exp_id,'output', 'best.json')
                            if environment.hdfs_client.exists(best_metric_record):
                                if not os.path.exists('/tmp/'):
                                    os.makedirs('/tmp/')
                                    
                                environment.hdfs_client.get(best_metric_record, '/tmp/')
                                if os.path.exists(os.path.join('/tmp/best.json')):
                                    with open('/tmp/best.json', 'r') as fp:
                                        exp_info['metric'].update(json.load(fp))

                        # 项目自动化优化流水线
                        if project_info['auto']:
                            self.schedule(project_info, exp_info)


class HttpClient(ClientBase):
    def __init__(self) -> None:
        super().__init__()
   

client_handler = LocalClient()
    
