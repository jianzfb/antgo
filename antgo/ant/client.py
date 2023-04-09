# encoding=utf-8
# @Time    : 23-03-22
# @File    : client.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from socket import SOCK_STREAM
import sys
import os
from unicodedata import name
from antvis.client.httprpc import *
from antgo.framework.helper.fileio import *
from antgo import config
from antgo import script
from antgo.utils.args import *
from antgo.framework.helper.tools.util import *
import time
from queue import PriorityQueue
from socketserver import StreamRequestHandler as Tcp
import socketserver
from socket import *
import threading
import shutil
import copy


'''
后台常住,需要定时获取项目最新进展 

'''
server_manager = None
file_lock = threading.Lock()

class BackgroundTCP(Tcp):
    def handle(self):
        while True:
            data = self.request.recv(1024)
            if not data:
                break

            data = data.decode('utf-8')
            if data == 'ping':
                content = {'status': True, 'message': 'ping recv'}
                self.request.sendall(json.dumps(content).encode('utf-8'))
                break

            try:
                global server_manager
                status = server_manager.trigger(data)
                content = {'status': status, 'message': ''}
                self.request.sendall(json.dumps(content).encode('utf-8'))
            except:
                logging.error(f'background server recv abnormal data {data}')

            break

def exp_basic_info():
    return {
        "exp": "", 
        "id": "",
        "branch": "", 
        "commit": "", 
        "metric": {}, 
        "dataset": {"test": "", "train": []},
        "checkpoint": "", 
        "create_time": time.strftime('%Y-%m-%dx%H-%M-%S',time.localtime(time.time())), 
        "finish_time": "",
        "state": "",    # running, finish, stop, default
        "stage": "",    # which stage of progress
    }


def dataset_basic_info():
    return {
        "tag": "",
        "num": 0,
        "status": True,
        'address': ""
    }

class ServerBase(object):
    def __init__(self, *args, **kwargs) -> None:
        super(ServerBase, self).__init__()
        self.root = kwargs.get('root', './')
        self.ext_module = kwargs.get('ext_module', None)
        self.timer = 10         # 10分钟一次监控, debug 10*60
        self.task_queue = PriorityQueue()
        self.task_set = set()       # 用于校验是否存在同类型任务

        # supervised            优先级2
        # semi-supervised       优先级3
        # distillation          优先级3
        # activelearning        优先级1
        # self.task_order = [('activelearning', 'label'),('supervised', 'activelearning'),('label', 'supervised'),('supervised','semi-supervised'),('supervised','distillation')]
        self.task_order = [('activelearning', 'label'),('supervised', 'activelearning'), ('label', 'supervised')]

        # self.task_order = [('activelearning', 'label'),('supervised', 'activelearning'),('label', 'supervised')]
        # self.task_order = [('supervised','semi-supervised')]
        self.task_priority = {'supervised':1, 'activelearning': 2, 'semi-supervised':3, 'distillation':3, 'label': 1}
        self.task_cmd = {
            "activelearning": "antgo activelearning",
            "label": "antgo tool label/start",
            "supervised": "antgo train",
            "semi-supervised": "antgo train",
            "distillation": "antgo train"
        }

        # 启动定时任务
        periodical_func = threading.Timer(self.timer, self.watch)
        periodical_func.start()

    def submit(self):
        raise NotImplementedError

    def watch(self):
        raise NotImplementedError

    def clear(self):
        self.task_queue.clear()

    def trigger(self, event):
        # 当开发者主动更新 1. 标签数据; 2. 无标签数据; 3. 更换product模型; 触发
        file_lock.acquire()
        project_name, project_event = event.split('.')
        project_info = {}
        if os.path.exists(os.path.join(config.AntConfig.task_factory, f'{project_name}.json')):
            with open(os.path.join(config.AntConfig.task_factory, f'{project_name}.json'), 'r') as fp:
                project_info = json.load(fp)        
    
        if len(project_info) == 0:
            logging.warn(f'Missing {project_name} project info.')
            file_lock.release()            
            return False

        if project_info['product'] == '' or project_info['product'] not in project_info['exp']:
            logging.warn(f'Missing project set of {project_name} project. Please antgo add product --exp=xxx')
            file_lock.release()
            return False

        if not project_info['auto']:
            logging.warn(f'Project {project_name} not auto optimize. Ignore auto trigger task.')
            file_lock.release()
            return False

        # 发现基于哪个项目实验，进行下一步任务
        product_exp_name = project_info['product']
        exp_info = copy.deepcopy(project_info['exp'][product_exp_name][0])
        if project_info['best']['exp'] != '' and project_info['best']['id'] != '':
            exp_info = copy.deepcopy( project_info['best'])
        exp_info['id'] = ''         # 清空exp id

        if project_event == 'train/label' or project_event == 'product':
            logging.info(f'Trigger {project_event}')
            # 启动监督训练，训练产品模型
            exp_info['checkpoint'] = '' # 清空checkpoint

            next_exp_stage = 'supervised'
            if next_exp_stage not in self.task_set:
                logging.info(f'Add {next_exp_stage} into task queue.')
                self.task_set.add(next_exp_stage)
                self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))     

            # 调度
            update_product_info = self.schedule(project_name, project_info, exp_info, auto_find_next_task=False)
            if update_product_info is not None:
                project_info = update_product_info

            # update project info
            with open(os.path.join(config.AntConfig.task_factory, f'{project_name}.json'), 'w') as fp:
                json.dump(project_info, fp)
            file_lock.release()
            return True
        elif project_event == 'train/unlabel':
            logging.info(f'Trigger {project_event}')
            # 启动主动学习，挑选等待标注样本
            next_exp_stage = 'activelearning'
            if len(project_info['tool']['activelearning']['config']) > 0:
                # 存在主动学习的配置方案
                if next_exp_stage not in self.task_set:
                    logging.info(f'Add {next_exp_stage} into task queue.')
                    self.task_set.add(next_exp_stage)
                    self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))       

            # 启动半监督学习，直接利用无标签样本训练产品模型
            next_exp_stage = 'semi-supervised'
            if len(project_info['tool']['semi']['config']) > 0:
                # 存在半监督学习的配置方案
                if next_exp_stage not in self.task_set:
                    logging.info(f'Add {next_exp_stage} into task queue.')
                    self.task_set.add(next_exp_stage)
                    self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))  

            # 调度
            update_product_info = self.schedule(project_name, project_info, exp_info, auto_find_next_task=False)
            if update_product_info is not None:
                project_info = update_product_info

            # update project info            
            with open(os.path.join(config.AntConfig.task_factory, f'{project_name}.json'), 'w') as fp:
                json.dump(project_info, fp)        

            file_lock.release()
            return True

        file_lock.release()
        return False

    def schedule(self, project_name, project_info, exp_info, auto_find_next_task=True):
        # exp_info 是当前已经运行完成的实验
        # 可以从project_info中获得，当前完成的实验处在哪个阶段
        exp_name = exp_info['exp']
        exp_id = exp_info['id']

        if not project_info['product'].startswith(exp_name):
            logging.warn(f'Exp {exp_name} not project {project_name} product model')
            return None

        if exp_name not in project_info['exp']:
            logging.warn(f'Exp {exp_name} not in project {project_name}.')
            return None

        if auto_find_next_task:
            for check_exp_info in project_info['exp'][exp_name]:
                if check_exp_info['id'] == exp_id:
                    exp_stage = check_exp_info['stage']
                    for order_info in self.task_order:
                        if exp_stage == order_info[0]:
                            next_exp_stage = order_info[1]
                            # 加入任务队列，如果任务队列中存在相同阶段实验，则放弃加入
                            if next_exp_stage not in self.task_set:
                                self.task_set.add(next_exp_stage)
                                self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))

        # 根据计算资源进行任务提交
        submitter_info = project_info['submitter']
        if len(submitter_info) == 0:
            # 没有提交配置，清空任务队列
            self.task_set.clear()
            while self.task_queue.qsize():
                self.task_queue.get()
            return

        submitter_method = submitter_info.get('method', '')
        if submitter_method == '':
            # 没有提交配置，清空任务队列
            self.task_set.clear()
            while self.task_queue.qsize():
                self.task_queue.get()
            return        

        submitter_gpu_num = submitter_info['gpu_num']
        submitter_cpu_num = submitter_info['cpu_num']
        submitter_memory = submitter_info['memory']

        # # debug
        # submitter_gpu_num = 0
        # submitter_method = 'local'

        # submitter_resource_check 检查资源
        submitter_resource_check = getattr(script, f'{submitter_method}_submit_resource_check_func', None)
        if submitter_resource_check is None:
            logging.warn(f'Submitter {submitter_method} not support.')
            return None

        # submitter_func 已经决定了如何提交任务
        submitter_func = getattr(script, f'{submitter_method}_submit_process_func', None)
        if submitter_func is None:
            logging.warn(f'Submitter {submitter_method} not support.')
            return None

        # 从任务队列中尝试取出所有等待任务进行提交
        while self.task_queue.qsize() > 0:
            if not submitter_resource_check(submitter_gpu_num, submitter_cpu_num, submitter_memory):
                logging.warn(f'Submitter {submitter_method} resource (gpu: {submitter_gpu_num}, cpu: {submitter_cpu_num}, memory: {submitter_memory}) not enough.')
                return None

            next_task = self.task_queue.get()
            next_task_cmd = self.task_cmd[next_task[1]]
            if next_task[1] == 'label':
                next_task_cmd += f" --exp={exp_name}.{exp_id}"
                next_task_cmd += f" --stage={next_task[1]}"
                next_task_cmd += f" --root={self.root}/{project_name}"
                next_task_cmd += f" --gpu-id=-1"
                # 任务的标注类别信息，标注类型，标注meta信息
                label_tags = ','.join([f'{tag}:{tag_i}' for tag_i, tag in enumerate(project_info['tool']['label']['category'])])
                label_type = project_info['tool']['label']['type']
                next_task_cmd += f" --tags={label_tags}"
                next_task_cmd += f" --type={label_type}"
                if self.ext_module is not None:
                    next_task_cmd += f" --ext-module={self.ext_module}"

                # 创建临时目录
                old_folder = os.path.abspath(os.curdir)
                temp_folder = f'{exp_name}.{exp_id}.label'
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)
                os.chdir(temp_folder)

                logging.info(f'Submit task command {next_task_cmd}')
                if submitter_func(project_name, next_task_cmd, submitter_gpu_num, submitter_cpu_num, submitter_memory, next_task[1]):
                    # 提交任务成功
                    logging.info(f"Success submit label task.")
                    # 修改当前实验的状态，标记为标注（仅对上一阶段为activelearning任务有效）
                    # 其余任务，在调度后不对原有任务的状态进行修正
                    # exp_info['stage'] = 'label'       # 调整当前实验阶段为标注阶段
                    # exp_info['state'] = 'running'     # 调整状态为运行状态
                    for check_exp_info in project_info['exp'][exp_name]:
                        if check_exp_info['id'] == exp_id:
                            check_exp_info['stage'] = 'label'               # 调整当前实验阶段为标注阶段
                            check_exp_info['state'] = 'running'             # 调整状态为运行状态

                    # 将next_task 从任务集合中删除 （任务队列已经没有此任务）
                    self.task_set.remove(next_task[1])
                else:
                    # 提交任务失败
                    logging.error(f'Fail submit label task.')
                    # 将next_task重新加入队列 （任务集合保持不变）
                    self.task_queue.put(next_task)
                    
                # 恢复原状
                # 切换回之前的目录为当前目录（仅对当前线程有效）
                os.chdir(old_folder) 
                # 对于local模型，不进行删除操作      
                if submitter_method != 'local':
                    if temp_folder is not None:
                        shutil.rmtree(temp_folder)
                return project_info

            # 其余涉及模型的任务（训练，预测等）
            next_task_cmd += f" --exp={exp_name}"
            next_task_cmd += f" --stage={next_task[1]}"
            next_task_cmd += f" --root={self.root}/{project_name}"
            # 基于提交脚本设置gpu_ids
            gpu_ids = ','.join([str(i) for i in range(submitter_gpu_num)])
            if submitter_gpu_num == 0:
                gpu_ids = '-1'
            next_task_cmd += f" --gpu-id={gpu_ids}"
            next_task_cmd += f" --checkpoint={exp_info['checkpoint']}"

            # 创建新实验记录
            auto_exp_info = exp_basic_info()
            auto_exp_info['exp'] = exp_info['exp']
            auto_exp_info['id'] = time.strftime('%Y-%m-%dx%H-%M-%S',time.localtime(time.time()))
            auto_exp_info['branch'] = exp_info['branch']
            auto_exp_info['commit'] = exp_info['commit']
            auto_exp_info['state'] = 'running'
            auto_exp_info['stage'] = next_task[1]
            auto_exp_info['checkpoint'] = exp_info['checkpoint']
            next_task_cmd += f" --id={auto_exp_info['id']}"

            logging.info(f'Submit task command {next_task_cmd}')
            
            # 准备代码环境
            old_folder = os.path.abspath(os.curdir)
            git_folder = None
            if os.path.exists(os.path.join(old_folder, '.git')):
                # 当前目录下存在git记录，默认为当前项目实际目录地址
                logging.warn(f'In current default folder {old_folder}, exists some project, wouldnt git clone from {project_info["git"]} again')
            else:
                # git_folder == project_name                
                git_folder = project_info["git"].split('/')[-1].split('.')[0]
                if not os.path.exists(f'./{project_name}'):
                    os.system(f'git clone {project_info["git"]}')    
                # 切换git_folder为当前目录（仅对当前线程有效）
                os.chdir(git_folder)

            if submitter_func(project_name, next_task_cmd, submitter_gpu_num, submitter_cpu_num, submitter_memory, next_task[1]):
                # 提交任务成功
                logging.info(f"Success submit task {self.task_cmd[next_task[1]]} of project {project_name}")

                # 将新增实验加入项目中
                project_info['exp'][exp_info['exp']].append(
                    auto_exp_info
                )

                # 将next_task 从任务集合中删除 （任务队列已经没有此任务）
                self.task_set.remove(next_task[1])
            else:
                # 提交任务失败 (重新加入任务队列)
                logging.error(f'Fail submit task {self.task_cmd[next_task[1]]} of project {project_name}')
                # 将next_task重新加入队列 （任务集合保持不变）
                self.task_queue.put(next_task)

            # 恢复原状
            if git_folder is not None:
                # 切换回之前的目录为当前目录（仅对当前线程有效）
                os.chdir(old_folder) 
            # 对于local模型，不进行删除操作      
            if submitter_method != 'local':
                if git_folder is not None:
                    shutil.rmtree(git_folder)

            return project_info


'''
默认项目全部信息存储在hdfs上
自动化触发, 依靠定时读取hdfs地址实现
'''
class LocalServer(ServerBase):
    def __init__(self, *args, **kwargs) -> None:
        super(LocalServer, self).__init__(*args, **kwargs)
        # antgo 监控root地址，定时从此目录下获得项目更新信息
        self.root = kwargs['root']
        # 目录结构如下
        # root 
        #   - project
        #        - activelearning
        #           - exp.id        # 在此实验之后，触发的主动学习进行标注样本挑选
        #               annotation.json
        #           - exp.id
        #               annotation.json
        #        - label
        #           - exp.id        # 针对主动学习挑选出来的样本，进行标注结果
        #               data.tar    # 压缩包数据格式 images/, annotation.json, meta.json
        #           - exp.id        # 在此实验后，进行的标注结果
        #               data.tar    # 压缩包数据格式 images/, annotation.json, meta.json
        #        - dataset
        #           - label
        #               exp-id-{}-tfrecord    # 针对标注后样本进行的重新打包
        #           - pseudo-label
        #               xxx.tfrecord          # 针对算法生成的伪标签数据进行的打包
        #           - unlabel
        #               xxx.tfrecord          # 针对上传的无标签数据
        #        - exp.id
        #               RUNNING/FINISH/STOP
        #               - output
        #                   - metric
        #                       best.pth
        #                       best.json
        #                   - checkpoint
        #                       ***.pth
        #        - exp.id
        #               RUNNING/FINISH/STOP
        #               - output
        #                   - metric
        #                       best.pth
        #                       best.json
        #                   - checkpoint
        #                       ***.pth        
        #   - project
        #        - exp.id

    def watch(self):
        # 获得所有项目配置信息
        file_lock.acquire()
        for project_file in os.listdir(config.AntConfig.task_factory):
            if not project_file.endswith('.json'):
                continue
            
            with open(os.path.join(config.AntConfig.task_factory, project_file), 'r') as fp:
                project_info = json.load(fp)

            # 项目名称    
            project_name = project_file.replace('.json', '')

            # 检查项目下所有任务的运行情况，对于完成的任务启动下一轮任务调度
            for exp_name, exp_list_info in project_info['exp'].items():
                for exp_info in exp_list_info:
                    exp_id = exp_info['id']
                    if exp_info['stage'] == 'label' and exp_info['state'] == 'running':
                        # 对于在运行的标注任务
                        project_state_folder = os.path.join(self.root, project_name, 'label', f'{exp_name}.{exp_id}')
                        is_exist = file_client_exists(os.path.join(project_state_folder, 'FINISH'))
                        if is_exist:
                            # 标注任务完成，开始调度下一轮任务
                            logging.info(f'Finish label task {exp_name}.{exp_id} project {project_name}.')
                            exp_info['state'] = 'finish'

                            # 获得新打包的标注数据集
                            # root, dataset, label
                            labeled_dataset_records = file_client_ls(os.path.join(self.root, project_name, 'dataset', 'label'))
                            if len(labeled_dataset_records) > 0:
                                existed_dataset_records = [
                                    aabb['address'] for aabb in project_info['dataset']['train']['label']
                                ]
                                temp_set = set()
                                for data_record_file in labeled_dataset_records:
                                    # 同一个数据文件，存在*-index和*-tfrecord
                                    finding_file = '-'.join(data_record_file.split('-')[:-1])
                                    if finding_file in temp_set:
                                        continue

                                    temp_set.add(finding_file)
                                    if finding_file not in existed_dataset_records:
                                        project_info['dataset']['train']['label'].append(
                                            {
                                                'tag': '',
                                                'num': 0,
                                                'status': True,
                                                'address': finding_file
                                            }
                                        )

                            # 保存项目新信息(在任务调度时可能需要重新读取项目信息，提前把更新内容保存)
                            with open(os.path.join(config.AntConfig.task_factory, project_file), 'w') as fp:
                                json.dump(project_info, fp)

                            # 项目自动化优化流水线(仅对产品模型进行自动化优化)
                            if project_info['auto'] and project_info['product'].startswith(exp_name):
                                logging.info(f'Schedule next task of {exp_name}.{exp_id} project {project_name}.')
                                update_product_info = self.schedule(project_name, project_info, exp_info)
                                if update_product_info is not None:
                                    project_info = update_product_info

                    elif exp_info['stage'] != 'label' and exp_info['state'] == 'running':
                        # 对于在运行的非标注任务
                        project_state_folder = os.path.join(self.root, project_name, f'{exp_name}.{exp_id}')
                        checkpoint_folder = os.path.join(self.root, project_name, f'{exp_name}.{exp_id}','output', 'checkpoint')
                        metric_folder = os.path.join(self.root, project_name, f'{exp_name}.{exp_id}','output', 'metric')

                        # FINISH 仅仅表示任务运行完毕
                        # 对于具体完成的任务是什么需要从project_info中查找
                        is_exist = file_client_exists(os.path.join(project_state_folder, 'FINISH'))
                        if is_exist:
                            # 任务完成状态
                            logging.info(f'Finish stage {exp_info["stage"]} task {exp_name}.{exp_id} project {project_name}.')
                            exp_info['state'] = 'finish'
                            exp_info['finish_time'] = time.strftime('%Y-%m-%dx%H-%M-%S',time.localtime(time.time()))

                            # 获得任务保存的checkpoint
                            if file_client_exists(os.path.join(metric_folder, 'best.pth')):
                                # 记录最佳指标模型（开启评估过程）
                                exp_info['checkpoint'] = os.path.join(metric_folder, 'best.pth')
                            elif file_client_exists(os.path.join(checkpoint_folder, 'latest.pth')):
                                # 记录最后的一个模型（未开启评估过程）
                                exp_info['checkpoint'] = os.path.join(checkpoint_folder, 'latest.pth')

                            # 获得任务保存的最佳指标结果
                            best_metric_record = os.path.join(metric_folder, 'best.json')
                            if file_client_exists(best_metric_record):
                                logging.info(f'Finding metric best.json of task {exp_name}.{exp_id}.')
                                if not os.path.exists('./temp'):
                                    os.makedirs('./temp')

                                file_client_get(best_metric_record, './temp')
                                if os.path.exists(os.path.join('./temp/best.json')):
                                    # 读取并更新实验指标信息
                                    with open('./temp/best.json', 'r') as fp:
                                        exp_info['metric'].update(json.load(fp))
                                    
                                    # 清理临时文件
                                    os.remove('./temp/best.json')

                                # 自动更新最佳模型（仅针对产品模型记录）
                                try:
                                    if project_info['product'].startswith(exp_name) and len(exp_info['metric']) > 0:
                                        # 当前完成的实验，存在指标计算，并且当前实验是产品模型
                                        logging.info(f'Update best metric of task {exp_name}.{exp_id} project {project_name}.')
                                        if len(project_info['best']) == 0 or len(project_info['best']['metric']) == 0:
                                            project_info['best'] = exp_info
                                        else:
                                            best_metric_keys = project_info['best']['metric']['score'].keys()
                                            exp_metric_keys = exp_info['metric']['score'].keys()
                                            if set(best_metric_keys) == set(exp_metric_keys):
                                                is_best = False
                                                for k in exp_info['metric']['score'].keys():
                                                    if exp_info['metric']['score'][k] < project_info['best']['metric']['score'][k]:
                                                        is_best = False
                                                
                                                if is_best:
                                                    project_info['best'] = exp_info
                                            else:
                                                logging.error('Abnormal exp metric keys not consistent with best record.')
                                except:
                                    logging.error("Abnormal analyze project best record.")

                            # 获得任务生成的数据（伪标签数据）
                            pseudo_label_dataset_records = file_client_ls(os.path.join(self.root, project_name, 'dataset', 'pseudo-label'))
                            if len(pseudo_label_dataset_records) > 0:
                                existed_dataset_records = [
                                    aabb['address'] for aabb in project_info['dataset']['train']['pseudo-label']
                                ]
                                temp_set = set()
                                for data_record_file in pseudo_label_dataset_records:
                                    # 同一个数据文件，存在*-index和*-tfrecord
                                    finding_file = '-'.join(data_record_file.split('-')[:-1])
                                    if finding_file in temp_set:
                                        continue

                                    temp_set.add(finding_file)
                                    if finding_file not in existed_dataset_records:
                                        project_info['dataset']['train']['label'].append(
                                            {
                                                'tag': '',
                                                'num': 0,
                                                'status': True,
                                                'address': finding_file
                                            }
                                        )

                            # 保存项目新信息(在任务调度时可能需要重新读取项目信息，提前把更新内容保存)
                            with open(os.path.join(config.AntConfig.task_factory, project_file), 'w') as fp:
                                json.dump(project_info, fp)
     
                            # 项目自动化优化流水线(仅对产品模型进行自动化优化)
                            if project_info['auto'] and project_info['product'].startswith(exp_name):
                                logging.info(f'Schedule next task of {exp_name}.{exp_id} project {project_name}.')
                                update_product_info = self.schedule(project_name, project_info, exp_info)
                                if update_product_info is not None:
                                    project_info = update_product_info                                

            # 保存项目新信息
            with open(os.path.join(config.AntConfig.task_factory, project_file), 'w') as fp:
                json.dump(project_info, fp)
        
        file_lock.release()
        # 重新启动下一轮触发
        periodical_func = threading.Timer(self.timer, self.watch)
        periodical_func.start()


class HttpServer(ServerBase):
    def __init__(self) -> None:
        super().__init__()


class ClientHandler(object):
    def __init__(self):
        super().__init__()
        self.host = 'localhost'
        # 默认端口
        self.port = 8886

        # 检查服务保存的端口信息
        config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
        config.AntConfig.parse_xml(config_xml)

        if os.path.exists(os.path.join(config.AntConfig.factory, '.server.info')):   
            with open(os.path.join(config.AntConfig.factory, '.server.info'), 'r') as fp:
                content = json.load(fp)
                self.port = content['port']    

    def alive(self):
        try:
            client = socket(AF_INET, SOCK_STREAM)
            client.connect((self.host, self.port))

            event = 'ping'
            client.send(event.encode('utf-8'))
            response = client.recv(1024)
            if not response:
                logging.error(f'Trigger event {event} fail.')
                return False

            return True        
        except:
            return False

    def trigger(self, event):
        try:
            client = socket(AF_INET, SOCK_STREAM)
            client.connect((self.host, self.port))

            client.send(event.encode('utf-8'))
            response = client.recv(1024)
            if not response:
                logging.error(f'Trigger event {event} fail.')
                return

            response = json.loads(response.decode('utf-8'))
            if response['status']:
                logging.info(f'Success to trigger {event}')
            else:
                logging.error(f'Abnormal to trigger {event}')

            client.close()
        except:
            logging.error(f'Trigger {event} fail (couldnt connect server).')


def get_client():
    client_handler = ClientHandler()
    return client_handler

def get_lock():
    global file_lock
    return file_lock

def launch_server(port, root, ext_module):
    # step1: 尝试使用httpserver，使用MLTALKER服务管理
    # step2: 使用localserver，常用于公网连接受限情况，基于文件服务实现同步
    host = ''
    port = 8886 if port == 0 else port

    # 检查是否已经存在服务进程
    if get_client().alive():
        logging.warn('Antgo server has running in background.')
        return    

    try:
        global server_manager
        server_manager = LocalServer(root=root, ext_module=ext_module)
        server = socketserver.TCPServer((host, port), BackgroundTCP)

        config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
        config.AntConfig.parse_xml(config_xml)

        with open(os.path.join(config.AntConfig.factory, '.server.info'), 'w') as fp:
            fp.write(json.dumps({
                'port': port
            }))
        logging.info(f'Antgo server on host {host} port {port}')
        server.serve_forever()
    except KeyboardInterrupt:
        logging.error('Antgo server CTRL+C,break.')
        sys.exit()      
    except:   
        logging.error('Launch antgo server fail.')
        sys.exit()


if __name__ == '__main__':
    DEFINE_int('port', '8886', '')
    DEFINE_nn_args()
    
    args = parse_args()
    if args.ext_module != '':
        logging.info('import extent module')
        load_extmodule(args.ext_module)

    # 启动服务        
    launch_server(args.port, args.root, args.ext_modeule)