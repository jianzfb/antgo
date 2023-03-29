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
from antvis.client.httprpc import *
from antgo.ant import environment
from antgo import config
from antgo import script
import time
from queue import PriorityQueue
from socketserver import StreamRequestHandler as Tcp
import socketserver
from socket import *
import threading


'''
后台常住,需要定时获取项目最新进展 

'''
server_manager = None

class BackgroundTCP(Tcp):
    def handle(self):
        while True:
            data = self.request.recv(1024)
            if not data:
                continue

            data = data.decode('utf-8')
            if data == 'ping':
                content = {'status': True, 'message': 'ping recv'}
                self.request.sendall(json.dumps(content).encode('utf-8'))
                continue

            try:
                global server_manager
                status = server_manager.trigger(data)
                content = {'status': status, 'message': ''}
                self.request.sendall(json.dumps(content).encode('utf-8'))
            except:
                logging.error(f'background server recv abnormal data {data}')

class ServerBase(object):
    def __init__(self, *args, **kwargs) -> None:
        super(ServerBase, self).__init__(*args, **kwargs)
        self.timer = 60*10 # 10分钟一次监控
        self.task_queue = PriorityQueue()
        self.task_set = set()       # 用于校验是否存在同类型任务
        
        # supervised            优先级2
        # semi-supervised       优先级3
        # distillation          优先级3
        # activelearning        优先级1
        self.task_order = [('activelearning', 'supervised'),('supervised', 'activelearning'), ('supervised','semi-supervised'),('supervised','distillation')]
        self.task_priority = {'activelearning': 1, 'supervised':2,'semi-supervised':3, 'distillation':3}
        self.task_cmd = {
            "activelearning": "",
            "supervised": "",
            "semi-supervised": "",
            "distillation": ""
        }

        # 启动定时任务
        periodical_func = threading.Timer(10, self.watch)
        periodical_func.daemon = True
        periodical_func.start()

    def submit(self):
        raise NotImplementedError

    def watch(self):
        raise NotImplementedError

    def clear(self):
        self.task_queue.clear()

    def trigger(self, event):
        # 当开发者主动更新 1. 标签数据; 2. 无标签数据; 3. 更换product模型; 触发
        if event == 'train/label':
            next_exp_stage = 'supervised'

            if next_exp_stage not in self.task_set:
                self.task_set.add(next_exp_stage)
                self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))     

            return True
        elif event == 'train/unlabel':
            next_exp_stage = 'activelearning'

            if next_exp_stage not in self.task_set:
                self.task_set.add(next_exp_stage)
                self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))       

            next_exp_stage = 'semi-supervised'
            if next_exp_stage not in self.task_set:
                self.task_set.add(next_exp_stage)
                self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))  

            return True
        elif event == 'product':
            next_exp_stage = 'supervised'
            if next_exp_stage not in self.task_set:
                self.task_set.add(next_exp_stage)
                self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))   

            return True
        return False

    def schedule(self, project_name, project_info, exp_info):
        exp_name = exp_info['name']
        exp_id = exp_info['id']

        # 自动化迭代流程
        # product模型下一阶段任务调度
        if project_info['product'].startswith(exp_name):
            if exp_name not in project_info['exp']:
                logging.warn(f'Exp {exp_name} not in project.')
            else:
                for exp_info in project_info['exp'][exp_name]:
                    if exp_info['id'] == exp_id:
                        exp_stage = exp_info['stage']
                        for order_info in self.task_order:
                            if exp_stage == order_info[0]:
                                next_exp_stage = order_info[1]
                                # 加入任务队列，如果任务队列中存在相同阶段实验，则放弃加入
                                if next_exp_stage not in self.task_set:
                                    self.task_set.add(next_exp_stage)
                                    self.task_queue.put((self.task_priority[next_exp_stage], next_exp_stage))

        # 根据计算资源进行任务提交
        submitter_info = project_info['submitter']
        submitter_is_local = submitter_info['local']
        submitter_method = submitter_info['method']
        submitter_gpu_num = submitter_info['gpu_num']
        submitter_cpu_num = submitter_info['cpu_num']
        submitter_memory = submitter_info['memory']

        submitter_resource_check = getattr(script, f'{submitter_method}_submit_resource_check_func', None)
        if submitter_resource_check is None:
            logging.warn(f'Submitter {submitter_method} not support.')
            return

        if not submitter_resource_check(submitter_gpu_num, submitter_cpu_num, submitter_memory):
            logging.warn(f'Submitter {submitter_method} resource (gpu: {submitter_gpu_num}, cpu: {submitter_cpu_num}, memory: {submitter_memory}) not enough.')
            return

        submitter_func = getattr(script, f'{submitter_method}_submit_process_func', None)
        next_task = self.task_queue.get()

        next_task_cmd = self.task_cmd[next_task[1]]
        if not submitter_is_local:
            next_task_cmd += " --cloud"

        if submitter_method != '':
            next_task_cmd += f" --{submitter_method}"

        if submitter_func(project_name, next_task_cmd, submitter_gpu_num, submitter_cpu_num, submitter_memory, next_task):
            # 提交任务成功
            logging.info(f"Success submit task {self.task_cmd[next_task[1]]}")
            self.task_set.pop(next_task[1])
        else:
            # 提交任务失败 (重新加入任务队列)
            logging.error(f'Fail submit task {self.task_cmd[next_task[1]]}')
            self.task_queue.put(next_task)

'''
默认项目全部信息存储在hdfs上
自动化触发, 依靠定时读取hdfs地址实现
'''
class LocalServer(ServerBase):
    def __init__(self, *args, **kwargs) -> None:
        super(LocalServer, self).__init__(*args, **kwargs)
        # antgo 监控root地址，定时从此目录下获得项目更新信息
        # self.root = kwargs['root']
        self.root = ''
        # 目录结构如下
        # root 
        #   - project
        #        - exp.id
        #               RUNNING/FINISH/STOP
        #               - output
        #                   - metric
        #                       best.json
        #                   - checkpoint
        #                       ***.pth
        #        - exp.id
        #               RUNNING/FINISH/STOP
        #               - output
        #                   - metric
        #                       best.json
        #                   - checkpoint
        #                       ***.pth        
        #   - project
        #        - exp.id

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
                    if exp_info['state'] == 'running':
                        project_state_folder = os.path.join(self.root, project_name, f'{exp_name}.{exp_id}')
                        checkpoint_folder = os.path.join(self.root, project_name, f'{exp_name}.{exp_id}','output', 'checkpoint')
                        is_exist = environment.hdfs_client.exists(os.path.join(project_state_folder, 'FINISH'))
                        if is_exist:
                            exp_info['state'] = 'finish'
                            exp_info['finish_time'] = time.strftime('%Y-%m-%dx%H:%M:%S',time.localtime(time.time()))

                            if environment.hdfs_client.exists(os.path.join(checkpoint_folder, 'best.pth')):
                                # 记录最佳指标模型（开启评估过程）
                                exp_info['checkpoint'] = os.path.join(checkpoint_folder, 'best.pth')
                            else:
                                # 记录最后的一个模型（未开启评估过程）
                                exp_info['checkpoint'] = os.path.join(checkpoint_folder, 'latest.pth')

                            best_metric_record = os.path.join(self.root, project_name, f'{exp_name}.{exp_id}','output', 'best.json')
                            if environment.hdfs_client.exists(best_metric_record):
                                if not os.path.exists('/tmp/'):
                                    os.makedirs('/tmp/')

                                environment.hdfs_client.get(best_metric_record, '/tmp/')
                                if os.path.exists(os.path.join('/tmp/best.json')):
                                    with open('/tmp/best.json', 'r') as fp:
                                        exp_info['metric'].update(json.load(fp))

                        # 项目自动化优化流水线
                        if project_info['auto']:
                            self.schedule(project_name, project_info, exp_info)


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


def get_client():
    client_handler = ClientHandler()
    return client_handler


def launch_server(port):
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
        server_manager = LocalServer()
        server = socketserver.TCPServer((host, port), BackgroundTCP)

        config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
        config.AntConfig.parse_xml(config_xml)

        with open(os.path.join(config.AntConfig.factory, '.server.info'), 'w') as fp:
            fp.write(json.dumps({
                'port': port
            }))
        server.serve_forever()
    except KeyboardInterrupt:
        logging.error('Antgo server CTRL+C,break.')
        sys.exit()      
    except:   
        logging.error('Launch antgo server fail.')
        sys.exit()


if __name__ == '__main__':
    # q = PriorityQueue()
    # q.put((2, 'code'))
    # q.put((1, 'eat'))
    # q.put((3, 'sleep'))

    # while not q.empty():
    #     next_item = q.get()
    #     print(next_item)

    host = ''
    port = 8886
    server_manager = LocalServer()
    server = socketserver.TCPServer((host, port), BackgroundTCP)
    server.serve_forever()