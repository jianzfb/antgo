# encoding=utf-8
# @Time    : 22-3-7
# @File    : help.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import git
import logging
import shutil
from antgo import config
import json
from pprint import pprint
import time
import yaml
from antgo.framework.helper.utils import Config
from antgo.ant.client import *


def prepare_project_environment(project_git, project_branch, project_commit):
    git_folder = project_git.split('/')[-1].split('.')[0]
    if os.path.exists(git_folder):
        logging.warn('Project code existed.')
        return
        
    # 无代码环境，处理流程
    logging.info(f"Clone project code from {project_git}")
    if project_branch is not None:
        if project_commit is not None:
            os.system(f'git clone {project_git} -b {project_branch}; git reset --soft {project_commit}')
        else:
            os.system(f'git clone {project_git}; -b {project_branch};')
    else:
        os.system(f'git clone {project_git}')
    
    os.chdir(f'./{git_folder}')


def check_project_environment(args):
    assert(args.project != '')
    if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} not exist.')
        return False
    
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp)

    # 检查是否在项目目录下
    is_in_project_folder = False
    try:
        rep = git.Repo('./')
        is_in_project_folder = True
        cur_git_address = ''
        for url_address in rep.remote().urls:
            cur_git_address = url_address
            break
        if cur_git_address.split('/')[-1] != project_info['git'].split('/')[-1]:
            logging.error('You are in an error project folder.')
            return False

        if rep.is_dirty() or len(rep.untracked_files) > 0:
            logging.error(f'Some uncomitted/untracked files in repo.')
            return False        
    except:
        is_in_project_folder = False

    project_git = project_info['git']
    if not is_in_project_folder:
        # 无代码环境，处理流程
        git_folder = project_git.split('/')[-1].split('.')[0]
        if not os.path.exists(git_folder): 
            if args.branch is not None:
                if args.commit is not None:
                    os.system(f'git clone {project_git} -b {args.branch}; git reset --soft {args.commit}')
                else:
                    os.system(f'git clone {project_git}; -b {args.branch};')
            else:
                os.system(f'git clone {project_git}')

        os.chdir(f'./{git_folder}')

    return True


def generate_project_exp_example(template_project_folder, target_folder, exp_name):
    # step1: 拷贝cifar10分类模型代码
    if exp_name is None:
        exp_name = 'cifar10'
    if not os.path.exists(os.path.join(target_folder, exp_name)):
        # 复制cifar10分类样例代码
        if exp_name is None or exp_name == '':
            exp_name = 'cifar10'
        shutil.copytree(os.path.join(template_project_folder, 'cifar10'), os.path.join(target_folder, exp_name))

        if exp_name != 'cifar10':
            # 修改main.py的帮助信息
            with open( os.path.join(target_folder, exp_name, 'main.py'), 'r') as fp:
                content = fp.read()
                content = content.replace('cifar10', exp_name)
            
            with open( os.path.join(target_folder, exp_name, 'main.py'), 'w') as fp:
                fp.write(content)

            # 修改标准配置文件 (lsp, pascal_voc)
            os.remove(os.path.join(target_folder, exp_name, 'configs', 'config.py'))
            if os.path.exists(os.path.join(template_project_folder, f'{exp_name}_config.py')):
                shutil.copy(os.path.join(template_project_folder, f'{exp_name}_config.py'), os.path.join(target_folder, exp_name, 'configs', 'config.py'))
            else:      
               shutil.copy(os.path.join(template_project_folder, 'config.py'), os.path.join(target_folder, exp_name, 'configs'))
    else:
        logging.warn('MVP code has existed in current path.')
        return

    # step2: 拷贝shell等说明代码
    if not os.path.exists(os.path.join(target_folder, 'system.py')):
        shutil.copy(os.path.join(template_project_folder, 'system.py'), target_folder)
    if not os.path.exists(os.path.join(target_folder, 'install.sh')):
        shutil.copy(os.path.join(template_project_folder, 'install.sh'), target_folder)
    if not os.path.exists(os.path.join(target_folder, 'launch.sh')):
        shutil.copy(os.path.join(template_project_folder, 'launch.sh'), target_folder)
    if not os.path.exists(os.path.join(target_folder, 'requirements.txt')):
        shutil.copy(os.path.join(template_project_folder, 'requirements.txt'), target_folder)
    if not os.path.exists(os.path.join(target_folder, 'README.md')):
        # 修改readme信息
        with open( os.path.join(template_project_folder, 'README.md'), 'r') as fp:
            content = fp.read()
            content = content.replace('cifar10', exp_name)

        with open( os.path.join(target_folder, 'README.md'), 'w') as fp:
            fp.write(content)

    if not os.path.exists(os.path.join(target_folder, '.gitignore')):
        shutil.copy(os.path.join(template_project_folder, '.gitignore'), target_folder)


def project_add_action(action_name, args):
    if args.project == '':
        logging.error('Must set --project')
        return
    
    if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} not existed.')
        return        

    get_lock().acquire()
    project_info = {}
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp) 

    if action_name == 'expert':
        if args.exp not in project_info['exp']:
            logging.error(f'Exp {args.exp} not in project {args.project}.')
            get_lock().release()
            return

        is_finding_exp = False
        for exp_info in project_info['exp'][args.exp]:
            if args.id == exp_info['id']:
                is_finding_exp = True
                break

        if not is_finding_exp:
            logging.error(f'Exp {args.exp} ID {args.id} not in project exp list.')
            get_lock().release()
            return
        
        exp_with_id = f'{args.exp}.{args.id}'
        if exp_with_id in project_info['expert']:
            logging.error(f'Exp {args.exp} ID {args.id} has existed in project {args.project} expert list')
            get_lock().release()
            return

        project_info['expert'].append(exp_with_id)
        pprint(project_info['expert'])
    elif action_name == 'product':
        if args.exp not in project_info['exp']:
            logging.error(f'Exp {args.exp} not in project exp list.')
            get_lock().release()
            return 
        project_info['product'] = args.exp

        # 因为要触发后台，任务调度服务，提前保存项目信息
        with open(os.path.join(config.AntConfig.task_factory, f'{args.project}.json'), 'w') as fp:
            json.dump(project_info, fp)  
        
        pprint(project_info['product'])
        get_lock().release()
        ######### trigger:start #########
        # 重新训练产品模型（监督训练->半监督训练->蒸馏训练）
        get_client().trigger(f'{args.project}.product')
        ######### trigger:end   #########        
        return
    elif action_name == 'baseline':
        if args.exp not in project_info['exp']:
            logging.error(f'Exp {args.exp} not in project {args.project}.')
            get_lock().release()
            return

        is_finding_exp = False
        for exp_info in project_info['exp'][args.exp]:
            if args.id == exp_info['id']:
                is_finding_exp = True
                break

        if not is_finding_exp:
            logging.error(f'Exp {args.exp} ID {args.id} not in project exp list.')
            get_lock().release()
            return

        exp_with_id = f'{args.exp}.{args.id}'
        project_info['baseline'] = exp_with_id
        pprint(project_info['baseline'])
    elif action_name == 'train/label':
        if args.address is None:
            logging.error('Must set --address')
            get_lock().release()
            return
        
        if not file_client_exists(f'{args.address}-index') or not file_client_exists(f'{args.address}-tfrecord'):
            logging.error(f'Data {args.address} dont exist')
            get_lock().release()
            return
        
        project_info['dataset']['train']['label'].append(
            {
                "tag": args.tags,
                "num": args.num,
                "status": True,
                'address': args.address
            }
        )

        # 因为要触发后台，任务调度服务，提前保存项目信息
        with open(os.path.join(config.AntConfig.task_factory, f'{args.project}.json'), 'w') as fp:
            json.dump(project_info, fp)  

        pprint(project_info['dataset']['train']['label'])
        get_lock().release()
        ######### trigger:start #########
        # 基于更新后的有标签数据，基于当前最佳产品模型，重新监督训练/蒸馏训练
        get_client().trigger(f'{args.project}.train/label')
        ######### trigger:end   #########
        return
    elif action_name == 'train/unlabel':
        if args.address is None:
            logging.error('Must set --address')
            get_lock().release()
            return
        
        if not file_client_exists(f'{args.address}-index') or not file_client_exists(f'{args.address}-tfrecord'):
            logging.error(f'Data {args.address} dont exist')
            get_lock().release()
            return

        project_info['dataset']['train']['unlabel'].append(
            {
                "tag": args.tags,
                "num": args.num,
                "status": True,
                'address': args.address
            }
        )

        # 因为要触发后台，任务调度服务，提前保存项目信息
        with open(os.path.join(config.AntConfig.task_factory, f'{args.project}.json'), 'w') as fp:
            json.dump(project_info, fp)  
        
        pprint(project_info['dataset']['train']['unlabel'])
        get_lock().release()
        ######### trigger:start #########
        # 基于更新后的无标签数据，基于当前最佳产品模型，重新半监督模型训练
        get_client().trigger(f'{args.project}.train/unlabel')
        ######### trigger:end   #########
        return
    elif action_name == 'test':
        if args.address is None:
            logging.error('Must set --address')
            get_lock().release()
            return
                
        project_info['dataset']['test'] = {
            "tag": args.tags,
            "num": args.num,
            "status": True,
            'address': args.address
        }
        pprint(project_info['dataset']['test'])
    elif action_name == 'val':
        if args.address is None:
            logging.error('Must set --address')
            get_lock().release()
            return
        
        project_info['dataset']['val'] = {
            "tag": args.tags,
            "num": args.num,
            "status": True,
            'address': args.address
        }
        project_info['dataset']['val']
    else:
        logging.error(f'Command {action_name} dont support.')

    # 在本地存储项目信息
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)
    get_lock().release()


def project_del_action(action_name, args):
    if args.project == '':
        logging.error('Must set --project')
        return

    if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} not existed.')
        return

    get_lock().acquire()
    project_info = {}
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp) 

    if action_name == 'expert':
        if args.exp not in project_info['exp']:
            logging.error(f'Exp {args.exp} not in project {args.project}.')
            get_lock().release()
            return
        
        exp_with_id = f'{args.exp}.{args.id}'
        if exp_with_id not in project_info['expert']:
            logging.error(f'{exp_with_id} not exist in project {args.project}')
            get_lock().release()
            return

        project_info['expert'].remove(exp_with_id)
        logging.info(f'Del expert {exp_with_id} of project {args.project}')
    elif action_name == 'train/label':
        after_list = []
        for info in project_info['dataset']['train']['label']:
            if info['tag'] == args.tags:
                continue
            after_list.append(info)
        
        project_info['dataset']['train']['label'] = after_list
        logging.info(f'Del train/label with tags {args.tags} of project {args.project}')
    elif action_name == 'train/unlabel':
        after_list = []
        for info in project_info['dataset']['train']['unlabel']:
            if info['tag'] == args.tags:
                continue
            after_list.append(info)
        
        project_info['dataset']['train']['unlabel'] = after_list
        logging.info(f'Del train/unlabel with tags {args.tags} of project {args.project}')
    elif action_name == 'test':
        project_info['dataset']['test'] = {}
        logging.info(f'Del test dataset of project {args.project}')
    elif action_name == 'val':
        project_info['dataset']['val'] = {}
        logging.info(f'Del val dataset of project {args.project}')
    else:
        logging.error(f'Command {action_name} dont support.')

    # 在本地存储项目信息
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)    

    get_lock().release()


def show_action(action_name, args):
    if action_name == 'best':
        if args.project == '':
            logging.error('Must set --project')
            return

        if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
            logging.error(f'Project {args.project} not existed.')
            return   
        
        project_info = {}
        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
            project_info = json.load(fp) 
                    
        pprint(project_info['best'])
    elif action_name == 'project':
        if args.project == '':
            all_projects = []
            for project_file in os.listdir(config.AntConfig.task_factory):
                if not project_file.endswith('.json'):
                    continue
                all_projects.append(project_file.replace('.json', ''))

            pprint(all_projects)
        else:
            if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
                logging.error(f'Project {args.project} not existed.')
                return

            project_info = {}
            with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
                project_info = json.load(fp) 

            pprint(project_info)
    elif action_name == "exp":
        if args.project == '':
            logging.error('Must set --project')
            return

        if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
            logging.error(f'Project {args.project} not existed.')
            return   

        project_info = {}
        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
            project_info = json.load(fp) 

        pprint(project_info['exp'])


def get_action(action_name, args):
    logging.error('Now dont support')


def update_project_config(sub_action_name, args):
    if args.project == '':
        logging.error('Must set --project')
        return

    if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} not existed.')
        return

    get_lock().acquire()
    project_info = {}
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp) 

    if sub_action_name == 'project/semi':
        # 更新半监督方案配置
        # args.name 来检查是否是内置算法
        if args.name == '' and args.config == '':
            # 清空配置
            project_info['tool']['semi']['method'] = ''
            project_info['tool']['semi']['config'] = dict()

            with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
                json.dump(project_info, fp)
            logging.info(f'Success remove {sub_action_name} config')  
            get_lock().release()
            return    
        if not os.path.exists(args.config):
            if args.name not in ['dense', 'detmpl', 'mpl']:
                logging.error(f'Missing config file, {sub_action_name} couldnt config')
                get_lock().release()
                return
            else:
                args.config = os.path.join(os.path.dirname(__file__), 'framework', 'helper', 'configs', sub_action_name.split('/')[-1], f'{args.name}_config.py')

        if not os.path.exists(args.config):
            logging.error(f'Config file {args.config} not exist')
            get_lock().release()
            return

        # 读取配置
        assert(args.config.endswith('.py'))
        cfg = Config.fromfile(args.config)

        # 设置半监督算法的名字        
        project_info['tool']['semi']['method'] = args.name
        # 设置半监督的配置
        project_info['tool']['semi']['config'].update(
            cfg._cfg_dict.to_dict()
        )

        # print info
        pprint(project_info['tool']['semi'])
    elif sub_action_name == 'project/distillation':
        # args.name 来检查是否是内置算法
        if args.name == '' and args.config == '':
            # 清空配置
            project_info['tool']['distillation']['method'] = ''
            project_info['tool']['distillation']['config'] = dict()

            with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
                json.dump(project_info, fp)
            logging.info(f'Success remove {sub_action_name} config')    
            get_lock().release()          
            return    

        if not os.path.exists(args.config):
            if args.name not in ['reviewkd']:
                logging.error(f'Missing config file, {sub_action_name} couldnt config')
                get_lock().release()
                return
            else:
                args.config = os.path.join(os.path.dirname(__file__), 'framework', 'helper', 'configs', sub_action_name.split('/')[-1], f'{args.name}_config.py')

        if not os.path.exists(args.config):
            logging.error(f'Config file {args.config} not exist')
            get_lock().release()
            return

        # 读取配置
        assert(args.config.endswith('.py'))
        cfg = Config.fromfile(args.config)

        # 设置蒸馏算法的名字        
        project_info['tool']['distillation']['method'] = args.name
        # 设置蒸馏的配置
        project_info['tool']['distillation']['config'].update(
            cfg._cfg_dict.to_dict()
        )

        # print info
        pprint(project_info['tool']['distillation'])
    elif sub_action_name == 'project/activelearning':
        # 更新主动学习方案配置
        if args.name == '' and args.config == '':
            # 清空配置
            project_info['tool']['activelearning']['method'] = ''
            project_info['tool']['activelearning']['config'] = dict()

            with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
                json.dump(project_info, fp)
            logging.info(f'Success remove {sub_action_name} config') 
            get_lock().release()             
            return    

        if not os.path.exists(args.config):
            if args.name not in ['dense', 'detmpl', 'mpl', 'ac']:
                logging.error(f'Missing config file, {sub_action_name} couldnt config')
                get_lock().release()
                return
            else:
                args.config = os.path.join(os.path.dirname(__file__), 'framework', 'helper', 'configs', sub_action_name.split('/')[-1], f'{args.name}_config.py')

        if not os.path.exists(args.config):
            logging.error(f'Config file {args.config} not exist')
            get_lock().release()
            return

        # 读取配置
        assert(args.config.endswith('.py'))
        cfg = Config.fromfile(args.config)

        # 设置主动学习算法的名字        
        project_info['tool']['activelearning']['method'] = args.name
        # 设置主动学习的配置
        project_info['tool']['activelearning']['config'].update(
            cfg._cfg_dict.to_dict()
        )
        
        # print info
        pprint(project_info['tool']['activelearning'])
    elif sub_action_name == 'project/label':
        assert(args.config.endswith('.py'))
        cfg = Config.fromfile(args.config)

        cfg.type = cfg.type.upper()
        if cfg.type not in ['', 'CLASS', 'RECT','POINT','POLYGON', 'SKELETON']:
            logging.error(f"Dont support label type {cfg.type}")
            get_lock().release()
            return

        if not isinstance(cfg.category, list):
            logging.error(f"Label category must be list type")
            get_lock().release()
            return

        if cfg.type == 'SKELETON':
            # 对于关键点标注，需要添加meta/skeleton信息
            if cfg.get('meta', None):
                logging.error('Must set meta/skeleton info for SKELETON label')
                get_lock().release()
                return

            if len(cfg.meta.get('skeleton', [])) == 0:
                logging.error('Must set meta/skeleton info for SKELETON label')
                get_lock().release()
                return

        project_info['tool']['label']['category'] = cfg.category    # 标注类别 ['clsname', 'clsname', ...]
        project_info['tool']['label']['type'] = cfg.type            # 标注类型
        if cfg.get('meta', None):
            project_info['tool']['label']['meta']['skeleton'] = cfg.meta.get('skeleton', [])

        # print info
        pprint(project_info['tool']['label'])
    elif sub_action_name == 'project/submitter':
        if args.ssh:
            project_info['submitter']['method'] = 'ssh'
        elif args.local:
            project_info['submitter']['method'] = 'local'
        else:
            # 检查本地是否存在定制化提交脚本
            submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'submit-config.yaml')    
            if not os.path.exists(submit_config_file):
                logging.error('No custom submit script config')
                get_lock().release()
                return

            with open(submit_config_file, encoding='utf-8', mode='r') as fp:
                config_content = yaml.safe_load(fp)
            script_folder = config_content['folder']
            script_file = config_content['script']
            if not os.path.exists(os.path.join(script_folder, script_file)):
                logging.error('Custom submit scrip launch file not exist.')
                get_lock().release()
                return

            project_info['submitter']['method'] = 'custom'

        if args.gpu >= 0:
            project_info['submitter']['gpu_num'] = args.gpu
        if args.cpu >= 0:
            project_info['submitter']['cpu_num'] = args.cpu
        if args.memory >= 0:
            project_info['submitter']['memory'] = args.memory

        # print info
        pprint(project_info['submitter'])
    else:
        logging.error(f"Dont support {sub_action_name}")
        get_lock().release()
        return

    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)

    logging.info(f'Success update {sub_action_name} config')    
    get_lock().release()
