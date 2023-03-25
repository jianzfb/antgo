# encoding=utf-8
# @Time    : 22-3-7
# @File    : help.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from genericpath import isdir

import os
import git
import logging
import shutil
from antgo import config
import json
from pprint import pprint

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
        if args.branch is not None:
            if args.commit is not None:
                os.system(f'git clone {project_git} -b {args.branch}; git reset --soft {args.commit}')
            else:
                os.system(f'git clone {project_git}; -b {args.branch};')
        else:
            os.system(f'git clone {project_git}')

        git_folder = project_git.split('/')[-1].split('.')[0]
        os.chdir(f'./{git_folder}')

    return True


def generate_project_exp_example(template_project_folder, target_folder):
    # step1: 拷贝cifar10分类模型代码
    if not os.path.exists(os.path.join(target_folder, 'cifar10')):
        shutil.copytree(os.path.join(template_project_folder, 'cifar10'), os.path.join(target_folder, 'cifar10'))
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
        shutil.copy(os.path.join(template_project_folder, 'README.md'), target_folder)


def exp_basic_info():
    return {
        "exp": "", 
        "branch": "", 
        "commit": "", 
        "metric": {}, 
        "dataset": {"test": "", "train": []},
        "checkpoint": "", 
        "create_time": "", 
        "finish_time": ""          
    }


def dataset_basic_info():
    return {
        "tag": "",
        "num": 0,
        "status": True,
        'address': ""
    }


def project_add_action(action_name, args):
    if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} not existed.')
        return        

    project_info = {}
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp) 
        
    if action_name == 'expert':
        assert(args.exp in project_info['exp'])
        if args.exp in project_info['expert']:
            logging.error(f'{args.exp} has existed in project {args.project}')
            return
        
        project_info['expert'].append(args.exp)
    elif action_name == 'product':
        assert(args.exp in project_info['exp'])        
        project_info['product'] = args.exp
    elif action_name == 'baseline':
        assert(args.exp in project_info['exp'])        
        project_info['baseline'] = args.exp        
    elif action_name == 'train/label':
        assert(args.address is not None)
        project_info['dataset']['train']['label'].append(
            {
                "tag": args.tags,
                "num": args.num,
                "status": True,
                'address': args.address
            }
        )
    elif action_name == 'train/unlabel':
        assert(args.address is not None)
        project_info['dataset']['train']['unlabel'].append(
            {
                "tag": args.tags,
                "num": args.num,
                "status": True,
                'address': args.address
            }
        )
    elif action_name == 'test':
        assert(args.address is not None)
        project_info['dataset']['test'] = {
            "tag": args.tags,
            "num": args.num,
            "status": True,
            'address': args.address
        }
    elif action_name == 'val':
        assert(args.address is not None)
        project_info['dataset']['val'] = {
            "tag": args.tags,
            "num": args.num,
            "status": True,
            'address': args.address
        }
    else:
        logging.error(f'Command {action_name} dont support.')

    # 在本地存储项目信息
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)


def project_del_action(action_name, args):
    if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} not existed.')
        return        

    project_info = {}
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp) 

    if action_name == 'expert':
        assert(args.exp in project_info['exp'])
        if args.exp not in project_info['expert']:
            logging.error(f'{args.exp} not exist in project {args.project}')
            return
        
        project_info['expert'].remove(args.exp)
    elif action_name == 'train/label':
        after_list = []
        for info in project_info['dataset']['train']['label']:
            if info['tag'] == args.tags:
                continue
            after_list.append(info)
        
        project_info['dataset']['train']['label'] = after_list
    elif action_name == 'train/unlabel':
        after_list = []
        for info in project_info['dataset']['train']['unlabel']:
            if info['tag'] == args.tags:
                continue
            after_list.append(info)
        
        project_info['dataset']['train']['unlabel'] = after_list
    elif action_name == 'test':
        project_info['dataset']['test'] = {}
    elif action_name == 'val':
        project_info['dataset']['val'] = {}
    else:
        logging.error(f'Command {action_name} dont support.')

    # 在本地存储项目信息
    with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)    


def show_action(action_name, args):
    if action_name == 'best':
        assert(args.project != '')
        if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
            logging.error(f'Project {args.project} not existed.')
            return   
        
        project_info = {}
        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
            project_info = json.load(fp) 
                    
        pprint(project_info['best'])
    elif action_name == 'project':
        all_projects = []
        for project_file in os.listdir(config.AntConfig.task_factory):
            if not project_file.endswith('.json'):
                continue
            all_projects.append(project_file.replace('.json', ''))
        
        pprint(all_projects)
    elif action_name == "exp":
        assert(args.project != '')
        if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
            logging.error(f'Project {args.project} not existed.')
            return   
        
        project_info = {}
        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
            project_info = json.load(fp) 
                    
        pprint(project_info['exp'])


def get_action(action_name, args):
    logging.error('Now dont support')
