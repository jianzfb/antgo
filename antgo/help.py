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


def check_project_environment(args):
    assert(args.project != '' and args.exp != '')
    if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} not exist.')
        return False
    
    with open(os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'project', 'project.json'), 'r') as fp:
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
        if cur_git_address.split('/')[-1] != project_info['git']:
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

    # step2: 拷贝shell等说明代码
    shutil.copy(os.path.join(template_project_folder, 'system.py'), target_folder)
    shutil.copy(os.path.join(template_project_folder, 'install.sh'), target_folder)
    shutil.copy(os.path.join(template_project_folder, 'launch.sh'), target_folder)
    shutil.copy(os.path.join(template_project_folder, 'requirements.txt'), target_folder)
    shutil.copy(os.path.join(template_project_folder, 'README.md'), target_folder)
