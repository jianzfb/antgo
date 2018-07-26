# -*- coding: UTF-8 -*-
# @Time : 2018/7/23
# @File : release.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.ant.base import *


class AntRelease(AntBase):
  def __init__(self,
               ant_context,
               ant_name,
               ant_token,
               ant_task,
               **kwargs):
    super(AntRelease, self).__init__(ant_name, ant_context, ant_token, **kwargs)
    self.html_template = kwargs.get('html_template', None)
    self.support_user_upload = kwargs.get('support_user_upload', False)
    self.support_user_input = kwargs.get('support_user_input', False)
    self.support_user_interaction = kwargs.get('support_user_interaction', False)
    self.support_upload_formats = kwargs.get('support_upload_formats', None)
    self.ant_task = ant_task

  def start(self):
    if self.token is None:
      logger.error('must set token')
      return

    # release on mltalker ai demo zoo
    # 1.step package codebase
    if self.ant_task is not None:
      shutil.copy(self.ant_task, self.main_folder)
    codebase_address, codebase_address_code = self.package_codebase()

    # 2.step release on mltalker
    cmd = "antgo demo"
    if self.html_template is not None:
      cmd = cmd + ' --html_template=%s'%self.html_template
    if not self.support_user_upload:
      cmd = cmd + ' --support_user_upload'
    if not self.support_user_input:
      cmd = cmd + ' --support_user_input'
    if not self.support_user_interaction:
      cmd = cmd + '  --support_user_interaction'
    if self.support_upload_formats is not None:
      cmd = cmd + ' --support_upload_formats=%s'%self.support_upload_formats

    if FLAGS.from_experiment() != "":
      cmd = cmd + ' --from_experiment=%s'%FLAGS.from_experiment()
    if self.ant_task is not None:
      cmd = cmd + ' --task=%s'%os.path.normpath(self.ant_task).split('/')[-1]

    result = self.mltalker_rpc.post_demozoo(codebase=codebase_address,
                                            codebase_code=codebase_address_code,
                                            cmd=cmd,
                                            description=getattr(self.context.params,'description', ''),
                                            running_os_platform=self.running_config['OS_PLATFORM'],
                                            running_os_version=self.running_config['OS_VERSION'],
                                            running_software_framework=self.running_config['SOFTWARE_FRAMEWORK'],
                                            running_cpu_model=self.running_config['CPU_MODEL'],
                                            running_cpu_num=self.running_config['CPU_NUM'],
                                            running_cpu_mem=self.running_config['CPU_MEM'],
                                            running_gpu_model=self.running_config['GPU_MODEL'],
                                            running_gpu_num=self.running_config['GPU_NUM'],
                                            running_gpu_mem=self.running_config['GPU_MEM'])

    if result['result'] == 'success':
      logger.info('successfully release on mltalker demo zoo')
    else:
      logger.error('fail to release on mltalker demo zoo')


