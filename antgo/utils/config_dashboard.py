from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo import config
import os
import antvis.client.mlogger as mlogger


def create_project_in_dashboard(project, experiment, auto_suffix=True):
    token = None
    if os.path.exists('./.token'):
        with open('./.token', 'r') as fp:
            token = fp.readline()

    # step 2: 检查antgo配置目录下的配置文件中是否有token
    if token is None or token == '':
        config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
        config.AntConfig.parse_xml(config_xml)
        token = getattr(config.AntConfig, 'server_user_token', '')
    if token == '' or token is None:
        print('No valid vibstring token, directly return')
        return None

    mlogger.config(project, experiment, token=token, auto_suffix=auto_suffix, server="BASELINE")
    return token
