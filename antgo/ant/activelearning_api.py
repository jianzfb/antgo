import os
import requests
from antgo.ant import flags
import time
import  urllib.parse
import uuid
from antgo.utils import logger

def get_file_name(headers):
    filename = ''
    if 'Content-Disposition' in headers and headers['Content-Disposition']:
        disposition_split = headers['Content-Disposition'].split(';')
        if len(disposition_split) > 1:
            if disposition_split[1].strip().lower().startswith('filename='):
                file_name = disposition_split[1].split('=')
                if len(file_name) > 1:
                    filename = urllib.parse.unquote(file_name[1])
    if filename == '':
        filename = str(uuid.uuid4())
    
    return filename

def activelearning_api_download():
    FLAGS = flags.AntFLAGS

    # 服务ip:port
    host_ip = FLAGS.host_ip()
    host_port = FLAGS.host_port()
    round = None
    for p in FLAGS.param().split(';'):
        k,v = p.split(":")
        if k == 'round':
            round = (int)(v)

    if round is None:
        logger.error('Need set param (eg. --param=round:0)')
        return    

    download_url = 'http://%s:%d/activelearning/download/'%(host_ip, host_port)
    down_res = requests.get(url=download_url, params={'round': round})
    file_name = get_file_name(down_res.headers)

    with open(file_name, "wb") as code:
        code.write(down_res.content)


def activelearning_api_upload():
    FLAGS = flags.AntFLAGS

    # 上传的文件路径
    file_path = FLAGS.file()

    # 服务ip:port
    host_ip = FLAGS.host_ip()
    host_port = FLAGS.host_port()

    round = None
    for p in FLAGS.param().split(';'):
        k,v = p.split(":")
        if k == 'round':
            round = (int)(v)
    
    if round is None:
        logger.error('Need set param (eg. --param=round:0)')
        return    

    if not os.path.exists(file_path):
        logger.error('%s file dont exist.'%file_path)
        return

    file_name = os.path.normpath(file_path).split('/')[-1]
    file_size = os.path.getsize(file_path)
    chunk_size = 1024*1024 # 1M
    chunk_num = (file_size + chunk_size - 1)//chunk_size
    finished_size = 0
    api_url = 'http://%s:%d/activelearning/upload/'%(host_ip, host_port)
    with open(file_path, 'rb') as fp:
        for chunk_id in range(chunk_num):
            remained_size = chunk_size
            if finished_size + chunk_size < file_size:
                remained_size = chunk_size
            else:
                remained_size = finished_size+chunk_size - file_size
            content = fp.read(remained_size)
            kwargs = {}
            kwargs.update({'sliceIndex': chunk_id, 'sliceNum': chunk_num, 'fileName': file_name, 'round': round})

            result = requests.post(api_url, kwargs, files={'file': content})
            if result.status_code not in [200, 201]:
                logger.error('Upload fail, exit.')
                return