from antgo.framework.helper.runner import get_dist_info
from antgo.framework.helper.fileio import *
import os


def running_flag(root):
    if root is None:
        return
    rank, _ = get_dist_info()        
    if rank == 0:
        file_client = \
                FileClient.infer_client(None, root)
        file_client.put(b'', f'{root}/RUNNING')


def finish_flag(root):
    if root is None:
        return        
    rank, _ = get_dist_info()
    
    if rank == 0:        
        file_client = \
                FileClient.infer_client(None, root)        
        file_client.put(b'', f'{root}/FINISH')


def stop_flag(root):
    if root is None:
        return        
    rank, _ = get_dist_info()
    
    if rank == 0:     
        file_client = \
                FileClient.infer_client(None, root)        
        file_client.put(b'', f'{root}/STOP')
