from antgo.framework.helper.runner import get_dist_info
from antgo.framework.helper.fileio import *
import os


def running_flag(root):
    rank, _ = get_dist_info()        

    if rank == 0:
        file_client = \
                FileClient.infer_client(None, root)
        os.system('touch RUNNING')
        file_client.put_text('', root)
        os.system('rm RUNNING')


def finish_flag(root):
    rank, _ = get_dist_info()
    
    if rank == 0:        
        file_client = \
                FileClient.infer_client(None, root)        
        os.system('touch FINISH')
        file_client.put_text('', root)
        os.system('rm FINISH')


def stop_flag(root):
    rank, _ = get_dist_info()
    
    if rank == 0:     
        file_client = \
                FileClient.infer_client(None, root)        
        os.system('touch STOP')
        file_client.put_text('', root)
        os.system('rm STOP')
