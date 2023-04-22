import os
import logging
from aligo import Aligo

def share_data_func(args):
    data_name = args.name
    shared_data_info_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resource', 'dataset')
    
    support_info_list = []
    for file_name in os.listdir(shared_data_info_folder):
        if file_name.startswith('vgg-face2'):
            continue
        support_info_list.append(file_name.split('.')[0])
    support_info_list.append('vgg-face2')

    if data_name is None:
        logging.error('--name must be set')
        print(f'share data support {support_info_list}')
        return
    
    if data_name not in support_info_list:
        print(f'share data only support {support_info_list}')
        return
    
    # 需要用户权限认证
    ali = Aligo()
    file_obj = ali.get_folder_by_path('/dataset/')
    if file_obj is None:
        ali.create_folder('/dataset')
    
    # 特殊处理vgg-face2
    if data_name == 'vgg-face2':
        with open(os.path.join(shared_data_info_folder, f'{data_name}-data.txt'), 'r') as fp:
            data_content = fp.read()
        with open(os.path.join(shared_data_info_folder, f'{data_name}-meta.txt'), 'r') as fp:
            meta_content = fp.read()
        
        file_obj = ali.get_folder_by_path(f'/dataset/{data_name}')
        if file_obj is None:
            ali.create_folder(f'/dataset/{data_name}')
        
        file_obj = ali.get_folder_by_path(f'/dataset/{data_name}/data')
        if file_obj is None:
            ali.create_folder(f'/dataset/{data_name}/data')
            file_obj = ali.get_folder_by_path(f'/dataset/{data_name}/data')
        
        ali.save_files_by_aligo(data_content, file_obj.file_id)
            
        file_obj = ali.get_folder_by_path(f'/dataset/{data_name}/meta')
        if file_obj is None:
            ali.create_folder(f'/dataset/{data_name}/meta')  
            file_obj = ali.get_folder_by_path(f'/dataset/{data_name}/meta')

        ali.save_files_by_aligo(meta_content, file_obj.file_id)      
        return
    
    with open(os.path.join(shared_data_info_folder, f'{data_name}.txt'), 'r') as fp:
        content = fp.read()
    
    file_obj = ali.get_folder_by_path(f'/dataset/{data_name}')
    if file_obj is None:
        ali.create_folder(f'/dataset/{data_name}')
        file_obj = ali.get_folder_by_path(f'/dataset/{data_name}')

    ali.save_files_by_aligo(content, file_obj.file_id)


# share_data_func('vgg-face2')