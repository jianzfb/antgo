import os
from aligo import Aligo

def main():
    ali = Aligo()

    shared_data_info_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resource', 'dataset')
    # 1.step ade20k
    file_obj = ali.get_file_by_path('/dataset/ade20k/ADEChallengeData2016.zip')
    result = ali.share_file_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'ade20k.txt'), 'w') as fp:
        fp.write(result)
    print('finish ade20k')

    # 2.step 3dpw
    file_obj = ali.get_file_by_path('/dataset/3dpw/3dpw.tar')
    result = ali.share_file_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, '3dpw.txt'), 'w') as fp:
        fp.write(result)
    print('finish 3dpw')

    # 3.step cityscapes
    file_obj = ali.get_file_by_path('/dataset/cityscapes/cityscapes.tar')
    result = ali.share_file_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'cityscapes.txt'), 'w') as fp:
        fp.write(result)
    print('finish cityscapes')


    # 4.step h36m
    file_obj = ali.get_folder_by_path('/dataset/h36m')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'h36m.txt'), 'w') as fp:
        fp.write(result)
    print('finish h36m')

    # 5.step lfw
    file_obj = ali.get_folder_by_path('/dataset/lfw')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder,'lfw.txt'), 'w') as fp:
        fp.write(result)
    print('finish lfw')


    # 6.step lip
    file_obj = ali.get_folder_by_path('/dataset/lip')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder,'lip.txt'), 'w') as fp:
        fp.write(result)
    print('finish lip')


    # 6.step voc
    file_obj = ali.get_folder_by_path('/dataset/voc')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'voc.txt'), 'w') as fp:
        fp.write(result)
    print('finish voc')

    # 7.step coco
    file_obj = ali.get_folder_by_path('/dataset/coco')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder,'coco.txt'), 'w') as fp:
        fp.write(result)
    print('finish coco')

    # 7.step flic
    file_obj = ali.get_folder_by_path('/dataset/flic')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'flic.txt'), 'w') as fp:
        fp.write(result)
    print('finish flic')

    # 8.step vgg-face2
    file_obj = ali.get_folder_by_path('/dataset/vgg-face2/data')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'vgg-face2-data.txt'), 'w') as fp:
        fp.write(result)

    file_obj = ali.get_folder_by_path('/dataset/vgg-face2/meta')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'vgg-face2-meta.txt'), 'w') as fp:
        fp.write(result)
    print('finish vgg-face2')

    # 9.step lsp
    file_obj = ali.get_folder_by_path('/dataset/lsp')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'lsp.txt'), 'w') as fp:
        fp.write(result)
    print('finish lsp')

    # 10.step mpii
    file_obj = ali.get_folder_by_path('/dataset/mpii')
    file_obj = ali.get_file_list(file_obj.file_id)
    result = ali.share_files_by_aligo(file_obj)

    with open(os.path.join(shared_data_info_folder, 'mpii.txt'), 'w') as fp:
        fp.write(result)
    print('finsh mpii')


if __name__ == '__main__':
  main()
