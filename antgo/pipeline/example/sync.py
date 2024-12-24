import os
from antgo.pipeline import *

# 数据生成模式1（基于贴图方式生成）
def layoutg_func(image_wo_bg):
    layout_image = image_wo_bg[:,:,:3]
    layout_id = image_wo_bg[:,:,-1]/255

    return {
        'layout_image': image_wo_bg,
        'layout_id': layout_id,
    }


with GroupRegister('layoutg') as layoutg_group:
    layoutg_group.image_decode.remote.removebg.demo.runas_op(
            [
                {
                    'folder': ''
                },
                {
                    'image': 'image',
                    'image_mask': 'image',
                    'image_wo_bg': 'image'
                },
                {
                    'func': layoutg_func
                }
            ],
            relation=[
                ['layout_image_path','image'],
                ['image', ('image_mask', 'image_wo_bg')],
                ['image_wo_bg', 'layout_info']
            ],
            input=['layout_image_path'],
            output=['layout_info']
        )


with GroupRegister('syncg') as syncg_group:
    syncg_group.image_decode.sync_op.save_sync_info_op(
        [
            {
                'folder': ''
            },
            {
                'min_scale': 0.5,
                'max_scale': 1.0,
            },
            {
                'folder': ''
            }
        ],
        relation=[
            ['image_path', 'image'],
            [('image', 'layout_info'), 'sync_info'],
            ['sync_info', 'sync_out']
        ],
        input=['image_path', 'layout_info'],
        output=['sync_out']
    )



# 数据模式生成2（基于AnyGS数据服务引擎）
with GroupRegister('removebg') as removebg_group:
    removebg_group.image_decode.remote.removebg.demo(
            [
                {
                    'folder': ''
                },
                {
                    'image': 'image',
                    'image_mask': 'image',
                    'image_wo_bg': 'image'
                }
            ],
            relation=[['image_path','image'],['image', ('image_mask', 'image_wo_bg')]],
            input=['image_path'],
            output=['image_mask', 'image_wo_bg']
        )


def augprompt_func(prompt, weather):
    aug_prompt = f'{prompt} with {weather}'
    return aug_prompt


with GroupRegister('augprompt') as augprompt_group:
    augprompt_group.runas_op(
        [
            {
                'func': augprompt_func
            }
        ],
        relation=[[('0','1'), '0']],
        input=['0', '1'],
        output=['0']
    )


def warp_info_func(image, info, message):
    warp_info = {
        'image': image,
        'bboxes': [json.loads(info)],
        'labels': [0]
    }
    return warp_info


with GroupRegister('anygs') as anygs_group:
    anygs_group.remote.anygs.demo.runas_op.save_sync_info_op(
        [
            {
                'polar_range': 40,
                'azimuth_range': 40,
                'image': 'image',
                'prompt': 'text',
                'min_obj_ratio': 'text',
                'max_obj_ratio': 'text',
                'sync_image': 'image',
                'sync_info': 'text',
                'sync_message': 'text'
            },
            {
                'func': warp_info_func
            },
            {
                'folder': ''
            }
        ],
        relation=[
            [
                ('image','prompt','min_obj_ratio','max_obj_ratio'), ('sync_image','sync_info','sync_message')
            ],
            [
                ('sync_image','sync_info','sync_message'), 'warp_info'
            ],
            [
                'warp_info', 'sync_out'
            ]
            ],
        input=['image','prompt','min_obj_ratio','max_obj_ratio'],
        output=['sync_out']
    )


def create_data_gen_base_pipe(folder, sample_num=10000, dataset_format='yolo'):
    data_gen_base_pipe = placeholder['bg_list', 'obj_list'](). \
        control.For.layoutg['obj_list', 'layout_info_list'](). \
        control.RandomChoice.syncg[('bg_list', 'layout_info_list'), 'sync_out'](
            sampling_num=sample_num, 
            syncg={
                'save_sync_info_op': {
                    'folder': folder,
                    'category_map':  {'obj': 0},
                    'sample_num': sample_num,
                    'dataset_format': dataset_format
                }
            }
        )
    return data_gen_base_pipe

def create_data_gen_pro_pipe(folder, sample_num=10000, dataset_format='yolo'):
    data_gen_pro_pipe = placeholder['obj_list', 'prompt', 'weather_list', 'min_obj_ratio', 'max_obj_ratio'](). \
        control.For.removebg['obj_list', ('obj_mask', 'obj_wo_bg')](). \
        control.For.augprompt[('prompt', 'weather_list'), 'prompt_list'](). \
        control.RandomChoice.anygs[('obj_wo_bg', 'prompt_list', 'min_obj_ratio', 'max_obj_ratio'), 'sync_out'](
            sampling_num=sample_num,
            anygs={
                'save_sync_info_op': {
                    'folder': folder,
                    'category_map':  {'object': 0},
                    'sample_num': sample_num,
                    'dataset_format': dataset_format
                }
            }
        )

    return data_gen_pro_pipe