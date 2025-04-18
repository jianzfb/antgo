import os
from antgo.pipeline import *

# 数据生成模式1（基于贴图方式生成）
@register
class layoutg_gen(object):
    def __init__(self):
        pass

    def __call__(self, image, anno):
        # 图像数据
        image = image[:,:,:3]
        image_h, image_w = image.shape[:2]

        # 标注数据
        # {'polygon': xxx, 'points': xxx}
        polygon = anno.get('polygon', None)
        points = anno.get('points', None)
        points = np.array(points, dtype=np.float32)

        # ploygon -> mask
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int64).reshape(-1,2)], 1)

        return {
            'layout_image': image,
            'layout_id': mask,
            'layout_points': points
        }


with GroupRegister['layout_image_path', 'layout_info']('layoutg') as layoutg_group:
    layoutg_group.image_decode.remote.removebg.demo.layoutg_gen(
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

                }
            ],
            relation=[
                ['layout_image_path','image'],
                ['image', ('image_mask', 'image_wo_bg')],
                ['image_wo_bg', 'layout_info']
            ]
        )


@register
class layoutg2_gen(object):
    def __init__(self):
        pass

    def __call__(self, image, anno):
        # 图像数据
        image = image[:,:,:3]
        image_h, image_w = image.shape[:2]

        # 标注数据
        # {'polygon': xxx, 'points': xxx}
        polygon = anno.get('polygon', None)
        points = anno.get('points', None)
        points = np.array(points, dtype=np.float32)

        # ploygon -> mask
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int64).reshape(-1,2)], 1)

        return {
            'layout_image': image,
            'layout_id': mask,
            'layout_points': points
        }


with GroupRegister['layout_image_path', 'layout_info']('layoutg2') as layoutg_group:
    layoutg_group.image_decode.json_load.layoutg2_gen(
            [
                {
                    'folder': ''
                },
                {
                    
                },
                {
                    
                }
            ],
            relation=[
                ['layout_image_path','image'],
                ['layout_image_path', 'anno'],
                [('image', 'anno'), 'layout_info']
            ]
        )


with GroupRegister[('image_path', 'layout_info'), 'sync_out']('syncg') as syncg_group:
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
        ]
    )



# 数据模式生成2（基于AnyGS数据服务引擎）
with GroupRegister['image_path', ('image_mask', 'image_wo_bg')]('removebg') as removebg_group:
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
            relation=[['image_path','image'],['image', ('image_mask', 'image_wo_bg')]]
        )


def augprompt_func(prompt, weather):
    aug_prompt = f'{prompt} with {weather}'
    return aug_prompt


with GroupRegister[('prompt', 'weather_list'), 'prompt_list']('augprompt') as augprompt_group:
    augprompt_group.runas_op(
        [
            {
                'func': augprompt_func
            }
        ],
        relation=[[('0','1'), '0']]
    )


def warp_info_func(image, info, message):
    warp_info = {
        'image': image,
        'bboxes': [json.loads(info)],
        'labels': [0]
    }
    return warp_info


with GroupRegister[('image','prompt','min_obj_ratio','max_obj_ratio'), 'sync_out']('anygs') as anygs_group:
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


def create_data_gen_base_pipe(folder, sample_num=10000, obj_num=1, dataset_format='yolo', stage='train', task='detect', prefix='data', is_auto=True, callback=None):
    if is_auto:
        if task == 'pose':
            print('Only detect,segment,classify task support auto mode')
            return None

    if is_auto:
        # 基于主体物分割模型，自动对目标图像进行分层以获得主体物
        data_gen_base_pipe = placeholder['bg_list', 'obj_list'](). \
            control.For.layoutg['obj_list', 'layout_info_list'](). \
            control.RandomChoice.syncg[('bg_list', 'layout_info_list'), 'sync_out'](
                sampling_num=sample_num, 
                sampling_group=(1, obj_num),
                syncg={
                    'save_sync_info_op': {
                        'folder': folder,
                        'category_map':  {'object': 0},
                        'sample_num': sample_num,
                        'dataset_format': dataset_format,
                        'callback': callback,
                        'stage': stage,
                        'mode': task,
                        'prefix': prefix
                    }
                }
            )
    else:
        # 基于解析文件，对目标图像进行拆解
        data_gen_base_pipe = placeholder['bg_list', 'obj_list'](). \
            control.For.layoutg2['obj_list', 'layout_info_list'](). \
            control.RandomChoice.syncg[('bg_list', 'layout_info_list'), 'sync_out'](
                sampling_num=sample_num, 
                sampling_group=(1, obj_num),
                syncg={
                    'save_sync_info_op': {
                        'folder': folder,
                        'category_map':  {'object': 0},
                        'sample_num': sample_num,
                        'dataset_format': dataset_format,
                        'callback': callback,
                        'stage': stage,
                        'mode': task,
                        'prefix': prefix
                    }
                }
            )

    return data_gen_base_pipe


def create_data_gen_pro_pipe(folder, sample_num=10000, obj_num=1, dataset_format='yolo', stage='train', task='detect',  prefix='data', callback=None):
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
                    'dataset_format': dataset_format,
                    'callback': callback,
                    'stage': stage,
                    'mode': task,
                    'prefix': prefix
                }
            }
        )

    return data_gen_pro_pipe