# from warp_kv import *
import sys
sys.path.append('/workspace/antgo')
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from dataloader import KVReader
from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper.reader import *
import cv2
import bytedrh2
from bytedrh2 import photonx
from numpy import linalg as LA


def get_keys(args):
    return KVReader(*args).list_keys()

def worker_init_fn(path, dataset, _):
    # worker_info = torch.utils.data.get_worker_info()
    # # dataset = worker_info.dataset
    # # Avoid "cannot pickle KVReader object" error
    # # print("dataset reader--------")
    # # print(dataset.reader)
    dataset.reader = KVReader(f'{path}', 4)


@DATASETS.register_module()
class KVPicoHandDataset(KVReaderBase):
    def __init__(self, path, pipeline=None):
        super().__init__(pipeline=pipeline)
        
        self.worker_init_fn = lambda worker_id: worker_init_fn(path, self, worker_id)
        # if not is_dist:
        #     self.worker_init_fn(0)
        
        # dataset_id
        self.dataset_id_list = ['2a42d33ee0008612']
        self.dataset_tag_list = ['test']
        bytedrh2.authenticate(host='rh2.bytedance.net',
                user_name='zhangjian.52',
                access_token='c6b6e313f30de2f3')       
        
        # 过滤多视角
        self.filter_multi_view_ids = [2,3]
        self.support_multi_view = True      # 支持多视角后，ViewNum x C x H x W
        
        # 支持时序
        self.history_length = 1
        self.support_history = False        # 支持时序后，HistoryLen x C x H x W
        
        # bbox标签过滤条件
        self.filter_bbox_label = 1  # 类别标签
        # bbox位置过滤
        self.filter_bbox_ratius_ratio = 0.9 # 距离中心距离
        
        self.context_features = dict()
        self.context_features["image"] = tf.io.FixedLenFeature([], dtype=tf.string)
        
        self.keys = []
        for dataset_tag, dataset_id in zip(self.dataset_tag_list, self.dataset_id_list):
            dataset = photonx.Dataset(id=dataset_id, mode = 'w') # 获取数据集对象
                        
            filter_view_id = [2,3]
            view_data_list =  {}
            ready_view_data = []
            for sample in dataset:
                image_file = sample['image_file']
                tag = sample['tag']
                clip_tag = sample['clip_tag']
                image_url = sample['image_url']
                view_id = sample['view_id']
                view_num = sample['view_num']
                timestamp = sample['timestamp']
                bboxes = sample['bboxes']
                cam_param = sample['cam_param']
                labels = sample['labels']
                height = sample['height']
                width = sample['width']
                sample_id = sample['_sid']
                
                # 仅过滤出 右手
                # 基于bbox标签过滤
                filter_bbox = None
                for bbox, label in zip(bboxes, labels):
                    if (int)(label) == self.filter_bbox_label:
                        filter_bbox = bbox
                        break
                if filter_bbox is None:
                    continue
                
                # 基于bbox位置过滤
                filter_bbox_cx = (filter_bbox[0] + filter_bbox[2])/2.0
                filter_bbox_cy = (filter_bbox[1] + filter_bbox[3])/2.0
                image_cx = width / 2.0
                image_cy = height / 2.0
                
                if (abs(filter_bbox_cx - image_cx) > image_cx * self.filter_bbox_ratius_ratio) or \
                    (abs(filter_bbox_cy - image_cy) > image_cy * self.filter_bbox_ratius_ratio):
                    continue

                # 解析
                K = np.eye(3)
                K[0,0] = cam_param['I'][1]
                K[1,1] = cam_param['I'][2]
                K[0,2] = cam_param['I'][3]
                K[1,2] = cam_param['I'][4]      
                
                xi = cam_param['I'][0]
                D = np.array(cam_param['I'][5:])
                
                view_key = f'{tag}/{clip_tag}/{timestamp}'

                if view_id in self.filter_multi_view_ids:
                    if view_key not in view_data_list:
                        view_data_list[view_key] = [None for _ in self.filter_multi_view_ids]

                    # response = requests.get(sample['image_url'], verify=False) # 通过meta中存储的url可以获得数据byte，想要存储或者直接处理都可以
                    # nparr = np.fromstring(response.content, np.uint8)
                    # img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    
                    view_data_list[view_key][filter_view_id.index(view_id)] = {
                        'image': None,
                        'bbox': np.array(filter_bbox),
                        'camera': {
                            'K': K,
                            'D': D,
                            'xi': xi,
                            'E': LA.inv(np.array(cam_param['E']).reshape((4,4)))
                        },
                        'image_metas':{
                            'image_file':image_file,
                            'transform_matrix': np.eye(3),
                            'scale_factor': (1,1,1,1),
                            'image_url': image_url,      
                            'image_key': f'{dataset_tag}-{sample_id}',        # 基于此从打包数据中获得具体图像数据                      
                        }
                    }
                                    
            # 删除不符合指定需求数据
            for _, value in view_data_list.items():
                if None in view_data_list:
                    continue
                
                self.keys.append(value)
    
    def reads(self, index):
        # index is [[],[],[]]
        assert(isinstance(index, list))
        
        image_keys = []
        image_annos = []
        for view_group_id in index:
            view_group = self.keys[view_group_id]
            
            image_keys.extend([v['image_metas']['image_key'] for v in view_group])
            image_annos.extend([v for v in view_group])
            
        raw_data = self.reader.read_many(image_keys)
        contexts = tf.io.parse_sequence_example(
            raw_data, context_features=self.context_features)[0]
        
        samples = []
        for img_raw, img_anno in zip(contexts['image'], image_annos):
            image = tf.image.decode_png(img_raw.numpy(), channels=1).numpy()
            h,w = image.shape[:2]
            image = image.reshape((h,w))
            img_anno.update({
                'image': image
            })
                        
            samples.append(img_anno)
        
        return samples

if __name__ == "__main__":
    # 测试无标签
    path = '/workspace/handtt/dataset/temp'
    kv_d = KVPicoHandDataset(path,  None)    
    num = len(kv_d)
    print(f'num {num}')
    for i in range(num):
        result = kv_d.reads([i])
        print(result)

    # # 测试有标签
    # path = '/home/byte_pico_zhangjian52_hdfs/data/activelearning/label/xx'
    # kv_d = KVLabelHandDetection(path,  num_worker=0)
    # num = len(kv_d)
    # part_list = [0]
    # aa = kv_d[part_list]
    # print(aa)
    # pass