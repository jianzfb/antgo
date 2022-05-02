from antgo.dataflow.dataset.my_det_dataset import *
from antgo.dataflow.dataset.my_seg_dataset import *
from antgo.dataflow.imgaug.operators import *
from antgo.dataflow.imgaug.batch_operators import *
from antgo.dataflow.dataset.reader import *

# # 检测数据集
# dataset = CustomDetDataset('train', '/root/paddlejob/workspace/env_run/portrait/dataset/merge', {'anno_json': 'annot/valid.json'})

# # data_folder = '/Users/jian/Downloads/factory/dataset/MNIST'
# # dataset = Mnist('train', data_folder)
# reader = Reader(dataset, [
#     DecodeImage(),
#     RandomCrop(),
#     Resize((256,192)),
#     RandomDistort(),
#     Rotation(15),
#     RandomFlipImage(0.0),
#     NormalizeImage(is_channel_first=False,is_scale=False),
#     Permute(to_bgr=False, channel_first=True)
# ], [
#     Gt2TTFKPTTarget(num_classes=1, down_ratio=4),
#     PadBatch(pad_to_stride=16)
# ]
# , 4, True, True, inputs_def={
#     'fields': ['image']
# })

# for data in reader.iterator_value():
#   print(data)

# 分割数据集
dataset = CustomSegDataset('train', '/root/paddlejob/workspace/env_run/portrait/seg/humanseg_new', {'anno_file': 'train.txt'})
reader = Reader(dataset, [
    DecodeImage(),
    Rotation(80),
    Resize((256,192)),
    # RandomPaddingCrop(),
    Resize((128,128)),
    RandomFlipImage(),
    RandomDistort(),
    # NormalizeImage(is_channel_first=False,is_scale=False),
    Permute(to_bgr=False, channel_first=False)
], [
]
, 4, True, True, inputs_def={
    'fields': ['image','semantic']
}, use_process=True, worker_num=1)

for data in reader.iterator_value():
    image = data[0][0]
    mask = data[0][1]
    cv2.imwrite('./data.png', image)
    cv2.imwrite("./mask.png",(mask*255).astype(np.uint8))
    print('ss')
