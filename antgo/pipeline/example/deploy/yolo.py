import os
from antgo.pipeline import *

# yolo detect
def generate_yolo_detect_pipe(engine, device, imgsz, model_path, mean_val, std_val, class_num, score_thre=0.25, iou_thre=0.7):
    pipe = placeholder['image'](np.uint8). \
        resize_op['image', 'resized_image'](out_size=(imgsz,imgsz)). \
        inference_onnx_op['resized_image', 'output0'](
            onnx_path=model_path, 
            mean=mean_val,
            std=std_val,
            engine=engine,
            engine_args={
                'device':device,
            }
        ). \
        deploy.YoloDetectDecodeOp[('image', 'output0'), 'obj_bboxes'](model_i_size=imgsz, class_num=class_num, score_thre=score_thre, iou_thre=iou_thre)
    return {
        'pipe': pipe,
        'input': [('image', 'EAGLEEYE_SIGNAL_BGR_IMAGE')],
        'output': [('obj_bboxes', 'EAGLEEYE_SIGNAL_TENSOR')]
    }


# yolo tracking
def generate_yolo_track_pipe(engine, device, imgsz, model_path, mean_val, std_val, class_num, score_thre=0.25, iou_thre=0.7):
    pipe = placeholder['image'](np.uint8). \
        resize_op['image', 'resized_image'](out_size=(imgsz,imgsz)). \
        inference_onnx_op['resized_image', 'output0'](
            onnx_path=model_path, 
            mean=mean_val,
            std=std_val,
            engine=engine,
            engine_args={
                'device':device,
            }
        ). \
        deploy.YoloDetectDecodeOp[('image', 'output0'), 'obj_bboxes'](model_i_size=imgsz, class_num=class_num, score_thre=score_thre, iou_thre=iou_thre). \
        deploy.TrackerOp['obj_bboxes', 'tracker_obj_bboxes'](frame_rate=25, track_thresh=score_thre, high_thresh=score_thre, match_thresh=0.8)
    return {
        'pipe': pipe,
        'input': [('image', 'EAGLEEYE_SIGNAL_BGR_IMAGE')],
        'output': [('tracker_obj_bboxes', 'EAGLEEYE_SIGNAL_TENSOR')]
    }


# yolo pose
def generate_yolo_pose_pipe(*args, **kwargs):
    return None

# yolo segment
def generate_yolo_segment_pipe(*args, **kwargs):
    return None

# yolo clssify
def generate_yolo_classify_pipe(*args, **kwargs):
    return None
