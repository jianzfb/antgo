#ifndef YOLODETECTDECODE_OP_H_
#define YOLODETECTDECODE_OP_H_
#include <iostream>
#include <vector>
#include "defines.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"

using namespace antgo;
template<class T>
struct SortElement{
	SortElement(){};
	SortElement(T v,unsigned int i):value(v),index(i){};
	T value;
	unsigned int index;
};

template<typename T>
struct DescendingSort{
	typedef  T			ElementType;
	bool operator()(const SortElement<T>& a,const SortElement<T>& b){
		return a.value > b.value;
	}
};

ANTGO_CLASS class YoloDetectDecodeOp{
public:
    YoloDetectDecodeOp(int model_i_size, int class_num, float score_thre=0.1f, float iou_thre=0.45){
        this->m_model_i_size = model_i_size;
        this->m_level_0_size = this->m_model_i_size / 8;
        this->m_level_1_size = this->m_model_i_size / 16;
        this->m_level_2_size = this->m_model_i_size / 32;
        this->m_class_num = class_num;
        this->m_score_thre = score_thre;
        this->m_iou_thre = iou_thre;
    }
    virtual ~YoloDetectDecodeOp(){}

    std::vector<unsigned int> sort(std::vector<std::vector<float>>& data){
        // num*5
        std::vector<SortElement<float>> temp_vector(data.size());
        unsigned int index = 0;
        for (unsigned int i = 0; i < data.size(); ++i){
            temp_vector[i] = SortElement<float>(data[i][4], i);
        }

        //sort
        DescendingSort<float> compare_op;
        std::sort(temp_vector.begin(),temp_vector.end(),compare_op);

        std::vector<unsigned int> result_index(data.size());
        index = 0;
        typename std::vector<SortElement<float>>::iterator iter,iend(temp_vector.end());
        for (iter = temp_vector.begin(); iter != iend; ++iter){
            result_index[index] = ((*iter).index);
            index++;
        }

        return result_index;
    }

    std::vector<float> get_ious(std::vector<std::vector<float>>& all_bbox, std::vector<float>& target_bbox, std::vector<unsigned int> order, unsigned int offset){
        std::vector<float> iou_list;
        for(unsigned int i=offset; i<order.size(); ++i){
            int index = order[i];
            float inter_x1 = std::max(all_bbox[index][0], target_bbox[0]);
            float inter_y1 = std::max(all_bbox[index][1], target_bbox[1]);

            float inter_x2 = std::min(all_bbox[index][2], target_bbox[2]);
            float inter_y2 = std::min(all_bbox[index][3], target_bbox[3]);

            float inter_w = std::max(inter_x2 - inter_x1, 0.0f);
            float inter_h = std::max(inter_y2 - inter_y1, 0.0f);

            float inter_area = inter_w*inter_h;
            float a_area = (all_bbox[index][2] - all_bbox[index][0])*(all_bbox[index][3] - all_bbox[index][1]);
            float b_area = (target_bbox[2] - target_bbox[0])*(target_bbox[3] - target_bbox[1]);
            float iou = inter_area / (a_area+b_area-inter_area);
            iou_list.push_back(iou);
        }

        return iou_list;
    }

    std::vector<unsigned int> nms(std::vector<std::vector<float>>& dets, float thresh) {
        std::vector<unsigned int> order = sort(dets);
        std::vector<unsigned int> keep;

        while (order.size() > 0) {
            unsigned int index = order[0];
            keep.push_back(index);
            if (order.size() == 1) {
                break;
            }

            std::vector<float> check_ious = get_ious(dets, dets[index], order, 1);
            std::vector<unsigned int> remained_order;
            for(int i=0; i<check_ious.size(); ++i){
                if(check_ious[i] < thresh){
                    remained_order.push_back(order[i + 1]);
                }
            }
            order = remained_order;
        }

        return keep;
    }

    cv::Mat dfl(cv::Mat position){
        int p_num = 4;
        int mc = position.rows / p_num;
        // float* position_ptr = (float*)position.data;
        // p_num x mc x position.cols

        cv::Mat res(p_num, position.cols, CV_32F, cv::Scalar(0.0f));
        for(int i=0; i<p_num; ++i){
            // float* p_ptr = position_ptr + i*mc*position.cols;
            // float* p_ptr = position.ptr<float>(i*mc);
            // 计算最大值
            std::vector<float> max_y(position.cols, -1e10f);
            for(int j=0; j<mc; ++j){
                // float* m_ptr = p_ptr + j*position.cols;
                float* m_ptr = position.ptr<float>(i*mc+j);
                for(int k=0; k<position.cols; ++k){
                    if(max_y[k]<m_ptr[k]){
                        max_y[k] = m_ptr[k];
                    }
                }
            }
            // softmax
            std::vector<float> sum_y(position.cols, 0.0f);
            cv::Mat e_y(mc, position.cols, CV_32F, cv::Scalar(0.0f));
            for(int j=0; j<mc; ++j){
                // float* m_ptr = p_ptr + j*position.cols;
                float* m_ptr = position.ptr<float>(i*mc+j);
                for(int k=0; k<position.cols; ++k){
                    e_y.at<float>(j,k) = std::exp(m_ptr[k] - max_y[k]);
                    sum_y[k] += e_y.at<float>(j,k);
                }
            }
            // 
            cv::Mat y(mc, position.cols, CV_32F, cv::Scalar(0.0f));
            for(int j=0; j<mc; ++j){
                // float* m_ptr = p_ptr + j*position.cols;
                // float* m_ptr = position.ptr<float>(i*mc+j);
                for(int k=0; k<position.cols; ++k){
                    y.at<float>(j,k) = e_y.at<float>(j,k) / sum_y[k];
                }
            }

            for(int j=0; j<mc; ++j){
                for(int k=0; k<position.cols; ++k){
                    res.at<float>(i,k) += y.at<float>(j,k) * j;
                }
            }
        }

        return res;
    }

    cv::Mat boxprocess(cv::Mat position, int grid_h, int grid_w, int stride_y, int stride_x){
        // 4 x ...
        position = dfl(position);

        cv::Mat res(4, grid_h*grid_w, CV_32F, cv::Scalar(0.0f));
        for(int i=0; i<grid_h; ++i){
            for(int j=0; j<grid_w; ++j){
                // x0
                res.at<float>(0, i*grid_w+j) = (j + 0.5 - position.at<float>(0,i*grid_w+j)) * stride_x;
                // y0
                res.at<float>(1, i*grid_w+j) = (i + 0.5 - position.at<float>(1,i*grid_w+j)) * stride_y;

                // x1
                res.at<float>(2, i*grid_w+j) = (j + 0.5 + position.at<float>(2,i*grid_w+j)) * stride_x;
                // y1
                res.at<float>(3, i*grid_w+j) = (i + 0.5 + position.at<float>(3,i*grid_w+j)) * stride_y;
            }
        }

        return res;
    }

    void filterbboxes(cv::Mat bboxes, cv::Mat box_confidences, cv::Mat box_class_probs, cv::Mat& select_boxes, cv::Mat& select_classes, cv::Mat& select_scores){
        int candidate = box_class_probs.rows;
        int class_num = box_class_probs.cols;

        std::vector<int> select_i;
        std::vector<int> select_class;
        std::vector<float> select_score;
        for(int i=0; i<candidate; ++i){
            float max_score = 0.0f;
            int max_class = -1;
            for(int j=0; j<class_num; ++j){
                if(box_class_probs.at<float>(i,j) > max_score){
                    max_score = box_class_probs.at<float>(i,j);
                    max_class = j;
                }
            }

            if(max_score > this->m_score_thre){
                select_i.push_back(i);
                select_class.push_back(max_class);
                select_score.push_back(max_score);
            }
        }

        select_boxes = cv::Mat(select_i.size(), 4, CV_32F);
        select_classes = cv::Mat(select_i.size(), 1, CV_32S);
        select_scores = cv::Mat(select_i.size(), 1, CV_32F);

        for(int i=0; i<select_i.size(); ++i){
            bboxes(cv::Range(select_i[i], select_i[i]+1), cv::Range::all()).copyTo(select_boxes(cv::Range(i,i+1), cv::Range::all()));
            select_classes.at<int>(i,0) = select_class[i];
            select_scores.at<float>(i,0) = select_score[i];
        }
    }

    void run(const CUCTensor* image, const CFTensor* features, CFTensor* obj_bboxes){
        float y_scale = (float)(image->dims[0]) / (float)(this->m_model_i_size);
        float x_scale = (float)(image->dims[1]) / (float)(this->m_model_i_size);

        int level_0_bbox_num = this->m_level_0_size*this->m_level_0_size;
        int level_1_bbox_num = this->m_level_1_size*this->m_level_1_size;
        int level_2_bbox_num = this->m_level_2_size*this->m_level_2_size;

        cv::Mat input_info(64+this->m_class_num, level_0_bbox_num+level_1_bbox_num+level_2_bbox_num, CV_32F, const_cast<float*>(features->data));
        cv::Mat level_0_info = input_info(cv::Range::all(), cv::Range(0, level_0_bbox_num));
        cv::Mat level_1_info = input_info(cv::Range::all(), cv::Range(level_0_bbox_num, level_0_bbox_num+level_1_bbox_num));
        cv::Mat level_2_info = input_info(cv::Range::all(), cv::Range(level_0_bbox_num+level_1_bbox_num, level_0_bbox_num+level_1_bbox_num+level_2_bbox_num));

        cv::Mat level_0_bbox_info = level_0_info(cv::Range(0, 64), cv::Range::all());
        cv::Mat level_1_bbox_info = level_1_info(cv::Range(0, 64), cv::Range::all());
        cv::Mat level_2_bbox_info = level_2_info(cv::Range(0, 64), cv::Range::all());

        cv::Mat level_0_bbox = boxprocess(level_0_bbox_info, this->m_level_0_size, this->m_level_0_size, 8, 8);
        cv::Mat level_1_bbox = boxprocess(level_1_bbox_info, this->m_level_1_size, this->m_level_1_size, 16, 16);
        cv::Mat level_2_bbox = boxprocess(level_2_bbox_info, this->m_level_2_size, this->m_level_2_size, 32, 32);

        cv::Mat bboxes(4, level_0_bbox_num+level_1_bbox_num+level_2_bbox_num, CV_32F);
        level_0_bbox.copyTo(bboxes(cv::Range(0, 4), cv::Range(0, level_0_bbox_num)));
        level_1_bbox.copyTo(bboxes(cv::Range(0, 4), cv::Range(level_0_bbox_num, level_0_bbox_num+level_1_bbox_num)));
        level_2_bbox.copyTo(bboxes(cv::Range(0, 4), cv::Range(level_0_bbox_num+level_1_bbox_num, level_0_bbox_num+level_1_bbox_num+level_2_bbox_num)));
        bboxes = bboxes.t();

        cv::Mat level_0_class = level_0_info(cv::Range(64, 64+this->m_class_num), cv::Range::all());
        cv::Mat level_1_class = level_1_info(cv::Range(64, 64+this->m_class_num), cv::Range::all());
        cv::Mat level_2_class = level_2_info(cv::Range(64, 64+this->m_class_num), cv::Range::all());

        cv::Mat classes_conf(this->m_class_num, level_0_bbox_num+level_1_bbox_num+level_2_bbox_num, CV_32F);
        level_0_class.copyTo(classes_conf(cv::Range::all(), cv::Range(0,level_0_bbox_num)));
        level_1_class.copyTo(classes_conf(cv::Range::all(), cv::Range(level_0_bbox_num, level_0_bbox_num+level_1_bbox_num)));
        level_2_class.copyTo(classes_conf(cv::Range::all(), cv::Range(level_0_bbox_num+level_1_bbox_num, level_0_bbox_num+level_1_bbox_num+level_2_bbox_num)));
        classes_conf = classes_conf.t();

        for(int i=0; i<classes_conf.rows; ++i){
            float* classes_conf_ptr = classes_conf.ptr<float>(i);
            for(int j=0; j<classes_conf.cols; ++j){
                classes_conf_ptr[j] = 1.0f/(1.0f+std::exp(-classes_conf_ptr[j]));
            }
        }

        cv::Mat scores = cv::Mat(level_0_bbox_num+level_1_bbox_num+level_2_bbox_num, 1, CV_32F, cv::Scalar(1.0f));
        cv::Mat select_boxes;
        cv::Mat select_classes; 
        cv::Mat select_scores;
        filterbboxes(bboxes, scores, classes_conf, select_boxes, select_classes, select_scores);

        std::vector<std::vector<float>> select_nms_bboxes;
        for(int c=0; c<this->m_class_num; ++c){
            std::vector<int> select_c_i;
            for(int i=0; i<select_classes.rows; ++i){
                if(select_classes.at<int>(i,0) == c){
                    select_c_i.push_back(i);
                }
            }
            if(select_c_i.size() == 0){
                continue;
            }

            std::vector<std::vector<float>> in_bboxes;
            for(int i=0; i<select_c_i.size(); ++i){
                int select_i = select_c_i[i];
                in_bboxes.push_back(std::vector<float>{
                    select_boxes.at<float>(select_i, 0),
                    select_boxes.at<float>(select_i, 1),
                    select_boxes.at<float>(select_i, 2),
                    select_boxes.at<float>(select_i, 3),
                    select_scores.at<float>(select_i, 0),
                    (float)(c),
                    (float)(select_i)
                });
            }
            std::vector<unsigned int> remain_index;
            remain_index = this->nms(in_bboxes, this->m_iou_thre);
            
            for(int i=0; i<remain_index.size(); ++i){
                int box_i = remain_index[i];
                select_nms_bboxes.push_back(in_bboxes[box_i]);
            }
        }

        // obj bboxes
        obj_bboxes->create2d(select_nms_bboxes.size(), 6);
        for(int i=0; i<select_nms_bboxes.size(); ++i){
            float* obj_bboxes_ptr = obj_bboxes->data + i*6;
            obj_bboxes_ptr[0] = select_nms_bboxes[i][0] * x_scale;  // x0
            obj_bboxes_ptr[1] = select_nms_bboxes[i][1] * y_scale;  // y0
            obj_bboxes_ptr[2] = select_nms_bboxes[i][2] * x_scale;  // x1
            obj_bboxes_ptr[3] = select_nms_bboxes[i][3] * y_scale;  // y1
            obj_bboxes_ptr[4] = select_nms_bboxes[i][4];            // score
            obj_bboxes_ptr[5] = select_nms_bboxes[i][5];            // label
        }
    }

private:
    int m_model_i_size;
    int m_level_0_size;
    int m_level_1_size;
    int m_level_2_size;
    int m_class_num;
    float m_score_thre;
    float m_iou_thre;
};

#endif