#ifndef TRACKER_OP_H_
#define TRACKER_OP_H_
#include <iostream>
#include <vector>
#include "defines.h"
#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/KalmanFilter.h"
#include "ByteTrack/STrack.h"
#include "ByteTrack/Rect.h"
#include "ByteTrack/Object.h"
#include "ByteTrack/lapjv.h"
#include "Eigen/Dense"

using namespace antgo;
ANTGO_CLASS class TrackerOp{
public:
    TrackerOp(int frame_rate, float track_thresh, float high_thresh, float match_thresh){
        m_tracker = new byte_track::BYTETracker(frame_rate, 100, track_thresh, high_thresh, match_thresh);
    }
    virtual ~TrackerOp(){
        delete m_tracker;
    }

    void run(const CFTensor* obj_bboxes, CFTensor* tracker_info){
        std::vector<byte_track::Object> observe_objs;
        int obj_bbox_num = obj_bboxes->dims[0];
        if(obj_bbox_num == 0){
            tracker_info->create2d(0, 7);   // x0,y0,x1,y1,label,score,tracker_id
            return;
        }
        int obj_bbox_dim = obj_bboxes->dims[1];
        float* obj_bbox_data = obj_bboxes->data;
        for(int obj_i=0; obj_i<obj_bbox_num; ++obj_i){
            float* obj_i_bbox_data = obj_bbox_data + obj_i*obj_bbox_dim;
            byte_track::Rect obj_rect(obj_i_bbox_data[0], obj_i_bbox_data[1], obj_i_bbox_data[2]-obj_i_bbox_data[0], obj_i_bbox_data[3]-obj_i_bbox_data[1]);
            float obj_prob = obj_i_bbox_data[4];
            int obj_label = obj_i_bbox_data[5];
            byte_track::Object obj(obj_rect, obj_label, obj_prob);
            observe_objs.push_back(obj);
        }
        std::vector<byte_track::BYTETracker::STrackPtr> trackers = this->m_tracker->update(observe_objs);
        int tracker_num = trackers.size();
        tracker_info->create2d(tracker_num, 7);
        float* tracker_bbox_data = tracker_info->data;
        for(int tracker_i=0; tracker_i<tracker_num; ++tracker_i){
            float* tracker_i_bbox_data = tracker_bbox_data + tracker_i * 7;
            byte_track::Rect tracker_rect = trackers[tracker_i]->getRect();
            tracker_i_bbox_data[0] = tracker_rect.tl_x();
            tracker_i_bbox_data[1] = tracker_rect.tl_y();
            tracker_i_bbox_data[2] = tracker_rect.br_x();
            tracker_i_bbox_data[3] = tracker_rect.br_y();

            tracker_i_bbox_data[4] = trackers[tracker_i]->getScore();       // score
            tracker_i_bbox_data[5] = trackers[tracker_i]->getLabel();       // label
            tracker_i_bbox_data[6] = trackers[tracker_i]->getTrackId();     // trackerid
        }
    }

private:
    byte_track::BYTETracker* m_tracker;
};

#endif