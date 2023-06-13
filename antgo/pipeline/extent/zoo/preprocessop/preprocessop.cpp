#include <iostream>
#include "defines.h"

using namespace eagleeye;
ANTGO_FUNC void preprocess_func(
    const CFTensor* mean, 
    const CFTensor* std, 
    const bool rgb2bgr,
    const CUCTensor* image, CFTensor* output){

    // subtract mean, div std
    const unsigned char* image_ptr = image->data;
    const float* mean_ptr = mean->data;
    const float* std_ptr = std->data;

    int batch_size = 0;
    int offset = 0;
    int hw = 0;
    if(image->dim_size == 4){
        // NxHxWx3
        output->create4d(image->dims[0], image->dims[3], image->dims[1], image->dims[2]);
        batch_size = image->dims[0];
        offset = image->dims[1] * image->dims[2] * image->dims[3];
        hw = image->dims[1] * image->dims[2];
    }
    else{
        // HxWx3
        output->create4d(1, image->dims[2], image->dims[0], image->dims[1]);
        batch_size = 1;
        offset = image->dims[0] * image->dims[1] * image->dims[2];
        hw = image->dims[0] * image->dims[1];
    }

    float* output_ptr = output->data;
    if(rgb2bgr){
        for(int b_i=0; b_i<batch_size; ++b_i){
            float* b_output_ptr = output_ptr + b_i * offset;

            for(int c=0; c<3; ++c){
                float* b_c_output_ptr = b_output_ptr + c*hw;
                for(int p=0; p<hw; ++p){
                    b_c_output_ptr[p] = ((float)(image_ptr[b_i*offset + p*3 + (2-c)]) - mean_ptr[2-c])/std_ptr[2-c];
                }
            }
        }
    }
    else{
        for(int b_i=0; b_i<batch_size; ++b_i){
            float* b_output_ptr = output_ptr + b_i * offset;

            for(int c=0; c<3; ++c){
                float* b_c_output_ptr = b_output_ptr + c*hw;
                for(int p=0; p<hw; ++p){
                    b_c_output_ptr[p] = ((float)(image_ptr[b_i*offset + p*3 + c]) - mean_ptr[c])/std_ptr[c];
                }
            }
        }
    }
}