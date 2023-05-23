#include <iostream>
#include "defines.h"

using namespace eagleeye;
EAGLEEYE_FUNC void preprocess_func(
    const CFTensor* meanv, 
    const CFTensor* stdv, 
    const bool to_chw,
    const bool needed_expand_batch_dim,
    const CUCTensor* image, CFTensor* output){

    // subtract mean, div std
    const unsigned char* image_ptr = image->data;
    const float* mean_ptr = meanv->data;
    const float* std_ptr = stdv->data;

    if(needed_expand_batch_dim){
        if(to_chw){
            output->create4d(1, image->dims[2], image->dims[0], image->dims[1]);
        }
        else{
            output->create4d(1, image->dims[0], image->dims[1], image->dims[2]);
        }
    }
    else{
        if(to_chw){
            output->create3d(image->dims[2], image->dims[0], image->dims[1]);
        }
        else{
            output->create3d(image->dims[0], image->dims[1], image->dims[2]);
        }
    }

    float* output_ptr = output->data;
    int num = image->dims[0] * image->dims[1] * image->dims[2];

    if(to_chw){
        int hw = image->dims[0]*image->dims[1];
        for(int c=0; c<3; ++c){
            output_ptr += c*hw;
            for(int p=0; p<hw; ++p){
                output_ptr[p] = ((float)(image_ptr[p*3+c]) - mean_ptr[c])/std_ptr[c];
            }
        }
    }
    else{
        for(int offset=0; offset<num; offset += 3){
            output_ptr[offset] = ((float)(image_ptr[offset]) - mean_ptr[0])/std_ptr[0];
            output_ptr[offset+1] = ((float)(image_ptr[offset+1]) - mean_ptr[1])/std_ptr[1];
            output_ptr[offset+2] = ((float)(image_ptr[offset+2]) - mean_ptr[2])/std_ptr[2];
        }
    }
}