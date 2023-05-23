#include <iostream>
#include "defines.h"

using namespace eagleeye;
EAGLEEYE_FUNC void preprocess_func(
    const CFTensor* meanv, 
    const CFTensor* stdv, 
    const CITensor* permute, 
    const bool needed_expand_batch_dim,
    const CUCTensor* image, CFTensor* output){

    // subtract mean, div std
    const unsigned char* image_ptr = image->data;
    const float* mean_ptr = meanv->data;
    const float* std_ptr = stdv->data;
    const int* permute_ptr = permute->data;

    if(needed_expand_batch_dim){
        output->create4d(1, image->dims[0], image->dims[1], image->dims[2]);
    }
    else{
        output->create3d(image->dims[0], image->dims[1], image->dims[2]);
    }

    float* output_ptr = output->data;
    int num = image->dims[0] * image->dims[1] * image->dims[2];
    for(int offset=0; offset<num; offset += 3){
        output_ptr[offset] = ((float)(image_ptr[offset]) - mean_ptr[0])/std_ptr[0];
        output_ptr[offset+1] = ((float)(image_ptr[offset+1]) - mean_ptr[1])/std_ptr[1];
        output_ptr[offset+2] = ((float)(image_ptr[offset+2]) - mean_ptr[2])/std_ptr[2];
    }
}