#include "defines.h"

extern "C" {
using namespace eagleeye;

EAGLEEYE_API void destroy_cdtensor(CDTensor* t){
    t->destroy();
}

EAGLEEYE_API void destroy_cftensor(CFTensor* t){
    t->destroy();
}

EAGLEEYE_API void destroy_citensor(CITensor* t){
    t->destroy();
}

EAGLEEYE_API void destroy_cuctensor(CUCTensor* t){
    t->destroy();
}

}
