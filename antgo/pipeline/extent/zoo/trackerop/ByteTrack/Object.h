#pragma once

#include "Rect.h"

namespace byte_track
{
struct Object
{
    Rect<float> rect;
    int label;
    float prob;

    Object(const Rect<float> &_rect,
           const int &_label,
           const float &_prob);
};
}