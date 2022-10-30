#pragma once

#include "cuda_utils.hpp"
#include <cuda_runtime_api.h>
#include <vector>

#define THREADS_PER_BLOCK 32;

void launch_resize_kernel(uint8_t* src_img, float src_h, float src_w,
                            float dst_h, float dst_w, uint8_t * dst_img);