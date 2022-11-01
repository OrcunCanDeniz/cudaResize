#include <cuda_runtime_api.h>
#include <memory>

#include <iostream>
#include <stdio.h>

void launch_resize_kernel(uint8_t* src_img, int channel_size, float src_h, float src_w, float dst_h, float dst_w, uint8_t * dst_img);