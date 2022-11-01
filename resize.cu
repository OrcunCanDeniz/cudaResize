#include "resize.hpp"
#include <iostream>
#include <stdio.h>
__global__ void resize_kernel(uint8_t* src_img, float scale_x, float scale_y, int dst_w, int dst_h, int src_w, int src_h, uint8_t* dst_img)
{
    int thread_ix = blockIdx.x * blockDim.x + threadIdx.x; // height dim
    int thread_iy = blockIdx.y * blockDim.y + threadIdx.y; // width dim

    if (thread_ix > dst_w || thread_iy > dst_h){return;}

    const float x_s = scale_x * thread_ix;
    const float y_s = scale_y * thread_iy;

    // pixel coords to interpolate between
    const int x_s_low = floor(x_s);
    const int y_s_low = floor(y_s);
    const int x_s_hi = ceil(x_s);
    const int y_s_hi = ceil(y_s);

    const float x_weight = 1 - (x_s - float(x_s_low)); // 1- since bigger difference should have the less weight
    const float y_weight = 1 - (y_s - float(y_s_low)); 

    const float ll = src_img[ y_s_low * src_w + x_s_low];
    const float lh = src_img[ y_s_low * src_w + x_s_hi];
    const float hl = src_img[ y_s_hi * src_w + x_s_low];
    const float hh = src_img[ y_s_hi * src_w + x_s_hi];

    dst_img[ thread_iy * dst_w + thread_ix ] = ll * x_weight * y_weight +
                                                lh * x_weight * (1.0f - y_weight) +
                                                hl * (1.0f- x_weight) * y_weight +
                                                hh * (1.0f - x_weight) * (1.0f - y_weight); 
}


void launch_resize_kernel(uint8_t* src_img, float src_h, float src_w, float dst_h, float dst_w, uint8_t * dst_img)
{
    float scale_x = src_w / dst_w;
    float scale_y = src_h / dst_h;

    int thread_num_block = 32;
    std::cout << "Block num X: " << ceil(dst_w/thread_num_block) << " Block num Y: " << ceil(dst_h/thread_num_block) << std::endl;
    dim3 grid( ceil(dst_w/thread_num_block), ceil(dst_h/thread_num_block));
    dim3 block(thread_num_block, thread_num_block);


    resize_kernel<<<grid,block>>>(src_img, scale_x, scale_y, dst_w, dst_h, src_w, src_h, dst_img);
    cudaDeviceSynchronize();
}