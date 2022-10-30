#include "resize.hpp"
#include <iostream>
#include <stdio.h>
__global__ void resize_kernel(uint8_t* src_img, float win_size_w, float win_size_h, int dst_w, int dst_h, uint8_t* dst_img)
{
                //  2          *      32     +       32
    int thread_ix = blockIdx.x * blockDim.x + threadIdx.x; // height dim
    int thread_iy = blockIdx.y * blockDim.y + threadIdx.y; // width dim

    if (thread_ix > dst_h || thread_iy > dst_w){return;}

    float start_idx_h = thread_ix * win_size_h;
    float end_idx_h = (thread_ix + 1) * win_size_h;

    float start_idx_w = thread_iy * win_size_w;
    float end_idx_w = (thread_iy + 1) * win_size_w;

    float weight_start_h = ceil(start_idx_h) - start_idx_h;
    float weight_end_h = end_idx_h - floor(end_idx_h);

    float weight_start_w = ceil(start_idx_w) - start_idx_w;
    float weight_end_w = end_idx_w - floor(end_idx_w);

    float acc = 0;
    float w_rate = 1;
    float h_rate = 1;

    if (thread_ix > 50 )
    {
        printf("%d ", thread_ix);} //max thread_ix = 63
    // for (int i = floor(start_idx_w); i <= ceil(end_idx_w); i++) 
    // {
    //     for (int j = floor(start_idx_h); j <= ceil(end_idx_h); j++) 
    //     {
    //         w_rate = 1;
    //         if (i == floor(start_idx_w)) 
    //         { 
    //             w_rate = weight_start_w;
    //         } else if (i == ceil(end_idx_w)) {
    //             w_rate = weight_end_w;
    //         }

    //         h_rate = 1;
    //         if (i == floor(start_idx_h)) 
    //         { 
    //             h_rate = weight_start_h;
    //         } else if (i == ceil(end_idx_h)) {
    //             h_rate = weight_end_h;
    //         }

    //         acc += src_img[i] * h_rate * w_rate;
    //                 //blockDim.y is the row width           //normalize too
    //     }
    // }

    dst_img[ thread_ix * dst_w + thread_iy ] = 100; 
}


void launch_resize_kernel(uint8_t* src_img, float src_h, float src_w,
                            float dst_h, float dst_w, uint8_t * dst_img)
{
    float scale_w = dst_w / src_w;
    float scale_h = dst_h / src_h;

    float win_size_w = 1.f / scale_w;
    float win_size_h = 1.f / scale_h;

    int win_num_y = ceil(dst_w/win_size_w);
    int win_num_x = ceil(dst_h/win_size_h);

    int thread_num_block = 32;
    std::cout << "Block num X: " << ceil(win_num_x/thread_num_block) << " Block num Y: " << ceil(win_num_y/thread_num_block) << std::endl;
    std::cout << "Windows num X: " << win_num_x << " Windows num Y: " << win_num_y << std::endl;
    dim3 grid(ceil(win_num_x/thread_num_block), ceil(win_num_y/thread_num_block));
    dim3 block(thread_num_block, thread_num_block);

    // std::cout<< block_x_size << " " << block_y_size << std::endl;

    resize_kernel<<<grid,block>>>(src_img, win_size_w, win_size_h, dst_w, dst_h, dst_img);
    cudaDeviceSynchronize();
}