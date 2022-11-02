#include "resize.hpp"

int thread_num_block = 18; // max 1024 threads can be launched from a block. So launch of max 18 * 18 * 3 threads is possible.

__device__ inline int resolveIndexNCHW(const int x, const int y, const int z, const int w, const int h)
{
    return z * w * h + y * w + x;
}

__device__ inline int resolveIndexNHWC(const int x, const int y, const int z, const int step, const int channel_size)
{
    return y * step + x * channel_size + z;
}

__global__ void resize_kernel(uint8_t* src_img, int channel_size, int src_step, int dst_step, float scale_x, float scale_y, int dst_w, int dst_h, int src_w, int src_h, uint8_t* dst_img)
{
    int thread_ix = blockIdx.x * blockDim.x + threadIdx.x; // height dim
    int thread_iy = blockIdx.y * blockDim.y + threadIdx.y; // width dim
    int thread_iz = threadIdx.z; // channel dim

    if (thread_ix > dst_w || thread_iy > dst_h){return;}

    const float x_s = scale_x * thread_ix;
    const float y_s = scale_y * thread_iy;
    
    // pixel coords to interpolate between
    const int x_s_low = floor(x_s);
    const int y_s_low = floor(y_s);
    const int x_s_hi = ceil(x_s);
    const int y_s_hi = ceil(y_s);

    //bilinear weights
    const float x_weight = 1 - (x_s - float(x_s_low)); // 1- since bigger distance should have less weight
    const float y_weight = 1 - (y_s - float(y_s_low)); 

               // xy
    const uint8_t ll = src_img[resolveIndexNHWC(x_s_low, y_s_low, thread_iz, src_step, channel_size)];
    const uint8_t lh = src_img[resolveIndexNHWC(x_s_low, y_s_hi, thread_iz, src_step, channel_size)];
    const uint8_t hl = src_img[resolveIndexNHWC(x_s_hi, y_s_low, thread_iz, src_step, channel_size)];
    const uint8_t hh = src_img[resolveIndexNHWC(x_s_hi, y_s_hi, thread_iz, src_step, channel_size)];

    const uint8_t pixel = ll * x_weight * y_weight +
                        lh * x_weight * (1.0f - y_weight) +
                        hl * (1.0f- x_weight) * y_weight +
                        hh * (1.0f - x_weight) * (1.0f - y_weight);
                                                                            hh * (1.0f - x_weight) * (1.0f - y_weight); 
                        hh * (1.0f - x_weight) * (1.0f - y_weight);
    dst_img[resolveIndexNHWC(thread_ix, thread_iy, thread_iz, dst_step, channel_size)] = pixel; //indexing for nhwc output
    // dst_img[resolveIndexNCHW(thread_ix, thread_iy, thread_iz, dst_w, dst_h)] = pixel; //nchw
}

void launch_resize_kernel(uint8_t* src_img, int channel_size, float src_h, float src_w, float dst_h, float dst_w, uint8_t * dst_img)
{
    float scale_x = src_w / dst_w;
    float scale_y = src_h / dst_h;


    dim3 grid(ceil(dst_w * thread_num_block), ceil(dst_h * thread_num_block));
    dim3 block(thread_num_block, thread_num_block, channel_size);
    resize_kernel<<<grid,block>>>(src_img, channel_size, channel_size*src_w, channel_size*dst_w, 
                                    scale_x, scale_y, dst_w, dst_h, src_w, src_h, dst_img);
    cudaDeviceSynchronize();
}