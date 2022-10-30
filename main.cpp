#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "resize.hpp"

using namespace cv;
int main()
{
    std::string image_path = "/home/orcun/cuda_resize/test.png";
    int width = 512;
    int height = 512;

    int out_w = 200;
    int out_h = 200;

    Mat mat_mono_disp(height, width, CV_8UC1);
    Mat mat_resize(out_h, out_w, CV_8UC1);
    size_t pitch_mono;
    size_t pitch_resize;
    uint8_t* img_d_;
    uint8_t* dst_img_d_;

    Mat mat_mono = imread(image_path, IMREAD_GRAYSCALE);
    if(mat_mono.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    size_t matrixLenMono = width;
    size_t matrixLenResize = out_w;
    cudaMallocPitch((void**)&img_d_, &pitch_mono, width, height);
    cudaMalloc((void**)&dst_img_d_, out_w * out_h * sizeof(uint8_t));

    cudaMemcpy2D(img_d_, pitch_mono, mat_mono.ptr(), width, matrixLenMono, height, cudaMemcpyHostToDevice);

    launch_resize_kernel(img_d_, height, width, out_h, out_w, dst_img_d_);
// void launch_resize_kernel(float* src_img, float& src_h, float& src_w,
//                             float& dst_h, float& dst_w, float * dst_img, cudaStream_t stream_)

    // copy back to host
    cudaMemcpy2D(mat_mono_disp.ptr(), matrixLenMono, img_d_, pitch_mono, matrixLenMono, height, cudaMemcpyDeviceToHost);
    cudaMemcpy(mat_resize.ptr(), dst_img_d_, out_h * out_w * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    imshow("Display window", mat_resize);
    int k = waitKey(0); // Wait for a keystroke in the window

    return 0;
}