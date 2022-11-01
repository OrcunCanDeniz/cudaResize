#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "resize.hpp"

using namespace cv;
int main()
{
    std::string image_path = "/home/orcun/cudaResize/test.png";
    int width = 512;
    int height = 512;

    int out_w = 600;
    int out_h = 600;

    Mat mat_mono_disp(height, width, CV_8UC1);
    Mat mat_resize(out_h, out_w, CV_8UC1);
    Mat cv_resize(out_h, out_w, CV_8UC1);
    size_t pitch_mono;
    size_t pitch_resize;
    uint8_t* img_d_;
    uint8_t* dst_img_d_;

    Mat mat_mono = imread(image_path, IMREAD_GRAYSCALE);
    cv::resize(mat_mono, cv_resize, cv::Size(out_w, out_h), cv::INTER_LINEAR);
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

    // copy back to host
    cudaMemcpy2D(mat_mono_disp.ptr(), matrixLenMono, img_d_, pitch_mono, matrixLenMono, height, cudaMemcpyDeviceToHost);
    cudaMemcpy(mat_resize.ptr(), dst_img_d_, out_h * out_w * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    imshow("CUDA Resize", mat_resize);
    imshow("CV Resize", cv_resize);
    int k = waitKey(0); // Wait for a keystroke in the window

    return 0;
}