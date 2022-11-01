#include <opencv2/opencv.hpp>
#include <iostream>

#include "resize.hpp"

using namespace cv;
int main()
{
    std::string image_path = "/home/orcun/cuda_resize/test.png";
    int width = 512;
    int height = 512;

    int out_w = 300;
    int out_h = 300;

    Mat gpu_resize_mat(out_h, out_w, CV_8UC3);
    Mat cv_resize_mat(out_h, out_w, CV_8UC3);
    uint8_t* src_img_d_;
    uint8_t* dst_img_d_;

    Mat src_mat = imread(image_path, IMREAD_COLOR);
    int channels = src_mat.channels();
    cv::resize(src_mat, cv_resize_mat, cv::Size(out_w, out_h), cv::INTER_LINEAR);
    if(src_mat.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    size_t in_img_size = channels * width * height * sizeof(uint8_t);
    size_t out_img_size = channels * out_w * out_h * sizeof(uint8_t);
    cudaMalloc((void**)&src_img_d_, in_img_size);
    cudaMalloc((void**)&dst_img_d_, out_img_size);
    cudaMemcpy(src_img_d_, src_mat.ptr(), in_img_size, cudaMemcpyHostToDevice);

    launch_resize_kernel(src_img_d_, channels, height, width, out_h, out_w, dst_img_d_);
    cv::resize(src_mat, cv_resize_mat, cv::Size(out_w, out_h), cv::INTER_LINEAR);

    // copy back to host
    cudaMemcpy(gpu_resize_mat.ptr(), dst_img_d_, out_img_size, cudaMemcpyDeviceToHost);

    imshow("CUDA Resize", gpu_resize_mat);
    imshow("CV Resize", cv_resize_mat);
    int k = waitKey(0); // Wait for a keystroke in the window

    return 0;
}