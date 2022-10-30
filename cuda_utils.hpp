#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

// #define CHECK_CUDA_ERROR(e) (check_error(e, __FILE__, __LINE__))

// void check_error(const cudaError_t e, const char * f, int n)
// {
//   if (e != cudaSuccess) {
//     std::stringstream s;
//     s << cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": " <<
//       cudaGetErrorString(e);
//     throw std::runtime_error{s.str()};
//   }
// }