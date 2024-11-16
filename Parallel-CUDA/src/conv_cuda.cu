#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include "conv_cuda.h"

__global__ void convolution_forward(const float* input, float* output, const float* filter, const float* bias, int depth, int height, int width, int f_y, int f_x, int f_d, int n_kernel, int out_H, int out_W, int stride, int padding, int batch_size) {
    int batch_idx = blockIdx.x;
    int kernel = blockIdx.y;
    int y_out = blockIdx.z * blockDim.y + threadIdx.y;
    int x_out = threadIdx.x;

    if (batch_idx < batch_size && kernel < n_kernel && y_out < out_H && x_out < out_W) {
        float sum = 0.0f;
        for (int layer = 0; layer < depth; ++layer) {
            for (int f_y_it = 0; f_y_it < f_y; ++f_y_it) {
                for (int f_x_it = 0; f_x_it < f_x; ++f_x_it) {
                    int y = y_out * stride + f_y_it - padding;
                    int x = x_out * stride + f_x_it - padding;
                    if (y >= 0 && y < height && x >= 0 && x < width) {
                        int input_idx = ((batch_idx * depth + layer) * height + y) * width + x;
                        int filter_idx = (((kernel * f_y + f_y_it) * f_x + f_x_it) * f_d + layer);
                        sum += input[input_idx] * filter[filter_idx];
                    }
                }
            }
        }
        int output_idx = ((batch_idx * n_kernel + kernel) * out_H + y_out) * out_W + x_out;
        output[output_idx] = sum + bias[kernel];
    }
}