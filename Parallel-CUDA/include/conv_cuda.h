#ifndef CONVOLUTION_CUDA_H
#define CONVOLUTION_CUDA_H

#include <cuda_runtime.h>

// Function declarations
void forward_pass_kernel(const float* input, float* output, int num_classes, int num_samples, int batch_size);
void error_evaluation_kernel(const float* output, const int* labels, float* errors, float* losses, int num_classes, int num_samples, int batch_size);

#endif // CONVOLUTION_CUDA_H