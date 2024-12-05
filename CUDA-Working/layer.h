#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int M, N, O;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
};


// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[8][13][13], float weight[8][3][3]);
__global__ void fp_bias_c1(float preact[8][13][13], float bias[8]);
__global__ void fp_preact_c2(float input[8][13][13], float preact[16][11][11], float weight[16][8][3][3]);
__global__ void fp_bias_c2(float preact[16][11][11], float bias[16]);
__global__ void fp_preact_f(float input[16][11][11], float preact[10], float weight[10][16][11][11]);
__global__ void fp_bias_f(float preact[10], float bias[10]);

// Back propagation kernels
__global__ void bp_weight_f(float d_weight[10][16][11][11], float d_preact[10], float p_output[16][11][11]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_c1(float d_output[8][13][13], float n_weight[16][8][3][3], float nd_preact[16][11][11]);
__global__ void bp_weight_c1(float d_weight[8][3][3], float d_preact[8][13][13], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[8], float d_preact[8][13][13]);
__global__ void bp_weight_c2(float d_weight[16][8][3][3], float d_preact[16][11][11], float p_output[8][13][13]);
__global__ void bp_bias_c2(float bias[16], float d_preact[16][11][11]);
__global__ void bp_output_c2(float d_output[16][11][11], float n_weight[10][16][11][11], float nd_preact[10]);
__global__ void bp_preact_c2(float d_preact[16][11][11], float d_output[16][11][11], float preact[16][11][11]);
__global__ void bp_preact_c1(float d_preact[8][13][13], float d_output[8][13][13], float preact[8][13][13]);
