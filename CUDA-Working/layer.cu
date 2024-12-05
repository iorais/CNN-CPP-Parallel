#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
	}
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}

__global__ void fp_preact_c1(float input[28][28], float preact[8][13][13], float weight[8][3][3])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 3*3*8*13*13;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 3);
		const int i2 = ((idx /= 3	) % 3);
		const int i3 = ((idx /= 3	) % 8);
		const int i4 = ((idx /= 8	) % 13);
		const int i5 = ((idx /= 13	) % 13);

		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
	}
}

__global__ void fp_bias_c1(float preact[8][13][13], float bias[8])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 8*13*13;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 8);
		const int i2 = ((idx /= 8	) % 13);
		const int i3 = ((idx /= 13	) % 13);

		preact[i1][i2][i3] += bias[i1];
	}
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		preact[i1][i2][i3] += bias[0];
	}
}
__global__ void fp_preact_c2(float input[8][13][13], float preact[16][11][11], float weight[16][8][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    // Total number of elements to process
    const int N = 3 * 3 * 8 * 16 * 11 * 11;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;

        // Extract indices from flattened loop counter
        const int k_h = ((idx /= 1) % 3);      // Kernel height
        const int k_w = ((idx /= 3) % 3);      // Kernel width
        const int in_c = ((idx /= 3) % 8);     // Input channel
        const int out_c = ((idx /= 8) % 16);   // Output channel
        const int h = ((idx /= 16) % 11);      // Output height
        const int w = ((idx /= 11) % 11);      // Output width

        // Accumulate the product of the input and the corresponding weight
        atomicAdd(&preact[out_c][h][w],
                  weight[out_c][in_c][k_h][k_w] * input[in_c][h + k_h][w + k_w]);
    }
}

__global__ void fp_bias_c2(float preact[16][11][11], float bias[16])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 16 * 11 * 11;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 16);
        const int i2 = ((idx /= 16) % 11);
        const int i3 = ((idx /= 11) % 11);

        preact[i1][i2][i3] += bias[i1];
    }
}

__global__ void bp_weight_c2(float d_weight[16][8][3][3], float d_preact[16][11][11], float p_output[8][13][13])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 16 * 8 * 3 * 3 * 11 * 11;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 16);
        const int i2 = ((idx /= 16) % 8);
        const int i3 = ((idx /= 8) % 3);
        const int i4 = ((idx /= 3) % 3);
        const int i5 = ((idx /= 3) % 11);
        const int i6 = ((idx /= 11) % 11);

        atomicAdd(&d_weight[i1][i2][i3][i4], d_preact[i1][i5][i6] * p_output[i2][i5 + i3][i6 + i4]);
    }
}

__global__ void bp_bias_c2(float bias[16], float d_preact[16][11][11])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 16 * 11 * 11;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 16);
        const int i2 = ((idx /= 16) % 11);
        const int i3 = ((idx /= 11) % 11);

        atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3]);
    }
}

__global__ void bp_output_c1(float d_output[8][13][13], float n_weight[16][8][3][3], float nd_preact[16][11][11])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 16 * 8 * 3 * 3 * 11 * 11;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 16);
        const int i2 = ((idx /= 16) % 8);
        const int i3 = ((idx /= 8) % 3);
        const int i4 = ((idx /= 3) % 3);
        const int i5 = ((idx /= 3) % 11);
        const int i6 = ((idx /= 11) % 11);

        atomicAdd(&d_output[i2][i5 + i3][i6 + i4], n_weight[i1][i2][i3][i4] * nd_preact[i1][i5][i6]);
    }
}

__global__ void bp_preact_c2(float d_preact[16][11][11], float d_output[16][11][11], float preact[16][11][11])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 16 * 11 * 11;  // Total elements in the output tensor

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 16);  // Channel index
        const int i2 = ((idx /= 16) % 11); // Row index
        const int i3 = ((idx /= 11) % 11); // Column index

        const float o = step_function(preact[i1][i2][i3]); // Step function output from forward pass

        // Compute gradient for pre-activation using the derivative of the step function
        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void fp_preact_f(float input[16][11][11], float preact[10], float weight[10][16][11][11])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*16*11*11;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 16);
		const int i3 = ((idx /= 16	) % 11);
		const int i4 = ((idx /= 11	) % 11);

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

__global__ void bp_weight_f(float d_weight[10][16][11][11], float d_preact[10], float p_output[16][11][11])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*16*11*11;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 16);
		const int i3 = ((idx /= 16	) % 11);
		const int i4 = ((idx /= 11	) % 11);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}



__global__ void bp_weight_c1(float d_weight[8][3][3], float d_preact[8][13][13], float p_output[28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 8*3*3*13*13;
	const float d = pow(13.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 8);
		const int i2 = ((idx /= 8	) % 3);
		const int i3 = ((idx /= 3	) % 3);
		const int i4 = ((idx /= 3	) % 13);
		const int i5 = ((idx /= 13	) % 13);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_c1(float bias[8], float d_preact[8][13][13])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 8*13*13;
	const float d = pow(13.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 8);
		const int i2 = ((idx /= 8	) % 13);
		const int i3 = ((idx /= 13	) % 13);

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_output_c2(float d_output[16][11][11], float n_weight[10][16][11][11], float nd_preact[10]) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    const int size = blockDim.x * gridDim.x;              // Total number of threads

    // Total elements in the output tensor
    const int N = 16 * 11 * 11;

    // Iterate over all elements of d_output in parallel
    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;

        // Decompose the index to 3D indices for the output
        const int out_c = ((idx /= 1) % 16);   // Output channel
        const int h = ((idx /= 16) % 11);      // Height index
        const int w = ((idx /= 11) % 11);      // Width index

        // Initialize gradient to 0
        float grad = 0.0f;

        // Accumulate gradients from the dense layer
        for (int dense_c = 0; dense_c < 10; ++dense_c) { // Iterate over the dense layer neurons
            grad += n_weight[dense_c][out_c][h][w] * nd_preact[dense_c];
        }

        // Assign computed gradient to d_output
        d_output[out_c][h][w] = grad;
    }
}

__global__ void bp_preact_c1(float d_preact[8][13][13], float d_output[8][13][13], float preact[8][13][13]) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    const int size = blockDim.x * gridDim.x;              // Total number of threads

    // Total number of elements in d_preact
    const int N = 8 * 13 * 13;

    // Iterate over all elements in d_preact
    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n) {
        int idx = n;

        // Decompose linear index into 3D indices for d_preact
        const int out_c = ((idx /= 1) % 8);   // Output channel
        const int h = ((idx /= 8) % 13);      // Height index
        const int w = ((idx /= 13) % 13);     // Width index

        // Compute the derivative of the activation function
        const float activation = step_function(preact[out_c][h][w]);
        const float activation_derivative = activation * (1 - activation);

        // Compute the gradient for the pre-activation value
        d_preact[out_c][h][w] = d_output[out_c][h][w] * activation_derivative;
    }
}