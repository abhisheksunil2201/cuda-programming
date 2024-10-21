#include <iostream>

__global void forward_pass(int batch_size, int n, int out_w, float *input,
                           float *weights, float *biases, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch_size && col < out_w) {
    output[row * out_w + col] = biases[col];
    for (int i = 0; i < n; i++) {
      output[row * out_w + col] +=
          weights[i * out_w + col] * input[row * n + i];
    }
  }
}

// we use ReLU as our activation function instead of sigmoid because of some
// reason - TODO: read up more on this ReLU is a function that returns x if x >
// 0 and 0 otherwise
__global__ void relu(int w, int h, float *input, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    float activation = input[row * w + col];
    output[row * w + col] = activation > 0 ? activation : 0;
  }
}

// softmax is used to convert the output of the neural network to a probability
// distribution
__global__ void softmax(int w, int h, float *input, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    float maxval = 0.f;
    for (int i = 1; i < w; i++) {
      maxval = max(maxval, input[row * w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i < w; i++) {
      divisor += exp(input[row * w + i] - maxval);
    }
    output[row * w + col] = exp(input[row * w + col] - maxval) / divisor;
  }
}

// loss function
__global__ void cross_entropy(int w, int h, float *prediction, float *real,
                              float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < h) {
    float loss = 0.f;
    for (int i = 0; i < w; i++) {
      loss -= real[idx * w + i] * log(max(1e-6, prediction[idx * w + i]));
    }
    output[idx] = loss;
  }
}

// initialize weights to random values
__global__ void init_rand(int w, int h, float *mat) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w) {
    curandState state;
    curant_init(42, row * w + col, 0, &state);
    mat[row * w + col] = curand_normal(&state) * sqrt(2.f / h);
  }
}

int main(int argc, char **argv) {}