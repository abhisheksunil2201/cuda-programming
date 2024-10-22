// TODO: needs to be completed. Need to understand how neural networks are
// trained in detail.

#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

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

__global__ void backward_pass(int batch_size, int n, int out_w, float *weights,
                              float *biases, float *d_l, float *out_d_l) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w) {
    float dl = 0.f;
    for (int i = 0; i < n; i++) {
      float w = weights[i * out_w + column];
      dl += w * d_l[row * n + i];
    }
    out_d_l[row * out_w + column] = dl;
  }
}

__global__ void update_layer(int w, int h, int batch_size, float lr,
                             float *weights, float *biases, float *activations,
                             float *d_l) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && column < w) {
    float dw = 0.f;
    float db = 0.f;
    for (int i = 0; i < batch_size; i++) {
      float act = activations[i * h + row];
      float dl = d_l[i * w + column];
      dw += act * dl;
      db += dl;
    }
    weights[row * w + column] -= lr * dw / batch_size;
    biases[column] -= lr * db / batch_size;
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

__global__ void relu_backwards(int w, int h, float *a, float *d_l, float *b) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && column < w) {
    float activation = a[row * w + column];
    b[row * w + column] = activation > 0.f ? d_l[row * w + column] : 0.f;
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

__global__ void cross_entropy_backward(int w, int h, float *predictions,
                                       float *real, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < h && col < w) {
    output[row * w + col] = predictions[row * w + col] - real[row * w + col];
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

void initLayer(float *weights, float *biases, int w, int h, int BLOCK_SIZE) {
  dim3 dimGrid =
      dim3(ceil(w / (float)BLOCK_SIZE), ceil(h / (float)BLOCK_SIZE), 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_rand<<<dimGrid, dimBlock>>>(w, h, weights);
  cudaPeekAtLastError();

  dimGrid = dim3(ceil(h / (float)BLOCK_SIZE), 1, 1);
  dimBlock = dim3(BLOCK_SIZE, 1, 1);
  init_rand<<<dimGrid, dimBlock>>>(1, h, biases);
  cudaPeekAtLastError();
}

int main(int argc, char **argv) {
  int train_length = 60000;

  float *input;
  float *labels;
  int input_size = 784;
  int labels_size = 10;

  float *mnist_train_x = new float[input_size * train_length];
  float *mnist_train_y = new float[labels_size * train_length];

  {
    Timer t("read mnist");
    read_mnist("./mnist_train.csv", train_length, mnist_train_x, mnist_train_y);
  }

  int size1 = 300;
  float *weights1;
  float *biases1;
  float *d_l1;

  int size2 = 100;
  float *weights2;
  float *biases2;
  float *d_l2;

  int size3 = 10;
  float *weights3;
  float *biases3;
  float *d_l3;

  int BLOCK_SIZE = 16;
  int BATCH_SIZE = 16;
  int EPOCHS = 10;
  float LR = 0.003f;
  dim3 dimGrid;
  dim3 dimBlock;

  float *out_h = new float[BATCH_SIZE * size3];
  float *loss_h = new float[BATCH_SIZE];

  float *x1;
  float *a1;
  float *x2;
  float *a2;
  float *x3;
  float *a3;
  float *loss;
  {
    Timer init("initialization");

    cudaMalloc((void **)&input, input_size * BATCH_SIZE * sizeof(float));
    cudaMalloc((void **)&labels, labels_size * BATCH_SIZE * sizeof(float));
    cudaMalloc((void **)&weights1, size1 * input_size * sizeof(float));
    cudaMalloc((void **)&biases1, size1 * sizeof(float));
    cudaMalloc((void **)&d_l1, size1 * BATCH_SIZE * sizeof(float));
    initLayer(weights1, biases1, size1, input_size, BLOCK_SIZE);

    cudaMalloc((void **)&weights2, size2 * size1 * sizeof(float));
    cudaMalloc((void **)&biases2, size2 * sizeof(float));
    cudaMalloc((void **)&d_l2, size2 * BATCH_SIZE * sizeof(float));
    initLayer(weights2, biases2, size2, size1, BLOCK_SIZE);

    cudaMalloc((void **)&weights3, size3 * size2 * sizeof(float));
    cudaMalloc((void **)&biases3, size3 * sizeof(float));
    cudaMalloc((void **)&d_l3, size3 * BATCH_SIZE * sizeof(float));
    initLayer(weights3, biases3, size3, size2, BLOCK_SIZE);

    cudaMalloc((void **)&x1, size1 * BATCH_SIZE * sizeof(float));
    cudaMalloc((void **)&a1, size1 * BATCH_SIZE * sizeof(float));

    cudaMalloc((void **)&x2, size2 * BATCH_SIZE * sizeof(float));
    cudaMalloc((void **)&a2, size2 * BATCH_SIZE * sizeof(float));

    cudaMalloc((void **)&x3, size3 * BATCH_SIZE * sizeof(float));
    cudaMalloc((void **)&a3, size3 * BATCH_SIZE * sizeof(float));

    cudaMalloc((void **)&loss, BATCH_SIZE * sizeof(float));
  }

  float total_time = 0.f;
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    float cum_loss = 0.f;
    int correct = 0;
    int total = 0;
    auto start_time = std::chrono::system_clock::now();
    for (int batch = 0; batch < train_length / BATCH_SIZE; batch++) {
      total += BATCH_SIZE;
      cudaMemcpy(input, &mnist_train_x[batch * BATCH_SIZE * input_size],
                 BATCH_SIZE * input_size * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(labels, &mnist_train_y[batch * BATCH_SIZE * labels_size],
                 BATCH_SIZE * labels_size * sizeof(float),
                 cudaMemcpyHostToDevice);

      dimGrid = dim3(ceil(size1 / (float)BLOCK_SIZE),
                     ceil(BATCH_SIZE / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward_pass<<<dimGrid, dimBlock>>>(BATCH_SIZE, input_size, size1, input,
                                          weights1, biases1, x1);
      cudaPeekAtLastError();

      relu<<<dimGrid, dimBlock>>>(size1, BATCH_SIZE, x1, a1);
      cudaPeekAtLastError();

      dimGrid = dim3(ceil(size2 / (float)BLOCK_SIZE),
                     ceil(BATCH_SIZE / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward_pass<<<dimGrid, dimBlock>>>(BATCH_SIZE, size1, size2, a1,
                                          weights2, biases2, x2);
      cudaPeekAtLastError();

      relu<<<dimGrid, dimBlock>>>(size2, BATCH_SIZE, x2, a2);
      cudaPeekAtLastError();

      dimGrid = dim3(ceil(size3 / (float)BLOCK_SIZE),
                     ceil(BATCH_SIZE / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      forward_pass<<<dimGrid, dimBlock>>>(BATCH_SIZE, size2, size3, a2,
                                          weights3, biases3, x3);
      cudaPeekAtLastError();

      softmax<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, x3, a3);
      cudaPeekAtLastError();

      dimGrid = dim3(ceil(size3 / (float)BLOCK_SIZE), 1, 1);
      dimBlock = dim3(BLOCK_SIZE, 1, 1);
      cross_entropy<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, a3, labels, loss);

      cudaDeviceSynchronize();

      cudaMemcpy(out_h, a3, BATCH_SIZE * size3 * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(loss_h, loss, BATCH_SIZE * sizeof(float),
                 cudaMemcpyDeviceToHost);

      for (int i = 0; i < BATCH_SIZE; i++) {
        float max_1 = 0.f;
        float max_2 = 0.f;
        int i1 = 0;
        int i2 = 0;
        for (int j = 0; j < labels_size; j++) {
          if (out_h[i * labels_size + j] > max_1) {
            max_1 = out_h[i * labels_size + j];
            i1 = j;
          }

          if (mnist_train_y[batch * BATCH_SIZE * labels_size + i * labels_size +
                            j] > max_2) {
            max_2 = mnist_train_y[batch * BATCH_SIZE * labels_size +
                                  i * labels_size + j];
            i2 = j;
          }
        }
        correct += (i1 == i2);
        cum_loss += loss_h[i];
      }

      dimGrid = dim3(ceil(size3 / (float)BLOCK_SIZE),
                     ceil(BATCH_SIZE / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      cross_entropy_backwards<<<dimGrid, dimBlock>>>(size3, BATCH_SIZE, a3,
                                                     labels, d_l3);
      cudaPeekAtLastError();

      dimGrid = dim3(ceil(size2 / (float)BLOCK_SIZE),
                     ceil(BATCH_SIZE / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      backward_pass<<<dimGrid, dimBlock>>>(BATCH_SIZE, size3, size2, weights3,
                                           biases3, d_l3, d_l2);
      cudaPeekAtLastError();

      relu_backwards<<<dimGrid, dimBlock>>>(size2, BATCH_SIZE, a2, d_l2, d_l2);

      dimGrid = dim3(ceil(size1 / (float)BLOCK_SIZE),
                     ceil(BATCH_SIZE / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

      backward_pass<<<dimGrid, dimBlock>>>(BATCH_SIZE, size2, size1, weights2,
                                           biases2, d_l2, d_l1);
      cudaPeekAtLastError();
      relu_backwards<<<dimGrid, dimBlock>>>(size1, BATCH_SIZE, a1, d_l1, d_l1);

      dimGrid = dim3(ceil(size3 / (float)BLOCK_SIZE),
                     ceil(size2 / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
      update_layer<<<dimGrid, dimBlock>>>(size3, size2, BATCH_SIZE, LR,
                                          weights3, biases3, a2, d_l3);
      dimGrid = dim3(ceil(size2 / (float)BLOCK_SIZE),
                     ceil(size1 / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
      update_layer<<<dimGrid, dimBlock>>>(size2, size1, BATCH_SIZE, LR,
                                          weights2, biases2, a1, d_l2);
      dimGrid = dim3(ceil(size1 / (float)BLOCK_SIZE),
                     ceil(input_size / (float)BLOCK_SIZE), 1);
      dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
      update_layer<<<dimGrid, dimBlock>>>(size1, input_size, BATCH_SIZE, LR,
                                          weights1, biases1, input, d_l1);
    }
    float val_loss = 0.f;
    int val_correct = 0;
    int val_total = 0;

    float epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now() - start_time)
                           .count();
    total_time += epoch_time;
    std::cout << "epoch " << epoch << " took " << epoch_time << "ms cum loss "
              << cum_loss << " accuracy " << (float)correct / total
              << " val loss " << val_loss << " val accuracy "
              << (float)val_correct / val_total << std::endl;
  }
  std::cout << "finished training, total time = " << total_time << " ms"
            << std::endl;
}