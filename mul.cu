#include <cassert>
#include <iostream>

__global__ void mul(float *a, float *b, float *c, int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && col < n) {
    float prod = 0.f;
    for (int i = 0; i < n; i++) {
      prod += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = prod;
  }
}

__global__ void mul_onedim(float *a, float *b, float *c, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int row = index / n;
  int col = index % n;
  if (row < n && col < n) {
    float prod = 0.f;
    for (int i = 0; i < n; i++) {
      prod += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = prod;
  }
}

int main() {
  int N = 1024;
  int BLOCK_SIZE = 32;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];
  float *d = new float[N * N];
  float *e = new float[N * N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        a[i * N + j] = 2;
      }
      b[i * N + j] = i + j;
    }
  }
  float *a_d;
  float *b_d;
  float *c_d;
  float *d_d;

  // allocating device memory
  cudaMalloc((void **)&a_d, N * N * sizeof(float));
  cudaMalloc((void **)&b_d, N * N * sizeof(float));
  cudaMalloc((void **)&c_d, N * N * sizeof(float));
  cudaMalloc((void **)&d_d, N * N * sizeof(float));

  // copy data from host to device
  cudaMemcpy(a_d, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(N / (float)BLOCK_SIZE), ceil(N / (float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  mul<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, N);

  cudaMemcpy(c, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  dim3 dimGrid_r(ceil(N * N / (float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock_r(BLOCK_SIZE, 1, 1);

  mul_onedim<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, N);

  cudaMemcpy(d, d_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert(c[i * N + j] == b[i * N + j] * 2);
      assert(d[i * N + j] == d[i * N + j] * 2);
    }
  }

  // freeing device memory
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  return 0;
}