#include<iostream>

__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

//host = CPU and device = GPU
int main() {
    int N = 4096;
    int BLOCK_SIZE = 256;
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2*i;
    }
    float *a_d;
    float *b_d;
    float *c_d;

    //allocating device memory
    cudaMalloc((void**) &a_d, N*sizeof(float));
    cudaMalloc((void**) &b_d, N*sizeof(float));
    cudaMalloc((void**) &c_d, N*sizeof(float));

    //copy data from host to device
    cudaMemcpy(a_d, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice);

    //launching kernel
    add<<<ceil(N/(float)BLOCK_SIZE), BLOCK_SIZE>>>(a_d, b_d, c_d, N);

    //copy data from device to host
    cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    //printing result
    for (int i = 0; i < 10; i++) {
        std::cout<<a[i]<<" "<<b[i]<<" "<<c[i]<<std::endl;
    }  

    //freeing device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}