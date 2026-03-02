#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void matmul_gpu_coalesced(const float *a, const float *b, float *c, int n){
    int row, col;
    row = blockDim.y*blockIdx.y+threadIdx.y;
    col = blockDim.x*blockIdx.x+threadIdx.x;
    if(row<n && col<n){
        float sum = 0.0f;
        const float *ar = a+row*n;
        for(int k=0; k<n; k++){
            sum += ar[k]*b[k*n+col];
        }
        c[row*n+col] = sum;
    }
}

void init_matrix(float *m, int n){
    for(int i=0; i<n*n; i++){
        m[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]){
    int n = 512;
    if(argc > 1){
        n = atoi(argv[1]);
    }
    size_t bytes = n*n*sizeof(float);
    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    float *c = (float*)malloc(bytes);
    init_matrix(a, n);
    init_matrix(b, n);
    float *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
    dim3 block(32,32);
    dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y);
    printf("=====================================\n");
    printf("Starting gpu coalesced matmul for N=%d\n", n);
    const auto start_time = std::chrono::system_clock::now();
    // matmul_cpu(a, b, c, n);
    matmul_gpu_coalesced<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    const auto now = std::chrono::system_clock::now();
    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);
    printf("Time taken: %f s\n", (float)std::chrono::duration_cast<std::chrono::microseconds>(now-start_time).count()/1e6);
    printf("=====================================\n\n");
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
    return 0;
}
