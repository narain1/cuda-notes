#include <stdio.h>
#include <cuda_runtime.h>

#define DEBUG

__global__ void sgemm(float *a, float *b, float *c, int n_ar, int n_ac,
                            int n_br, int n_bc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_ar || col >= n_bc) return;

    float sum = 0;
    for (int k=0; k<n_ac; k++) {  // everything is being fetched from global mem
        sum += a[row * n_ac + k] * b[k * n_bc + col];
    }

    c[row * n_bc + col] = sum;
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

extern "C" {
    void cuda_gemm(float *a, float *b, float *c, int aw, int ah, int bw, int bh) {
        float *d_a, *d_b, *d_c;
        
        cudaMalloc((void **)&d_a, aw * ah * sizeof(float));
        cudaMalloc((void **)&d_b, bw * bh * sizeof(float));
        cudaMalloc((void **)&d_c, aw * bh * sizeof(float));

        cudaMemcpy(d_a, a, aw * ah * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, bw * bh * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim(cdiv(bw, blockDim.x), cdiv(ah, blockDim.y));

        sgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c,
                                        ah, aw,
                                        bw, bh);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "mm kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        cudaMemcpy(c, d_c, aw * bh *sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

    }
}
