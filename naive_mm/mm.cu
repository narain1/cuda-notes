#include <stdio.h>

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

extern "C" {
    void cuda_gemm(float *a, float *b, float *c, int aw, int ah, int bw, int bh) {
        unsigned char *d_a, *d_b, *d_c;
        
        cudaMalloc((void **)&d_a, aw * ah);
        cudaMalloc((void **)&d_b, bw * bh);
        cudaMalloc((void **)&d_c, aw * bh);

        cudaMemcpy(d_a, a, aw * ah, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, bw * bh, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(ceil((float)bh / blockDim.x), ceil((float)aw/blockDim.y));

        sgemm<<<threadsPerBlock, numBlocks>>>(d_a, d_b, d_c,
                                        aw, ah,
                                        bw, bh)



    }
}