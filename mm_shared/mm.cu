#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void mm_kernel(float *a, float *b, float *c, int ar, int k, int bc) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph=0; ph<ceil(ar/(float)TILE_WIDTH); ph++) {
        if (row < ar && ph * TILE_WIDTH + tx < k)
            ds_M[ty][tx] = a[row * k + ph * TILE_WIDTH + tx];
        else
            ds_M[ty][tx] = 0;

        if (col < bc && ph * TILE_WIDTH + ty < k)
            ds_N[ty][tx] = b[(ph * TILE_WIDTH + ty) * bc + col];
        else
            ds_N[ty][tx] = 0;

        __syncthreads();

        for (int k=0; k<TILE_WIDTH; k++)
            Pvalue += ds_M[ty][k] * ds_N[k][tx];

        __syncthreads();
    }

    if (row < k && col < bc)
        c[row * bc + col] = Pvalue;
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

extern "C" {
    void mm(float *a, float *b, float *c, int aw, int ah, int bw, int bh) {
        float *d_a, *d_b, *d_c;

        if (aw != bh) {
            fprintf(stderr, "error: mm dimension not equal\n");
            exit(EXIT_FAILURE);
        }

        cudaMalloc((void **)&d_a, aw * ah * sizeof(float));
        cudaMalloc((void **)&d_b, bw * bh * sizeof(float));
        cudaMalloc((void **)&d_c, aw * bh * sizeof(float));

        cudaMemcpy(d_a, a, aw * ah * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, bw * bh * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim(cdiv(bw, blockDim.x), cdiv(ah, blockDim.y));

        mm_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c,
                ah, aw, bh);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "mm kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        cudaMemcpy(c, d_c, aw * bh * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
}
