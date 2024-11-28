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

// extern "C" {
//    void cuda_gemm(float *a, float *b, float *c, int aw, int ah, int bw, int bh) {
//        float *d_a, *d_b, *d_c;
//        :v
//        cudaMalloc((void **)&d_a, aw * ah * sizeof(float));
//        cudaMalloc((void **)&d_b, bw * bh * sizeof(float));
//        cudaMalloc((void **)&d_c, aw * bh * sizeof(float));
//
//        cudaMemcpy(d_a, a, aw * ah, cudaMemcpyHostToDevice);
//        cudaMemcpy(d_b, b, bw * bh, cudaMemcpyHostToDevice);
//
//        dim3 threadsPerBlock(16, 16);
//        dim3 numBlocks(cdiv(bw, threadsPerBlock.x), cdiv(ah, threadsPerBlock.y));
//
//        sgemm<<<threadsPerBlock, numBlocks>>>(d_a, d_b, d_c,
//                                        ah, aw,
//                                        bw, bh);
//
//        cudaError_t cudaStatus = cudaGetLastError();
//        if (cudaStatus != cudaSuccess) {
//            fprintf(stderr, "mm kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        }
//
//        cudaMemcpy(c, d_c, aw * bh, cudaMemcpyDeviceToHost);
//
//        cudaFree(d_a);
//        cudaFree(d_b);
//        cudaFree(d_c);
//
//    }
//}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int ar = 2048, ac = 2048, br = 2048, bc = 2048;

    a = (float *) malloc(ar * ac * sizeof(float));
    b = (float *) malloc(br * bc * sizeof(float));
    c = (float *) calloc(ar * bc, sizeof(float));

    cudaMalloc((void **) &d_a, ar * ac * sizeof(float));
    cudaMalloc((void **) &d_b, br * bc * sizeof(float));
    cudaMalloc((void **) &d_c, ar * bc * sizeof(float));

    srand(time(NULL));
    for (int i=0; i<ar; i++)
        for (int j=0; j<ac; j++)
            a[i * ac + j] = rand() / (float) RAND_MAX;

    for (int i=0; i<br; i++)
        for (int j=0; j<bc; j++)
            b[i * bc + j] = rand() / (float) RAND_MAX;

    cudaMemcpy(d_a, a, ar * ac * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, br * bc * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(cdiv(ar, blockDim.x), cdiv(bc, blockDim.y));
    sgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c,
                                ar, ac,
                                br, bc);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mm kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaMemcpy(c, d_c, ar * bc * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", c[2047 * 2047]);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
