#include <cuda_runtime.h>
#include <stdio.h>

__global__ void blur_kernel(unsigned char *in, unsigned char *out, int width, int height, int blur_radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int BLURSIZE = blur_radius;

    if (col < width && row < height) {
        for (int channel = 0; channel < 3; channel++) {
            int pixVal = 0;
            int pixels = 0;
            for (int blurRow = -BLURSIZE; blurRow <= BLURSIZE; blurRow++) {
                for (int blurCol = -BLURSIZE; blurCol <= BLURSIZE; blurCol++) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int idx = ((curRow * width + curCol) * 3) + channel;
                        pixVal += in[idx];
                        pixels++;
                    }
                }
            }
            int idx = ((row * width + col) * 3) + channel;
            out[idx] = pixVal / pixels;
        }
    }
}


extern "C" {
    void blur(unsigned char *in, unsigned char *out, int width, int height, int channels, int blur_radius) {
        unsigned char *d_in, *d_out;
        size_t num_bytes = width * height * channels;

        cudaMalloc((void **)&d_in, num_bytes);
        cudaMalloc((void **)&d_out, num_bytes);

        cudaMemcpy(d_in, in, num_bytes, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1 ) / threadsPerBlock.x,
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        blur_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height, blur_radius);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "blur kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        cudaMemcpy(out, d_out, num_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);
    }
}