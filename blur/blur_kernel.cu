#include <cuda_runtime.h>
#include <stdio.h>

#define BLURSIZE 3

__global__ void blur_kernel(unsigned char *in, unsigned char *out, int width, int height, int channel) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int baseOffset = channel * height * width;
        if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;

        // averaging surrounding pixels
        for (int blurRow=-BLURSIZE; blurRow < BLURSIZE; blurRow++) {
            for (int blurCol=-BLURSIZE; blurCol < BLURSIZE; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal += in[baseOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        }
        out[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}

extern "C" {
    void blur(unsigned char *in, unsigned char *out, int width, int height, int channels) {
        unsigned char *d_in, *d_out;
        size_t num_bytes = width * height * channels;

        cudaMalloc((void **)&d_in, num_bytes);
        cudaMalloc((void **)&d_out, num_bytes);

        cudaMemcpy(d_in, in, num_bytes, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1 ) / threadsPerBlock.x,
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        for (int channel=0; channel<channels; channel++) {
            blur_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height, channel);
        }

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "blur kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        cudaMemcpy(out, d_out, num_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);
    }
}