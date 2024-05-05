#include <cuda_runtime.h>

__global__ void rgbToGrayscaleKernel(unsigned char *data, unsigned char *o, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rgbIndex = (x + y * width) * 3;
        int grayIndex = x + y * width;
        unsigned char r = data[rgbIndex];
        unsigned char g = data[rgbIndex + 1];
        unsigned char b = data[rgbIndex + 2];

        o[grayIndex] = (unsigned char)(0.2989f * r + 0.5870f * g + 0.1140f * b);

    }
}

extern "C" {
    void rgbToGrayscale(unsigned char *data, unsigned char *o, int width, int height) {
        unsigned char *d_rgb, *d_gray;


        // Allocate memory on the device
        cudaMalloc(&d_rgb, width * height * 3);
        cudaMalloc(&d_gray, width * height);
        
        // Copy data from host to device
        cudaMemcpy(d_rgb, data, width * height * 3, cudaMemcpyHostToDevice);

        dim3 dimBlock(16, 16);
        dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

        // Launch the kernel
        rgbToGrayscaleKernel<<<dimGrid, dimBlock>>>(d_rgb, d_gray, width, height);

        // Copy the converted grayscale image back to the host
        cudaMemcpy(o, d_gray, width * height, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_rgb);
        cudaFree(d_gray);
    }
}
