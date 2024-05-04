#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

void load_array(const char* filename, float** out_array, size_t* out_size) {
    FILE *file = fopen(filename, "rb");  // Open the file in binary read mode
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Seek to the end of the file to determine its size
    fseek(file, 0, SEEK_END);
    size_t num_elements = ftell(file) / sizeof(float);  // Assuming the file consists of integers
    fseek(file, 0, SEEK_SET);  // Reset the file pointer to the beginning of the file

    // Allocate memory for the array
    *out_array = (float*)malloc(num_elements * sizeof(float));
    if (*out_array == NULL) {
        fclose(file);
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Read the file into the array
    size_t read_count = fread(*out_array, sizeof(float), num_elements, file);
    if (read_count != num_elements) {
        free(*out_array);
        fclose(file);
        fprintf(stderr, "Failed to read file\n");
        exit(EXIT_FAILURE);
    }

    // Close the file
    fclose(file);

    // Output the size of the array
    *out_size = num_elements;
}

void print_head(float *arr) {
    for (int i=0; i<5; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
} 

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void vecAddKernel(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n) c[i] = a[i] + b[i];
}

int main() {
  float *a, *b;
  size_t size;

  load_array("vec1.bin", &a, &size);
  load_array("vec2.bin", &b, &size);

  float *c = (float *)malloc(size * sizeof(float));

  // allocating memory in gpu for tensors to be loaded
  size_t mem_size = size * sizeof(float);
  float *ad, *bd, *cd;
  cudaMalloc((void **)&ad, mem_size);
  cudaMalloc((void **)&bd, mem_size);
  cudaMalloc((void **)&cd, mem_size);

  // moving tensors
  cudaMemcpy(ad, a, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, mem_size, cudaMemcpyHostToDevice);

  // defining thread count and calculation blocks
  const unsigned int numThreads = 256;
  unsigned int numBlocks = cdiv(size, numThreads);

  vecAddKernel<<<numBlocks, numThreads>>>(ad, bd, cd, size);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(c, cd, mem_size, cudaMemcpyDeviceToHost);

  // free tensors on device 
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);

  printf("array size: %lu \n", size);
  print_head(a);
  print_head(b);
  print_head(c);

  free(a);
  free(b);
  free(c);
  return 0;
}
