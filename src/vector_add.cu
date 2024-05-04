#include <stdio.h>

__global__ vec_add(int *int1, int *int2, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = in1[i] + in2[i];
}
