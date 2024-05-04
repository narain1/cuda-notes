#include <stdio.h>

void load_tensor(const char *fname, float *arr, size_t size) {
  FILE *file = fopen(fname, 'rb');
  if (file == NULL) {

