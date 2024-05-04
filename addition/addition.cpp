#include <stdio.h>
#include <stdlib.h>

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

int main() {
  float *a, *b;
  size_t size;

  load_array("vec1.bin", &a, &size);
  load_array("vec2.bin", &b, &size);

  printf("array size: %lu \n", size);
  print_head(a);
  print_head(b);

  free(a);
  free(b);
  return 0;
}
