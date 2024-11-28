#define TILE_SIZE 32

void mm_cpu(const float *a, const float *b, float *c, unsigned int n) {
  for (unsigned int row_tile = 0; row_tile < n/TILE_SIZE; ++row_tile) {
    for (unsigned int col_tile = 0; col_tile < n/TILE_SIZE; ++ col_tile) {
      for (unsigned int itile = 0; itile < n/TILE_SIZE; ++itile) {
        for unsigned int row = row_rile * TILE_SIZE; ++itile)
