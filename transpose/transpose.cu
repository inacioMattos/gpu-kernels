#include "../common/check.cuh"
#include <utility>

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose(const float* input, float* output, int rows, int cols) {
  // Padding avoids shared-memory bank conflicts when reading columns.
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < cols && y + j < rows)
      tile[threadIdx.y + j][threadIdx.x] = input[size_t(y + j) * cols + x];
  }
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < rows && y + j < cols)
      output[size_t(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

int main() {
  for (auto shape : {std::pair<int, int>{1, 1}, {1, 73}, {91, 1}, {32, 32}, {31, 65}, {513, 1001}}) {
    int rows = shape.first, cols = shape.second;
    size_t n = size_t(rows) * cols;
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) input[i] = float(i);
    DeviceBuffer src(n), dst(n);
    src.upload(input);
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose<<<grid, block>>>(src.data, dst.data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    auto result = dst.download(n);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (result[size_t(c) * rows + r] != input[size_t(r) * cols + c]) {
          std::fprintf(stderr, "transpose failed: %dx%d at (%d,%d)\n", rows, cols, r, c);
          return 1;
        }
      }
    }
  }
  std::puts("transpose: PASS");
}
