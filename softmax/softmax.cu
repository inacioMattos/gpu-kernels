#include "../common/check.cuh"
#include <math_constants.h>
#include <algorithm>

constexpr int BLOCK = 256;

// One block per row. Finite FP32 inputs, any positive row width.
__global__ void softmax(const float* input, float* output, int cols) {
  __shared__ float values[BLOCK];
  unsigned t = threadIdx.x;
  size_t base = size_t(blockIdx.x) * cols;
  float maximum = -CUDART_INF_F;
  for (int c = t; c < cols; c += BLOCK) maximum = fmaxf(maximum, input[base + c]);
  values[t] = maximum;
  __syncthreads();
  for (unsigned stride = BLOCK / 2; stride; stride /= 2) {
    if (t < stride) values[t] = fmaxf(values[t], values[t + stride]);
    __syncthreads();
  }
  maximum = values[0];
  // All threads must read the maximum before values[] is reused for sums.
  __syncthreads();
  float sum = 0;
  for (int c = t; c < cols; c += BLOCK) sum += expf(input[base + c] - maximum);
  values[t] = sum;
  __syncthreads();
  for (unsigned stride = BLOCK / 2; stride; stride /= 2) {
    if (t < stride) values[t] += values[t + stride];
    __syncthreads();
  }
  sum = values[0];
  for (int c = t; c < cols; c += BLOCK)
    output[base + c] = expf(input[base + c] - maximum) / sum;
}

int main() {
  constexpr int rows = 5;
  for (int cols : {1, 7, 255, 256, 257, 4099}) {
    size_t n = size_t(rows) * cols;
    std::vector<float> input(n);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        // Equal values, large positive/negative offsets, and a sharp peak.
        float value = float((c * 17) % 101 - 50) / 5;
        if (r == 0) value = 3;
        if (r == 1) value += 1000;
        if (r == 2) value -= 1000;
        if (r == 3) value = c == cols / 2 ? 1000 : -1000;
        input[size_t(r) * cols + c] = value;
      }
    }
    DeviceBuffer src(n), dst(n);
    src.upload(input);
    softmax<<<rows, BLOCK>>>(src.data, dst.data, cols);
    CUDA_CHECK(cudaGetLastError());
    auto result = dst.download(n);
    for (int r = 0; r < rows; ++r) {
      size_t base = size_t(r) * cols;
      double maximum = *std::max_element(input.begin() + base, input.begin() + base + cols);
      double denominator = 0, total = 0;
      for (int c = 0; c < cols; ++c) denominator += std::exp(input[base + c] - maximum);
      for (int c = 0; c < cols; ++c) {
        double expected = std::exp(input[base + c] - maximum) / denominator;
        float actual = result[base + c];
        if (actual < 0 || !close(actual, expected, 1e-7, 2e-5)) {
          std::fprintf(stderr, "softmax failed: cols=%d row=%d col=%d\n", cols, r, c);
          return 1;
        }
        total += actual;
      }
      if (std::abs(total - 1) > 1e-5) {
        std::fprintf(stderr, "softmax row sum failed: %g\n", total);
        return 1;
      }
    }
  }
  std::puts("softmax: PASS");
}
