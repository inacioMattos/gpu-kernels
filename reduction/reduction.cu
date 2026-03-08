#include "../common/check.cuh"

constexpr int BLOCK = 256;

// Each block reduces up to 512 values. Repeat on partial sums until one remains.
__global__ void reduce_sum(const float* input, float* output, size_t n) {
  __shared__ float values[BLOCK];
  unsigned t = threadIdx.x;
  size_t i = size_t(blockIdx.x) * (2 * BLOCK) + t;
  float sum = i < n ? input[i] : 0.0f;
  if (i + BLOCK < n) sum += input[i + BLOCK];
  values[t] = sum;
  __syncthreads();
  for (unsigned stride = BLOCK / 2; stride > 0; stride /= 2) {
    if (t < stride) values[t] += values[t + stride];
    __syncthreads();
  }
  if (t == 0) output[blockIdx.x] = values[0];
}

float sum_gpu(const std::vector<float>& input) {
  if (input.empty()) return 0.0f;
  size_t n = input.size();
  DeviceBuffer data(n), scratch((n + 2 * BLOCK - 1) / (2 * BLOCK));
  data.upload(input);
  float* src = data.data;
  float* dst = scratch.data;
  while (n > 1) {
    size_t blocks = (n + 2 * BLOCK - 1) / (2 * BLOCK);
    reduce_sum<<<static_cast<unsigned>(blocks), BLOCK>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
    float* tmp = src;
    src = dst;
    dst = tmp;
    n = blocks;
  }
  float result;
  CUDA_CHECK(cudaMemcpy(&result, src, sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

int main() {
  for (size_t n : {size_t(0), size_t(1), size_t(511), size_t(512), size_t(513), size_t(1000003)}) {
    std::vector<float> input(n);
    double expected = 0;
    for (size_t i = 0; i < n; ++i) {
      input[i] = float(int(i % 31) - 15) * 0.01f;
      expected += input[i];
    }
    float result = sum_gpu(input);
    if (!close(result, expected, 1e-4, 1e-4)) {
      std::fprintf(stderr, "reduction failed: n=%zu got=%g expected=%g\n", n, result, expected);
      return 1;
    }
  }
  std::puts("reduction: PASS");
}
