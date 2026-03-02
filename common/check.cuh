#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

inline void cuda_check(cudaError_t status, const char* expr, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "line %d: %s: %s\n", line, expr, cudaGetErrorString(status));
    std::exit(1);
  }
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __LINE__)

struct DeviceBuffer {
  float* data = nullptr;
  explicit DeviceBuffer(size_t n) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(float)));
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void upload(const std::vector<float>& v) {
    CUDA_CHECK(cudaMemcpy(data, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  std::vector<float> download(size_t n) const {
    std::vector<float> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
  }
};

inline bool close(float actual, double expected, double atol = 1e-5, double rtol = 1e-5) {
  return std::isfinite(actual) && std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}
