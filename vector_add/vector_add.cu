#include "../common/check.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(blockDim.x) * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  for (size_t n : {size_t(1), size_t(255), size_t(256), size_t(257), size_t(1000003)}) {
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
      a[i] = float(int(i % 101) - 50) / 8;
      b[i] = float(int(i % 53) - 26) / 16;
    }
    DeviceBuffer da(n), db(n), dc(n);
    da.upload(a);
    db.upload(b);
    vector_add<<<128, 256>>>(da.data, db.data, dc.data, n);
    CUDA_CHECK(cudaGetLastError());
    auto result = dc.download(n);
    for (size_t i = 0; i < n; ++i) {
      if (result[i] != a[i] + b[i]) {
        std::fprintf(stderr, "vector_add failed: n=%zu i=%zu\n", n, i);
        return 1;
      }
    }
  }
  std::puts("vector_add: PASS");
}
