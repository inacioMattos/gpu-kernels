// Benchmark the original kernels without timing allocation or host transfers.
#define main original_matmul_main
#include "matmul.cu"
#undef main
#include "../common/check.cuh"
#include <algorithm>
#include <fstream>
#include <random>
#include <string>

void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS error: %d\n", int(status));
    std::exit(1);
  }
}

const char* names[] = {"cublas_fp32", "cublas_default", "naive", "shared_tiled",
  "register_1d", "register_2d_v1", "register_2d_v2", "vectorized_k6",
  "warp_single_k7", "warp_multi_k8", "warp_vectorized_k10"};
constexpr int COUNT = sizeof(names) / sizeof(names[0]);

void launch(int algo, int n, float* a, float* b, float* c, cublasHandle_t strict, cublasHandle_t normal) {
  float alpha = 1, beta = 0;
  switch (algo) {
    case 0: case 1:
      blas_check(cublasSgemm(algo == 0 ? strict : normal, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n, &alpha, b, n, a, n, &beta, c, n));
      break;
    case 2:
      sgemm_naive<<<dim3(n / 16, n / 16), dim3(16, 16)>>>(n,n,n,1,0,a,b,c); break;
    case 3:
      sgemm_tiled<<<dim3(n / TILE_WIDTH, n / TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH)>>>(n,n,n,1,0,a,b,c); break;
    case 4:
      sgemm_4_register_tiling<<<dim3(n / BN_4, n / BM_4), BM_4 * BN_4 / TM_4>>>(n,n,n,1,0,a,b,c); break;
    case 5:
      sgemm_5_register_2dtiling<<<dim3(n / BN, n / BM), BM * BN / (TM * TM)>>>(n,n,n,1,0,a,b,c); break;
    case 6:
      sgemm_5_register_2dtiling_v2<<<dim3(n / BN, n / BM), BM * BN / (TM * TM)>>>(n,n,n,1,0,a,b,c); break;
    case 7:
      sgemm_6_register_2dtiling_vectorized_As<<<dim3(n / K6_BN, n / K6_BM), K6_BM * K6_BN / (K6_TM * K6_TM)>>>(n,n,n,1,0,a,b,c); break;
    case 8:
      sgemm_7_warptiling_single_iter<<<dim3(n / K7_BN, n / K7_BM), K7_BM * K7_BN / (K7_TM * K7_TN)>>>(n,n,n,1,0,a,b,c); break;
    case 9:
      sgemm_8_warptiling<<<dim3(n / K8_BN, n / K8_BM), K8_BM * K8_BN / (K8_TM * K8_TN * K8_WITER)>>>(n,n,n,1,0,a,b,c); break;
    case 10:
      sgemm_10_vec_smem<<<dim3(n / K10_BN, n / K10_BM), K10_NUM_THREADS>>>(n,n,n,1,0,a,b,c); break;
  }
  CUDA_CHECK(cudaGetLastError());
}

int positive_int(const char* text) {
  char* end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (!*text || *end || value < 1 || value > 16384) {
    std::fprintf(stderr, "Expected an integer in [1,16384]: %s\n", text);
    std::exit(2);
  }
  return int(value);
}

int main(int argc, char** argv) {
  if (argc > 5) {
    std::fprintf(stderr, "Usage: %s [N=2048] [rounds=7] [iterations=10] [samples.csv]\n", argv[0]);
    return 2;
  }
  int n = argc > 1 ? positive_int(argv[1]) : 2048;
  int rounds = argc > 2 ? positive_int(argv[2]) : 7;
  int iterations = argc > 3 ? positive_int(argv[3]) : 10;
  if (n % 256) {
    std::fprintf(stderr, "N must be a multiple of 256 for these kernel configurations.\n");
    return 2;
  }
  std::ofstream samples(argc > 4 ? argv[4] : "build/matmul-samples.csv");
  if (!samples) { std::fprintf(stderr, "Cannot open sample output\n"); return 2; }
  samples << "n,kernel,round,iterations,mean_us\n";
  samples << std::setprecision(9);

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  int runtime, driver, blas_version;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  cublasHandle_t strict, normal;
  blas_check(cublasCreate(&strict));
  blas_check(cublasCreate(&normal));
  blas_check(cublasSetMathMode(strict, CUBLAS_PEDANTIC_MATH));
  blas_check(cublasSetMathMode(normal, CUBLAS_DEFAULT_MATH));
  blas_check(cublasGetVersion(strict, &blas_version));
  std::cout << "# GPU=" << props.name << ", SMs=" << props.multiProcessorCount
            << ", compute=" << props.major << '.' << props.minor << '\n'
            << "# CUDA runtime=" << runtime << ", driver API=" << driver << ", cuBLAS=" << blas_version << '\n'
            << "# N=" << n << ", rounds=" << rounds << ", iterations=" << iterations
            << ", warmups=3, alpha=1, beta=0, seed=1704, input=uniform[0,1)\n"
            << "# cublas_fp32=CUBLAS_PEDANTIC_MATH; cublas_default=CUBLAS_DEFAULT_MATH\n";
  size_t count = size_t(n) * n;
  std::vector<float> a(count), b(count);
  std::mt19937 rng(1704);
  for (size_t i = 0; i < count; ++i) {
    a[i] = (rng() >> 8) * (1.0f / 16777216.0f);
    b[i] = (rng() >> 8) * (1.0f / 16777216.0f);
  }
  DeviceBuffer da(count), db(count), dc(count), reference(count);
  da.upload(a); db.upload(b);
  launch(0, n, da.data, db.data, reference.data, strict, normal);
  auto expected = reference.download(count);
  std::vector<double> max_abs(COUNT, 0), max_rel(COUNT, 0);
  for (int k = 0; k < COUNT; ++k) {
    // NaN-fill catches missing output writes as well as non-finite results.
    CUDA_CHECK(cudaMemset(dc.data, 0xff, count * sizeof(float)));
    launch(k, n, da.data, db.data, dc.data, strict, normal);
    auto actual = dc.download(count);
    for (size_t i = 0; i < count; ++i) {
      double diff = std::abs(double(actual[i]) - expected[i]);
      if (!std::isfinite(actual[i]) || diff > 1e-5 + 1e-4 * std::abs(expected[i])) {
        std::fprintf(stderr, "%s FAIL at %zu: got %g expected %g\n", names[k], i, actual[i], expected[i]);
        return 1;
      }
      max_abs[k] = std::max(max_abs[k], diff);
      max_rel[k] = std::max(max_rel[k], diff / std::max(1e-30, std::abs(double(expected[i]))));
    }
    std::cerr << names[k] << ": PASS\n";
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<std::vector<double>> times(COUNT);
  for (int round = 0; round < rounds; ++round) {
    for (int offset = 0; offset < COUNT; ++offset) {
      int k = (offset + round * 3) % COUNT;
      for (int i = 0; i < 3; ++i) launch(k, n, da.data, db.data, dc.data, strict, normal);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(start));
      for (int i = 0; i < iterations; ++i) launch(k, n, da.data, db.data, dc.data, strict, normal);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      double us = double(ms) * 1000 / iterations;
      times[k].push_back(us);
      samples << n << ',' << names[k] << ',' << round << ',' << iterations << ',' << us << '\n';
    }
  }
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[v.size() / 2] + v[(v.size() - 1) / 2]) / 2;
  };
  double baseline = median(times[0]);
  double naive = median(times[2]);
  std::cout << "kernel,n,median_us,min_us,max_us,tflops,speedup_vs_naive,percent_cublas_fp32,max_abs_error,max_rel_error,status\n";
  std::cout << std::setprecision(9);
  for (int k = 0; k < COUNT; ++k) {
    double med = median(times[k]);
    std::cout << names[k] << ',' << n << ',' << med << ','
              << *std::min_element(times[k].begin(), times[k].end()) << ','
              << *std::max_element(times[k].begin(), times[k].end()) << ','
              << (2.0 * n * n * n / (med * 1e6)) << ',' << naive / med << ','
              << (100 * baseline / med) << ',' << max_abs[k] << ',' << max_rel[k] << ",PASS\n";
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  blas_check(cublasDestroy(strict)); blas_check(cublasDestroy(normal));
}
