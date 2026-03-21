#include <cublas_v2.h>

#include <cmath>
#include <iostream>

float lcgRandom() {
  static int seed = 1704;
  seed = seed * 1664525 + 1013904223;  // Classic LCG constants
  return (seed & 0xFFFF) / 65535.0f;   // Map to [0, 1]
}

void randomFill(float* vec, int len) {
  for (int i = 0; i < len; i++) {
    vec[i] = lcgRandom();
  }
}

enum MatmulAlgorithm {
  Naive,
  Warptiling,
};

__global__ void sgemm_naive(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
  const uint col = threadIdx.x + blockIdx.x * blockDim.x;
  const uint row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row >= M || col >= N) return;

  float acc = 0.0;
  for (int i = 0; i < K; i++) {
    acc += A[i + row * K] * B[col + i * N];
  }

  float C_val = 0.0;
  if (beta != 0) C_val = C[col + row * N];
  C[col + row * N] = (alpha * acc) + (beta * C_val);
}

void matmul(float* C_h, float* A_h, int A_m, int A_n, float* B_h, int B_m, int B_n, MatmulAlgorithm algo) {
  if (A_n != B_m) {
    std::cout << "A_n needs to be equal to B_m" << std::endl;
    return;
  }

  float *A_d, *B_d, *C_d;
  cudaMalloc((void**)&A_d, sizeof(float) * A_m * A_n);
  cudaMalloc((void**)&B_d, sizeof(float) * B_m * B_n);
  cudaMalloc((void**)&C_d, sizeof(float) * A_m * B_n);

  cudaMemcpy(A_d, A_h, sizeof(float) * A_m * A_n, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, sizeof(float) * B_m * B_n, cudaMemcpyHostToDevice);

  // kernel invocation
  if (algo == Naive) {
    dim3 blockDim = dim3(16, 16, 1);
    dim3 gridDim = dim3(ceil(B_n / (float)blockDim.x), ceil(A_m / (float)blockDim.y), 1);

    cudaDeviceSynchronize();
    sgemm_naive<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  cudaMemcpy(C_h, C_d, sizeof(float) * A_m * B_n, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void compareResults(float* ref, float* test, int len, const char* label, float rel_tol = 1e-4f, float abs_tol = 1e-5f) {
  int mismatches = 0;
  float maxRelDiff = 0.0f;
  float maxAbsDiff = 0.0f;
  int worstIdx = 0;

  for (int i = 0; i < len; i++) {
    float diff = fabs(ref[i] - test[i]);
    float tol = abs_tol + rel_tol * fabs(ref[i]);
    if (diff > tol) {
      mismatches++;
    }

    float relDiff = diff / (fabs(ref[i]) + 1e-30f);
    if (relDiff > maxRelDiff) {
      maxRelDiff = relDiff;
      maxAbsDiff = diff;
      worstIdx = i;
    }
  }

  if (mismatches == 0) {
    std::cout << label << ": PASS (max rel diff = " << maxRelDiff << ", max abs diff = " << maxAbsDiff << ")" << std::endl;
  } else {
    std::cout << label << ": FAIL — " << mismatches << "/" << len << " mismatches" << std::endl;
    std::cout << "  worst: ref[" << worstIdx << "] = " << ref[worstIdx] << ", test[" << worstIdx << "] = " << test[worstIdx] << ", abs diff = " << maxAbsDiff << ", rel diff = " << maxRelDiff
              << std::endl;
  }
}

int main() {
  int A_m = 2048;
  int A_n = 2048;
  int B_n = 2048;

  float* A = new float[A_m * A_n];
  float* B = new float[A_n * B_n];
  float* C = new float[A_m * B_n];

  randomFill(A, A_m * A_n);
  randomFill(B, A_n * B_n);

  float* D = new float[A_m * B_n];
  matmul(D, A, A_m, A_n, B, A_n, B_n, Naive);

  return 0;
}
