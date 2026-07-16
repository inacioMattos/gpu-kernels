#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#define BR 4
#define BC 32

enum AttentionAlgorithm {
  Naive,
  FlashAttention,
};

void cudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

float lcgRandom() {
  static uint32_t seed = 1704;
  seed = seed * 1664525u + 1013904223u;
  return 2.0f * (seed & 0xFFFF) / 65535.0f - 1.0f;
}

void randomFill(float* vec, size_t len) {
  for (size_t i = 0; i < len; i++) {
    vec[i] = lcgRandom();
  }
}

__device__ float warpMax(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFF, value, offset));
  }
  return value;
}

__device__ float warpSum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }
  return value;
}

__global__ void attention_1_scores(int N, int D, bool causal, const float* Q, const float* K, float* S) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t inputOffset = (size_t)blockIdx.z * N * D;
  const size_t scoreOffset = (size_t)blockIdx.z * N * N;

  if (row >= N || col >= N) return;

  float product = 0.0f;
  for (int d = 0; d < D; d++) {
    product += Q[inputOffset + (size_t)row * D + d] * K[inputOffset + (size_t)col * D + d];
  }
  S[scoreOffset + (size_t)row * N + col] = causal && col > row ? -INFINITY : product * rsqrtf((float)D);
}

__global__ void attention_1_softmax(int N, float* S) {
  const int lane = threadIdx.x;
  float* row = S + ((size_t)blockIdx.y * N + blockIdx.x) * N;
  float maxValue = -INFINITY;

  for (int col = lane; col < N; col += 32) {
    maxValue = fmaxf(maxValue, row[col]);
  }
  maxValue = warpMax(maxValue);

  float sum = 0.0f;
  for (int col = lane; col < N; col += 32) {
    float value = expf(row[col] - maxValue);
    row[col] = value;
    sum += value;
  }
  sum = warpSum(sum);

  for (int col = lane; col < N; col += 32) {
    row[col] /= sum;
  }
}

__global__ void attention_1_output(int N, int D, const float* P, const float* V, float* O) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t inputOffset = (size_t)blockIdx.z * N * D;
  const size_t scoreOffset = (size_t)blockIdx.z * N * N;

  if (row >= N || col >= D) return;

  float product = 0.0f;
  for (int k = 0; k < N; k++) {
    product += P[scoreOffset + (size_t)row * N + k] * V[inputOffset + (size_t)k * D + col];
  }
  O[inputOffset + (size_t)row * D + col] = product;
}

template <int BD, bool CAUSAL>
__global__ void attention_2_flash(int N, int D, const float* Q, const float* K, const float* V, float* O) {
  const int lane = threadIdx.x % 32;
  const int rowInTile = threadIdx.x / 32;
  const int rowOffset = blockIdx.x * BR;
  const int row = rowOffset + rowInTile;
  const size_t headOffset = (size_t)blockIdx.y * N * D;

  __shared__ float Qs[BR][BD];
  __shared__ float KVs[BD * (BC + 1)];
  __shared__ float Ps[BR][BC];

  float product[BD / 32] = {0.0f};
  float runningMax = -INFINITY;
  float runningSum = 0.0f;
  const float scale = rsqrtf((float)D);

  for (int t = threadIdx.x; t < BR * BD; t += BR * 32) {
    const int r = t / BD;
    const int d = t % BD;
    Qs[r][d] = rowOffset + r < N && d < D ? Q[headOffset + (size_t)(rowOffset + r) * D + d] : 0.0f;
  }

  const int keyLimit = CAUSAL ? min(N, rowOffset + BR) : N;
  for (int keyOffset = 0; keyOffset < keyLimit; keyOffset += BC) {
    for (int t = threadIdx.x; t < BC * BD; t += BR * 32) {
      const int k = t / BD;
      const int d = t % BD;
      KVs[d * (BC + 1) + k] = keyOffset + k < N && d < D ? K[headOffset + (size_t)(keyOffset + k) * D + d] : 0.0f;
    }
    __syncthreads();

    float score = 0.0f;
#pragma unroll 4
    for (int d = 0; d < BD; d++) {
      score = fmaf(Qs[rowInTile][d], KVs[d * (BC + 1) + lane], score);
    }

    const int key = keyOffset + lane;
    const bool valid = row < N && key < N && (!CAUSAL || key <= row);
    score = valid ? score * scale : -INFINITY;
    const float nextMax = fmaxf(runningMax, warpMax(score));
    const float correction = runningSum > 0.0f ? expf(runningMax - nextMax) : 0.0f;
    const float weight = valid ? expf(score - nextMax) : 0.0f;
    runningSum = runningSum * correction + warpSum(weight);
    runningMax = nextMax;
    Ps[rowInTile][lane] = weight;

#pragma unroll
    for (int d = 0; d < BD / 32; d++) {
      product[d] *= correction;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < BC * BD; t += BR * 32) {
      const int k = t / BD;
      const int d = t % BD;
      KVs[t] = keyOffset + k < N && d < D ? V[headOffset + (size_t)(keyOffset + k) * D + d] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BC; k++) {
      const float weight = Ps[rowInTile][k];
#pragma unroll
      for (int d = 0; d < BD / 32; d++) {
        product[d] = fmaf(weight, KVs[k * BD + d * 32 + lane], product[d]);
      }
    }
    __syncthreads();
  }

  if (row < N) {
#pragma unroll
    for (int d = 0; d < BD / 32; d++) {
      const int col = d * 32 + lane;
      if (col < D) O[headOffset + (size_t)row * D + col] = product[d] / runningSum;
    }
  }
}

template <int BD>
void launchFlash(int N, int D, int heads, bool causal, const float* Q, const float* K, const float* V, float* O) {
  dim3 blockDim(BR * 32);
  dim3 gridDim((N + BR - 1) / BR, heads);
  if (causal) {
    attention_2_flash<BD, true><<<gridDim, blockDim>>>(N, D, Q, K, V, O);
  } else {
    attention_2_flash<BD, false><<<gridDim, blockDim>>>(N, D, Q, K, V, O);
  }
}

void launchAttention(int N, int D, int heads, bool causal, const float* Q, const float* K, const float* V, float* O, float* S, AttentionAlgorithm algo) {
  if (algo == Naive) {
    dim3 blockDim(16, 16);
    attention_1_scores<<<dim3((N + 15) / 16, (N + 15) / 16, heads), blockDim>>>(N, D, causal, Q, K, S);
    cudaCheck(cudaGetLastError());
    attention_1_softmax<<<dim3(N, heads), 32>>>(N, S);
    cudaCheck(cudaGetLastError());
    attention_1_output<<<dim3((D + 15) / 16, (N + 15) / 16, heads), blockDim>>>(N, D, S, V, O);
  } else if (D <= 32) {
    launchFlash<32>(N, D, heads, causal, Q, K, V, O);
  } else if (D <= 64) {
    launchFlash<64>(N, D, heads, causal, Q, K, V, O);
  } else if (D <= 128) {
    launchFlash<128>(N, D, heads, causal, Q, K, V, O);
  } else {
    launchFlash<256>(N, D, heads, causal, Q, K, V, O);
  }
  cudaCheck(cudaGetLastError());
}

float attention(float* O_h, const float* Q_h, const float* K_h, const float* V_h, int B, int H, int N, int D, bool causal, AttentionAlgorithm algo, int iterations = 1) {
  if (B <= 0 || H <= 0 || (int64_t)B * H > 65535 || N <= 0 || N > 1048560 || D <= 0 || D > 256 || iterations <= 0) {
    std::cerr << "Expected B, H > 0, B * H <= 65535, 1 <= N <= 1048560, 1 <= D <= 256, iterations > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int heads = B * H;
  const size_t bytes = (size_t)heads * N * D * sizeof(float);
  float *Q_d, *K_d, *V_d, *O_d;
  float* S_d = nullptr;
  cudaCheck(cudaMalloc((void**)&Q_d, bytes));
  cudaCheck(cudaMalloc((void**)&K_d, bytes));
  cudaCheck(cudaMalloc((void**)&V_d, bytes));
  cudaCheck(cudaMalloc((void**)&O_d, bytes));
  if (algo == Naive) cudaCheck(cudaMalloc((void**)&S_d, (size_t)heads * N * N * sizeof(float)));

  cudaCheck(cudaMemcpy(Q_d, Q_h, bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(K_d, K_h, bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(V_d, V_h, bytes, cudaMemcpyHostToDevice));
  launchAttention(N, D, heads, causal, Q_d, K_d, V_d, O_d, S_d, algo);
  cudaCheck(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  cudaCheck(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    launchAttention(N, D, heads, causal, Q_d, K_d, V_d, O_d, S_d, algo);
  }
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));

  float elapsed;
  cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));
  cudaCheck(cudaMemcpy(O_h, O_d, bytes, cudaMemcpyDeviceToHost));
  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));
  cudaCheck(cudaFree(Q_d));
  cudaCheck(cudaFree(K_d));
  cudaCheck(cudaFree(V_d));
  cudaCheck(cudaFree(O_d));
  if (S_d) cudaCheck(cudaFree(S_d));
  return elapsed / iterations;
}

bool compareResults(const float* ref, const float* test, size_t len, const char* label, float rel_tol = 1e-4f, float abs_tol = 1e-5f) {
  size_t mismatches = 0;
  float maxAbsDiff = 0.0f;
  for (size_t i = 0; i < len; i++) {
    const float diff = fabsf(ref[i] - test[i]);
    if (!std::isfinite(ref[i]) || !std::isfinite(test[i]) || diff > abs_tol + rel_tol * fabsf(ref[i])) mismatches++;
    maxAbsDiff = std::max(maxAbsDiff, diff);
  }
  std::cout << label << ": " << (mismatches == 0 ? "PASS" : "FAIL") << " (max abs diff = " << maxAbsDiff << ", mismatches = " << mismatches << ")" << std::endl;
  return mismatches == 0;
}

#ifndef FLASH_ATTENTION_NO_MAIN
int main() {
  const int B = 1;
  const int H = 4;
  const int N = 512;
  const int D = 64;
  const size_t len = (size_t)B * H * N * D;

  float* Q = new float[len];
  float* K = new float[len];
  float* V = new float[len];
  float* ref = new float[len];
  float* result = new float[len];
  randomFill(Q, len);
  randomFill(K, len);
  randomFill(V, len);

  bool passed = true;
  for (bool causal : {false, true}) {
    std::cout << "B=" << B << " H=" << H << " N=" << N << " D=" << D << " causal=" << causal << std::endl;
    float naiveMs = attention(ref, Q, K, V, B, H, N, D, causal, Naive, 20);
    float flashMs = attention(result, Q, K, V, B, H, N, D, causal, FlashAttention, 20);
    std::cout << "naive: " << naiveMs * 1000 << " us, flash: " << flashMs * 1000 << " us" << std::endl;
    passed &= compareResults(ref, result, len, "naive v flash");
  }

  delete[] Q;
  delete[] K;
  delete[] V;
  delete[] ref;
  delete[] result;
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
