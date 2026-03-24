#pragma once
#include <cuda_runtime.h>

// Restored from the AI-assisted matmul experiment in pmpp-book commit ec79370.
// Public imported revision: 8dfe7c0 (matmul/matmul.cu). Kernel and config unchanged.
// Requires M/N multiples of 128 and K a multiple of 16.
namespace k10_experiment {
constexpr unsigned WARPSIZE = 32;
// k10 (vectorized shared loads): best at the big tile.
#ifndef K10_BM
#define K10_BM 128
#endif
#ifndef K10_BN
#define K10_BN 128
#endif
#ifndef K10_BK
#define K10_BK 16
#endif
#ifndef K10_TM
#define K10_TM 8
#endif
#ifndef K10_TN
#define K10_TN 4
#endif
#ifndef K10_WM
#define K10_WM 64
#endif
#ifndef K10_WN
#define K10_WN 64
#endif
#ifndef K10_WNITER
#define K10_WNITER 4
#endif
#ifndef K10_NUM_THREADS
#define K10_NUM_THREADS 128
#endif

// k10: k8 warptiling with the warptile register loads VECTORIZED (LDS.128). k8's profile shows
// the top true-stall is short_scoreboard (0.79 c/inst, shared-load latency) + dispatch_stall.
// k8 loaded regM/regN with TM/TN scalar LDS.32; since the TM (resp. TN) elements are contiguous
// in shared, each group is one float4 LDS.128 — 4x fewer shared-load instructions, less MIO
// pressure. Requires TM,TN multiples of 4. Same layout/config as k8, no extra registers/smem.
#ifndef K10_MINBLOCKS
#define K10_MINBLOCKS 1
#endif
__global__ void __launch_bounds__(K10_NUM_THREADS, K10_MINBLOCKS)
    sgemm_10_vec_smem(int M, int N, int K, float alpha, float beta, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (K10_BN / K10_WN);
  const uint warpRow = warpIdx / (K10_BN / K10_WN);

  constexpr uint WMITER = (K10_WM * K10_WN) / (WARPSIZE * K10_TM * K10_TN * K10_WNITER);
  constexpr uint WSUBM = K10_WM / WMITER;
  constexpr uint WSUBN = K10_WN / K10_WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / K10_TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / K10_TN);

  // Pad the As column stride so it is NOT a multiple of 32 (must stay multiple of 4 for float4).
  // The transpose store writes with stride BM; with BM=64 (mult. of 32) the 2 innerColA lanes
  // collide on the same bank (ncu: ~2.7-way store conflict). Padding to BM+4 breaks the collision.
#define K10_ASTRIDE (K10_BM + 4)
  __shared__ float As[K10_BK * K10_ASTRIDE];  // transposed, padded
  __shared__ float Bs[K10_BK * K10_BN];

  A += cRow * K10_BM * K;
  B += cCol * K10_BN;
  C += (cRow * K10_BM + warpRow * K10_WM) * N + cCol * K10_BN + warpCol * K10_WN;

  const uint innerRowA = threadIdx.x / (K10_BK / 4);
  const uint innerColA = threadIdx.x % (K10_BK / 4);
  constexpr uint rowStrideA = (K10_NUM_THREADS * 4) / K10_BK;
  const uint innerRowB = threadIdx.x / (K10_BN / 4);
  const uint innerColB = threadIdx.x % (K10_BN / 4);
  constexpr uint rowStrideB = K10_NUM_THREADS / (K10_BN / 4);

  float threadResults[WMITER * K10_TM * K10_WNITER * K10_TN] = {0.0};
  float regM[WMITER * K10_TM] = {0.0};
  float regN[K10_WNITER * K10_TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += K10_BK) {
    for (uint offset = 0; offset + rowStrideA <= K10_BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<const float4*>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * K10_ASTRIDE + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * K10_ASTRIDE + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * K10_ASTRIDE + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * K10_ASTRIDE + innerRowA + offset] = tmp.w;
    }
    for (uint offset = 0; offset + rowStrideB <= K10_BK; offset += rowStrideB)
      reinterpret_cast<float4*>(&Bs[(innerRowB + offset) * K10_BN + innerColB * 4])[0] = reinterpret_cast<const float4*>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    __syncthreads();

#pragma unroll
    for (uint dotIdx = 0; dotIdx < K10_BK; ++dotIdx) {
#pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint i = 0; i < K10_TM; i += 4)
          reinterpret_cast<float4*>(&regM[wSubRowIdx * K10_TM + i])[0] = reinterpret_cast<float4*>(&As[(dotIdx * K10_ASTRIDE) + warpRow * K10_WM + wSubRowIdx * WSUBM + threadRowInWarp * K10_TM + i])[0];
#pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < K10_WNITER; ++wSubColIdx)
        for (uint i = 0; i < K10_TN; i += 4)
          reinterpret_cast<float4*>(&regN[wSubColIdx * K10_TN + i])[0] = reinterpret_cast<float4*>(&Bs[(dotIdx * K10_BN) + warpCol * K10_WN + wSubColIdx * WSUBN + threadColInWarp * K10_TN + i])[0];

#pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < K10_WNITER; ++wSubColIdx)
#pragma unroll
          for (uint resIdxM = 0; resIdxM < K10_TM; ++resIdxM)
#pragma unroll
            for (uint resIdxN = 0; resIdxN < K10_TN; ++resIdxN)
              threadResults[(wSubRowIdx * K10_TM + resIdxM) * (K10_WNITER * K10_TN) + (wSubColIdx * K10_TN) + resIdxN] += regM[wSubRowIdx * K10_TM + resIdxM] * regN[wSubColIdx * K10_TN + resIdxN];
    }
    A += K10_BK;
    B += K10_BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < K10_WNITER; ++wSubColIdx) {
      float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < K10_TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < K10_TN; resIdxN += 4) {
          const int i = (wSubRowIdx * K10_TM + resIdxM) * (K10_WNITER * K10_TN) + wSubColIdx * K10_TN + resIdxN;
          float* dst = &C_interim[(threadRowInWarp * K10_TM + resIdxM) * N + threadColInWarp * K10_TN + resIdxN];
          float4 tmp;
          if (beta == 0) {
            tmp.x = alpha * threadResults[i + 0];
            tmp.y = alpha * threadResults[i + 1];
            tmp.z = alpha * threadResults[i + 2];
            tmp.w = alpha * threadResults[i + 3];
          } else {
            tmp = reinterpret_cast<float4*>(dst)[0];
            tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          }
          reinterpret_cast<float4*>(dst)[0] = tmp;
        }
      }
    }
  }
}


}  // namespace k10_experiment
