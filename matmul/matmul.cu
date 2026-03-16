#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>

#define WALL_START(name)         \
  struct timespec _start_##name; \
  clock_gettime(CLOCK_MONOTONIC, &_start_##name);
#define WALL_END(name)                                                                                                                                                    \
  {                                                                                                                                                                       \
    struct timespec _end_##name;                                                                                                                                          \
    clock_gettime(CLOCK_MONOTONIC, &_end_##name);                                                                                                                         \
    std::cout << #name << ": " << ((_end_##name.tv_sec - _start_##name.tv_sec) * 1000000L + (_end_##name.tv_nsec - _start_##name.tv_nsec) / 1000L) << " us" << std::endl; \
  }

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

void printMatrix(const float* mat, int rows, int cols, const char* name = nullptr) {
  if (name) std::cout << name << " ";
  std::cout << "(" << rows << "x" << cols << ")\n";

  int colWidth = 6;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      char buf[32];
      int len = snprintf(buf, sizeof(buf), "%.6g", mat[c + r * cols]);
      if (len > 0) colWidth = std::max(colWidth, len);
    }
  }
  colWidth += 2;

  for (int r = 0; r < rows; r++) {
    std::cout << "  ";
    for (int c = 0; c < cols; c++) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.6g", mat[c + r * cols]);
      std::cout << std::setw(colWidth) << buf;
    }
    std::cout << "\n";
  }
}

enum MatmulAlgorithm {
  Cublas,
  Naive,
  Tiled,
  TiledWith1DRegisterTiling,
  TiledWith2DRegisterTiling,
  TiledWith2DRegisterTilingAsVectorized,
  K7VectorizedSmem,
  K8Warptiling,
  K9DoubleBuffer,
  K10VecSmem,
  K11DbVec,
};

#define TM 8
#define BK 8
#define BM 128
#define BN 128

// k6 (square 2D register tiling, vectorized GMEM): own overridable config for a fair sweep.
#ifndef K6_BM
#define K6_BM 64
#endif
#ifndef K6_BN
#define K6_BN 256
#endif
#ifndef K6_BK
#define K6_BK 32
#endif
#ifndef K6_TM
#define K6_TM 8
#endif

__global__ void sgemm_6_register_2dtiling_vectorized_As(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C) {
  // Compile-time validity for the (BM,BN,BK,TM) config. The compute phase tiles the BMxBN output
  // in TMxTM blocks (needs BM,BN divisible by TM); the float4 loaders need BK,BN divisible by 4.
  if (K6_BK % 4 != 0 || K6_BN % 4 != 0 || K6_BM % K6_TM != 0 || K6_BN % K6_TM != 0) {
    return;
  }

  __shared__ float As[K6_BK][K6_BM];
  __shared__ float Bs[K6_BK][K6_BN];

  float product[K6_TM][K6_TM] = {0.0};

  const uint totalThreads = (K6_BM * K6_BN) / (K6_TM * K6_TM);

  const uint blockRowOffset = blockIdx.y * K6_BM;
  const uint blockColOffset = blockIdx.x * K6_BN;

  const uint computeRowInTile = K6_TM * ((threadIdx.x * K6_TM) / K6_BN);
  const uint computeColInTile = (threadIdx.x * K6_TM) % K6_BN;

  const uint CrowOffset = blockRowOffset + computeRowInTile;
  const uint CcolOffset = blockColOffset + computeColInTile;

  // float4 groups per row in each tile. General grid-stride loaders below fill the SAME shared
  // contents as the original hand-rolled loops, but cover any thread count / tile shape.
  const uint a4PerRow = K6_BK / 4;  // along K  (As stored transposed [BK][BM])
  const uint b4PerRow = K6_BN / 4;  // along N  (Bs stored [BK][BN])

  for (uint bkIdx = 0; bkIdx < K; bkIdx += K6_BK) {
    // LOADING PHASE: A tile -> As (transposed), float4 along K
    for (uint t = threadIdx.x; t < K6_BM * a4PerRow; t += totalThreads) {
      const uint mRow = t / a4PerRow;
      const uint kCol = (t % a4PerRow) * 4;
      const uint gRow = blockRowOffset + mRow;
      const uint gCol = bkIdx + kCol;
      float4 tmp = {0, 0, 0, 0};
      if (gRow < M && gCol + 3 < K) tmp = reinterpret_cast<float4*>(&A[gRow * K + gCol])[0];
      As[kCol + 0][mRow] = tmp.x;
      As[kCol + 1][mRow] = tmp.y;
      As[kCol + 2][mRow] = tmp.z;
      As[kCol + 3][mRow] = tmp.w;
    }
    // LOADING PHASE: B tile -> Bs, float4 along N
    for (uint t = threadIdx.x; t < K6_BK * b4PerRow; t += totalThreads) {
      const uint kRow = t / b4PerRow;
      const uint nCol = (t % b4PerRow) * 4;
      const uint gRow = bkIdx + kRow;
      const uint gCol = blockColOffset + nCol;
      float4 tmp = {0, 0, 0, 0};
      if (gRow < K && gCol + 3 < N) tmp = reinterpret_cast<float4*>(&B[gRow * N + gCol])[0];
      Bs[kRow][nCol + 0] = tmp.x;
      Bs[kRow][nCol + 1] = tmp.y;
      Bs[kRow][nCol + 2] = tmp.z;
      Bs[kRow][nCol + 3] = tmp.w;
    }

    __syncthreads();

    // COMPUTE PHASE START
    float Atmp[K6_TM] = {0.0};
    float Btmp[K6_TM] = {0.0};
    for (uint dotIdx = 0; dotIdx < K6_BK; dotIdx++) {
      for (uint tm = 0; tm < K6_TM; tm++) {
        const uint fromAsRow = (K6_TM * ((threadIdx.x * K6_TM) / K6_BN)) + tm;
        const uint fromAsCol = dotIdx;
        Atmp[tm] = As[fromAsCol][fromAsRow];
      }

      // K6_BM = K6_BN = 128
      // K6_TM = 8
      // tn = 0
      for (uint tn = 0; tn < K6_TM; tn++) {
        const uint fromBsRow = dotIdx;

        // idx = 0 -> (0 * 8) % 128 + 0 = 0
        // idx = 1 -> 8
        // idx = 2 -> 16
        // ...
        // idx= 10 -> 80
        // ...
        // idx= 15 -> 120
        // idx= 16 -> 0
        // idx= 31 -> 120
        const uint fromBsCol = ((threadIdx.x * K6_TM) % K6_BN) + tn;
        Btmp[tn] = Bs[fromBsRow][fromBsCol];
      }

      for (uint tm = 0; tm < K6_TM; tm++) {
        for (uint tn = 0; tn < K6_TM; tn++) {
          product[tm][tn] += Atmp[tm] * Btmp[tn];
        }
      }
    }

    __syncthreads();
  }

  for (uint tm = 0; tm < K6_TM; tm++) {
    for (uint tn = 0; tn < K6_TM; tn++) {
      const uint row = CrowOffset + tm;
      const uint col = CcolOffset + tn;

      if (col >= N || row >= M) continue;

      const uint Cidx = col + row * N;
      if (beta == 0) C[Cidx] = alpha * product[tm][tn];
      else C[Cidx] = alpha * product[tm][tn] + beta * C[Cidx];
    }
  }
}

// k8: warptiling. Resolves the structural ~4-way Bs shared-load bank conflict that
// k6/k7 could not (proven: float4 reads still showed 33.7M conflicts, and forcing
// higher occupancy made it slower — the L1/shared pipe, not occupancy, is the limit).
// Each warp owns a WM x WN sub-tile; within it threads map with stride TN so the
// float4 Bs reads span distinct banks, and each thread iterates WMITER x WNITER
// sub-tiles, reusing each shared load across more FMAs (higher arithmetic intensity).
// Block tile 64x128, 128 threads/block. (Boehm kernel-10 layout.)
#define WARPSIZE 32

// --- per-kernel tile configs (each independently overridable via -D) ---
// k8 (scalar shared loads): tuned to its own best (small tile).
#ifndef K8_BM
#define K8_BM 64
#endif
#ifndef K8_BN
#define K8_BN 128
#endif
#ifndef K8_BK
#define K8_BK 8
#endif
#ifndef K8_TM
#define K8_TM 4
#endif
#ifndef K8_TN
#define K8_TN 4
#endif
#ifndef K8_WM
#define K8_WM 32
#endif
#ifndef K8_WN
#define K8_WN 64
#endif
#ifndef K8_WNITER
#define K8_WNITER 2
#endif
#ifndef K8_NUM_THREADS
#define K8_NUM_THREADS 128
#endif

// k9 (double buffer, scalar loads): best at the big tile.
#ifndef K9_BM
#define K9_BM 128
#endif
#ifndef K9_BN
#define K9_BN 128
#endif
#ifndef K9_BK
#define K9_BK 16
#endif
#ifndef K9_TM
#define K9_TM 8
#endif
#ifndef K9_TN
#define K9_TN 4
#endif
#ifndef K9_WM
#define K9_WM 64
#endif
#ifndef K9_WN
#define K9_WN 64
#endif
#ifndef K9_WNITER
#define K9_WNITER 4
#endif
#ifndef K9_NUM_THREADS
#define K9_NUM_THREADS 128
#endif

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

// k11 (double buffer + vectorized loads): best at the big tile.
#ifndef K11_BM
#define K11_BM 128
#endif
#ifndef K11_BN
#define K11_BN 128
#endif
#ifndef K11_BK
#define K11_BK 16
#endif
#ifndef K11_TM
#define K11_TM 8
#endif
#ifndef K11_TN
#define K11_TN 4
#endif
#ifndef K11_WM
#define K11_WM 64
#endif
#ifndef K11_WN
#define K11_WN 64
#endif
#ifndef K11_WNITER
#define K11_WNITER 4
#endif
#ifndef K11_NUM_THREADS
#define K11_NUM_THREADS 128
#endif
__global__ void __launch_bounds__(K8_NUM_THREADS) sgemm_8_warptiling(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (K8_BN / K8_WN);
  const uint warpRow = warpIdx / (K8_BN / K8_WN);

  constexpr uint WMITER = (K8_WM * K8_WN) / (WARPSIZE * K8_TM * K8_TN * K8_WNITER);
  constexpr uint WSUBM = K8_WM / WMITER;
  constexpr uint WSUBN = K8_WN / K8_WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / K8_TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / K8_TN);

  __shared__ float As[K8_BM * K8_BK];  // stored transposed: As[k * BM + m]
  __shared__ float Bs[K8_BK * K8_BN];

  A += cRow * K8_BM * K;
  B += cCol * K8_BN;
  C += (cRow * K8_BM + warpRow * K8_WM) * N + cCol * K8_BN + warpCol * K8_WN;

  const uint innerRowA = threadIdx.x / (K8_BK / 4);
  const uint innerColA = threadIdx.x % (K8_BK / 4);
  constexpr uint rowStrideA = (K8_NUM_THREADS * 4) / K8_BK;
  const uint innerRowB = threadIdx.x / (K8_BN / 4);
  const uint innerColB = threadIdx.x % (K8_BN / 4);
  constexpr uint rowStrideB = K8_NUM_THREADS / (K8_BN / 4);

  float threadResults[WMITER * K8_TM * K8_WNITER * K8_TN] = {0.0};
  float regM[WMITER * K8_TM] = {0.0};
  float regN[K8_WNITER * K8_TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += K8_BK) {
    for (uint offset = 0; offset + rowStrideA <= K8_BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4*>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * K8_BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * K8_BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * K8_BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * K8_BM + innerRowA + offset] = tmp.w;
    }
    for (uint offset = 0; offset + rowStrideB <= K8_BK; offset += rowStrideB) {
      reinterpret_cast<float4*>(&Bs[(innerRowB + offset) * K8_BN + innerColB * 4])[0] = reinterpret_cast<float4*>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    for (uint dotIdx = 0; dotIdx < K8_BK; ++dotIdx) {
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint i = 0; i < K8_TM; ++i) regM[wSubRowIdx * K8_TM + i] = As[(dotIdx * K8_BM) + warpRow * K8_WM + wSubRowIdx * WSUBM + threadRowInWarp * K8_TM + i];
      for (uint wSubColIdx = 0; wSubColIdx < K8_WNITER; ++wSubColIdx)
        for (uint i = 0; i < K8_TN; ++i) regN[wSubColIdx * K8_TN + i] = Bs[(dotIdx * K8_BN) + warpCol * K8_WN + wSubColIdx * WSUBN + threadColInWarp * K8_TN + i];

      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint wSubColIdx = 0; wSubColIdx < K8_WNITER; ++wSubColIdx)
          for (uint resIdxM = 0; resIdxM < K8_TM; ++resIdxM)
            for (uint resIdxN = 0; resIdxN < K8_TN; ++resIdxN)
              threadResults[(wSubRowIdx * K8_TM + resIdxM) * (K8_WNITER * K8_TN) + (wSubColIdx * K8_TN) + resIdxN] += regM[wSubRowIdx * K8_TM + resIdxM] * regN[wSubColIdx * K8_TN + resIdxN];
    }
    A += K8_BK;
    B += K8_BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < K8_WNITER; ++wSubColIdx) {
      float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < K8_TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < K8_TN; resIdxN += 4) {
          const int i = (wSubRowIdx * K8_TM + resIdxM) * (K8_WNITER * K8_TN) + wSubColIdx * K8_TN + resIdxN;
          float* dst = &C_interim[(threadRowInWarp * K8_TM + resIdxM) * N + threadColInWarp * K8_TN + resIdxN];
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

// k11: k10 (vectorized LDS.128 warptile loads) + DOUBLE BUFFERING done right. Two shared buffers
// so the per-tile sequence is load(next)->compute(cur)->commit->ONE barrier (vs two in k10),
// cutting the barrier stall and overlapping the global LDG for tile t+1 with the FMAs of tile t.
#define K11_NA (K11_BM / ((K11_NUM_THREADS * 4) / K11_BK))
#define K11_NB (K11_BK / (K11_NUM_THREADS / (K11_BN / 4)))
#ifndef K11_MINBLOCKS
#define K11_MINBLOCKS 1
#endif
__global__ void __launch_bounds__(K11_NUM_THREADS, K11_MINBLOCKS) sgemm_11_db_vec(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (K11_BN / K11_WN);
  const uint warpRow = warpIdx / (K11_BN / K11_WN);

  constexpr uint WMITER = (K11_WM * K11_WN) / (WARPSIZE * K11_TM * K11_TN * K11_WNITER);
  constexpr uint WSUBM = K11_WM / WMITER;
  constexpr uint WSUBN = K11_WN / K11_WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / K11_TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / K11_TN);

  __shared__ float As[2][K11_BK * K11_BM];
  __shared__ float Bs[2][K11_BK * K11_BN];

  A += cRow * K11_BM * K;
  B += cCol * K11_BN;
  C += (cRow * K11_BM + warpRow * K11_WM) * N + cCol * K11_BN + warpCol * K11_WN;

  const uint innerRowA = threadIdx.x / (K11_BK / 4);
  const uint innerColA = threadIdx.x % (K11_BK / 4);
  constexpr uint rowStrideA = (K11_NUM_THREADS * 4) / K11_BK;
  const uint innerRowB = threadIdx.x / (K11_BN / 4);
  const uint innerColB = threadIdx.x % (K11_BN / 4);
  constexpr uint rowStrideB = K11_NUM_THREADS / (K11_BN / 4);

  float threadResults[WMITER * K11_TM * K11_WNITER * K11_TN] = {0.0};
  float regM[WMITER * K11_TM];
  float regN[K11_WNITER * K11_TN];
  float4 aReg[K11_NA];
  float4 bReg[K11_NB];

  const uint nTiles = K / K11_BK;

#define K11_STORE_SHARED(buf)                                                                                                                                  \
  {                                                                                                                                                            \
    uint j = 0;                                                                                                                                                \
    for (uint o = 0; o + rowStrideA <= K11_BM; o += rowStrideA, ++j) {                                                                                          \
      As[buf][(innerColA * 4 + 0) * K11_BM + innerRowA + o] = aReg[j].x;                                                                                        \
      As[buf][(innerColA * 4 + 1) * K11_BM + innerRowA + o] = aReg[j].y;                                                                                        \
      As[buf][(innerColA * 4 + 2) * K11_BM + innerRowA + o] = aReg[j].z;                                                                                        \
      As[buf][(innerColA * 4 + 3) * K11_BM + innerRowA + o] = aReg[j].w;                                                                                        \
    }                                                                                                                                                          \
    j = 0;                                                                                                                                                     \
    for (uint o = 0; o + rowStrideB <= K11_BK; o += rowStrideB, ++j) reinterpret_cast<float4*>(&Bs[buf][(innerRowB + o) * K11_BN + innerColB * 4])[0] = bReg[j]; \
  }
#define K11_LOAD_GLOBAL(Aptr, Bptr)                                                                                                                       \
  {                                                                                                                                                       \
    uint j = 0;                                                                                                                                           \
    for (uint o = 0; o + rowStrideA <= K11_BM; o += rowStrideA, ++j) aReg[j] = reinterpret_cast<float4*>(&(Aptr)[(innerRowA + o) * K + innerColA * 4])[0]; \
    j = 0;                                                                                                                                                \
    for (uint o = 0; o + rowStrideB <= K11_BK; o += rowStrideB, ++j) bReg[j] = reinterpret_cast<float4*>(&(Bptr)[(innerRowB + o) * N + innerColB * 4])[0]; \
  }

  K11_LOAD_GLOBAL(A, B);
  K11_STORE_SHARED(0);
  __syncthreads();

  uint cur = 0;
  for (uint tile = 0; tile < nTiles; ++tile) {
    const bool hasNext = (tile + 1) < nTiles;
    if (hasNext) {
      float* An = A + (tile + 1) * K11_BK;
      float* Bn = B + (tile + 1) * K11_BK * N;
      K11_LOAD_GLOBAL(An, Bn);
    }

    for (uint dotIdx = 0; dotIdx < K11_BK; ++dotIdx) {
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint i = 0; i < K11_TM; i += 4)
          reinterpret_cast<float4*>(&regM[wSubRowIdx * K11_TM + i])[0] = reinterpret_cast<float4*>(&As[cur][(dotIdx * K11_BM) + warpRow * K11_WM + wSubRowIdx * WSUBM + threadRowInWarp * K11_TM + i])[0];
      for (uint wSubColIdx = 0; wSubColIdx < K11_WNITER; ++wSubColIdx)
        for (uint i = 0; i < K11_TN; i += 4)
          reinterpret_cast<float4*>(&regN[wSubColIdx * K11_TN + i])[0] = reinterpret_cast<float4*>(&Bs[cur][(dotIdx * K11_BN) + warpCol * K11_WN + wSubColIdx * WSUBN + threadColInWarp * K11_TN + i])[0];

      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint wSubColIdx = 0; wSubColIdx < K11_WNITER; ++wSubColIdx)
          for (uint resIdxM = 0; resIdxM < K11_TM; ++resIdxM)
            for (uint resIdxN = 0; resIdxN < K11_TN; ++resIdxN)
              threadResults[(wSubRowIdx * K11_TM + resIdxM) * (K11_WNITER * K11_TN) + (wSubColIdx * K11_TN) + resIdxN] += regM[wSubRowIdx * K11_TM + resIdxM] * regN[wSubColIdx * K11_TN + resIdxN];
    }

    if (hasNext) {
      K11_STORE_SHARED(cur ^ 1);
      __syncthreads();
      cur ^= 1;
    }
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < K11_WNITER; ++wSubColIdx) {
      float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < K11_TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < K11_TN; resIdxN += 4) {
          const int i = (wSubRowIdx * K11_TM + resIdxM) * (K11_WNITER * K11_TN) + wSubColIdx * K11_TN + resIdxN;
          float* dst = &C_interim[(threadRowInWarp * K11_TM + resIdxM) * N + threadColInWarp * K11_TN + resIdxN];
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
#undef K11_STORE_SHARED
#undef K11_LOAD_GLOBAL
}

// k9: k8 warptiling + DOUBLE BUFFERING. After k8, compute (63%) is the heavier pipe but the SMs
// still stall on global/L2 latency at each K-tile's load->sync->compute->sync boundary. Here we
// keep two shared buffers and, while computing tile t out of one buffer, issue the GLOBAL loads
// for tile t+1 into REGISTERS (LDG.128 in flight). Their latency overlaps the FMAs; we then write
// the prefetched registers into the other shared buffer with a single barrier per iteration.
// Same conflict-free transposed As layout and config as k8.
#define K9_NA (K9_BM / ((K9_NUM_THREADS * 4) / K9_BK))  // float4s of A prefetched per thread
#define K9_NB (K9_BK / (K9_NUM_THREADS / (K9_BN / 4)))  // float4s of B prefetched per thread
__global__ void __launch_bounds__(K9_NUM_THREADS) sgemm_9_doublebuffer(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (K9_BN / K9_WN);
  const uint warpRow = warpIdx / (K9_BN / K9_WN);

  constexpr uint WMITER = (K9_WM * K9_WN) / (WARPSIZE * K9_TM * K9_TN * K9_WNITER);
  constexpr uint WSUBM = K9_WM / WMITER;
  constexpr uint WSUBN = K9_WN / K9_WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / K9_TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / K9_TN);

  __shared__ float As[2][K9_BK * K9_BM];  // transposed: As[buf][k * BM + m]
  __shared__ float Bs[2][K9_BK * K9_BN];

  A += cRow * K9_BM * K;
  B += cCol * K9_BN;
  C += (cRow * K9_BM + warpRow * K9_WM) * N + cCol * K9_BN + warpCol * K9_WN;

  const uint innerRowA = threadIdx.x / (K9_BK / 4);
  const uint innerColA = threadIdx.x % (K9_BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / K9_BK;
  const uint innerRowB = threadIdx.x / (K9_BN / 4);
  const uint innerColB = threadIdx.x % (K9_BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (K9_BN / 4);

  float threadResults[WMITER * K9_TM * K9_WNITER * K9_TN] = {0.0};
  float regM[WMITER * K9_TM] = {0.0};
  float regN[K9_WNITER * K9_TN] = {0.0};

  // register staging for the prefetched (next) tile
  float4 aReg[K9_NA];
  float4 bReg[K9_NB];

  const uint nTiles = K / K9_BK;

  // --- prologue: stage tile 0 into registers, then into As[0]/Bs[0] ---
  {
    uint j = 0;
    for (uint offset = 0; offset + rowStrideA <= K9_BM; offset += rowStrideA, ++j) aReg[j] = reinterpret_cast<float4*>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
    j = 0;
    for (uint offset = 0; offset + rowStrideB <= K9_BK; offset += rowStrideB, ++j) bReg[j] = reinterpret_cast<float4*>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
  {
    uint j = 0;
    for (uint offset = 0; offset + rowStrideA <= K9_BM; offset += rowStrideA, ++j) {
      As[0][(innerColA * 4 + 0) * K9_BM + innerRowA + offset] = aReg[j].x;
      As[0][(innerColA * 4 + 1) * K9_BM + innerRowA + offset] = aReg[j].y;
      As[0][(innerColA * 4 + 2) * K9_BM + innerRowA + offset] = aReg[j].z;
      As[0][(innerColA * 4 + 3) * K9_BM + innerRowA + offset] = aReg[j].w;
    }
    j = 0;
    for (uint offset = 0; offset + rowStrideB <= K9_BK; offset += rowStrideB, ++j) reinterpret_cast<float4*>(&Bs[0][(innerRowB + offset) * K9_BN + innerColB * 4])[0] = bReg[j];
  }
  __syncthreads();

  uint cur = 0;
  for (uint tile = 0; tile < nTiles; ++tile) {
    const bool hasNext = (tile + 1) < nTiles;

    // issue global loads for tile+1 into registers NOW (overlap with compute below)
    if (hasNext) {
      float* An = A + (tile + 1) * K9_BK;
      float* Bn = B + (tile + 1) * K9_BK * N;
      uint j = 0;
      for (uint offset = 0; offset + rowStrideA <= K9_BM; offset += rowStrideA, ++j) aReg[j] = reinterpret_cast<float4*>(&An[(innerRowA + offset) * K + innerColA * 4])[0];
      j = 0;
      for (uint offset = 0; offset + rowStrideB <= K9_BK; offset += rowStrideB, ++j) bReg[j] = reinterpret_cast<float4*>(&Bn[(innerRowB + offset) * N + innerColB * 4])[0];
    }

    // compute from the current shared buffer
    for (uint dotIdx = 0; dotIdx < K9_BK; ++dotIdx) {
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint i = 0; i < K9_TM; ++i) regM[wSubRowIdx * K9_TM + i] = As[cur][(dotIdx * K9_BM) + warpRow * K9_WM + wSubRowIdx * WSUBM + threadRowInWarp * K9_TM + i];
      for (uint wSubColIdx = 0; wSubColIdx < K9_WNITER; ++wSubColIdx)
        for (uint i = 0; i < K9_TN; ++i) regN[wSubColIdx * K9_TN + i] = Bs[cur][(dotIdx * K9_BN) + warpCol * K9_WN + wSubColIdx * WSUBN + threadColInWarp * K9_TN + i];

      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for (uint wSubColIdx = 0; wSubColIdx < K9_WNITER; ++wSubColIdx)
          for (uint resIdxM = 0; resIdxM < K9_TM; ++resIdxM)
            for (uint resIdxN = 0; resIdxN < K9_TN; ++resIdxN)
              threadResults[(wSubRowIdx * K9_TM + resIdxM) * (K9_WNITER * K9_TN) + (wSubColIdx * K9_TN) + resIdxN] += regM[wSubRowIdx * K9_TM + resIdxM] * regN[wSubColIdx * K9_TN + resIdxN];
    }

    // commit the prefetched registers into the other buffer, then flip
    if (hasNext) {
      uint j = 0;
      for (uint offset = 0; offset + rowStrideA <= K9_BM; offset += rowStrideA, ++j) {
        As[cur ^ 1][(innerColA * 4 + 0) * K9_BM + innerRowA + offset] = aReg[j].x;
        As[cur ^ 1][(innerColA * 4 + 1) * K9_BM + innerRowA + offset] = aReg[j].y;
        As[cur ^ 1][(innerColA * 4 + 2) * K9_BM + innerRowA + offset] = aReg[j].z;
        As[cur ^ 1][(innerColA * 4 + 3) * K9_BM + innerRowA + offset] = aReg[j].w;
      }
      j = 0;
      for (uint offset = 0; offset + rowStrideB <= K9_BK; offset += rowStrideB, ++j) reinterpret_cast<float4*>(&Bs[cur ^ 1][(innerRowB + offset) * K9_BN + innerColB * 4])[0] = bReg[j];
      __syncthreads();
      cur ^= 1;
    }
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < K9_WNITER; ++wSubColIdx) {
      float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < K9_TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < K9_TN; resIdxN += 4) {
          const int i = (wSubRowIdx * K9_TM + resIdxM) * (K9_WNITER * K9_TN) + wSubColIdx * K9_TN + resIdxN;
          float* dst = &C_interim[(threadRowInWarp * K9_TM + resIdxM) * N + threadColInWarp * K9_TN + resIdxN];
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

// k7: same tiling as k6, but the inner-loop register caches and the C store-out
// are vectorized (float4 / LDS.128 + STG.128). k6's profile showed 33.5M shared-load
// bank conflicts (scalar LDS.32 over a stride-8 column map) and C stores using only
// 4 of 32 sectors. As is stored transposed [BK][BM] so a column slice across BM is
// contiguous; Bs[dot][col..] is contiguous in BN — both are float4-loadable.
// Assumes BM,BN divisible by 4 and (for the vectorized store) BN divides N, BM divides M.
__global__ void sgemm_7_vectorized_smem(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C) {
  __shared__ float As[BK][BM];  // transposed: As[k][m]
  __shared__ float Bs[BK][BN];

  float product[TM][TM] = {0.0};
  float regM[TM];
  float regN[TM];

  const uint totalThreads = (BM * BN) / (TM * TM);

  // --- load-phase thread maps (float4 granularity), identical to k6 ---
  const uint subtitleAHeight = totalThreads / BK;  // rows of (transposed) As filled per pass
  const uint subtitleBHeight = totalThreads / BN;

  const uint subtitleARow = (threadIdx.x * 4) / BK;
  const uint subtitleACol = (threadIdx.x * 4) % BK;
  const uint subtitleBRow = (threadIdx.x * 4) / BN;
  const uint subtitleBCol = (threadIdx.x * 4) % BN;

  // --- compute-phase output tile (one 8x8 tile per thread) ---
  const uint threadCol = threadIdx.x % (BN / TM);  // 0..15
  const uint threadRow = threadIdx.x / (BN / TM);  // 0..15
  const uint computeRow = threadRow * TM;          // row base inside the 128x128 tile
  const uint computeCol = threadCol * TM;          // col base inside the 128x128 tile

  const uint CrowOffset = blockIdx.y * BM + computeRow;
  const uint CcolOffset = blockIdx.x * BN + computeCol;

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const uint fromARowOffset = blockIdx.y * BM;
    const uint fromBColOffset = blockIdx.x * BN;

    // load A tile (transposed into As) with float4 GMEM reads
    for (uint loadOffset = 0; loadOffset < BM / 4; loadOffset += subtitleAHeight) {
      const uint toAsRow = subtitleARow + loadOffset;  // index along BM/4
      const uint toAsCol = subtitleACol;               // index along BK
      float4 tmp = reinterpret_cast<float4*>(&A[(bkIdx + toAsCol) + (fromARowOffset + toAsRow) * K])[0];
      As[toAsCol + 0][toAsRow] = tmp.x;
      As[toAsCol + 1][toAsRow] = tmp.y;
      As[toAsCol + 2][toAsRow] = tmp.z;
      As[toAsCol + 3][toAsRow] = tmp.w;
    }

    // load B tile with float4 GMEM + SMEM writes
    for (uint loadOffset = 0; loadOffset < BK / 4; loadOffset += subtitleBHeight) {
      const uint toBsRow = subtitleBRow + loadOffset;
      const uint toBsCol = subtitleBCol;
      reinterpret_cast<float4*>(&Bs[toBsRow][toBsCol])[0] = reinterpret_cast<float4*>(&B[fromBColOffset + toBsCol + (bkIdx + toBsRow) * N])[0];
    }

    __syncthreads();

    for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
      // float4 register-cache loads (LDS.128) — conflict-free, 4x fewer instructions
      reinterpret_cast<float4*>(&regM[0])[0] = reinterpret_cast<float4*>(&As[dotIdx][computeRow + 0])[0];
      reinterpret_cast<float4*>(&regM[4])[0] = reinterpret_cast<float4*>(&As[dotIdx][computeRow + 4])[0];
      reinterpret_cast<float4*>(&regN[0])[0] = reinterpret_cast<float4*>(&Bs[dotIdx][computeCol + 0])[0];
      reinterpret_cast<float4*>(&regN[4])[0] = reinterpret_cast<float4*>(&Bs[dotIdx][computeCol + 4])[0];

      for (uint tm = 0; tm < TM; tm++) {
        for (uint tn = 0; tn < TM; tn++) {
          product[tm][tn] += regM[tm] * regN[tn];
        }
      }
    }

    __syncthreads();
  }

  // vectorized C store-out (STG.128)
  for (uint tm = 0; tm < TM; tm++) {
    const uint row = CrowOffset + tm;
    if (row >= M) continue;
    for (uint tn = 0; tn < TM; tn += 4) {
      const uint col = CcolOffset + tn;
      if (col + 3 >= N) {  // ragged tail: fall back to scalar
        for (uint t = tn; t < TM && CcolOffset + t < N; t++) {
          const uint idx = (CcolOffset + t) + row * N;
          C[idx] = beta == 0 ? alpha * product[tm][t] : alpha * product[tm][t] + beta * C[idx];
        }
        continue;
      }
      const uint idx = col + row * N;
      float4 out;
      if (beta == 0) {
        out.x = alpha * product[tm][tn + 0];
        out.y = alpha * product[tm][tn + 1];
        out.z = alpha * product[tm][tn + 2];
        out.w = alpha * product[tm][tn + 3];
      } else {
        float4 prev = reinterpret_cast<float4*>(&C[idx])[0];
        out.x = alpha * product[tm][tn + 0] + beta * prev.x;
        out.y = alpha * product[tm][tn + 1] + beta * prev.y;
        out.z = alpha * product[tm][tn + 2] + beta * prev.z;
        out.w = alpha * product[tm][tn + 3] + beta * prev.w;
      }
      reinterpret_cast<float4*>(&C[idx])[0] = out;
    }
  }
}

__global__ void sgemm_5_register_2dtiling(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
  // total thread count needs to be divisible by BK
  if (blockDim.x % BK != 0) {
    return;
  }

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  float product[TM][TM] = {0.0};

  const uint subtitleAHeight = blockDim.x / BK;
  const uint subtitleAWidth = BK;

  const uint subtitleBHeight = BK;
  const uint subtitleBWidth = blockDim.x / BK;

  // idx = 1, row = 0, col = 1
  const uint subtitleARow = threadIdx.x / subtitleAWidth;
  const uint subtitleACol = threadIdx.x % subtitleAWidth;

  // width = 2
  // idx = 0, row = 0, col = 0
  // idx = 1, row = 0, col = 1 <--- CHOSEN
  // idx = 2, row = 1, col = 0
  // idx = 3, row = 1, col = 1
  // idx = 4, row = 2, col = 0
  const uint subtitleBRow = threadIdx.x / subtitleBWidth;
  const uint subtitleBCol = threadIdx.x % subtitleBWidth;

  const uint blockRowOffset = blockIdx.y * BM;
  const uint blockColOffset = blockIdx.x * BN;

  const uint computeRowInTile = TM * ((threadIdx.x * TM) / BN);
  const uint computeColInTile = (threadIdx.x * TM) % BN;

  /*
        const uint threadCol = threadIdx.x % (BN / TM);   // 0..15 with stride 1
        const uint threadRow = threadIdx.x / (BN / TM);   // 0..15

        const uint computeRowInTile = threadRow * TM;
        const uint computeColInTile = threadCol * TM;
  */

  const uint CrowOffset = blockRowOffset + computeRowInTile;
  const uint CcolOffset = blockColOffset + computeColInTile;

  for (uint tile = 0; tile < ceil((float)K / BK); tile++) {
    // 0
    const uint fromARowOffset = blockIdx.y * BM;
    // 1 * 3 = 3
    const uint fromAColOffset = tile * BK;

    // likely wrong; needs to take into consideration the blockIdx
    const uint fromBRowOffset = tile * BK;
    const uint fromBColOffset = blockIdx.x * BN;

    for (uint subtitle = 0; subtitle < ceil((float)BN / subtitleAHeight); subtitle++) {
      // 0 + 1 * 2 = 2
      const uint toAsRow = subtitleARow + subtitle * subtitleAHeight;
      // 1
      const uint toAsCol = subtitleACol;
      // 0 + 2 = 2
      const uint fromARow = fromARowOffset + toAsRow;
      // 3 + 1 = 4
      const uint fromACol = fromAColOffset + toAsCol;

      if (fromARow >= M || fromACol >= K) As[toAsRow][toAsCol] = 0.0;
      else As[toAsRow][toAsCol] = A[fromACol + fromARow * K];
    }

    for (uint subtitle = 0; subtitle < ceil((float)BN / subtitleBWidth); subtitle++) {
      // subtitle = 1
      // toBsRow = 0
      const uint toBsRow = subtitleBRow;

      // toBsCol = 1 + 1 * 2 = 3
      const uint toBsCol = subtitleBCol + subtitle * subtitleBWidth;

      // fromBRow = 0 + 0 = 0
      const uint fromBRow = fromBRowOffset + toBsRow;

      // fromBCol =
      const uint fromBCol = fromBColOffset + toBsCol;

      if (fromBRow >= K || fromBCol >= N) Bs[toBsRow][toBsCol] = 0.0;
      else Bs[toBsRow][toBsCol] = B[fromBCol + fromBRow * N];
    }

    __syncthreads();

    float Atmp[TM] = {0.0};
    float Btmp[TM] = {0.0};
    for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
      for (uint tm = 0; tm < TM; tm++) {
        // depends on: TM, total threads, BN, threadIdx
        const uint fromAsRow = (TM * ((threadIdx.x * TM) / BN)) + tm;
        // idx = 0, 8 * (0*8 / 8) + 0 = 0
        // idx = 1, 8 * (1*8 / 8) + 0 = 8
        const uint fromAsCol = dotIdx;
        Atmp[tm] = As[fromAsRow][fromAsCol];
      }

      for (uint tn = 0; tn < TM; tn++) {
        const uint fromBsRow = dotIdx;
        const uint fromBsCol = ((threadIdx.x * TM) % BN) + tn;
        Btmp[tn] = Bs[fromBsRow][fromBsCol];
      }

      for (uint tm = 0; tm < TM; tm++) {
        for (uint tn = 0; tn < TM; tn++) {
          product[tm][tn] += Atmp[tm] * Btmp[tn];
        }
      }
    }

    __syncthreads();
  }

  for (uint tm = 0; tm < TM; tm++) {
    for (uint tn = 0; tn < TM; tn++) {
      const uint row = CrowOffset + tm;
      const uint col = CcolOffset + tn;

      if (col >= N || row >= M) continue;

      const uint Cidx = col + row * N;
      if (beta == 0) C[Cidx] = alpha * product[tm][tn];
      else C[Cidx] = alpha * product[tm][tn] + beta * C[Cidx];
    }
  }
}

__global__ void sgemm_5_register_2dtiling_v2(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
  // total thread count needs to be divisible by BK
  if (blockDim.x % BK != 0 || blockDim.x % BN != 0) {
    return;
  }

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  float product[TM][TM] = {0.0};

  const uint totalThreads = (BM * BN) / (TM * TM);

  const uint subtitleAHeight = totalThreads / BK;
  const uint subtitleAWidth = BK;

  const uint subtitleBHeight = totalThreads / BN;  // rows loaded per pass
  const uint subtitleBWidth = BN;                  // full row width

  const uint subtitleARow = threadIdx.x / subtitleAWidth;
  const uint subtitleACol = threadIdx.x % subtitleAWidth;

  const uint subtitleBRow = threadIdx.x / subtitleBWidth;
  const uint subtitleBCol = threadIdx.x % subtitleBWidth;

  const uint blockRowOffset = blockIdx.y * BM;
  const uint blockColOffset = blockIdx.x * BN;

  const uint computeRowInTile = TM * ((threadIdx.x * TM) / BN);
  const uint computeColInTile = (threadIdx.x * TM) % BN;

  const uint CrowOffset = blockRowOffset + computeRowInTile;
  const uint CcolOffset = blockColOffset + computeColInTile;

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const uint fromARowOffset = blockIdx.y * BM;
    const uint fromAColOffset = bkIdx;

    const uint fromBRowOffset = bkIdx;
    const uint fromBColOffset = blockIdx.x * BN;

    // LOADING PHASE: loading into As
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += subtitleAHeight) {
      const uint toAsRow = subtitleARow + loadOffset;
      const uint toAsCol = subtitleACol;

      const uint fromARow = fromARowOffset + toAsRow;
      const uint fromACol = fromAColOffset + toAsCol;

      if (fromARow >= M || fromACol >= K) As[toAsRow][toAsCol] = 0.0;
      else As[toAsRow][toAsCol] = A[fromACol + fromARow * K];
    }

    // LOADING PHASE: loading into Bs
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += subtitleBHeight) {
      const uint toBsRow = subtitleBRow + loadOffset;
      const uint toBsCol = subtitleBCol;

      const uint fromBRow = fromBRowOffset + toBsRow;
      const uint fromBCol = fromBColOffset + toBsCol;

      if (fromBRow >= K || fromBCol >= N) Bs[toBsRow][toBsCol] = 0.0;
      else Bs[toBsRow][toBsCol] = B[fromBCol + fromBRow * N];
    }

    __syncthreads();

    // COMPUTE PHASE START
    float Atmp[TM] = {0.0};
    float Btmp[TM] = {0.0};
    const int threadCol = threadIdx.x % (BN / TM);
    const int threadRow = threadIdx.x / (BN / TM);
    for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
      for (uint tm = 0; tm < TM; tm++) {
        const uint fromAsRow = (TM * ((threadIdx.x * TM) / BN)) + tm;
        const uint fromAsCol = dotIdx;
        Atmp[tm] = As[fromAsRow][fromAsCol];
      }

      for (uint tn = 0; tn < TM; tn++) {
        const uint fromBsRow = dotIdx;
        const uint fromBsCol = ((threadIdx.x * TM) % BN) + tn;
        Btmp[tn] = Bs[fromBsRow][fromBsCol];
      }

      for (uint tm = 0; tm < TM; tm++) {
        for (uint tn = 0; tn < TM; tn++) {
          product[tm][tn] += Atmp[tm] * Btmp[tn];
        }
      }
    }

    __syncthreads();
  }

  for (uint tm = 0; tm < TM; tm++) {
    for (uint tn = 0; tn < TM; tn++) {
      const uint row = CrowOffset + tm;
      const uint col = CcolOffset + tn;

      if (col >= N || row >= M) continue;

      const uint Cidx = col + row * N;
      if (beta == 0) C[Cidx] = alpha * product[tm][tn];
      else C[Cidx] = alpha * product[tm][tn] + beta * C[Cidx];
    }
  }
}

#define TM_4 8
#define BK_4 8
#define BM_4 64
#define BN_4 64
// 2141198334 fp32 loads
__global__ void sgemm_4_register_tiling(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
  const uint Ccol = (blockIdx.x * BN_4) + (threadIdx.x % BN_4);
  const uint Crow = (blockIdx.y * BM_4) + ((threadIdx.x / BN_4) * TM_4);

  __shared__ float As[BM_4][BK_4];
  __shared__ float Bs[BK_4][BN_4];

  float product[TM_4] = {0.0};

  // The A row and B col this thread will be loading from GMEM into SREM
  // It's the same value throughout this thread lifecycle
  const uint Arow = (blockIdx.y * BM_4) + (threadIdx.x / BK_4);
  const uint Bcol = (blockIdx.x * BN_4) + threadIdx.x % BN_4;

  const uint toAsRow = threadIdx.x / BK_4;
  const uint toAsCol = threadIdx.x % BK_4;

  const uint toBsRow = threadIdx.x / BN_4;
  const uint toBsCol = threadIdx.x % BN_4;

  for (uint tile = 0; tile < ceil((float)K / BK_4); tile++) {
    const uint Acol = (threadIdx.x % BK_4) + tile * BK_4;

    if (Acol < K && Arow < M) As[toAsRow][toAsCol] = A[Acol + Arow * K];
    else As[toAsRow][toAsCol] = 0.0;

    const uint Brow = (threadIdx.x / BN_4) + tile * BK_4;

    if (Brow < K && Bcol < N) Bs[toBsRow][toBsCol] = B[Bcol + Brow * N];
    else Bs[toBsRow][toBsCol] = 0.0;

    __syncthreads();
    for (uint k = 0; k < BK_4; k++) {
      const uint fromAsRow = (threadIdx.x / BN_4) * TM_4;
      const uint fromAsCol = k;

      const uint fromBsRow = k;
      const uint fromBsCol = (threadIdx.x % BN_4);

      float Btmp = Bs[fromBsRow][fromBsCol];
      for (uint tm = 0; tm < TM_4; tm++) {
        product[tm] += As[fromAsRow + tm][fromAsCol] * Btmp;
      }
    }
    __syncthreads();
  }

  for (uint tm = 0; tm < TM_4; tm++) {
    const uint col = Ccol;
    const uint row = Crow + tm;

    if (col >= N || row >= M) continue;

    const uint Cidx = col + row * N;
    if (beta == 0) C[Cidx] = alpha * product[tm];
    else C[Cidx] = alpha * product[tm] + beta * C[Cidx];
  }
}

#define TILE_WIDTH 16
// 8564793336 fp32 loads
__global__ void sgemm_tiled(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
  const uint col = threadIdx.x + (blockIdx.x * blockDim.x);
  const uint row = threadIdx.y + (blockIdx.y * blockDim.y);

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  float product = 0.0;

  for (uint tile = 0; tile < ceil((float)K / TILE_WIDTH); tile++) {
    const uint Acol = threadIdx.x + tile * TILE_WIDTH;

    if (Acol < K && row < M) As[threadIdx.y][threadIdx.x] = A[Acol + row * K];
    else As[threadIdx.y][threadIdx.x] = 0.0;

    const uint Brow = threadIdx.y + tile * TILE_WIDTH;

    if (Brow < K && col < N) Bs[threadIdx.y][threadIdx.x] = B[col + Brow * N];
    else Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (uint t = 0; t < TILE_WIDTH; t++) {
      product += As[threadIdx.y][t] * Bs[t][threadIdx.x];
    }
    __syncthreads();
  }

  if (col >= N || row >= M) return;

  const uint Cidx = col + row * N;
  if (beta == 0) C[Cidx] = alpha * product;
  else C[Cidx] = alpha * product + beta * C[Cidx];
}

// A is MxK
// B is KxN
// C, then, is MxN

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

void matmul(float* C_h, float* A_h, int A_m, int A_n, float* B_h, int B_m, int B_n, MatmulAlgorithm algo, cublasHandle_t handle) {
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

    // Warm-up
    sgemm_naive<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(naive);
    sgemm_naive<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(naive);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == Cublas) {
    cublasCreate(&handle);
    float alpha = 1;
    float beta = 0;

    // Warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_n, A_m, A_n, &alpha, B_d, B_n, A_d, A_n, &beta, C_d, B_n);

    cudaDeviceSynchronize();
    WALL_START(cublas);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_n, A_m, A_n,  // N, M, K
                &alpha, B_d, B_n,                                 // ldb in row-major = N
                A_d, A_n,                                         // lda in row-major = K
                &beta, C_d, B_n);
    cudaDeviceSynchronize();
    WALL_END(cublas);
  }

  else if (algo == Tiled) {
    dim3 blockDim = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim = dim3(ceil((float)B_n / TILE_WIDTH), ceil((float)A_m / TILE_WIDTH), 1);

    // Warm-up
    sgemm_tiled<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(tiled);
    sgemm_tiled<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(tiled);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == TiledWith1DRegisterTiling) {
    dim3 blockDim((BM_4 * BN_4) / TM_4);  // 512 threads, 1D
    dim3 gridDim(ceil((float)B_n / BN_4), ceil((float)A_m / BM_4));

    // Warm-up
    sgemm_4_register_tiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(tiled_1d_register_tiling);
    sgemm_4_register_tiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(tiled_1d_register_tiling);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == TiledWith2DRegisterTiling) {
    dim3 blockDim((BM * BN) / (BK * BK));
    dim3 gridDim(ceil((float)B_n / BN), ceil((float)A_m / BM));

    // Warm-up
    sgemm_5_register_2dtiling_v2<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(tiled_2d_register_tiling);
    sgemm_5_register_2dtiling_v2<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(tiled_2d_register_tiling);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == TiledWith2DRegisterTilingAsVectorized) {
    dim3 blockDim((K6_BM * K6_BN) / (K6_TM * K6_TM));
    dim3 gridDim(ceil((float)B_n / K6_BN), ceil((float)A_m / K6_BM));

    // Warm-up
    sgemm_6_register_2dtiling_vectorized_As<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(k6_vectorized_As_tiled_2d_register_tiling);
    sgemm_6_register_2dtiling_vectorized_As<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(k6_vectorized_As_tiled_2d_register_tiling);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == K7VectorizedSmem) {
    dim3 blockDim((BM * BN) / (BK * BK));
    dim3 gridDim(ceil((float)B_n / BN), ceil((float)A_m / BM));

    // Warm-up
    sgemm_7_vectorized_smem<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(k7_vectorized_smem);
    sgemm_7_vectorized_smem<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(k7_vectorized_smem);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == K8Warptiling) {
    dim3 blockDim(K8_NUM_THREADS);
    dim3 gridDim(ceil((float)B_n / K8_BN), ceil((float)A_m / K8_BM));

    // Warm-up
    sgemm_8_warptiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(k8_warptiling);
    sgemm_8_warptiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(k8_warptiling);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == K9DoubleBuffer) {
    dim3 blockDim(K9_NUM_THREADS);
    dim3 gridDim(ceil((float)B_n / K9_BN), ceil((float)A_m / K9_BM));

    // Warm-up
    sgemm_9_doublebuffer<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(k9_doublebuffer);
    sgemm_9_doublebuffer<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(k9_doublebuffer);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == K10VecSmem) {
    dim3 blockDim(K10_NUM_THREADS);
    dim3 gridDim(ceil((float)B_n / K10_BN), ceil((float)A_m / K10_BM));
    sgemm_10_vec_smem<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_START(k10_vec_smem);
    sgemm_10_vec_smem<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(k10_vec_smem);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == K11DbVec) {
    dim3 blockDim(K11_NUM_THREADS);
    dim3 gridDim(ceil((float)B_n / K11_BN), ceil((float)A_m / K11_BM));
    sgemm_11_db_vec<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_START(k11_db_vec);
    sgemm_11_db_vec<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(k11_db_vec);
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

  cublasHandle_t handle;
  cublasCreate(&handle);

  matmul(C, A, A_m, A_n, B, A_n, B_n, Cublas, handle);

  float* D = new float[A_m * B_n];
  matmul(D, A, A_m, A_n, B, A_n, B_n, Naive, handle);

  compareResults(C, D, A_m * B_n, "cublas v naive");

  float* E = new float[A_m * B_n];
  matmul(E, A, A_m, A_n, B, A_n, B_n, Tiled, handle);

  compareResults(C, E, A_m * B_n, "cublas v tiled");

  float* F = new float[A_m * B_n];
  matmul(F, A, A_m, A_n, B, A_n, B_n, TiledWith1DRegisterTiling, handle);

  compareResults(C, F, A_m * B_n, "cublas v tiled with 1D register tiling");

  float* G = new float[A_m * B_n];
  matmul(G, A, A_m, A_n, B, A_n, B_n, TiledWith2DRegisterTiling, handle);

  compareResults(C, G, A_m * B_n, "cublas v tiled with 2D register tiling");

  float* H = new float[A_m * B_n];
  matmul(H, A, A_m, A_n, B, A_n, B_n, TiledWith2DRegisterTilingAsVectorized, handle);

  compareResults(C, H, A_m * B_n, "cublas v K6 vectorized tiled with 2D register tiling");

  float* I = new float[A_m * B_n];
  matmul(I, A, A_m, A_n, B, A_n, B_n, K7VectorizedSmem, handle);

  compareResults(C, I, A_m * B_n, "cublas v K7 vectorized smem (float4 inner loop + store)");

  float* J = new float[A_m * B_n];
  matmul(J, A, A_m, A_n, B, A_n, B_n, K8Warptiling, handle);

  compareResults(C, J, A_m * B_n, "cublas v K8 warptiling");

  float* L = new float[A_m * B_n];
  matmul(L, A, A_m, A_n, B, A_n, B_n, K9DoubleBuffer, handle);

  compareResults(C, L, A_m * B_n, "cublas v K9 double-buffered warptiling");

  float* P = new float[A_m * B_n];
  matmul(P, A, A_m, A_n, B, A_n, B_n, K10VecSmem, handle);

  compareResults(C, P, A_m * B_n, "cublas v K10 warptiling + vectorized smem loads");

  float* Q = new float[A_m * B_n];
  matmul(Q, A, A_m, A_n, B, A_n, B_n, K11DbVec, handle);

  compareResults(C, Q, A_m * B_n, "cublas v K11 double-buffered + vectorized smem loads");

  return 0;
}
