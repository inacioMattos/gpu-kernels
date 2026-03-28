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
  WarptilingSingleIter,
  Warptiling,
  WarptilingVectorizedShared,
};

#define TM 8
#define BK 8
#define BM 128
#define BN 128

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

#ifndef K10_MINBLOCKS
#define K10_MINBLOCKS 1
#endif
#define K10_ASTRIDE (K10_BM + 4)

// Requires M/N multiples of 128 and K a multiple of 16.
__global__ void __launch_bounds__(K10_NUM_THREADS, K10_MINBLOCKS)
    sgemm_10_vec_smem(int M, int N, int K, float alpha, float beta, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / 32;
  const uint warpCol = warpIdx % (K10_BN / K10_WN);
  const uint warpRow = warpIdx / (K10_BN / K10_WN);

  constexpr uint WMITER = (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  constexpr uint WSUBM = K10_WM / WMITER;
  constexpr uint WSUBN = K10_WN / K10_WNITER;

  const uint threadIdxInWarp = threadIdx.x % 32;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / K10_TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / K10_TN);

  __shared__ float As[K10_BK * K10_ASTRIDE];
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
    for (uint offset = 0; offset + rowStrideB <= K10_BK; offset += rowStrideB) {
      reinterpret_cast<float4*>(&Bs[(innerRowB + offset) * K10_BN + innerColB * 4])[0] = reinterpret_cast<const float4*>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

#pragma unroll
    for (uint dotIdx = 0; dotIdx < K10_BK; ++dotIdx) {
#pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < K10_TM; i += 4) {
          reinterpret_cast<float4*>(&regM[wSubRowIdx * K10_TM + i])[0] =
              reinterpret_cast<float4*>(&As[(dotIdx * K10_ASTRIDE) + warpRow * K10_WM + wSubRowIdx * WSUBM + threadRowInWarp * K10_TM + i])[0];
        }
      }
#pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < K10_WNITER; ++wSubColIdx) {
        for (uint i = 0; i < K10_TN; i += 4) {
          reinterpret_cast<float4*>(&regN[wSubColIdx * K10_TN + i])[0] = reinterpret_cast<float4*>(&Bs[(dotIdx * K10_BN) + warpCol * K10_WN + wSubColIdx * WSUBN + threadColInWarp * K10_TN + i])[0];
        }
      }

#pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < K10_WNITER; ++wSubColIdx)
#pragma unroll
          for (uint resIdxM = 0; resIdxM < K10_TM; ++resIdxM)
#pragma unroll
            for (uint resIdxN = 0; resIdxN < K10_TN; ++resIdxN) {
              threadResults[(wSubRowIdx * K10_TM + resIdxM) * (K10_WNITER * K10_TN) + (wSubColIdx * K10_TN) + resIdxN] += regM[wSubRowIdx * K10_TM + resIdxM] * regN[wSubColIdx * K10_TN + resIdxN];
            }
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

#ifndef K8_BM
#define K8_BM 64
#endif
#ifndef K8_BN
#define K8_BN 128
#endif
#ifndef K8_BK
#define K8_BK 32
#endif
#ifndef K8_TM
#define K8_TM 8
#endif
#ifndef K8_TN
#define K8_TN 4
#endif
#define K8_WMX 2
#define K8_WITER 2

__global__ void sgemm_8_warptiling(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C) {
  // Compile-time validity for the (BM,BN,BK,TM) config. The compute phase tiles the BMxBN output
  // in TMxTM blocks (needs BM,BN divisible by TM); the float4 loaders need BK,BN divisible by 4.
  if (K8_BK % 4 != 0 || K8_BN % 4 != 0 || K8_BM % K8_TM != 0 || K8_BN % K8_TM != 0) {
    return;
  }

  if (32 % K8_WMX != 0) {
    return;
  }

  // WARPTILING VARS
  const uint K8_WNX = 32 / K8_WMX;  // 8

  const uint WM = K8_WMX * K8_TM;              // 32
  const uint WN_SINGLE_ITER = K8_WNX * K8_TN;  // 32
  const uint WN = WN_SINGLE_ITER * K8_WITER;   // 64

  const uint warpIdx = threadIdx.x / 32;  // 13
  const uint lane = threadIdx.x % 32;     // 8

  const uint warptileColOffset = (warpIdx * WN) % K8_BN;         // 64
  const uint warpTileRowOffset = WM * ((warpIdx * WN) / K8_BN);  // 192

  const uint colInWarptile = (lane * K8_TN) % WN_SINGLE_ITER;            // 32 (wrong)
  const uint rowInWarptile = K8_TM * ((lane * K8_TN) / WN_SINGLE_ITER);  // 0 (wrong)

  // WARPTILING END

  __shared__ float As[K8_BK][K8_BM];
  __shared__ float Bs[K8_BK][K8_BN];

  float product[K8_TM][K8_TN * K8_WITER] = {0.0};

  const uint totalThreads = (K8_BM * K8_BN) / (K8_TM * K8_TN * K8_WITER);

  const uint blockRowOffset = blockIdx.y * K8_BM;
  const uint blockColOffset = blockIdx.x * K8_BN;

  const uint computeRowInTile = K8_TM * ((threadIdx.x * K8_TN) / K8_BN);
  const uint computeColInTile = (threadIdx.x * K8_TN) % K8_BN;

  const uint CrowOffset = blockRowOffset + warpTileRowOffset + rowInWarptile;
  const uint CcolOffset = blockColOffset + warptileColOffset + colInWarptile;

  // float4 groups per row in each tile. General grid-stride loaders below fill the SAME shared
  // contents as the original hand-rolled loops, but cover any thread count / tile shape.
  const uint a4PerRow = K8_BK / 4;  // along K  (As stored transposed [BK][BM])
  const uint b4PerRow = K8_BN / 4;  // along N  (Bs stored [BK][BN])

  for (uint bkIdx = 0; bkIdx < K; bkIdx += K8_BK) {
    // LOADING PHASE: A tile -> As (transposed), float4 along K
    for (uint t = threadIdx.x; t < K8_BM * a4PerRow; t += totalThreads) {
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
    for (uint t = threadIdx.x; t < K8_BK * b4PerRow; t += totalThreads) {
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
    float Atmp[K8_TM] = {0.0};
#pragma unroll
    for (uint dotIdx = 0; dotIdx < K8_BK; dotIdx++) {
#pragma unroll
      for (uint tm = 0; tm < K8_TM; tm += 4) {
        const uint fromAsRow = warpTileRowOffset + rowInWarptile + tm;
        const uint fromAsCol = dotIdx;

        float4 tmp = reinterpret_cast<float4*>(&As[fromAsCol][fromAsRow])[0];
        Atmp[tm + 0] = tmp.x;
        Atmp[tm + 1] = tmp.y;
        Atmp[tm + 2] = tmp.z;
        Atmp[tm + 3] = tmp.w;
      }

#pragma unroll
      for (uint warpIter = 0; warpIter < K8_WITER; warpIter++) {
        float Btmp[K8_TN] = {0.0};

#pragma unroll
        for (uint tn = 0; tn < K8_TN; tn += 4) {
          const uint fromBsRow = dotIdx;
          const uint fromBsCol = (warptileColOffset + (warpIter * WN_SINGLE_ITER)) + colInWarptile + tn;

          float4 tmp = reinterpret_cast<float4*>(&Bs[fromBsRow][fromBsCol])[0];
          Btmp[tn + 0] = tmp.x;
          Btmp[tn + 1] = tmp.y;
          Btmp[tn + 2] = tmp.z;
          Btmp[tn + 3] = tmp.w;
        }

#pragma unroll
        for (uint tm = 0; tm < K8_TM; tm++) {
#pragma unroll
          for (uint tn = 0; tn < K8_TN; tn++) {
            product[tm][tn + warpIter * K8_TN] += Atmp[tm] * Btmp[tn];
          }
        }
      }
    }

    __syncthreads();
  }

  for (uint warpIter = 0; warpIter < K8_WITER; warpIter++) {
    for (uint tm = 0; tm < K8_TM; tm++) {
      for (uint tn = 0; tn < K8_TN; tn++) {
        const uint row = CrowOffset + tm;
        const uint col = CcolOffset + tn + (warpIter * WN_SINGLE_ITER);

        if (col >= N || row >= M) continue;

        const uint Cidx = col + row * N;
        if (beta == 0) C[Cidx] = alpha * product[tm][tn + warpIter * K8_TN];
        else C[Cidx] = alpha * product[tm][tn + warpIter * K8_TN] + beta * C[Cidx];
      }
    }
  }
}

// k6 (square 2D register tiling, vectorized GMEM): own overridable config for a fair sweep.
#ifndef K7_BM
#define K7_BM 128
#endif
#ifndef K7_BN
#define K7_BN 128
#endif
#ifndef K7_BK
#define K7_BK 8
#endif
#ifndef K7_TM
#define K7_TM 8
#endif
#ifndef K7_TN
#define K7_TN 4
#endif
#define K7_WMX 4

__global__ void sgemm_7_warptiling_single_iter(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C) {
  // Compile-time validity for the (BM,BN,BK,TM) config. The compute phase tiles the BMxBN output
  // in TMxTM blocks (needs BM,BN divisible by TM); the float4 loaders need BK,BN divisible by 4.
  if (K7_BK % 4 != 0 || K7_BN % 4 != 0 || K7_BM % K7_TM != 0 || K7_BN % K7_TM != 0) {
    return;
  }

  if (32 % K7_WMX != 0) {
    return;
  }

  // WARPTILING VARS
  const uint K7_WNX = 32 / K7_WMX;

  const uint WM = K7_WMX * K7_TM;
  const uint WN = K7_WNX * K7_TN;

  const uint warpIdx = threadIdx.x / 32;
  const uint lane = threadIdx.x % 32;

  const uint warptileColOffset = (warpIdx * WN) % K7_BN;
  const uint warpTileRowOffset = WM * ((warpIdx * WN) / K7_BN);

  const uint colInWarptile = (lane * K7_TN) % WN;
  const uint rowInWarptile = K7_TM * ((lane * K7_TN) / WN);

  // WARPTILING END

  __shared__ float As[K7_BK][K7_BM];
  __shared__ float Bs[K7_BK][K7_BN];

  float product[K7_TM][K7_TN] = {0.0};

  const uint totalThreads = (K7_BM * K7_BN) / (K7_TM * K7_TN);

  const uint blockRowOffset = blockIdx.y * K7_BM;
  const uint blockColOffset = blockIdx.x * K7_BN;

  const uint computeRowInTile = K7_TM * ((threadIdx.x * K7_TN) / K7_BN);
  const uint computeColInTile = (threadIdx.x * K7_TN) % K7_BN;

  const uint CrowOffset = blockRowOffset + warpTileRowOffset + rowInWarptile;
  const uint CcolOffset = blockColOffset + warptileColOffset + colInWarptile;

  // float4 groups per row in each tile. General grid-stride loaders below fill the SAME shared
  // contents as the original hand-rolled loops, but cover any thread count / tile shape.
  const uint a4PerRow = K7_BK / 4;  // along K  (As stored transposed [BK][BM])
  const uint b4PerRow = K7_BN / 4;  // along N  (Bs stored [BK][BN])

  for (uint bkIdx = 0; bkIdx < K; bkIdx += K7_BK) {
    // LOADING PHASE: A tile -> As (transposed), float4 along K
    for (uint t = threadIdx.x; t < K7_BM * a4PerRow; t += totalThreads) {
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
    for (uint t = threadIdx.x; t < K7_BK * b4PerRow; t += totalThreads) {
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
    float Atmp[K7_TM] = {0.0};
    float Btmp[K7_TN] = {0.0};
    for (uint dotIdx = 0; dotIdx < K7_BK; dotIdx++) {
      for (uint tm = 0; tm < K7_TM; tm++) {
        const uint fromAsRow = warpTileRowOffset + rowInWarptile + tm;
        const uint fromAsCol = dotIdx;
        Atmp[tm] = As[fromAsCol][fromAsRow];
      }

      for (uint tn = 0; tn < K7_TN; tn++) {
        const uint fromBsRow = dotIdx;
        const uint fromBsCol = warptileColOffset + colInWarptile + tn;
        Btmp[tn] = Bs[fromBsRow][fromBsCol];
      }

      for (uint tm = 0; tm < K7_TM; tm++) {
        for (uint tn = 0; tn < K7_TN; tn++) {
          product[tm][tn] += Atmp[tm] * Btmp[tn];
        }
      }
    }

    __syncthreads();
  }

  for (uint tm = 0; tm < K7_TM; tm++) {
    for (uint tn = 0; tn < K7_TN; tn++) {
      const uint row = CrowOffset + tm;
      const uint col = CcolOffset + tn;

      if (col >= N || row >= M) continue;

      const uint Cidx = col + row * N;
      if (beta == 0) C[Cidx] = alpha * product[tm][tn];
      else C[Cidx] = alpha * product[tm][tn] + beta * C[Cidx];
    }
  }
}

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

  else if (algo == WarptilingSingleIter) {
    dim3 blockDim((K7_BM * K7_BN) / (K7_TM * K7_TN));
    dim3 gridDim(ceil((float)B_n / K7_BN), ceil((float)A_m / K7_BM));

    // Warm-up
    sgemm_7_warptiling_single_iter<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(warptiling_single_iter);
    sgemm_7_warptiling_single_iter<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(warptiling_single_iter);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == Warptiling) {
    dim3 blockDim((K8_BM * K8_BN) / (K8_TM * K8_TN * K8_WITER));
    dim3 gridDim(ceil((float)B_n / K8_BN), ceil((float)A_m / K8_BM));

    // Warm-up
    sgemm_8_warptiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(warptiling);
    sgemm_8_warptiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(warptiling);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "Kernel crashed :[" << std::endl << cudaGetErrorString(err) << std::endl;
    }
  }

  else if (algo == WarptilingVectorizedShared) {
    dim3 blockDim(K10_NUM_THREADS);
    dim3 gridDim(B_n / K10_BN, A_m / K10_BM);

    sgemm_10_vec_smem<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(warptiling_vectorized_shared);
    sgemm_10_vec_smem<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(warptiling_vectorized_shared);

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
    std::cout << label << ": FAIL - " << mismatches << "/" << len << " mismatches" << std::endl;
    std::cout << "  worst: ref[" << worstIdx << "] = " << ref[worstIdx] << ", test[" << worstIdx << "] = " << test[worstIdx] << ", abs diff = " << maxAbsDiff << ", rel diff = " << maxRelDiff
              << std::endl;
  }
}

int main() {
  int A_m = 4096;
  int A_n = 4096;
  int B_n = 4096;

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

  // compareResults(C, D, A_m * B_n, "cublas v naive");

  // float* E = new float[A_m * B_n];
  // matmul(E, A, A_m, A_n, B, A_n, B_n, Tiled, handle);

  // compareResults(C, E, A_m * B_n, "cublas v tiled");

  // float* F = new float[A_m * B_n];
  // matmul(F, A, A_m, A_n, B, A_n, B_n, TiledWith1DRegisterTiling, handle);

  // compareResults(C, F, A_m * B_n, "cublas v tiled with 1D register tiling");

  // float* G = new float[A_m * B_n];
  // matmul(G, A, A_m, A_n, B, A_n, B_n, TiledWith2DRegisterTiling, handle);

  // compareResults(C, G, A_m * B_n, "cublas v tiled with 2D register tiling");

  float* H = new float[A_m * B_n];
  matmul(H, A, A_m, A_n, B, A_n, B_n, TiledWith2DRegisterTilingAsVectorized, handle);

  // compareResults(C, H, A_m * B_n, "cublas v K6 vectorized tiled with 2D register tiling");

  float* I = new float[A_m * B_n];
  matmul(I, A, A_m, A_n, B, A_n, B_n, WarptilingSingleIter, handle);

  // compareResults(C, I, A_m * B_n, "warptiling single iter");

  float* J = new float[A_m * B_n];
  matmul(J, A, A_m, A_n, B, A_n, B_n, Warptiling, handle);

  compareResults(C, J, A_m * B_n, "warptiling");

  float* L = new float[A_m * B_n];
  matmul(L, A, A_m, A_n, B, A_n, B_n, WarptilingVectorizedShared, handle);

  compareResults(C, L, A_m * B_n, "warptiling vectorized shared");

  return 0;
}
