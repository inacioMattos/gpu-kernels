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
};

#define TM 8
#define BK 8
#define BM 128
#define BN 128
__global__ void sgemm_5_register_2dtiling(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
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

  const int threadCol = threadIdx.x % (BN / TM);
  const int threadRow = threadIdx.x / (BN / TM);

  float Atmp[TM] = {0.0};
  float Btmp[TM] = {0.0};

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

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        Atmp[i] = As[(threadRow * TM + i)][dotIdx];
      }
      for (uint i = 0; i < TM; ++i) {
        Btmp[i] = Bs[dotIdx][threadCol * TM + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TM; ++resIdxN) {
          product[resIdxM][resIdxN] += Atmp[resIdxM] * Btmp[resIdxN];
        }
      }
    }

    // COMPUTE PHASE START
    /*
    float Atmp[TM] = {0.0};
    float Btmp[TM] = {0.0};
    const int threadCol = threadIdx.x % (BN / TM);
    const int threadRow = threadIdx.x / (BN / TM);
    for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
      for (uint tm = 0; tm < TM; tm++) {
        const uint fromAsRow = (TM * ((threadIdx.x * TM) / BN)) + tm;
        const uint fromAsCol = dotIdx;
        Atmp[tm] = As[threadRow * TM + tm][dotIdx];
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
    */

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
  const int TN = 8;
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;      // move BK columns to right
    B += BK * N;  // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
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
    sgemm_5_register_2dtiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);

    cudaDeviceSynchronize();
    WALL_START(tiled_2d_register_tiling);
    sgemm_5_register_2dtiling<<<gridDim, blockDim>>>(A_m, B_n, A_n, 1, 0, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    WALL_END(tiled_2d_register_tiling);

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

  return 0;
}
