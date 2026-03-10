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

// 2141198334 fp32 loads
__global__ void sgemm_4_register_tiling(int M, int N, int K, float alpha, float beta, const float* A, const float* B, float* C) {
  const uint Ccol = (blockIdx.x * BN) + (threadIdx.x % BN);
  const uint Crow = (blockIdx.y * BM) + ((threadIdx.x / BN) * TM);

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  float product[TM] = {0.0};

  // The A row and B col this thread will be loading from GMEM into SREM
  // It's the same value throughout this thread lifecycle
  const uint Arow = (blockIdx.y * BM) + (threadIdx.x / BK);
  const uint Bcol = (blockIdx.x * BN) + threadIdx.x % BN;

  const uint toAsRow = threadIdx.x / BK;
  const uint toAsCol = threadIdx.x % BK;

  const uint toBsRow = threadIdx.x / BN;
  const uint toBsCol = threadIdx.x % BN;

  for (uint tile = 0; tile < ceil((float)K / BK); tile++) {
    const uint Acol = (threadIdx.x % BK) + tile * BK;

    if (Acol < K && Arow < M) As[toAsRow][toAsCol] = A[Acol + Arow * K];
    else As[toAsRow][toAsCol] = 0.0;

    const uint Brow = (threadIdx.x / BN) + tile * BK;

    if (Brow < K && Bcol < N) Bs[toBsRow][toBsCol] = B[Bcol + Brow * N];
    else Bs[toBsRow][toBsCol] = 0.0;

    __syncthreads();
    for (uint k = 0; k < BK; k++) {
      const uint fromAsRow = (threadIdx.x / BN) * TM;
      const uint fromAsCol = k;

      const uint fromBsRow = k;
      const uint fromBsCol = (threadIdx.x % BN);

      float Btmp = Bs[fromBsRow][fromBsCol];
      for (uint tm = 0; tm < TM; tm++) {
        product[tm] += As[fromAsRow + tm][fromAsCol] * Btmp;
      }
    }
    __syncthreads();
  }

  for (uint tm = 0; tm < TM; tm++) {
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
    dim3 blockDim((BM * BN) / TM);  // 512 threads, 1D
    dim3 gridDim(ceil((float)B_n / BN), ceil((float)A_m / BM));

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
