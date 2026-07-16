#define FLASH_ATTENTION_NO_MAIN
#include "flash-attention.cu"

#include <vector>

void attentionCpu(float* O, const float* Q, const float* K, const float* V, int B, int H, int N, int D, bool causal) {
  std::vector<double> scores(N);
  for (int head = 0; head < B * H; head++) {
    const size_t offset = (size_t)head * N * D;
    for (int row = 0; row < N; row++) {
      const int limit = causal ? row + 1 : N;
      double maxValue = -INFINITY;
      for (int col = 0; col < limit; col++) {
        double product = 0.0;
        for (int d = 0; d < D; d++) {
          product += (double)Q[offset + (size_t)row * D + d] * K[offset + (size_t)col * D + d];
        }
        scores[col] = product / std::sqrt((double)D);
        maxValue = std::max(maxValue, scores[col]);
      }

      double sum = 0.0;
      for (int col = 0; col < limit; col++) {
        scores[col] = std::exp(scores[col] - maxValue);
        sum += scores[col];
      }
      for (int d = 0; d < D; d++) {
        double product = 0.0;
        for (int col = 0; col < limit; col++) {
          product += scores[col] * V[offset + (size_t)col * D + d];
        }
        O[offset + (size_t)row * D + d] = product / sum;
      }
    }
  }
}

bool testAttention(int B, int H, int N, int D, bool causal, int pattern) {
  const size_t len = (size_t)B * H * N * D;
  std::vector<float> Q(len), K(len), V(len), ref(len), result(len);
  randomFill(Q.data(), len);
  randomFill(K.data(), len);
  randomFill(V.data(), len);

  if (pattern == 1) {
    std::fill(Q.begin(), Q.end(), 0.0f);
  } else if (pattern == 2) {
    for (size_t i = 0; i < len; i++) {
      Q[i] = 8.0f;
      K[i] = 8.0f + (float)((i / D) % N) / 128.0f;
    }
  }

  attentionCpu(ref.data(), Q.data(), K.data(), V.data(), B, H, N, D, causal);
  std::cout << "B=" << B << " H=" << H << " N=" << N << " D=" << D << " causal=" << causal << " pattern=" << pattern << std::endl;
  bool passed = true;
  for (AttentionAlgorithm algo : {Naive, FlashAttention}) {
    attention(result.data(), Q.data(), K.data(), V.data(), B, H, N, D, causal, algo);
    passed &= compareResults(ref.data(), result.data(), len, algo == Naive ? "cpu v naive" : "cpu v flash", 5e-4f, 5e-5f);
  }
  return passed;
}

int main() {
  const int shapes[][4] = {
      {1, 1, 1, 1},
      {1, 2, 3, 7},
      {2, 3, 31, 32},
      {1, 2, 32, 33},
      {2, 1, 33, 64},
      {1, 2, 59, 65},
      {2, 2, 65, 128},
      {1, 1, 97, 129},
      {1, 2, 129, 256},
  };

  bool passed = true;
  for (const auto& shape : shapes) {
    for (bool causal : {false, true}) {
      passed &= testAttention(shape[0], shape[1], shape[2], shape[3], causal, 0);
    }
  }
  for (int pattern : {1, 2}) {
    for (bool causal : {false, true}) {
      passed &= testAttention(2, 2, 97, 64, causal, pattern);
    }
  }
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
