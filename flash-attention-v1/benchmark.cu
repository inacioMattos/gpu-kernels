#define FLASH_ATTENTION_NO_MAIN
#include "flash-attention.cu"

#include <fstream>
#include <iomanip>
#include <vector>

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./benchmark samples.csv summary.csv\n";
    return EXIT_FAILURE;
  }
  std::ofstream samples(argv[1]), summary(argv[2]);
  if (!samples || !summary) return EXIT_FAILURE;
  samples << "B,H,N,D,causal,round,iterations,naive_us,flash_us\n" << std::setprecision(9);
  summary << "B,H,N,D,causal,naive_median_us,flash_median_us,speedup,naive_min_us,naive_max_us,flash_min_us,flash_max_us,status\n" << std::setprecision(9);

  cudaDeviceProp props;
  cudaCheck(cudaGetDeviceProperties(&props, 0));
  std::cout << "GPU: " << props.name << "\n";
  constexpr int B = 1, H = 4, rounds = 7, iterations = 20;
  const int shapes[][2] = {{512, 64}, {1024, 64}, {2048, 64}, {1024, 128}, {2048, 128}};
  for (const auto& shape : shapes) {
    const int N = shape[0], D = shape[1];
    const size_t len = size_t(B) * H * N * D;
    std::vector<float> Q(len), K(len), V(len), ref(len), result(len);
    randomFill(Q.data(), len);
    randomFill(K.data(), len);
    randomFill(V.data(), len);
    for (bool causal : {false, true}) {
      std::vector<double> naiveTimes, flashTimes;
      std::cout << "B=" << B << " H=" << H << " N=" << N << " D=" << D << " causal=" << causal << std::endl;
      for (int round = 0; round < rounds; ++round) {
        double naiveUs = 0, flashUs = 0;
        for (int step = 0; step < 2; ++step) {
          const bool flash = (step + round) % 2;
          float ms = attention(flash ? result.data() : ref.data(), Q.data(), K.data(), V.data(),
                               B, H, N, D, causal, flash ? FlashAttention : Naive, iterations);
          if (flash) flashUs = ms * 1000.0;
          else naiveUs = ms * 1000.0;
        }
        if (!compareResults(ref.data(), result.data(), len, "naive v flash")) return EXIT_FAILURE;
        naiveTimes.push_back(naiveUs);
        flashTimes.push_back(flashUs);
        samples << B << ',' << H << ',' << N << ',' << D << ',' << causal << ',' << round << ','
                << iterations << ',' << naiveUs << ',' << flashUs << '\n';
      }
      double naiveUs = median(naiveTimes), flashUs = median(flashTimes);
      summary << B << ',' << H << ',' << N << ',' << D << ',' << causal << ',' << naiveUs << ','
              << flashUs << ',' << naiveUs / flashUs << ','
              << *std::min_element(naiveTimes.begin(), naiveTimes.end()) << ','
              << *std::max_element(naiveTimes.begin(), naiveTimes.end()) << ','
              << *std::min_element(flashTimes.begin(), flashTimes.end()) << ','
              << *std::max_element(flashTimes.begin(), flashTimes.end()) << ",PASS\n";
    }
  }
}
