# Flash Attention v1

I built this while following [Flash Attention Version 1](https://www.lowlevelml.com/blog/flash-attention-1)
by Vishal Padia and Sriram Govindan at Low Level ML. This is a plain CUDA
implementation of tiled attention with online softmax.

The operation is `softmax(Q K^T / sqrt(D)) V`, with FP32 tensors laid out as
`[B, H, N, D]`. Both causal and non-causal attention are supported.

## Main ideas

- The naive baseline computes scores, softmax, and the output in three kernels.
- The fused kernel processes four query rows and 32 key/value rows per tile.
  It keeps a running maximum, normalization sum, and output accumulator,
  rescaling them as each tile arrives.
- Scores stay on chip instead of materializing an `N × N` matrix in global
  memory. At `B=1, H=4, N=2048`, this avoids the baseline's 64 MiB score buffer.
  The causal kernel also skips future key/value tiles.

## Results

RTX 5070 Ti, `B=1`, `H=4`, FP32. These compare the two implementations in this
folder. The baseline is the included naive CUDA code, not PyTorch SDPA or the
FlashAttention library.

| N | D | Causal | Naive (µs) | Flash (µs) | Speedup |
|---:|---:|:---:|---:|---:|---:|
| 512 | 64 | No | 260.0 | 102.0 | 2.55× |
| 512 | 64 | Yes | 260.7 | 71.9 | 3.63× |
| 1024 | 64 | No | 1092.4 | 386.8 | 2.82× |
| 1024 | 64 | Yes | 1096.7 | 243.8 | 4.50× |
| 2048 | 64 | No | 4395.6 | 1427.2 | 3.08× |
| 2048 | 64 | Yes | 4453.6 | 827.5 | 5.38× |
| 1024 | 128 | No | 2025.8 | 792.3 | 2.56× |
| 1024 | 128 | Yes | 2025.5 | 482.8 | 4.20× |
| 2048 | 128 | No | 7922.4 | 2927.1 | 2.71× |
| 2048 | 128 | Yes | 7931.4 | 1615.7 | 4.91× |

CUDA-event timing, one warmup and 20 iterations per round, seven rounds.
The table shows median batch means, with allocation and transfers excluded.
Execution order alternates each round. GPU clocks were not locked.

All 44 CPU-reference comparisons passed, including partial tiles, head dimensions
from 1 to 256, causal masks, uniform scores, and large logits. Compute Sanitizer
memcheck reported zero errors. All 70 benchmark comparisons against naive also
passed. CPU tolerance is `5e-5 + 5e-4 * abs(reference)`; benchmark tolerance is
`1e-5 + 1e-4 * abs(reference)`.

[Raw samples, summaries, and validation output](results/) are included.

## Build and run

From this directory, with CUDA installed:

```sh
make ARCH=sm_120
make check
./flash-attention

make benchmark ARCH=sm_120
mkdir -p results-local
./benchmark results-local/samples.csv results-local/summary.csv
```

Use the architecture flag for your GPU. The recorded build used CUDA 13.3 and
`-O3 -std=c++17 -lineinfo -arch=sm_120`.

This is a forward-pass learning implementation. Head dimensions from 1 to 256
are supported; dropout, backward, and Tensor Core paths are not implemented.
