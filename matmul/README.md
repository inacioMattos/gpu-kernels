# Matmul: getting close to cuBLAS on my RTX 5070 Ti

This started in `my-own/matmul` in my PMPP repo. I worked from one thread per
output element through shared-memory tiling, register tiling, vectorized loads,
and finally warp tiling. The useful part was figuring out *why* each version was
slow, including a few changes that sounded good but didn't help.

The kernels live together in [matmul.cu](matmul.cu), including K10's vectorized
warp tiling. K10 and K11 were AI-assisted; I wrote the other matmul kernels
from scratch by hand. The optimization log was written with AI.

## Optimization-log scoreboard

These are the historical measurements from my
[optimization log](notes/optimization-log.md), on the **RTX 5070 Ti** with
**M=N=K=2048**, FP32 inputs, and row-major matrices. The progression went from
**5900 µs to 557 µs**: **10.59× faster than naive**, **30.9 TFLOP/s**, and about
**98% of cuBLAS throughput** against the recorded **548 µs** baseline.

| Kernel | Time | Time / cuBLAS time | TFLOP/s | Main idea |
|---|---:|---:|---:|---|
| cuBLAS | 548 µs | 1.00× | 31.4 | Reference |
| Naive | 5900 µs | 10.8× | 2.9 | One thread per output element |
| Shared-memory tiling | 4200 µs | 7.7× | 4.1 | Stage 16×16 tiles per block |
| 1-D register tiling | 1450 µs | 2.65× | 11.9 | Eight outputs per thread |
| 2-D register tiling | 1015 µs | 1.85× | 16.9 | An 8×8 register tile per thread |
| K6: vectorized global loads | 765 µs | 1.40× | 22.5 | `float4` loads and transposed A staging |
| K7: vectorized shared loads/stores | 768 µs | 1.40× | 22.4 | Little gain with the old thread mapping |
| K8: warp tiling | 675 µs | 1.23× | 25.5 | Change the warp's shared-memory access pattern |
| K8b: warp tiling in K6's structure | 655 µs | 1.20× | 26.3 | Similar performance, more familiar code |
| K9: double buffering | 578 µs | 1.05× | 29.8 | Prefetch through registers |
| **K10: warp tiling + vectorized shared loads** | **557 µs** | **1.02×** | **30.9** | **Best recorded result: ≈98% of cuBLAS** |
| K11: double buffering + vectorized loads | 571 µs | 1.04× | 30.1 | Added staging did not beat K10 |

Times, ratios, and TFLOP/s retain the log's rounding. The ratio column measures
**latency**, so lower is better; throughput relative to cuBLAS is the inverse.
Kernel names here follow the log: its K7 is a vectorized shared-memory
experiment, not the single-iteration warp kernel in the current `matmul.cu`.
These are recorded experiment results; the fresh rerun is linked below.

## Where the gains came from

1. **Reuse data inside a block: 5900 -> 4200 µs (1.40×).** Shared-memory tiling
   lets threads cooperatively load tiles of A and B, then reuse those values
   across the block. Each thread still computes only one output, leaving lots
   of shared-memory traffic per multiply-add.
2. **Compute more outputs per thread: 4200 -> 1450 -> 1015 µs.** With 1-D register
   tiling, one B value contributes to eight output rows: **2.90× over shared
   tiling**. An 8×8 register tile reuses both operands: eight A values and eight
   B values feed 64 FMAs. That adds another **1.43×**.
3. **Make the compiler's job easier.** The separate
   [V1/V2 profiling experiment](notes/register-tiling-profile.md) recorded
   **2130 -> 1176 µs (1.81×)** for the same basic 2-D tiling algorithm. Constant
   tile geometry enabled loop unrolling and reduced branch/address work. The
   profile recorded **37% more executed instructions in V1**. Those timings
   come from that profiling experiment, not the scoreboard run.
4. **Move contiguous values together: 1015 -> 765 µs (1.33×).** K6 uses `float4`
   global loads and stages A transposed in shared memory. That reduces load
   instruction count and arranges the values for register reuse. Simply
   vectorizing shared loads on the old mapping, K7, gave **768 µs** - essentially
   flat. Wider loads alone didn't fix the access pattern.
5. **Design the warp's access pattern: 765 -> 675 µs (1.13×).** Warp tiling gives
   each warp a structured output region and more reuse across subtiles. In a
   separately profiled configuration, the [K6 notes](notes/k6-profile.md)
   recorded shared-load bank conflicts falling from **33.5 million to 142,000**
   after the warp-tiling change. K8b expressed the same idea in a structure
   closer to K6 and recorded **655 µs**.
6. **Vectorize the warp-tile loads and retune: down to 557 µs.** K10 combines
   vectorized shared loads/output stores with a 128×128 block tile, BK=16,
   128 threads, and padded A staging. The log records `dispatch_stall` falling
   **0.71 -> 0.31**, `mio_throttle` **0.17 -> 0.10**, and compute throughput
   **63% -> 66%** in its profiling comparison. The combined result is **1.21×
   faster than the scoreboard's K8**, and within about **2% of cuBLAS latency**.

The tile optimum changed with the instruction mix: the smaller tile worked
well for K8, while K10 preferred the larger tile. Comparing several versions
with one shared configuration initially hid that distinction. The lesson was
to retune after a structural change.

Two useful dead ends: forcing higher occupancy made one version slower, and
explicit double buffering did not beat the best vectorized warp-tiled version
(**571 µs for K11 versus 557 µs for K10**). The scoreboard's K9 result also
changes configuration, so its improvement cannot all be attributed to
prefetching. These explanations and profiler counters come from the archived
experiments.

## Fresh rerun and reproducibility

The **October 1, 2026** rerun uses repeated CUDA-event measurements.
K10 measured **574 µs versus 541 µs** for strict FP32 cuBLAS at 2048²
(**94.3% of cuBLAS throughput**), and **4.885 ms versus 4.200 ms** at 4096²
(**86.0%**). It did not reproduce the historical 98% result. The current K8
measured **639 µs** at 2048² (**84.7%**). Full results are in
[matmul/results/rtx-5070-ti](results/rtx-5070-ti).

The following procedure and validation details describe that fresh rerun.
Run these commands from the repository root:

```sh
make benchmark ARCH=sm_120
python3 matmul/run_benchmarks.py --out build/matmul-results
```

- CUDA 13.3, nvcc 13.3.73, driver 610.43.03, cuBLAS version code 130600.
- Built with `-O3 -std=c++17 -lineinfo -arch=sm_120`, without `--use_fast_math`.
- Three warmups before each batch, ten launches per batch, seven rounds.
  The CSV summaries report the **median of the seven batch means**. Kernel order
  rotates between rounds; this is a repeated-input, warm-cache benchmark.
- [CUDA events](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers)
  time GPU work. Allocation, host/device copies, and correctness checks are
  outside the timed region. Throughput uses `2 × M × N × K / elapsed time`.
- The reference uses `cublasSgemm` with
  [`CUBLAS_PEDANTIC_MATH`](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t)
  for strict FP32. Default cuBLAS math is measured separately. The custom
  kernels use FP32 multiply-adds; this is not a TF32/FP16 Tensor Core comparison.
- Every output element passes against strict cuBLAS with
  `abs_error <= 1e-5 + 1e-4 × abs(reference)`. Maximum relative error across the
  custom kernels is **3.39e-6 at 2048²** and **4.98e-6 at 4096²**. Inputs are
  deterministic uniform values in [0,1), seed 1704.
- Compute Sanitizer memcheck also passed all variants at 256² with zero errors.
  Before measuring, I fixed K8's `float4` register-loading loop to advance by
  four elements; advancing by one wrote past the end of the register array.
- The GPU also drives the desktop. Clocks were not locked, so these are local
  measurements, not a guarantee across sizes, GPUs, or thermal conditions.

Raw batch timings, full summaries, correctness output, source hashes, and
environment snapshots are in [matmul/results/rtx-5070-ti](results/rtx-5070-ti).
The harness currently accepts square sizes divisible by 256; it does not claim
arbitrary-shape coverage for these optimized kernels.

TODO: more matmul edge cases, and measurements with signed inputs and nonzero beta.
