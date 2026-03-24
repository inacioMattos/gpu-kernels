# gpu-kernels

Working through CUDA and PMPP. Mostly small experiments to understand where the
time goes.

- `vector_add/` — grid-stride loop
- `reduction/` — shared-memory sum, multiple passes
- `transpose/` — 32x32 tiles with padding
- `softmax/` — row-wise max/sum reductions
- `matmul/` — naive through register/warp tiling, compared against cuBLAS

Build with nvcc. Set `ARCH` for your card.

```sh
make ARCH=sm_120
make check
make matmul
./build/matmul
```

The four small examples check against CPU results and exit nonzero on failure.
Matmul uses 4096x4096 by default; change `main()` to try the other kernels.
Its comparisons print PASS/FAIL, so check the output.

Notes:

- Optimized matmul kernels still assume suitable tile sizes/alignment.
- Softmax expects finite inputs.
- `matmul/sweep_k8.py` runs from `matmul/`. Its architecture is hardcoded, and
  WMX/WITER are fixed in the CUDA source despite appearing in the sweep.
- Some matmul history contains AI-assisted experiments.

## Matmul: getting close to cuBLAS on my RTX 5070 Ti

This started in `my-own/matmul` in my PMPP repo. I worked from one thread per
output element through shared-memory tiling, register tiling, vectorized loads,
and finally warp tiling. The useful part was figuring out *why* each version was
slow, including a few changes that sounded good but didn't help.

There are two branches of that work here: my current implementation ends at K8
in [matmul.cu](matmul/matmul.cu), while
[k10_experiment.cuh](matmul/k10_experiment.cuh) restores the later optimization
experiment from an earlier revision. That experiment was AI-assisted; it is
kept separate so the results are clear about which implementation is running.

### Measured results

Fresh run on **October 1, 2026**, on my **RTX 5070 Ti (16 GB)**. Square, row-major
FP32 matrices, `C = A × B`, with alpha=1 and beta=0.

At **2048×2048**, K10 takes **574 µs**, compared with **541 µs** for strict FP32
cuBLAS: **94.3% of cuBLAS throughput**, **29.93 TFLOP/s**, and **10.81× faster
than naive**. My current K8 takes **639 µs**, or **84.7% of cuBLAS throughput**.
The gap is larger at 4096×4096: K10 reaches **86.0%** of cuBLAS throughput.

| Implementation | 2048² time (µs) | 4096² time (µs) | Speedup over naive, 2048² | cuBLAS throughput, 2048² |
|---|---:|---:|---:|---:|
| Naive | 6205.5 | 52391.4 | 1.00× | 8.7% |
| Shared-memory tiles | 4377.3 | 35513.9 | 1.42× | 12.4% |
| 1-D register tiling | 1516.7 | 12282.2 | 4.09× | 35.7% |
| 2-D register tiling, v1 | 1606.2 | 16959.5 | 3.86× | 33.7% |
| 2-D register tiling, v2 | 1116.4 | 8252.6 | 5.56× | 48.5% |
| K6: vectorized global loads | 772.1 | 6845.3 | 8.04× | 70.1% |
| K7: single-iteration warp tiling | 872.3 | 6644.1 | 7.11× | 62.1% |
| K8: multi-iteration warp tiling | 639.4 | 5176.1 | 9.71× | 84.7% |
| K10: restored vectorized warp-tile experiment | **574.0** | **4884.5** | **10.81×** | **94.3%** |
| cuBLAS, strict FP32 | 541.5 | 4199.9 | 11.46× | 100.0% |
| cuBLAS, default math | 541.8 | 4221.0 | 11.45× | 99.9% |

The [older optimization log](matmul/notes/optimization-log.md) recorded **557 µs
versus 548 µs**, about **98% of cuBLAS throughput**, at 2048². That was an earlier
timing experiment. The table above is the current repeated CUDA-event result;
it does not reproduce the old 98% figure.

### Where the gains came from

1. **Reuse data inside a block.** The naive kernel has each thread walk a row of
   A and a column of B. Shared-memory tiling lets a block load operand tiles
   together and reuse them. That first step gives **1.42×** at 2048², but each
   thread still does little work per shared-memory load.
2. **Compute more outputs per thread.** With 1-D register tiling, one B value
   contributes to several output rows: **2.89× over shared tiling**. A 2-D
   register tile reuses both A and B: eight A values and eight B values feed
   64 FMAs. V2 adds another **1.36×** over 1-D tiling.
3. **Give the compiler constant loop bounds.** V1 and V2 are a useful detour:
   the basic 2-D tiling idea wasn't enough. The
   [earlier profile](matmul/notes/register-tiling-profile.md) found extra loop,
   branch, and address-calculation work in V1. Compile-time tile geometry helped
   the compiler unroll the loads and expose independent work. In this rerun,
   V2 is **1.44× faster at 2048²** and **2.06× faster at 4096²** than V1.
4. **Move contiguous values together.** K6 uses `float4` global loads and stages
   A transposed in shared memory. Fewer load instructions and a layout suited
   to the register tiles bring another **1.45× over V2** at 2048². The
   [K6 profiling notes](matmul/notes/k6-profile.md) then pointed to shared-memory
   bank conflicts and load latency as the next things to investigate.
5. **Make the warp's access pattern part of the design.** Warp tiling assigns
   each warp a structured output region and reuses operands across subtiles.
   My multi-iteration K8 is **1.21× faster than K6** at 2048². The single-iteration
   K7 is slower than K6 at that size, which is a good reminder that changing the
   mapping alone doesn't guarantee a win.
6. **Tune the shared loads and tile shape together.** The restored K10 combines
   a 128×128 block tile, BK=16, 128 threads, padded shared storage for A, and
   vectorized shared loads/output stores. It is another **1.11× over current K8**
   at 2048². That comparison changes several details, so it isn't an isolated
   measurement of vectorization alone.

Two lessons from the older experiments: forcing higher occupancy made one
version slower, and explicit double buffering did not reliably beat the best
warp-tiled version. More resident warps and more staging are useful only when
they address the actual bottleneck. Those profiler observations are archived
notes; the fresh run above measures latency and correctness, not new stall or
bank-conflict counters.

### Reproducing the numbers

```sh
make benchmark ARCH=sm_120
python3 matmul/run_benchmarks.py --out build/matmul-results
```

- CUDA 13.3, nvcc 13.3.73, driver 610.43.03, cuBLAS version code 130600.
- Built with `-O3 -std=c++17 -lineinfo -arch=sm_120`, without `--use_fast_math`.
- Three warmups before each batch, ten launches per batch, seven rounds.
  The table reports the **median of the seven batch means**. Kernel order
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
environment snapshots are in [matmul/results/rtx-5070-ti](matmul/results/rtx-5070-ti).
The harness currently accepts square sizes divisible by 256; it does not claim
arbitrary-shape coverage for these optimized kernels.

TODO: more matmul edge cases, and measurements with signed inputs and nonzero beta.
