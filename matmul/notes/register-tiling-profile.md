> Archived experiment notes from `pmpp-book/my-own/matmul`. These are historical
> measurements and interpretations, not the current benchmark results. See the
> repository README for the reproducible rerun. Kernel numbering/configurations
> in these notes differ from the latest `matmul.cu`.

# sgemm5 v1 vs v2 — why v1 is ~1.5–1.8× slower

**HW:** RTX 5070 Ti (Blackwell, sm_120). **Shapes:** M=N=K=2048, BM=BN=128, BK=8, TM=TN=8, 256 threads/block.
Standalone harness (`harness/harness.cu`), `-O3 -lineinfo -arch=sm_120`.

## Result
| | v1 (`sgemm_5_register_2dtiling`) | v2 (`..._v2`) | v1/v2 |
|---|---|---|---|
| Duration | 2130 us | 1176 us | **1.81×** |
| SM throughput | 33.9% | 50.2% | |
| Mem throughput | 43.0% | 74.7% | |
| Instructions executed | 4.99e8 | 3.64e8 | 1.37× |
| Branch instructions | 1.37e7 | 4.72e6 | 2.89× |
| **stall long_scoreboard** (cyc/issue) | **4.28** | **0.94** | **4.53×** |
| stall wait | 1.17 | 0.41 | 2.86× |
| shared-mem inst | 4.61e7 | 4.61e7 | 1.00× |
| global-ld inst | 4.19e6 | 4.33e6 | 0.97× |
| bank conflicts | 6.77e7 | 6.73e7 | 1.01× |
| registers / occupancy | 93 / 30.7% | 93 / 29.7% | — |

The two kernels are **algorithmically identical** — same thread→tile mapping, same SMEM
access pattern, same FFMA count, same global-load count, same bank-conflict count, same
occupancy. So the slowdown is **not** the algorithm, memory layout, or occupancy.

## Root cause: v1's loops don't unroll
v1 derives all its loop bounds and strides from **`blockDim.x` (a runtime value)** and uses
**`ceil((float)K / BK)`** for the k-loop:

```cpp
const uint subtitleAHeight = blockDim.x / BK;             // runtime
for (tile = 0; tile < ceil((float)K / BK); tile++)        // float ceil, runtime trip count
  for (subtitle = 0; subtitle < ceil((float)BN / subtitleAHeight); subtitle++)  // runtime
```

v2 derives everything from **compile-time constants** (`numThreadsBlocktile = BM*BN/(TM*TN)`,
`strideA = numThreadsBlocktile / BK`, `for (bkIdx=0; bkIdx<K; bkIdx+=BK)`), so `nvcc` knows
every trip count (4, 4, 8) and fully unrolls.

SASS confirms it (per-kernel instruction histogram):

| SASS | v2 | v1 |
|---|---|---|
| ISETP (predicate) | 1 | **160** |
| BRA / BSSY | 4 / 0 | **40 / 16** |
| IMAD | 25 | **121** |
| I2F + F2I | 0 | **2** (from `ceil((float)…)`) |
| FFMA | 128 | 144 |

## Why that costs 1.8×, precisely
Two compounding effects, both visible in NCU:

1. **Lost memory-level parallelism → exposed global latency.** This is the dominant cost.
   In v2 the SMEM-fill loop is unrolled, so the 4 independent `LDG`s per phase are issued
   back-to-back and sit in flight together; their ~hundreds-of-cycle latency overlaps. In v1
   the same loop is a real loop: each iteration computes its address (dependent IMAD chain),
   issues one `LDG`, then a branch must resolve before the next address/load — the loads
   serialize and each one's latency is exposed. That is exactly the **4.5× `long_scoreboard`
   stall** (4.28 vs 0.94 cyc/issue) — same number of loads, far worse overlap. It also shows
   up as v1's much lower memory throughput (43% vs 75%).

2. **Instruction overhead.** The non-unrolled loops add +37% total instructions, 2.9× branches,
   ~5× the IMAD/ISETP integer-address work, plus float↔int conversions for `ceil()`. This is
   pure overhead competing for issue slots with the FFMAs (v1 issue-active 34% vs 48%).

Net: v1 stalls more on memory *and* spends more cycles on bookkeeping, so its FMA pipe runs at
21% vs v2's 36%.

## Fix
Make v1's bounds compile-time constant so it unrolls like v2:
- Replace `blockDim.x` with a `constexpr`/`#define` thread count (256) — derive `subtitleAHeight`
  etc. from constants, not `blockDim.x`.
- Replace `ceil((float)K / BK)` with the integer `for (tile=0; tile*BK < K; tile++)` (or `K/BK`
  when divisible) to drop the I2F/F2I and give a constant trip count.
- Add `#pragma unroll` on the SMEM-load loops as insurance.

That alone should recover essentially all of the gap; the two kernels do identical work.
