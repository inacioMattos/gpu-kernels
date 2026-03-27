> Archived experiment notes from `pmpp-book/my-own/matmul`. These are historical
> measurements and interpretations, not the current benchmark results. See the
> [matmul README](../README.md) for the reproducible rerun. Kernel numbering/configurations
> in these notes differ from the latest `matmul.cu`.

# k6 SGEMM profiling - RTX 5070 Ti (sm_120), M=N=K=2048

Kernel: `sgemm_6_register_2dtiling_vectorized_As`
BM=BN=128, BK=8, TM=8 (8×8 reg tile), 256 threads/block, grid 16×16.

cuBLAS 549 µs · k6 786 µs (wall) · ncu duration 936 µs (profiling overhead). Gap ≈ 43%.

## Headline numbers

| metric | value | read |
|---|---|---|
| Compute (SM) throughput | **52.3%** | neither pipe saturated |
| Memory throughput | **63.4%** | memory-pipe heavier than compute |
| DRAM throughput | **5.6%** | **not** DRAM-bandwidth bound |
| L1/TEX hit rate | 17.5% | streaming, expected |
| L2 hit rate | 89.3% | A/B re-fetched from L2 cheaply |
| Achieved occupancy | **30.2%** (theoretical 33.3%) | low |
| Registers / thread | **90** → Block Limit (Registers) = **2** | occupancy is register-limited |
| Warp cycles / issued inst | 6.45 | high latency per issue |
| Eligible warps / scheduler | **1.49** of 3.62 active | not enough to hide latency |

## Where the cycles go (pcsamp stall reasons)

| reason | % | meaning |
|---|---|---|
| not_selected | 24.3 | (a warp *was* ready - contention, not a true stall) |
| **long_scoreboard** | **17.3** | waiting on **global** loads of A/B |
| selected | 15.7 | actual issue |
| **short_scoreboard** | **15.3** | waiting on **shared** loads (Atmp/Btmp) |
| dispatch_stall | 7.7 | |
| barrier | 7.5 | `__syncthreads` |
| mio_throttle | 4.3 | LSU/shared pipe saturated |

Per-line (harness lines): the global A/B load region (long_scoreboard ≈10.8k samples) and the
inner FMA + `Bs[..][..]` shared read (short_scoreboard ≈11k) dominate.

## Root cause

Two exposed latencies, both because **occupancy is pinned at 33%** (90 regs/thread → only 2
blocks/SM). With just 1.49 eligible warps/scheduler there aren't enough warps to hide either:

1. **Shared-memory bank conflicts (biggest fixable item).**
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 33.5M` (vs 2.1M on stores).
   NCU: shared **loads ≈ 5-way** conflicts. The inner loop reads `Bs[dot][col+tn]` one float at a
   time (LDS.32) with a column stride of TM=8 across threads → repeated bank collisions and 8×
   the instruction count it needs.

2. **Global store coalescing.** NCU: store pattern uses **only 4 of 32 sectors** per request.
   The C write-out is scalar with the same stride-8 thread→column mapping.

3. **DRAM is idle (5.6%)** - this kernel is latency-bound on the L1/shared path, not bandwidth.

## Plan (ranked by evidence × effort)

1. **Vectorize the inner-loop register caches to float4 (LDS.128).** `As` is already stored
   transposed `[BK][BM]`, so `As[dot][row..row+3]` is contiguous; `Bs[dot][col..col+3]` is
   contiguous too. Replace the 8 scalar `Atmp`/`Btmp` reads with 2× `float4` each. Cuts shared-load
   instructions 4×, slashes the 33.5M conflicts, relieves short_scoreboard + mio_throttle.
   *This is the canonical "kernel 6 finishing move" and the highest ROI.*
2. **Vectorize the C store (float4).** Fixes the 4/32-sector store pattern.
3. **Warp-tiling (kernel 7).** Restructure so each warp owns a contiguous output sub-tile →
   conflict-free smem access + more ILP per thread; the path to >33% effective latency hiding.
4. Later: double-buffered/`cp.async` global→shared (kernel 9/10), autotune BK/TM/TN.

Start with 1+2: smallest diff, directly targets the two metrics that are red.

---

## Outcome (what was actually done)

| kernel | time | vs cuBLAS (548µs) |
|---|---|---|
| k6 (baseline) | 786 µs | +43% |
| k7 (float4 inner loop + store) | 763 µs | +39% |
| **k8 (warptiling, BK=16, 64×128, 128 thr)** | **654 µs** | **+19%** |

- **k7** (plan items 1+2: float4 LDS.128 register caches + STG.128 store): only ~3%.
  Re-profiling showed the 33.5M shared-load conflicts were **unchanged** - they're structural to
  the warp's stride-8 thread→column map, not the scalar-vs-vector access width.
- **Occupancy is NOT the lever:** forcing 3 blocks/SM via `__launch_bounds__(256,3)` (0 spills,
  33%→50% occ) made k6 *slower* (845 µs). More warps just contend on the already-65%-busy
  L1/shared pipe. This redirected the effort to *reducing* shared traffic, not hiding its latency.
- **k8 warptiling** (plan item 3) was the real fix. Each warp owns a WM×WN sub-tile; threads map
  with stride TN so float4 `Bs` reads hit distinct banks, and each thread sweeps WMITER×WNITER
  sub-tiles → more FMAs per shared load. Verified by re-profile:
  - shared-LD bank conflicts **33.5M → 142K** (236×)
  - Compute (SM) **52% → 63%**, eligible warps/sched **1.49 → 2.17**, occupancy **33% → 42%**
- Tuning: 64×128 block / 128 threads / BK=16 beat the textbook 128×128 / 256-thread configs on
  the 5070 Ti - smaller tiles → 512 blocks → better wave balance (the 128×128 config ran only
  ~1.8 waves, leaving a tail).

Remaining +19% gap: compute (63%) is now the heavier pipe. Closing it needs double-buffered
`cp.async` global→shared prefetch to overlap the load phase with compute - diminishing returns
for a hand-written FP32 SGEMM.
