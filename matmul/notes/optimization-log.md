> **Authorship:** This optimization log was written with AI. I wrote the matmul
> kernels described here from scratch by hand, except K10 and K11, which were
> AI-assisted.

> Archived experiment notes from `pmpp-book/my-own/matmul`. These are historical
> measurements and interpretations, not the current benchmark results. See the
> [matmul README](../README.md) for the reproducible rerun. Kernel numbering/configurations
> in these notes differ from the latest `matmul.cu`.

# SGEMM on an RTX 5070 Ti (sm_120): from naïve to cuBLAS parity

A profile-driven optimization log for FP32 `C = αA·B + βC`, M = N = K = 2048, row-major.
Each kernel is presented as **Hypothesis -> Plan -> Results**: why the *previous* kernel leaves
performance on the table, exactly what we change, and what the change actually bought (with Nsight
Compute evidence). Part II is written so that a CUDA beginner can read it top-to-bottom and
re-derive every kernel from scratch - the GPU concepts are explained where they first matter.

---

## Setup & methodology

- **GPU:** RTX 5070 Ti, Blackwell consumer, sm_120, 16 GB, 70 SMs, 65536 32-bit registers/SM,
  FP32 peak ≈ 44 TFLOP/s.
- **Workload:** 2048³ SGEMM. FLOP = 2·M·N·K = 1.72×10¹⁰. Checked against cuBLAS at rel-tol 1e-4;
  all kernels report `max rel diff ≈ 4e-6` -> genuine FP32 (cuBLAS is **not** using TF32 tensor
  cores here, which would show ~1e-3 - a fair CUDA-core-vs-CUDA-core race).
- **Build:** `nvcc -O3 -arch=sm_120 matmul.cu -o matmul -lcublas`.
- **Profiling:** `ncu --set full` and `--set source` on standalone `-lineinfo` harnesses
  (`profile/k6_baseline/`, `profile/k8/`).
- **Reading the numbers:** absolute µs **drift with GPU temperature** - sustained benchmarking
  throttles clocks (I saw the *same* binary swing 760->1100 µs). The trustworthy figure is the
  **same-run cuBLAS:k_x ratio** (both timed back-to-back in one process).

## Scoreboard

| # | kernel | time | vs cuBLAS | TFLOP/s | one-line idea |
|---|--------|------|-----------|---------|---------------|
| - | cuBLAS | 548 µs | 1.00× | 31.4 | reference |
| 0 | naïve | 5900 µs | 10.8× | 2.9 | one thread per C element |
| 1 | tiled (16×16 SMEM) | 4200 µs | 7.7× | 4.1 | block-level shared staging |
| 2 | 1-D register tiling | 1450 µs | 2.65× | 11.9 | TM=8 outputs/thread |
| 3 | 2-D register tiling | 1015 µs | 1.85× | 16.9 | 8×8 outputs/thread |
| 4 | k6 vectorized GMEM (tuned `64/256/32/8`) | 765 µs | 1.40× | 22.5 | float4 global loads + transposed As |
| 5 | k7 vectorized SMEM/store | 768 µs | 1.40× | 22.4 | **dead end - adds nothing** |
| 6 | k8 warptiling | 675 µs | 1.23× | 25.5 | conflict-free shared loads |
| 6b | k8b warptiling, k6-style | 655 µs | 1.20× | 26.3 | **same perf as k8, reads like k6** (see interlude) |
| 7 | k9 double-buffer (reg-prefetch) | 578 µs | 1.05× | 29.8 | overlap global loads |
| 8 | **k10 warptiling + vec SMEM loads** | **557 µs** | **1.02×** | **30.9** | **best - ≈98% of cuBLAS** |
| 9 | k11 double-buffer + vec loads | 571 µs | 1.04× | 30.1 | confirm db is a wash |

---

# Part I - the classic ladder (naïve -> 2-D register tiling)

The standard PMPP progression. Reasoning is from first principles + timings (no per-kernel ncu;
their bottlenecks are analytically obvious).

**Kernel 0 - naïve.** One thread per output; each thread streams a full row of A and column of B
from DRAM. Every A element is re-read N times, every B element N times. Arithmetic intensity =
2 FLOP / 8 bytes = **0.25 FLOP/byte** -> pinned to DRAM bandwidth, FP32 units idle. -> 5900 µs.

**Kernel 1 - shared tiling.** Stage a 16×16 tile of A and B into `__shared__` once, reuse it 16×
across the block -> ~16× less DRAM traffic. But each thread still computes **one** output and issues
**2 shared loads per FMA** -> now shared-memory-throughput bound. -> 4200 µs (~1.4×).

**Kernel 2 - 1-D register tiling.** One thread computes a **column of TM=8 outputs**; it loads one
B value from shared and reuses it against TM A values in registers -> SMEM-load:FMA ratio drops ~TM×.
-> 1450 µs (~2.9× over tiled).

**Kernel 3 - 2-D register tiling.** One thread computes an **8×8 block**; TM A-loads + TN B-loads
feed TM·TN FMAs, so intensity per shared load ≈ (TM·TN)/(TM+TN). -> 1015 µs, ~16.9 TFLOP/s. The
register tile is now efficient; what's still crude is *how* the global and shared loads are issued.

---

# Part II - the profiled push to cuBLAS (k6 -> k11)

> **This is the heart of the report.** Read the mental-model box first; every kernel below leans on it.

## The mental model you need (read once)

A GPU hides latency with *parallelism*, not speed. To reason about a GEMM kernel you need six facts:

**1. The memory hierarchy (fast -> slow).**
- **Registers** - per-thread, ~0 latency, but a hard budget: 65536 32-bit registers per SM, shared
  by every thread resident on it.
- **Shared memory (SMEM)** - per-block scratchpad, ~30-cycle latency, software-managed. Organized
  into **32 banks** (see fact 4).
- **L1 / L2 cache -> global memory (DRAM)** - ~200–400-cycle latency to DRAM. L2 is ~89% hit here
  because A/B tiles get re-read.
- The whole game of a fast GEMM is: pull each operand from DRAM **once**, into SMEM, then into
  registers, and do as many FMAs as possible before touching SMEM/DRAM again.

**2. Warps & latency hiding.** Threads run in **warps of 32**, in lockstep (SIMT). Each cycle a
warp **scheduler** picks one *eligible* warp (one whose next instruction's inputs are ready) and
issues it. If a warp is waiting on a load, the scheduler runs a different warp instead. So latency
is hidden **only if there are enough other warps to switch to**. "Enough warps" = **occupancy**
(fact 5). When a load result isn't back yet and no other warp is eligible, the SM stalls - that
stall is attributed to a **scoreboard** (fact 6).

**3. Coalescing (global loads).** When the 32 threads of a warp read **consecutive** global
addresses, the hardware merges them into a few 32-byte/128-byte sectors. Scattered or strided
addresses waste sectors (e.g. "4 of 32 sectors used" = 8× wasted bandwidth). Rule of thumb: make
consecutive lanes touch consecutive addresses.

**4. Shared-memory banks & conflicts.** SMEM is 32 banks × 4 bytes. A 4-byte word at word-index
`w` lives in **bank `w % 32`**. In one shared-memory instruction, if two lanes of a warp address
**different words in the same bank**, the hardware **serializes** them - an *N-way bank conflict*
costs N× the cycles. The one exception: if lanes read the **same** word it's a free **broadcast**.
*Bank conflicts are a property of which banks which lanes touch - not of the load instruction's
width.* (This single fact decides k7 vs k8.)

**5. Occupancy & the register math.** Occupancy = resident warps ÷ max warps (sm_120 max ≈ 48
warps/SM = 1536 threads). What caps it is usually **registers**: `blocks_per_SM = floor(65536 /
(threads_per_block × regs_per_thread))`. Example (k6): 256 threads × 90 regs = 23040 -> only
`65536/23040 = 2` blocks fit -> 2×256/32 = 16 warps = 16/48 = **33%**. More occupancy *can* hide
latency - but only if the bottleneck is latency and not a saturated pipe (k7's occupancy
experiment shows the failure mode).

**6. Reading warp stalls (ncu).** `ncu` samples why warps couldn't issue. The two that matter here:
- **`long_scoreboard`** - waiting on a **global/L2** load. Fix: hide it with more warps, or overlap
  it with compute (prefetch).
- **`short_scoreboard`** - waiting on a **shared-memory** load. Fix: fewer/cheaper SMEM loads
  (vectorize), or fewer bank conflicts.
- `mio_throttle` - the SMEM/load-store pipe itself is saturated (too many SMEM instructions).
- `barrier` - waiting at `__syncthreads()`.
- `not_selected` - a warp *was* ready but the scheduler picked another (a sign of *enough*
  parallelism, not a real stall).

Two derived knobs we'll keep returning to:
- **Arithmetic intensity** = FLOPs per byte moved (from DRAM, or from SMEM). Higher = less
  bottlenecked by memory. Register tiling and warptiling both exist to raise it.
- **Vectorized load (`float4`)** = one instruction (`LDS.128`/`LDG.128`) moving 16 bytes (4 floats).
  Fewer instructions for the same data -> relieves the load-store pipe. Requires 16-byte alignment
  (the base index must be a multiple of 4 floats) and 4 contiguous elements.

With that, here is each kernel.

---

## Kernel 4 (k6) - vectorize the global loads + transpose As

### Hypothesis - why kernel 3 (2-D register tiling) is slow

Kernel 3's register tile is efficient, but its **global loads are scalar 32-bit**. Two costs:
1. A 32-bit `LDG` moves 4 bytes per instruction; the load-store pipe issues 4× more instructions
   than it needs to. A 128-bit `float4` load moves 16 bytes in one instruction (fact 3 + the
   vectorized-load knob).
2. The inner product needs a **column** of the A-tile (a fixed M-row across the BK depth). In
   row-major A that column is contiguous along K - good - but to *also* read it conveniently in the
   compute loop we want A laid out in SMEM **transposed** as `As[k][m]`, so that a slice over m is
   contiguous and itself `float4`-friendly.

### Plan - concretely

Block computes a `BM×BN` output tile; each thread an 8×8 (`TM×TM`) register block.

```cuda
__shared__ float As[BK][BM];   // TRANSPOSED: As[k][m]  (note [BK][BM], not [BM][BK])
__shared__ float Bs[BK][BN];   // normal:     Bs[k][n]
float product[TM][TM] = {0};   // this thread's 8×8 accumulator, lives in registers
```

Per K-tile, load the global tiles **with float4**:
- A tile: a thread reads 4 consecutive K-elements of one A row in one `float4`, and **scatters
  them transposed** into As (`As[k+0][m]=tmp.x; As[k+1][m]=tmp.y; …`). The transpose happens on the
  store side, for free.
- B tile: a thread reads 4 consecutive N-elements of one B row in one `float4` and writes them
  contiguously into Bs.

Then the compute loop (unchanged from kernel 3): for each `dotIdx` in `0..BK`, pull `Atmp[8]` from
`As[dotIdx][...]` and `Btmp[8]` from `Bs[dotIdx][...]`, do 64 FMAs.

The thread->output map is the thing to internalize, because it's where k6's problem hides:
```cuda
// thread t owns the 8×8 tile starting at:
computeCol = (threadIdx.x * TM) % BN;          // column inside the BN-wide tile
computeRow = TM * ((threadIdx.x * TM) / BN);   // row inside the BM-tall tile
```

### Results

765 µs after tuning (Part III), ≈22.5 TFLOP/s, **1.40× cuBLAS**. This is the kernel I profiled in
depth - the launch pad for everything after.

### The k6 diagnosis (this is what every later kernel reacts to)

`ncu --set full` + `--set source`:

| metric | value | what it means |
|---|---|---|
| Compute (SM) throughput | 52.3% | the math pipe is busy only half the time |
| Memory throughput | 63.4% | the **L1/shared** pipe is the heavier one |
| DRAM throughput | **5.6%** | we are **not** DRAM-bandwidth-bound |
| L2 hit rate | 89.3% | re-read A/B tiles come cheaply from L2 |
| Registers/thread | 90 -> 2 blocks/SM | occupancy is **register-limited** |
| Achieved occupancy | 30% (max would be 33%) | few warps to hide latency |
| Eligible warps/scheduler | **1.49** | scheduler usually has nothing ready to issue |
| **shared-LD bank conflicts** | **33.5 million** | the smoking gun |
| global-store sectors used | **4 of 32** | the C write-out is uncoalesced |

Warp-stall sampling: `long_scoreboard` 17% (waiting on global A/B), `short_scoreboard` 15% (waiting
on shared loads), `barrier` 7%, `mio_throttle` 4%.

**Now do the bank-conflict math by hand** (config BM=BN=128, BK=8, TM=8, 256 threads). In the
compute loop a thread reads `Btmp[tn] = Bs[dotIdx][computeCol + tn]`. Take a single warp
(lanes 0–31) and `tn = 0`. Each lane's column is `computeCol = (lane·8) % 128`:

```
lane:   0   1   2   3   4   5  …  15   16  …  31
col:    0   8  16  24  32  40  … 120    0  … 120   (lane 16 wraps: 16·8=128, 128%128=0)
```

The Bs row sits at word offset `dotIdx·128` (a multiple of 32 -> bank 0), so the **bank** each lane
hits is `col % 32`:

```
col:   0   8  16  24  32  40  48  56  …
bank:  0   8  16  24   0   8  16  24  …   ← only 4 distinct banks {0,8,16,24}!
```

Lanes 0,4,8,12 all want bank 0 (but *different* words 0,32,64,96) -> that's a **4-way conflict**.
Lanes 16–31 re-read lanes 0–15's columns -> free broadcasts. Net: a 4-way bank conflict on every
`Bs` read, ~5-way once averaged with the `As` reads. **That is the 33.5 M conflicts**, and it's
why `short_scoreboard` is high: every shared load of B takes ~4× longer than it should, and 33%
occupancy can't hide it.

**Takeaway that sets up the next three kernels:** k6 is **latency-bound on the shared-memory path**,
the dominant cause is a **structural bank conflict baked into the thread->column map** (stride 8),
and DRAM is idle. So the fix must change *which banks the lanes touch*.

---

## Kernel 5 (k7) - vectorize the shared loads & store -> dead end #1 (and the key lesson)

### Hypothesis

"The 33.5 M conflicts and the 4/32-sector stores are both *scalar instruction* problems. As is
transposed (so `As[dotIdx][m..m+3]` is contiguous) and Bs is contiguous, so I can replace the 8
scalar `Atmp`/`Btmp` reads with 2× `float4` `LDS.128`, and replace the scalar C store with a
`float4` `STG.128`. Fewer, wider shared loads ⇒ fewer conflicts and coalesced stores."

### Plan

```cuda
// was: for (tn=0; tn<8; ++tn) Btmp[tn] = Bs[dotIdx][computeCol+tn];   // 8× LDS.32
reinterpret_cast<float4*>(&Btmp[0])[0] = *reinterpret_cast<float4*>(&Bs[dotIdx][computeCol+0]); // LDS.128
reinterpret_cast<float4*>(&Btmp[4])[0] = *reinterpret_cast<float4*>(&Bs[dotIdx][computeCol+4]);
// …same for Atmp; and a float4 STG.128 for the C write-out
```

### Results - nothing (768 µs), and *why* is the whole point

Re-profile: shared-LD bank conflicts are **still 33.7 M, unchanged**. Walk the math: a `float4`
read at column `computeCol = (lane·8)%128` occupies words `[8·lane .. 8·lane+3]`. Lane 0 -> banks
0–3, lane 1 -> banks 8–11, lane 2 -> banks 16–19, lane 3 -> banks 24–27, **lane 4 -> banks 32–35 ≡ 0–3
again** -> collides with lane 0. The float4 made each access *wider* but the lanes still land on the
same 4 bank-groups. **A bank conflict is a property of the thread->data *mapping*, not the load
width** (fact 4). You cannot vectorize your way out of a bad mapping - you must restructure which
lane owns which columns. That realization is what forces warptiling.

---

## Interlude - the occupancy experiment -> dead end #2 (rules out a tempting fix)

### Hypothesis

"k6/k7 are at 33% occupancy with only 1.49 eligible warps/scheduler. Maybe there just aren't enough
warps to hide the shared/global latency. Force more warps."

### Plan

`__launch_bounds__(256, 3)` tells `ptxas`: guarantee at least 3 blocks fit per SM. To do that it
must keep registers ≤ `65536/(256·3) = 85`. More blocks -> more warps -> 50% occupancy.

### Results - *slower* (845 µs)

`ptxas` hit 85 regs with **0 spills**, occupancy went 33->50% - and it got worse. Why: the
bottleneck wasn't "too few warps", it was a **saturated shared-memory pipe** (Memory throughput
already 63%). Adding warps just throws *more* SMEM instructions at the same congested pipe ->
`mio_throttle` rises. **Occupancy is not the lever; reducing SMEM traffic per FMA is.** This is the
hinge of the whole report: it redirects from "hide the latency" to "remove the work."

---

## Kernel 6 (k8) - warptiling -> the first real win (−12%)

### Hypothesis

Two changes must happen together: (a) **kill the structural Bs conflict** by remapping which lane
owns which output columns, and (b) **raise arithmetic intensity per shared load** so the SMEM pipe
stops being the bottleneck. **Warptiling** does both. The idea: insert a *warp-tile* level between
the block tile and the thread tile. The 32 lanes of a warp cooperate on one contiguous `WM×WN`
region, arranged so that consecutive lanes own *adjacent* columns (stride `TN`, not stride `TM`).
And each thread sweeps `WMITER×WNITER` sub-tiles within the warp region, so every value it loads
from SMEM feeds even more FMAs.

### Plan - the three-level decomposition (this is the whole kernel)

A block tile `BM×BN` is split into warps; each warp owns `WM×WN`; within a warp each thread owns a
small `TM×TN` tile, and *repeats* it `WMITER×WNITER` times to cover the warp region.

```cuda
const uint warpIdx = threadIdx.x / 32;
const uint warpCol = warpIdx % (BN / WN);          // which warp-column in the block
const uint warpRow = warpIdx / (BN / WN);

// how many TM×TN sub-tiles each thread repeats, and the resulting sub-tile pitch
constexpr uint WMITER = (WM*WN) / (32 * TM * TN * WNITER);
constexpr uint WSUBM  = WM / WMITER;               // M-size of one warp sub-tile
constexpr uint WSUBN  = WN / WNITER;               // N-size of one warp sub-tile

const uint lane = threadIdx.x % 32;
const uint threadColInWarp = lane % (WSUBN / TN);  // ← consecutive lanes -> adjacent columns
const uint threadRowInWarp = lane / (WSUBN / TN);

// register caches + the per-thread accumulator
float regM[WMITER*TM], regN[WNITER*TN];
float threadResults[WMITER*TM * WNITER*TN] = {0};
```

Inner loop, per `dotIdx`:
```cuda
// load this thread's slice of As/Bs for the current k into registers
for (wSubRow=0; wSubRow<WMITER; ++wSubRow)
  for (i=0; i<TM; ++i)
    regM[wSubRow*TM+i] = As[dotIdx*BM + warpRow*WM + wSubRow*WSUBM + threadRowInWarp*TM + i];
for (wSubCol=0; wSubCol<WNITER; ++wSubCol)
  for (i=0; i<TN; ++i)
    regN[wSubCol*TN+i] = Bs[dotIdx*BN + warpCol*WN + wSubCol*WSUBN + threadColInWarp*TN + i];
// outer-product accumulate over all sub-tiles
for (wSubRow…) for (wSubCol…) for (rm<TM) for (rn<TN)
    threadResults[…] += regM[wSubRow*TM+rm] * regN[wSubCol*TN+rn];
```

### Why this is conflict-free - do the math again (config WSUBN=32, TN=4 -> `WSUBN/TN = 8`)

`threadColInWarp = lane % 8`. The `Bs` read for a thread starts at column
`warpCol*WN + wSubCol*WSUBN + threadColInWarp*TN`, i.e. the bank-relevant part is
`threadColInWarp*4 = (lane%8)*4`:

```
lane:            0   1   2   3   4   5   6   7   | 8  …
threadColInWarp: 0   1   2   3   4   5   6   7   | 0  …   (lane 8 wraps back to col-group 0)
float4 words:   0-3 4-7 8-11 … 28-31            | 0-3…
banks:          0-3 4-7 8-11 …28-31  ← all 32 banks covered, ZERO conflict
```

Lanes 0–7 cover banks 0–31 with eight distinct `float4` groups (no conflict); lanes 8–15, 16–23,
24–31 have the *same* `threadColInWarp` and re-read those words -> free broadcasts. The stride-8
collision of k6 is gone because lanes now advance by `TN=4` *contiguous* columns instead of `TM=8`
*strided* ones.

### Results

675 µs, **1.23× cuBLAS**. Re-profile confirms the hypothesis exactly:

| metric | k6 | k8 |
|---|---|---|
| shared-LD bank conflicts | 33.5 M | **0.14 M** (236× fewer) |
| Compute (SM) throughput | 52% | **63%** |
| eligible warps/scheduler | 1.49 | **2.17** |
| achieved occupancy | 30% | **42%** |

The −93 µs (~12%) is the single biggest real step in Part II - and it came straight from the
"change the mapping" lesson k7 taught us.

---

## Interlude - k8 made legible: deriving warptiling from k6, one change at a time

k8 is fast, but if you just learned k6 and then open k8, it looks like a *different program*:
warp/sub-tile machinery (`WMITER`, `WSUBM`, `threadColInWarp`), a flat `threadResults[]` array
with hand-computed strides, and pointer arithmetic on `A`/`B`/`C`. None of that is essential to the
*idea*. This interlude builds a kernel - **`sgemm_8b_warptiled_k6style`** (in the source) - that is
**exactly as fast as k8** but is *k6 with two small, local edits*. If you can read k6, you can read
this, and from it re-derive k8. We get there in three steps.

### Step 0 - what we keep from k6, and the one line that's wrong

Recall k6's skeleton (everything here is **kept verbatim** in k8b):

```cuda
__shared__ float As[BK][BM];          // transposed: As[k][m]
__shared__ float Bs[BK][BN];
float product[TM][TM] = {0};          // this thread's output tile, in registers

for (bkIdx = 0; bkIdx < K; bkIdx += BK) {
  /* grid-stride float4 loaders fill As (transposed) and Bs */   __syncthreads();
  float Atmp[TM], Btmp[TM];
  for (dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (tm) Atmp[tm] = As[dotIdx][computeRow + tm];     // a column of As
    for (tn) Btmp[tn] = Bs[dotIdx][computeCol + tn];     // a row of Bs
    for (tm) for (tn) product[tm][tn] += Atmp[tm]*Btmp[tn];
  }                                                       __syncthreads();
}
/* store product back to C */
```

The **only** thing wrong with k6 is the two lines that decide which output each thread owns:

```cuda
computeCol = (threadIdx.x * TM) % BN;          // ← stride TM across threads
computeRow = TM * ((threadIdx.x * TM) / BN);
```

As shown in the k6 diagnosis, within one warp this makes `Btmp[tn] = Bs[dotIdx][computeCol+tn]`
land the 32 lanes on only ~4 banks -> the 33.5 M-conflict disaster. Everything else about k6 is fine.
So: **change only the mapping.**

### Step 1 - make the mapping warp-aware (the bank-conflict fix)

A warp is 32 lanes that issue one shared-load instruction together (mental-model fact 2 & 4).
Bank conflicts are decided *within a warp*, so we must control the layout *of the warp*, not of the
whole block. Replace the two bad lines with: **lay the warp's 32 lanes out as a small `WMX × WNX`
grid** (with `WMX·WNX = 32`), so that consecutive lanes own **adjacent** `TN`-wide columns.

**The new knobs, and their values in k8b's default config.** You pick `WMX` (lanes of a warp along
M); `WNX` is forced by `WMX·WNX = 32`. Everything else is derived:

| symbol | meaning | formula | default value |
|---|---|---|---|
| `TM`,`TN` | this thread's output tile (rows × cols) | (chosen) | 8 × 4 |
| `WMX` | **threads** of the warp stacked along **M** (not C rows - each owns `TM` rows) | (chosen) | **4** |
| `WNX` | **threads** of the warp along **N** (each owns `TN` cols) | `32 / WMX` | **8** |
| `WSUBN` | C-columns in one column-group (a warp-row's width) | `WNX · TN` | 32 |
| `WM` | warp-tile height, in C-rows | `WMX · TM` | 32 |
| `WN` | warp-tile width, in C-cols (step 1: one column-group) | `WNX · TN` | 32 |

So in step 1 a warp is a **4 lanes tall × 8 lanes wide** grid of threads, and since each thread owns
an `8 × 4` tile, the warp covers `WM × WN` = `32 × 32` of C. (`WNITER` does not exist yet - there is
one column-group per thread. **Step 2 introduces it**: it lets each thread own `WNITER`
column-groups, widening the warp tile to `WN = WNITER · WSUBN` - e.g. `2 · 32 = 64` for `WNITER=2`.)

```cuda
const uint warpIdx = threadIdx.x / 32;
const uint laneIdx = threadIdx.x % 32;
const uint warpsAlongN = BN / WN;                                // # of warp-tiles across N (a COUNT of warps, = 128/32 = 4 in step 1; NOT the width WN)
const uint warpRowOffset = (warpIdx / warpsAlongN) * WM;         // delinearize warpIdx -> (row,col); this warp's top row
const uint warpColOffset = (warpIdx % warpsAlongN) * WN;         // this warp's left col
const uint threadColInWarp = laneIdx % WNX;     // 0..WNX-1  -> adjacent lanes, adjacent columns
const uint threadRowInWarp = laneIdx / WNX;     // 0..WMX-1
computeRow = warpRowOffset + threadRowInWarp * TM;
computeCol = warpColOffset + threadColInWarp * TN;
```

That is the entire change for step 1. **The concrete config used for the rest of this interlude**
(so you can check the arithmetic yourself, and so steps 1–3 differ in exactly one variable):

```
BM = BN = 128,  BK = 8,  TM = 8,  TN = 4,  WMX = 4  (-> WNX = 32/WMX = 8),  WNITER = 1
⇒ WSUBN = WNX·TN = 32,  WM = WMX·TM = 32,  WN = WNX·TN = 32  (one column-group)
⇒ threads/block = BM·BN/(TM·TN) = 128·128/32 = 512 ;  warps/block = 512/32 = 16 (a 4×4 warp grid)
⇒ each thread owns a TM×TN = 8×4 tile ;  a warp owns WM×WN = 32×32 of C
```

(Step 2 will flip `WNITER` to 2 and change *nothing else*; step 3 adds `#pragma unroll`. Block tile
`BM/BN/BK` stays fixed across all three, so the only moving parts are `WNITER` and the pragma.)

**Why it's conflict-free** - do the same lane->bank table we did for k6, now with `WNX=8, TN=4`
(so a warp is 4 lanes tall × 8 lanes wide). `As` is `[BK][BM]` and `Bs` is `[BK][BN]` with `BM=BN=128`,
both multiples of 32, so the `dotIdx·BM`/`dotIdx·BN` row offset contributes bank 0 and the bank is
just `(column or row within the tile) % 32`:

```
Bs read, Btmp[tn] at column  computeCol = warpColOffset + threadColInWarp*4,  threadColInWarp = lane % 8
  lane:            0   1   2   3   4   5   6   7   | 8 …(wraps)
  col within tile: 0   4   8  12  16  20  24  28   | 0 …
  bank (=col%32):  0   4   8  12  16  20  24  28   | 0 …   ← 8 distinct banks, ZERO conflict
  (lanes 8..31 repeat threadColInWarp 0..7 -> they read the SAME word -> free broadcast)

As read, Atmp[tm] at row  computeRow = warpRowOffset + threadRowInWarp*8,  threadRowInWarp = lane / 8
  threadRowInWarp: 0   1   2   3       (only 4 distinct values; 8 lanes share each -> broadcast)
  row stride 8 ->   banks 0, 8, 16, 24  ← 4 distinct banks, ZERO conflict
```

Contrast with k6, where `col = (lane·8)%128` gave `banks 0,8,16,24,0,8,16,24…` -> lanes 0 & 4 collide.
The fix is purely *which lane touches which column*; not one byte of the compute or the loaders
changes. The compute loop is still `product[TM][TN]` (k6 used `[TM][TM]`; we just allow `TN ≠ TM`).

**Result of step 1 alone** (this config, `WNITER=1`): the 33.5 M conflicts vanish (PASS), and at the
fixed `128/128/8` tile it runs **736 µs** - versus ~790 µs for k6 at the *same* tile. The conflicts
are gone, yet it's *barely* faster. Why so little? Because we fixed *latency* but not *work*.

### Step 2 - restore arithmetic intensity with a `WNITER` loop

Count the shared traffic per thread per `dotIdx` in the step-1 kernel (with `TM=8, TN=4`): it loads
`8 Atmp + 4 Btmp = 12` floats from SMEM and does `TM·TN = 32` FMAs -> **2.7 FMAs per shared load**.
k8 does **4.0** (it loads 16, does 64). That gap *is* the remaining slowness: too many trips to the
shared-memory pipe per unit of math (mental-model: arithmetic intensity, now measured at the SMEM
level).

The cure is the one idea k8 has that step-1 doesn't: **let each thread own several column-groups and
reuse its `Atmp` across all of them.** If a thread keeps its `TM` A-values and multiplies them
against `WNITER` separate `TN`-wide groups of B (spread `WSUBN = WNX·TN` apart so each group stays
conflict-free), then those same 8 A-loads now feed `TM · WNITER·TN` FMAs. Concretely, the compute
loop grows **one** loop and `product` grows **one** dimension:

```cuda
float Atmp[TM];
float Btmp[WNITER*TN];
float product[TM][WNITER*TN] = {0};        // was product[TM][TM]
…
for (dotIdx = 0; dotIdx < BK; ++dotIdx) {
  for (tm) Atmp[tm] = As[dotIdx][computeRow + tm];           // load A ONCE…
  for (w = 0; w < WNITER; ++w) {                              // …reuse it across WNITER B-groups
    uint colBase = warpColOffset + w*WSUBN + threadColInWarp*TN;
    for (tn) Btmp[w*TN + tn] = Bs[dotIdx][colBase + tn];
  }
  for (tm) for (j = 0; j < WNITER*TN; ++j)
    product[tm][j] += Atmp[tm] * Btmp[j];
}
```

That's it - still k6-shaped (transposed `As`, scalar reads, a 2-D `product`, the `dotIdx` loop), with
a single extra `w` loop. With `WNITER=2`, `TM=8`, `TN=4` a thread now does `8 + 8 = 16` loads for
`8·8 = 64` FMAs -> **4.0**, matching k8. (This is literally k8 with `WMITER` fixed at 1 - and k8's own
best config *also* uses `WMITER=1`, so fixing it costs us nothing while keeping the M side simple.)

**Why `WNITER>1` is forced, not optional.** Conflict-free needs *both* `WNX·TN ≤ 32` (for the Bs
read) *and* `WMX·TM ≤ 32` (for the As read), with `WMX·WNX = 32`. With a big square thread tile
(`TM=TN=8`) that's `WNX ≤ 4` and `WMX ≤ 4`, i.e. `WMX·WNX ≤ 16 < 32` - **impossible**. The only way
to get a big per-thread tile *and* conflict-free banks *and* a sane thread count is to split the warp
region into sub-tiles - which is exactly what `WNITER` (and, in full k8, `WMITER`) is for. Warptiling
isn't an arbitrary structure; it's the unique shape that satisfies all three constraints at once.

### Step 3 - the `#pragma unroll` that makes or breaks it

Step 2, compiled naively, ran at **~912 µs - slower than k6.** The profile was damning:
`short_scoreboard` = **3.71 cycles/instruction** (k8's is 0.79), yet bank conflicts were near zero.
The cause: `product[tm][j]` indexed by a *runtime* loop variable `j`. **Registers are not
addressable** - if the compiler can't resolve every index at compile time, it cannot keep `product`
in registers, so each `+=` recomputes addresses and stalls. ptxas reported 0 spills but used all 80
registers on data with **zero scratch** - the tell-tale sign.

The fix is to force every tile loop to unroll, so all `product[tm][j]` indices become compile-time
constants and collapse to named registers:

```cuda
#pragma unroll
for (dotIdx …) {
  #pragma unroll
  for (tm) Atmp[tm] = …;
  #pragma unroll
  for (w …) { … #pragma unroll for (tn) Btmp[…] = …; }
  #pragma unroll
  for (tm) #pragma unroll for (j …) product[tm][j] += Atmp[tm]*Btmp[j];
}
```

k8 got this unrolling *implicitly* because its loop bounds are all `constexpr` template-ish constants
in a fully static nest; k8b's `WNITER` loop needed it spelled out. After adding it: **100 registers
(16 scratch, like k8), `short_scoreboard` back to normal, 653 µs.** This is the single most important
practical lesson in the whole report for a beginner writing register-tiled kernels: *a register array
indexed by a non-unrolled loop silently falls out of registers.*

### The payoff

All k8b rows below are at the **same** block tile `BM=BN=128, BK=8, TM=8, TN=4, WMX=4` - the only
things changing are `WNITER` and the `#pragma unroll`, so each row isolates one idea:

| stage | config change | time |
|---|---|---|
| k6 at this tile | (baseline, stride-`TM` map) | ~790 µs |
| k8b step 1 | warp-aware mapping, `WNITER=1` (conflict-free) | **736 µs** |
| k8b step 2 | `WNITER=2`, **no `#pragma unroll`** | ~912 µs (`product[][]` falls out of registers) |
| **k8b step 3** | `WNITER=2`, **+ `#pragma unroll`** | **668 µs** |
| k8 (reference) | the "native" warptiling structure | ~655 µs |

(k6's *own* best is 765 µs at its tuned `64/256/32/8` tile; the ~782 here is k6 at this fixed tile,
for an apples-to-apples baseline.) Note step 2 is *slower than step 1* - the naïve `WNITER` add
backfires until the pragma lets `product` stay in registers; that reversal is the whole lesson of
step 3.

`sgemm_8b_warptiled_k6style` is **within noise of k8** (668 vs ~655 µs) and is, line for line, k6 plus
(1) a warp-aware mapping and (2) one extra loop. That is warptiling, demystified: it is not a new
algorithm, it is k6 with the thread->column map fixed and the register reuse turned back up.

---

## Kernel 7 (k9) - double buffering (register prefetch) -> dead end #3

### Hypothesis

After k8, **compute (63%) is now the heavier pipe**, but the SMs still stall at each K-tile boundary
because the loop is strictly serial: `load tile -> __syncthreads -> compute -> __syncthreads -> load
next`. The remaining `short_scoreboard` (0.79 c/inst) and `long_scoreboard` (0.54) are those load
latencies showing through. **Double buffering** breaks the serialization: keep *two* SMEM buffers,
and while the SM computes from buffer A, issue the global loads for the next tile into registers
(latency overlaps the FMAs); then write them into buffer B and flip. Bonus: one `__syncthreads` per
iteration instead of two.

### Plan

```cuda
__shared__ float As[2][...], Bs[2][...];
float4 aReg[NA], bReg[NB];                 // register staging for the next tile
// prologue: load tile 0 -> regs -> As[0]/Bs[0]; __syncthreads
for (tile…){
  if (next) load global tile t+1 INTO aReg/bReg;   // LDG issued now, overlaps compute below
  compute from buffer `cur`;                       // …the FMAs run while the LDG is in flight
  if (next){ store aReg/bReg -> buffer cur^1; __syncthreads; cur ^= 1; }
}
```

### Results - no net gain (654 µs at k8's config)

Re-profile shows exactly why: the register staging pushed registers **96 -> 126**, which dropped
occupancy **42% -> 30%** (fact 5: fewer warps fit). The overlap won back about as much as the lost
occupancy cost - a wash. And there's a deeper reason: with `-O3`, **`ptxas` was already hoisting the
global `LDG` early** (the loads have no dependency until the SMEM write), so much of the overlap
existed implicitly. Manual double buffering mostly just bought register pressure. (At the *big*-tile
config k9 looks like a win over k8 - but only because big-tile k8 is itself bottlenecked; k8 at its
own tuned config already captured it.)

---

## Kernel 8 (k10) - vectorize the warptile register loads -> the polish win (≈98% of cuBLAS)

### Hypothesis

Look at k8's inner loop: `regM`/`regN` are still loaded from SMEM with **scalar `LDS.32`** in a
`for i<TM` loop. k8's top true-stalls are `short_scoreboard` (0.79) and `dispatch_stall` (0.71  -
the issue port is clogged with too many tiny load instructions). The `TM` elements a thread needs
(`As[…+threadRowInWarp*TM + i]`, i=0..TM-1) are **contiguous in SMEM**, so each group is exactly one
`float4 LDS.128` -> **4× fewer shared-load instructions**, less MIO pressure, fewer dispatch stalls.

**Why does vectorizing work here when it did nothing in k7?** Because warptiling already made the
mapping conflict-free. In k7 we vectorized a *bad* mapping (conflicts unchanged). Here we're cutting
instruction count on an *already-good* mapping - no conflicts to fight, pure issue-rate win. Same
tool, opposite outcome, decided entirely by whether the layout was fixed first.

### Plan

```cuda
// was 8 scalar LDS.32 per thread per dotIdx; now 2 float4 LDS.128 (TM=8) and 1 (TN=4):
for (i=0; i<TM; i+=4)
  reinterpret_cast<float4*>(&regM[wSubRow*TM+i])[0] =
      *reinterpret_cast<float4*>(&As[dotIdx*ASTRIDE + warpRow*WM + wSubRow*WSUBM + threadRowInWarp*TM + i]);
```
Plus three cheap polish moves the profile pointed at:
- `__restrict__` on A/B/C pointers + `#pragma unroll` on the K-loop and FMA nest -> lets `ptxas`
  schedule freely and fully unroll the 64-FMA inner product.
- **Pad the As column stride `BM -> BM+4`** (kept a multiple of 4 so `float4` stays aligned). The
  ncu rule engine flagged a 2.7-way conflict on the *transpose store* (`As[k][m]` writes with
  stride BM; BM=64 is a multiple of 32 so the lanes collide). Stride 68 isn't a multiple of 32 ->
  the collision breaks. (It turned out off-critical-path, but it's the right instinct and free.)

### Results

557 µs, **1.02× cuBLAS (≈98%)**. Re-profile: `dispatch_stall` **0.71 -> 0.31**, `mio_throttle`
**0.17 -> 0.10**, compute **63 -> 66%**. With far fewer SMEM instructions the *tile optimum shifted*
to the big tile (BK=16) - see Part III. What's left is `short/long_scoreboard` latency the compiler
already pipelines, against a 66%-busy FMA pipe - i.e. we're now genuinely close to the math limit.

---

## Kernel 9 (k11) - double buffer + vectorized loads -> confirms double-buffering is a wash

### Hypothesis

"Maybe double buffering finally pays once the SMEM loads are cheap (k10) - combine k10's vectorized
inner loop with k9's two-buffer prefetch."

### Plan

k10's `float4` compute loop + k9's prologue/prefetch/flip structure, one barrier per iteration
(110 registers, no spills).

### Results

571 µs - slightly *worse* than k10. Final confirmation of the k9 finding: the compiler already
pipelines the global loads, so explicit double buffering only adds register/SMEM cost here. k10 is
the keeper.

---

# Part III - autotuning, dead ends, and the fair comparison

## Per-kernel configs (why the comparison is now fair, and how to sweep)

The tunable knobs of the warptiling kernels are: `BM,BN` (block tile), `BK` (depth), `TM,TN`
(thread tile), `WM,WN` (warp tile), `WNITER` (sub-tiles per warp in N), and `NUM_THREADS`. They are
not free - they're tied by `NUM_THREADS = (BM·BN)/(TM·TN)` (one thread per `TM×TN` output) and
`num_warps = (BM/WM)·(BN/WN)`, and `WMITER` must come out a positive integer. An invalid combo
either won't compile or fails the correctness check, so a sweep self-filters.

Originally all of k8–k11 shared one `K8_*` config and k6/k7 shared fixed globals - so most kernels
were measured at *someone else's* optimum. Fixed: **each kernel now has its own `-D`-overridable
config** (`K6_*`, `K8_*`, `K10_*`, …), `WARPSIZE` is global, and k6's brittle hand-rolled loaders
were replaced with general grid-stride loaders (identical SMEM contents and perf) so it could be
swept too.

**The tile optimum moves with the instruction mix.** The textbook `128/128, BK16, TM8/TN4,
WM64/WN64, WNITER4` config *lost* at the k8 stage (768 µs) but *won* at k10 (557 µs) - because
k10's vectorized loads changed the per-iteration instruction balance. **Lesson: re-sweep after
every structural change**; there is no globally "best" tile.

Tuned optima found by sweeping:
- **k6:** `BM64 BN256 BK32 TM8` -> 765 µs (only ~2% over its default; the algorithm caps at ~1.40×
  no matter the tile - the conflicts are structural, not a tuning artifact).
- **k8:** small tile `BM64 BN128 BK8 TM4 TN4 WM32 WN64 WNITER2` -> 653–675 µs.
- **k10/k9/k11:** big tile `BM128 BN128 BK16 TM8 TN4 WM64 WN64 WNITER4`, 128 threads -> 557 µs (k10).

**What the fair comparison reveals:** tuned k6 (765) ≈ k7 (768) -> **k7 contributed nothing**; the
real wins are **k8 (warptiling)** then **k10 (vectorized loads)**. The earlier impression that "k8
was flat and k9 was the jump" was purely an artifact of the shared config.

## Dead ends (documented so they're not re-tried)

| attempt | result | why |
|---|---|---|
| k7: vectorize SMEM loads on the old tiling | flat | a conflict is the *mapping*, not the width |
| `__launch_bounds__` forcing 50% occupancy | slower (845 µs) | adds warps to an already-saturated SMEM pipe |
| k9 / k11 double buffering | wash | `ptxas` already pipelines the LDG; staging costs registers |
| smaller tiles to cut the tail (waves = 1.83) | slower | lost arithmetic intensity > tail saved |
| `-Xptxas -dlcm=cg`, `--use_fast_math` | neutral | not bandwidth- or transcendental-bound |
| As stride padding (BM+4) | neutral here | the store conflict overlaps compute (off critical path) |

## KernelWiki cross-check (Blackwell/Hopper KB)

Consulted for anything that could push past cuBLAS. Its high-value techniques - **128-byte TMA
swizzling, tcgen05/TMEM, warp specialization, CLC persistent kernels, ping-pong, NVFP4/FP8 block
scaling** - are all bound to **datacenter tensor-core GEMM (SM100/SM90, FP16/FP8/FP4)** and don't
transfer to FP32 **CUDA-core** SGEMM on a consumer sm_120 part. The two portable ideas: **XOR
shared-memory swizzle** (a cleaner bank-conflict fix than padding - same effect as our As pad, off
the critical path here) and **wider/256-bit loads + cache policies** (these target *memory-bound*
kernels; k10 is compute-bound -> neutral).

## Why not beat cuBLAS outright

k10 sits at compute 66% / memory 61%, DRAM idle. The remaining ~2% is `short/long_scoreboard`
latency the compiler already pipelines against a 66%-busy FMA pipe. Closing it needs hand-scheduled
SASS (cuBLAS is hand-tuned assembly) - not reachable from CUDA C, and far past diminishing returns.
≈98% of cuBLAS in true FP32 is effective parity for a hand-written SGEMM (cf. Boehm's ~93–96% after
exhaustive autotuning on an A6000).

## Reproduce

```bash
nvcc -O3 -arch=sm_120 matmul.cu -o matmul -lcublas && ./matmul
# per-kernel tile sweeps (each prefix independent):
nvcc -O3 -arch=sm_120 -DK10_BK=16 -DK10_TM=8 -DK10_WNITER=4 matmul.cu -o m -lcublas
nvcc -O3 -arch=sm_120 -DK6_BM=64 -DK6_BN=256 -DK6_BK=32 -DK6_TM=8 matmul.cu -o m -lcublas
# ncu reports live in profile/k6_baseline/ and profile/k8/
```
