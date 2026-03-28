#!/usr/bin/env python3
"""
sweep_k8.py - autotune the k8 warptiling kernel (sgemm_8_warptiling) in matmul.cu.

Non-destructive: never edits matmul.cu. It compiles the binary once per config with
`-DK8_*=...` overrides, runs it best-of-N (to ride out thermal/clock noise), parses the
`warptiling:` timing + PASS/FAIL, and ranks the valid configs.

The k8 config knobs (must match the #ifndef names in matmul.cu):
    K8_BM, K8_BN, K8_BK, K8_TM, K8_TN, K8_WMX, K8_WITER

Derived geometry (the script enforces all of these so we don't compile/launch garbage):
    WNX  = 32 / WMX            (lanes of a warp along N; WMX along M, WMX*WNX = 32)
    WM   = WMX * TM            (warp-tile height, in C rows)
    WN   = WITER * WNX * TN    (warp-tile width,  in C cols)
    threads/block = BM*BN / (TM*TN*WITER)

Usage:
    python3 sweep_k8.py                 # full sweep with the default search space
    python3 sweep_k8.py --runs 6        # more repeats per config (less noise, slower)
    python3 sweep_k8.py --cooldown 1.5  # sleep between configs to fight throttling
    python3 sweep_k8.py --conflict-free # only try configs with zero bank conflicts
    python3 sweep_k8.py --csv out.csv   # also dump every result to CSV
"""

import argparse
import itertools
import re
import subprocess
import time

SRC = "matmul.cu"
ARCH = "sm_120"
BIN = "/tmp/mm_sweep_k8"
SMEM_LIMIT = 48 * 1024  # static __shared__ cap (bytes). Beyond this needs opt-in dynamic smem.
MAX_THREADS = 1024
AS_PAD = 4  # set to 0 if your As is NOT padded; only affects the smem estimate

# ---------------------------------------------------------------------------
# Search space - edit freely. Keep it small-ish; each config costs one nvcc build.
# ---------------------------------------------------------------------------
SPACE = dict(
    BM=[64, 128, 256],
    BN=[64, 128, 256],
    BK=[8, 16, 32],
    TM=[4, 8],
    TN=[4, 8],
    WMX=[2, 4, 8],
    WITER=[1, 2, 4],
)


def geometry(bm, bn, bk, tm, tn, wmx, witer):
    """Return derived geometry + a list of reasons the config is invalid (empty = valid)."""
    reasons = []
    if 32 % wmx:
        return None, ["WMX must divide 32"]
    wnx = 32 // wmx
    wm = wmx * tm
    wn = witer * wnx * tn
    threads = (bm * bn) // (tm * tn * witer)
    smem = (bk * (bm + AS_PAD) + bk * bn) * 4

    if bk % 4:                 reasons.append("BK%4")
    if bn % 4:                 reasons.append("BN%4 (float4 B load)")
    if bm % tm:                reasons.append("BM%TM")
    if bn % tn:                reasons.append("BN%TN")
    if bm % wm:                reasons.append(f"BM%WM (WM={wm})")
    if bn % wn:                reasons.append(f"BN%WN (WN={wn})")
    if (bm * bn) % (tm * tn * witer): reasons.append("threads not integral")
    if threads % 32:           reasons.append(f"threads%32 (threads={threads})")
    if threads > MAX_THREADS:  reasons.append(f"threads>{MAX_THREADS} ({threads})")
    if threads < 32:           reasons.append(f"threads<32 ({threads})")
    if smem > SMEM_LIMIT:      reasons.append(f"smem>{SMEM_LIMIT} ({smem})")

    g = dict(wnx=wnx, wm=wm, wn=wn, threads=threads, smem=smem,
             conflict_free=(wnx * tn <= 32 and wmx * tm <= 32))
    return g, reasons


def ptxas_regs(defs):
    """Compile and return (ok, regs, spill_bytes) for sgemm_8_warptiling, or (False,..) on build error."""
    r = subprocess.run(
        ["nvcc", "-O3", f"-arch={ARCH}", "--ptxas-options=-v", *defs, SRC, "-o", BIN, "-lcublas"],
        capture_output=True, text=True)
    if r.returncode != 0:
        return False, None, None
    txt = r.stdout + r.stderr
    # find the ptxas block for our kernel
    regs = spill = None
    blocks = txt.split("Compiling entry function")
    for b in blocks:
        if "sgemm_8_warptiling" in b:
            m = re.search(r"Used (\d+) registers", b)
            if m: regs = int(m.group(1))
            s = re.search(r"(\d+) bytes spill stores", b)
            if s: spill = int(s.group(1))
            break
    return True, regs, spill


def run_binary(runs):
    """Run the compiled binary `runs` times; return (verdict, best_k8_us, best_cublas_us)."""
    best_k8 = best_cub = None
    verdict = "NO_OUTPUT"
    for _ in range(runs):
        p = subprocess.run([BIN], capture_output=True, text=True)
        out = p.stdout
        if p.returncode != 0:
            verdict = "CRASH"
            continue
        m = re.search(r"^warptiling: (\d+) us", out, re.M)
        if m:
            t = int(m.group(1))
            best_k8 = t if best_k8 is None else min(best_k8, t)
        c = re.search(r"^cublas: (\d+) us", out, re.M)
        if c:
            t = int(c.group(1))
            best_cub = t if best_cub is None else min(best_cub, t)
        v = re.search(r"^warptiling: (PASS|FAIL)", out, re.M)
        if v:
            verdict = v.group(1)
    return verdict, best_k8, best_cub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4, help="repeats per config (best-of-N)")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to sleep between configs")
    ap.add_argument("--conflict-free", action="store_true", help="only sweep zero-conflict configs")
    ap.add_argument("--csv", type=str, default=None, help="write all results to this CSV")
    args = ap.parse_args()

    keys = ["BM", "BN", "BK", "TM", "TN", "WMX", "WITER"]
    combos = list(itertools.product(*(SPACE[k] for k in keys)))

    # pre-filter for validity
    valid = []
    for c in combos:
        cfg = dict(zip(keys, c))
        g, reasons = geometry(**{k.lower(): v for k, v in cfg.items()})
        if reasons:
            continue
        if args.conflict_free and not g["conflict_free"]:
            continue
        valid.append((cfg, g))

    print(f"{len(combos)} combos, {len(valid)} pass the geometry constraints "
          f"({'conflict-free only' if args.conflict_free else 'including conflicting'}).\n")

    results = []
    for i, (cfg, g) in enumerate(valid, 1):
        defs = [f"-DK8_{k}={v}" for k, v in cfg.items()]
        tag = " ".join(f"{k}={v}" for k, v in cfg.items())
        ok, regs, spill = ptxas_regs(defs)
        if not ok:
            print(f"[{i}/{len(valid)}] BUILD_ERR | {tag}")
            continue
        verdict, k8, cub = run_binary(args.runs)
        cf = "CF" if g["conflict_free"] else "conf"
        ratio = (k8 / cub) if (k8 and cub) else None
        results.append(dict(cfg=cfg, **g, regs=regs, spill=spill,
                            verdict=verdict, k8=k8, cublas=cub, ratio=ratio))
        rstr = f"{k8}us" if k8 else "-"
        ratiostr = f"{ratio:.3f}x" if ratio else "-"
        print(f"[{i}/{len(valid)}] {verdict:5s} {rstr:>7s} ({ratiostr} cublas) "
              f"regs={regs} thr={g['threads']} smem={g['smem']//1024}K {cf} | {tag}")
        if args.cooldown:
            time.sleep(args.cooldown)

    # ranking: only PASS configs with a timing
    good = [r for r in results if r["verdict"] == "PASS" and r["k8"]]
    good.sort(key=lambda r: r["ratio"] if r["ratio"] else r["k8"])
    print("\n================ TOP VALID (PASS) CONFIGS ================")
    print(f"{'k8':>6} {'ratio':>7} {'regs':>4} {'thr':>4} {'CF':>4}  config")
    for r in good[:15]:
        tag = " ".join(f"{k}={v}" for k, v in r["cfg"].items())
        print(f"{r['k8']:>5}u {r['ratio']:.3f}x {r['regs']:>4} {r['threads']:>4} "
              f"{'yes' if r['conflict_free'] else 'no':>4}  {tag}")

    fails = [r for r in results if r["verdict"] in ("FAIL", "CRASH")]
    if fails:
        print(f"\n{len(fails)} configs FAILED/CRASHED (correctness bug at that config - "
              f"e.g. the totalThreads÷WITER issue surfaces at larger BK).")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["BM", "BN", "BK", "TM", "TN", "WMX", "WITER",
                        "verdict", "k8_us", "cublas_us", "ratio", "regs", "spill",
                        "threads", "WM", "WN", "smem", "conflict_free"])
            for r in results:
                c = r["cfg"]
                w.writerow([c["BM"], c["BN"], c["BK"], c["TM"], c["TN"], c["WMX"], c["WITER"],
                            r["verdict"], r["k8"], r["cublas"], r["ratio"], r["regs"], r["spill"],
                            r["threads"], r["wm"], r["wn"], r["smem"], r["conflict_free"]])
        print(f"\nWrote {len(results)} rows to {args.csv}")


if __name__ == "__main__":
    main()
