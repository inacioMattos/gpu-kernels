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

TODO: better timing harness, more matmul edge cases.
