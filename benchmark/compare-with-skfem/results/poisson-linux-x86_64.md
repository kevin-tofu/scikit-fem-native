# skfn vs. scikit-fem: Poisson assembly

Generated: 2026-07-31T14:17:33.975512+00:00

Environment: `Python 3.12.12, NumPy 2.5.1, SciPy 1.18.0, skfn 0.1.0, scikit-fem 12.0.2, Linux-6.8.0-136-generic-x86_64-with-glibc2.35`

Warm-cache median timings; solve time is excluded.

| DoFs | Elements | skfn K [ms] | skfem K [ms] | K speedup | skfn f [ms] | skfem f [ms] | f speedup | total speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 289 | 512 | 0.114 | 0.507 | 4.43x | 0.072 | 0.145 | 2.00x | 3.49x |
| 1089 | 2048 | 0.335 | 1.015 | 3.03x | 0.134 | 0.176 | 1.31x | 2.54x |
| 4225 | 8192 | 1.234 | 4.162 | 3.37x | 0.385 | 0.262 | 0.68x | 2.73x |
| 16641 | 32768 | 4.846 | 16.932 | 3.49x | 1.364 | 0.739 | 0.54x | 2.85x |
| 66049 | 131072 | 19.805 | 58.554 | 2.96x | 5.618 | 3.002 | 0.53x | 2.42x |

Basis construction:

| DoFs | skfn [ms] | skfem [ms] | speedup |
|---:|---:|---:|---:|
| 289 | 41.130 | 0.870 | 0.02x |
| 1089 | 143.198 | 1.180 | 0.01x |
| 4225 | 582.240 | 3.715 | 0.01x |
| 16641 | 2338.934 | 14.234 | 0.01x |
| 66049 | 9319.100 | 46.406 | 0.00x |
