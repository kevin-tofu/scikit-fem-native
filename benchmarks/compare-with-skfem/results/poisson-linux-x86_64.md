# skfemntv vs. scikit-fem: Poisson assembly

Generated: 2026-07-31T15:13:28.351151+00:00

Environment: `Python 3.12.12, NumPy 2.5.1, SciPy 1.18.0, skfemntv 0.1.0, scikit-fem 12.0.2, Linux-6.8.0-136-generic-x86_64-with-glibc2.35`

Warm-cache median of 7 timed runs after 2 warm-up runs; solve time is excluded.

| DoFs | Elements | skfemntv K [ms] | skfemntv K parallel [ms] | threads | skfem K [ms] | K speedup | parallel K speedup | skfemntv f [ms] | skfemntv f parallel [ms] | threads | skfem f [ms] | skfemntv f speedup | parallel f speedup | skfemntv total speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 289 | 512 | 0.118 | 0.120 | 4 | 0.510 | 4.33x | 4.24x | 0.053 | 0.057 | 4 | 0.147 | 2.77x | 2.60x | 3.85x |
| 1089 | 2048 | 0.351 | 0.350 | 4 | 1.030 | 2.93x | 2.94x | 0.088 | 0.263 | 4 | 0.180 | 2.05x | 0.68x | 2.76x |
| 4225 | 8192 | 1.299 | 0.738 | 4 | 3.100 | 2.39x | 4.20x | 0.242 | 0.259 | 4 | 0.265 | 1.09x | 1.02x | 2.18x |
| 16641 | 32768 | 5.110 | 2.013 | 4 | 11.869 | 2.32x | 5.90x | 0.856 | 0.489 | 4 | 0.747 | 0.87x | 1.53x | 2.12x |
| 66049 | 131072 | 20.745 | 7.708 | 4 | 50.285 | 2.42x | 6.52x | 3.567 | 2.283 | 4 | 3.608 | 1.01x | 1.58x | 2.22x |
| 263169 | 524288 | 83.540 | 32.766 | 4 | 228.109 | 2.73x | 6.96x | 14.034 | 4.373 | 4 | 15.363 | 1.09x | 3.51x | 2.50x |
| 1050625 | 2097152 | 334.092 | 133.080 | 4 | 986.948 | 2.95x | 7.42x | 55.221 | 17.437 | 4 | 94.526 | 1.71x | 5.42x | 2.78x |

Basis construction:

| DoFs | skfemntv 1 thread [ms] | skfemntv parallel [ms] | threads | skfem [ms] | speedup |
|---:|---:|---:|---:|---:|---:|
| 289 | 0.541 | 0.450 | 4 | 0.848 | 1.57x |
| 1089 | 0.840 | 1.131 | 4 | 1.269 | 1.51x |
| 4225 | 4.358 | 3.213 | 4 | 4.071 | 0.93x |
| 16641 | 15.968 | 12.317 | 4 | 14.693 | 0.92x |
| 66049 | 66.336 | 52.929 | 4 | 46.299 | 0.70x |
| 263169 | 253.927 | 205.531 | 4 | 168.523 | 0.66x |
| 1050625 | 1180.653 | 975.396 | 4 | 640.917 | 0.54x |
