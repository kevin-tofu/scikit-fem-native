# skfemntv vs. scikit-fem: Poisson assembly

Generated: 2026-08-02T15:43:25.157221+00:00

Environment: `Python 3.12.12, NumPy 2.5.1, SciPy 1.18.0, skfemntv 0.1.0, scikit-fem 12.0.2, Linux-6.8.0-136-generic-x86_64-with-glibc2.35`

Warm-cache median of 1 timed runs after 2 warm-up runs; solve time is excluded.

| DoFs | Elements | skfemntv K [ms] | skfemntv K parallel [ms] | threads | skfem K [ms] | K speedup | parallel K speedup | skfemntv f [ms] | skfemntv f parallel [ms] | threads | skfem f [ms] | skfemntv f speedup | parallel f speedup | skfemntv total speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 289 | 512 | 0.128 | 0.130 | 4 | 0.524 | 4.10x | 4.04x | 0.067 | 0.394 | 4 | 0.184 | 2.74x | 0.47x | 3.63x |
| 1089 | 2048 | 0.351 | 0.648 | 4 | 1.061 | 3.02x | 1.64x | 0.109 | 0.231 | 4 | 0.202 | 1.85x | 0.87x | 2.75x |
| 4225 | 8192 | 1.270 | 0.979 | 4 | 3.878 | 3.05x | 3.96x | 0.278 | 0.321 | 4 | 0.298 | 1.07x | 0.93x | 2.70x |
| 16641 | 32768 | 5.033 | 2.230 | 4 | 15.944 | 3.17x | 7.15x | 0.917 | 0.537 | 4 | 1.577 | 1.72x | 2.94x | 2.94x |
| 66049 | 131072 | 20.096 | 7.978 | 4 | 58.688 | 2.92x | 7.36x | 3.415 | 1.361 | 4 | 6.326 | 1.85x | 4.65x | 2.77x |
| 263169 | 524288 | 80.093 | 32.563 | 4 | 226.491 | 2.83x | 6.96x | 13.298 | 4.516 | 4 | 15.990 | 1.20x | 3.54x | 2.60x |
| 1050625 | 2097152 | 321.605 | 134.251 | 4 | 1007.764 | 3.13x | 7.51x | 53.182 | 18.624 | 4 | 102.570 | 1.93x | 5.51x | 2.96x |

Basis construction:

| DoFs | skfemntv 1 thread [ms] | skfemntv parallel [ms] | threads | skfem [ms] | speedup |
|---:|---:|---:|---:|---:|---:|
| 289 | 0.267 | 0.279 | 4 | 0.597 | 2.23x |
| 1089 | 0.653 | 0.458 | 4 | 1.034 | 1.58x |
| 4225 | 2.276 | 1.083 | 4 | 2.729 | 1.20x |
| 16641 | 7.423 | 3.869 | 4 | 9.837 | 1.33x |
| 66049 | 25.754 | 13.055 | 4 | 36.459 | 1.42x |
| 263169 | 95.461 | 43.031 | 4 | 139.101 | 1.46x |
| 1050625 | 361.732 | 162.724 | 4 | 544.621 | 1.51x |
