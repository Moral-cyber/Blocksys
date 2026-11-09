# Three-partition development comparison

Frozen original post-hoc protocol: fixed lambda 10, TRAIN fitting and designated client holdout scoring. These are development results; the protected original TEST values are not mixed into this table. Each partition uses the same three model-seed numbers. All eighteen configurations are retained.

| Task | Partition | Zero NLL mean ± SD | Prior NLL mean ± SD | Δ NLL | Δ macro accuracy (pp) | NLL improved seeds | Improved client-seed cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| cifar | 20261217 | 0.653559 ± 0.014043 | 0.612447 ± 0.013954 | -0.041112 | +0.0731 | 3/3 | 60/60 |
| cifar | 20261105 | 0.691176 ± 0.017000 | 0.652115 ± 0.017437 | -0.039062 | +0.0928 | 3/3 | 57/60 |
| cifar | 20261106 | 0.631376 ± 0.021375 | 0.585063 ± 0.021274 | -0.046313 | +0.1684 | 3/3 | 60/60 |
| fashion | 20261219 | 0.180172 ± 0.012122 | 0.166384 ± 0.011603 | -0.013788 | +0.2919 | 3/3 | 60/60 |
| fashion | 20261105 | 0.220082 ± 0.012545 | 0.209755 ± 0.012321 | -0.010327 | +0.1663 | 3/3 | 58/60 |
| fashion | 20261106 | 0.167248 ± 0.009793 | 0.156392 ± 0.008740 | -0.010856 | +0.1006 | 3/3 | 58/60 |

The SD above describes the three model runs conditional on a partition. It is not a partition-level confidence interval. Negative Δ NLL favors the prior center. No significance or universal generalization claim is inferred from these cells.

## Every development configuration

| Task | Partition | Model seed | Zero macro NLL | Prior macro NLL | Δ NLL | Zero macro acc (%) | Prior macro acc (%) | Zero weighted NLL | Prior weighted NLL | Zero weighted acc (%) | Prior weighted acc (%) | Improved clients |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar | 20261217 | 20261231 | 0.638950 | 0.599279 | -0.039670 | 76.8313 | 77.0238 | 0.649950 | 0.625763 | 76.2240 | 76.2160 | 20/20 |
| cifar | 20261217 | 20261232 | 0.654771 | 0.610988 | -0.043783 | 76.3806 | 76.4767 | 0.660827 | 0.634873 | 76.0400 | 75.9360 | 20/20 |
| cifar | 20261217 | 20261233 | 0.666956 | 0.627073 | -0.039884 | 75.8800 | 75.8107 | 0.679092 | 0.654761 | 74.9920 | 75.0400 | 20/20 |
| cifar | 20261105 | 20261231 | 0.674073 | 0.635205 | -0.038869 | 76.0668 | 76.2287 | 0.653318 | 0.631135 | 75.2820 | 75.2740 | 19/20 |
| cifar | 20261105 | 20261232 | 0.691384 | 0.651105 | -0.040279 | 76.0219 | 76.0698 | 0.667270 | 0.644239 | 74.9940 | 74.8820 | 19/20 |
| cifar | 20261105 | 20261233 | 0.708072 | 0.670035 | -0.038037 | 74.8290 | 74.8975 | 0.681465 | 0.659381 | 74.0981 | 74.1861 | 19/20 |
| cifar | 20261106 | 20261231 | 0.614248 | 0.568117 | -0.046131 | 78.5998 | 78.6510 | 0.578911 | 0.555572 | 79.1617 | 79.1457 | 20/20 |
| cifar | 20261106 | 20261232 | 0.624550 | 0.578134 | -0.046417 | 77.9513 | 78.3128 | 0.590044 | 0.566472 | 78.7777 | 78.8577 | 20/20 |
| cifar | 20261106 | 20261233 | 0.655330 | 0.608938 | -0.046393 | 76.9055 | 76.9980 | 0.618292 | 0.595129 | 77.6978 | 77.8098 | 20/20 |
| fashion | 20261219 | 20261241 | 0.166183 | 0.152989 | -0.013194 | 94.4415 | 94.7626 | 0.138485 | 0.130266 | 95.3930 | 95.4797 | 20/20 |
| fashion | 20261219 | 20261242 | 0.187565 | 0.172826 | -0.014740 | 93.7658 | 93.9378 | 0.154264 | 0.145269 | 94.9263 | 95.0197 | 20/20 |
| fashion | 20261219 | 20261243 | 0.186767 | 0.173338 | -0.013429 | 93.7423 | 94.1250 | 0.158078 | 0.149310 | 94.6730 | 94.7797 | 20/20 |
| fashion | 20261105 | 20261241 | 0.205596 | 0.195531 | -0.010066 | 92.5445 | 92.7327 | 0.173452 | 0.165792 | 93.7600 | 93.8933 | 19/20 |
| fashion | 20261105 | 20261242 | 0.227391 | 0.216658 | -0.010733 | 91.5409 | 91.6409 | 0.191671 | 0.183565 | 92.9600 | 93.0133 | 20/20 |
| fashion | 20261105 | 20261243 | 0.227259 | 0.217077 | -0.010182 | 91.3921 | 91.6027 | 0.192910 | 0.185019 | 92.8667 | 92.9867 | 19/20 |
| fashion | 20261106 | 20261241 | 0.156241 | 0.146480 | -0.009760 | 94.5477 | 94.6146 | 0.145665 | 0.138828 | 94.8470 | 94.8403 | 19/20 |
| fashion | 20261106 | 20261242 | 0.170507 | 0.159697 | -0.010810 | 94.0986 | 94.1869 | 0.160074 | 0.152468 | 94.3670 | 94.4537 | 20/20 |
| fashion | 20261106 | 20261243 | 0.174996 | 0.162997 | -0.011998 | 93.8873 | 94.0339 | 0.163101 | 0.155023 | 94.2471 | 94.3670 | 19/20 |

Source paths and SHA-256 identities for the original probe and all twelve new evaluations are in `combined_development.json`. This script reads saved development scores only; it does not fit predictors, forward images, modify checkpoints, or read official TEST.
