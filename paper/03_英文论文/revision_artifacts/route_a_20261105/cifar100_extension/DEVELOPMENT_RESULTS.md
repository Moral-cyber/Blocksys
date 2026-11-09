# Complete CIFAR-100 development results

One partition (20261107), three fixed model seeds, final round20; no TEST scoring or choice. This is an added-dataset check of the same fixed method, not a new mechanism or state-of-the-art comparison.

| Model seed | Zero macro NLL | Prior macro NLL | Prior-minus-zero NLL | Zero macro acc (%) | Prior macro acc (%) |
|---:|---:|---:|---:|---:|---:|
| 20261231 | 2.414892 | 2.276566 | -0.138326 | 35.6822 | 35.2217 |
| 20261232 | 2.435797 | 2.294607 | -0.141190 | 35.2223 | 35.3416 |
| 20261233 | 2.434240 | 2.297562 | -0.136678 | 34.8967 | 34.7999 |

NLL improves in 3/3 model runs and 60/60 client-seed cells. The mean prior-minus-zero NLL is -0.138731. Every arm/client, Brier/ECE and macro/weighted metric remains in JSON/CSV. The 800 B predictor excludes local fit memory and the FP32 EF residual.
