# Sparse Update Benchmark Summary

Source: `sparse_clean_20c_15r_3s.csv`

Final-round mean (standard deviation across seeds):

| Dataset | Alpha | Method | Accuracy | Client acc. std | Min client acc. | Cumulative bytes | Compression ratio | Residual norm |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Cifar10CleanA0p1 | 0.1 | block_topk_ef | 0.4066 (0.0107) | 0.1005 (0.0161) | 0.2506 (0.0528) | 65798400 (0) | 2.497 (0.000) | 0.4181 (0.0294) |
| Cifar10CleanA0p1 | 0.1 | dense | 0.4186 (0.0117) | 0.0986 (0.0121) | 0.2627 (0.0311) | 164268000 (0) | 1.000 (0.000) | 0.0000 (0.0000) |
| Cifar10CleanA0p1 | 0.1 | global_topk | 0.4093 (0.0090) | 0.0947 (0.0118) | 0.2500 (0.0269) | 65716800 (0) | 2.500 (0.000) | 0.0000 (0.0000) |
| Cifar10CleanA0p1 | 0.1 | global_topk_ef | 0.4102 (0.0108) | 0.1008 (0.0215) | 0.2557 (0.0659) | 65716800 (0) | 2.500 (0.000) | 0.3562 (0.0329) |

Interpretation guardrail: this summary reports measured outcomes only; it does not establish that novelty scoring protects minority classes or improves heterogeneous FL in general.
