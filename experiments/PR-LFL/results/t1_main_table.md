# P0 existing-data results

One fixed partition and shared fitted model per seed. Client-seed pairs are repeated, dependent observations, not sixty independent training replicates. Pooled Wilcoxon p-values are nominal descriptive calculations, not valid population-level evidence of sixty independent replications. Client-cluster percentile intervals describe variation over observed client clusters conditional on these three seeds and the fixed partition; they omit new partition and training-run uncertainty.

## cifar

NLL paired differences: 60/60 improve, 0/60 worsen, median prior-minus-zero = -0.02570773; nominal pooled Wilcoxon two-sided p = 1.7347235e-18. All 3/3 seed-level means improve. Client-cluster bootstrap (1000 draws; three seeds kept within each client): mean delta -0.03575086, 95% percentile interval [-0.05248377, -0.02328366]. 20/20 client means improve.

| Endpoint | Zero center | Prior center | Difference |
|---|---:|---:|---:|
| macro_accuracy | 0.7539034671 | 0.7556805534 | +0.0017770863 |
| weighted_accuracy | 0.7475333333 | 0.7471333333 | -0.0004000000 |
| macro_nll | 0.6803677623 | 0.6446169008 | -0.0357508614 |
| weighted_nll | 0.6787784958 | 0.6570664934 | -0.0217120024 |
| client_accuracy_q10 | 0.6027078966 | 0.6065036443 | +0.0037957477 |

## fashion

NLL paired differences: 60/60 improve, 0/60 worsen, median prior-minus-zero = -0.01185078; nominal pooled Wilcoxon two-sided p = 1.7347235e-18. All 3/3 seed-level means improve. Client-cluster bootstrap (1000 draws; three seeds kept within each client): mean delta -0.01823567, 95% percentile interval [-0.02734456, -0.01089916]. 20/20 client means improve.

| Endpoint | Zero center | Prior center | Difference |
|---|---:|---:|---:|
| macro_accuracy | 0.9280099419 | 0.9297399369 | +0.0017299950 |
| weighted_accuracy | 0.9461666667 | 0.9472666667 | +0.0011000000 |
| macro_nll | 0.2062309628 | 0.1879952973 | -0.0182356655 |
| weighted_nll | 0.1588086046 | 0.1486097222 | -0.0101988824 |
| client_accuracy_q10 | 0.8577691759 | 0.8566495009 | -0.0011196750 |

Sample-weighted accuracy equals global pooled test accuracy for each seed, because every original TEST example is assigned once. Macro averages weight clients equally, whereas sample weighting uses their assigned TEST sizes. This difference in client weighting changes the measured contribution of each client; it should not be described as proof that minority-class scarcity alone caused the difference.

Caption safety: every client improves is false for these main-control results if any mean or seed-level pair is positive. Plot actual prior-minus-zero differences around a zero reference line and label the observed count.

No P1/P2 experiment was performed by this script.
