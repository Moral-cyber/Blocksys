# Primary sources and retrieval record

Checked on 2026-11-04. Two works only; three PDFs because the FedMPQ preprint was subsequently checked against the formal CVPR paper. No training, no author-code or third-party implementation read. Page numbers below refer to PDF page indices starting at 1.

## 1. FedFQ

Haowei Li, Weiying Xie, Hangyu Ye, Jitao Ma, Shuran Ma, Yunsong Li. **FedFQ: Federated Learning with Fine-Grained Quantization.** arXiv:2408.08977v1, 16 Aug 2024.

- Primary: https://arxiv.org/pdf/2408.08977
- Saved: `FedFQ_primary.pdf`, extracted `FedFQ_primary.txt` (9 pages).
- Page 3, Eq.4: compression of local update h=theta_local-theta_global.
- Page 3, Eqs.5–7: stochastic uniform quantizer, unbiasedness, variance bound q=d/4^b.
- Page 3, Eqs.10–12: mixed quantizer variance coefficient q_f=sum_j d*|h_j|^2/(4^(b_j)*||h||^2).
- Page 4, Eq.17: minimize q_f subject to sum_j b_j=B.
- Page 4, Algorithm 1: CGSA with coordinate choices {0,2,4,8}; amplitude-sort initialization; objective based on squared update values. Page 5 explains raising precision for large components, reducing it for small components.
- Pages 3 and 4 rendered and visually checked: `FedFQ_page3.png`, `FedFQ_page4.png`. Confirmed 4^(b_j), not 4*b_j.
- Page 5 setup: 100 clients, 10 selected per round, 5 local updates, CIFAR10 batch50/lr0.15; these are not the current experimental settings. CGSA initial temperature1000, cooling0.95, 100 iterations in reported experiment.
- Original budget equation counts sum of bit-widths. It is not by itself an actual serialized-byte accounting specification for an implementation with bit maps/scales/headers/indices.

## 2. FedMPQ (mixed precision, not the homonymous product-quantization paper)

Huancheng Chen and Haris Vikalo. **Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices.** CVPR 2024, pp. 6138–6148. DOI:10.1109/CVPR52733.2024.00587. Method name in paper: FedMPQ.

- Formal primary: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Mixed-Precision_Quantization_for_Federated_Learning_on_Resource-Constrained_Heterogeneous_Devices_CVPR_2024_paper.pdf
- Saved: `FedMPQ_CVPR2024_primary.pdf`, extracted `FedMPQ_CVPR2024_primary.txt` (11 pages).
- Initially retrieved same-work preprint: https://arxiv.org/pdf/2311.18129 ; saved `FedMPQ_primary.pdf`, `FedMPQ_primary.txt`. Final formula and algorithm claims were cross-checked against formal paper.
- Page 3, Eq.2: average precision constraint b_n dot m / ||m||_1 <= v_n, where m is number of parameters per layer.
- Page 4, Eqs.11–13: group Lasso sum over bit-plane L2 norms; layer weights M_l/M; sparsity delta=bit-plane L0 count/M_l; prune MSBs when delta<=epsilon.
- Page 5, Eq.15 and Algorithm 1: client models converted back to full precision, global model and bit-widths aggregated; p_n proportional to v_n*|D_n|. This modifies aggregation relative to standard sample-weighted FedAvg when budgets differ.
- Page 5 prose: layer-wise precision retained after sparse training is interpreted as sensitivity/importance under local data distribution. It does not define a gradient-square or Hessian score for bit allocation.
- Page 6, Algorithm 2: order d=argDescending(m elementwise*(Delta b +1)); prune earlier in order, grow from reverse order, with nominal bit limits1–8. The growing loop increments while current average v<v_n, so may cross v_n on the last step; do not copy it as proof of a strict actual-byte cap. Report deliberately says it adjusts around the average-bit-width budget.
- Page 6 setup: epsilon0.03, lambda0.01, SGD lr0.1/momentum0.9/weightdecay0.0005, batch64, 50 rounds, 5 local epochs, 10 clients, fraction0.5. Main comparison fixes activations at4bits and simulates heterogeneous client budgets.

## Search record

OpenAlex endpoints (no API key needed for these successful public requests):

1. https://api.openalex.org/works?search=FedFQ&per_page=3
   - Raw JSON saved as `search_fedfq_openalex.json`.
   - Selected exact matching title, arXiv DOI 10.48550/arXiv.2408.08977, OpenAlex W4402502962.
2. https://api.openalex.org/works?search=FedMPQ&per_page=3
   - Raw JSON saved as `search_fedmpq_openalex.json`.
   - Selected mixed-precision CVPR work, DOI 10.1109/CVPR52733.2024.00587, OpenAlex W4402754051.
   - Search also returns a homonymous multi-codebook product-quantization work. Its full text was not read or used; scope remained two works.

LITERATURE_CHECK.md suggestions about a gradient-square proxy, 256-coordinate blocks, 2/3/4-bit greedy allocation, EF, actual-byte accounting and matched codec baselines are comparison advice for the current candidate, not claimed author implementations or proven convergence guarantees.
