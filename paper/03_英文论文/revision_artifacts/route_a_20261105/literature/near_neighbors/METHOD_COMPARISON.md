# Near-neighbor method comparison (read-only evidence review)

## PRIOR / pFedBreD: verified full text, real conceptual overlap

Source: https://arxiv.org/html/2310.09183v2 ; local `prior_full.html` and extracted `prior_full.txt`. This is the 2023-11-10 v2 author full text, matched by title/authors to Crossref DOI 10.52202/075280-1234. Crossref says NeurIPS36 (2023), pages28378–28392. `prior_append_only.bib` is an optional append-only entry; no manuscript bibliography was changed.

The broad idea “personalized prior as a regularization center” is already present. Eq8 explicitly optimizes a local objective `f_i(theta_i)+lambda D_{g*}(theta_i,mu_i(w))` inside a global envelope. Eq11 again gives `min_theta {Psi(theta,w)+lambda D_{g*}(theta,mu_Phi(w))}`. Eq12 defines the global problem through this centered Bregman-Moreau envelope. Section5/ Eq14 chooses spherical-Gaussian priors (scaled squared norm) and a learned/meta-step mean `mu_i(w)=w-eta grad Phi_i(w)`. AppendixA.5 explicitly calls this “personalized regularization (or personalized prior in Bayesian learning)”. Algorithm1 alternates center generation, local model optimization, global-model updates and aggregation.

PR-LFL's evidence-backed distinction is narrower and concrete:
- The center is a centered logarithm of observed local class frequencies (ten class-bias coordinates), not a personalized full-model prior mean generated from global weights and meta-gradients.
- The CNN/logits are frozen; the twenty-dimensional diagonal-scale/bias convex fit happens after shared training and does not change the federated optimizer or participate in model aggregation.
- The stationarity identity is fitted class probability mass minus TRAIN label counts, not a new generic MAP/proximal/Bregman principle.
- The class-identity permutation controls and 80B predictor/54,067B shared-message tradeoff are the specific experimental/mechanism story. They do not establish universal novelty of prior-centered regularization or a general non-overlap theorem.

A concise related-work sentence suitable for root consideration:
“PRIOR/pFedBreD learns personalized centers for Bregman-regularized model training; here the center is the observed local log class prior in a convex post-hoc predictor on frozen logits.”

Use `shi2023prior` only after root approves/merges the optional bib. Do not claim the generic prior-as-center concept is new. No evidence here proves that every possible special case of pFedBreD excludes the PR-LFL objective; it is a broad framework. The practical contribution must be the particular information source, parameterization and cost/validation, not the algebraic existence of a center.

## LaSCal: method overlap unresolved in this bounded search

Verified Crossref DOI10.52202/079017-2088 and title/authors; proceedings.com product page has no method abstract/full text. OpenReview endpoint returned403. arXiv exact title searches returned zero; author search found `2312.08586v1`, titled “Estimating calibration error under label shift without labels”, whose abstract proposes importance reweighting of labeled source distribution for consistent/asymptotically unbiased target calibration-error estimation. That related author preprint is not automatically the identical LaSCal paper; do not use it to assert LaSCal does or does not use PR-LFL's quadratic class-bias center. Official ECVA/venue full text remains the necessary source.

## FedNPC: method overlap unresolved in this bounded search

Verified Crossref title “FedNPC: Stochastic Noise-driven Post-hoc Classifier Calibration Method for Federated Long-Tailed Learning”, DOI10.1109/cvprf71909.2026.00743, CVPR2026, author JintongGao etal. Crossref has no abstract. Official IEEE document11697702 returned an empty response; arXiv `all:FedNPC` returned zero records; CVF guessed slug404 and CVPR2026 index timed out. These retrieval failures say nothing about actual overlap. Obtain a real CVF index URL/IEEE abstract before adding any method distinction.

## Artifact scope
The search was bounded to the three requested neighbors. No main tex, existing bib, evidence/checks or protected provenance was changed. HTTP outcomes, URLs and raw retrieved sources are archived here. The pFedMe baseline work continues independently; none of this literature metadata is a baseline experiment result.
