# Scope addendum from the user

2026-11-04, before any official TEST access. The user specifies that three training seeds are sufficient for this conference paper. Use exactly the three declared seeds per task; do not add more seeds.

The finite paper scope is two tasks (CIFAR-10 and FashionMNIST), one declared alpha0.1 partition per task, 20 rounds, paired CE/FedLC and dense/low-bit controls, TRAIN-only calibration deletions and restricted alternatives. Three paired TopK and FedRoD-linear reference runs are evaluated on CIFAR-10. They are not presented as two-task full reproductions.

The original prospective NLL criteria remain unchanged. Additional moderate-alpha and longer-training experiments described as later follow-up in PLAN.md are deferred from this finite paper; no claim of robustness across alpha, round counts, architectures or participation regimes is permitted. No new training hyperparameter search. A lambda sensitivity on saved TRAIN logits can be shown only as a diagnostic with all fixed values reported and without replacing the lambda10 candidate. It is optional, not a gate to a new claimed mechanism.

The final model set and TRAIN-fitted states must be locked before official TEST access. TEST outcomes are reported without further model selection. Negative developer records remain archived. The new scope does not rewrite any STOP or redefine any previously assessed endpoint.
