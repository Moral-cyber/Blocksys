# TopK two-method saved-array adapter fixture

PASS: fabricated 2 methods × 20 clients, seven variants. This is synthetic integration evidence only.

- Wrapper SHA256: `867b7145c934978be60c7f406b9e1d1b46b0fb27cd3dc7dd164f62a37da858a4`
- Harness SHA256: `e518716f918b536ca16e2c7fce2b5ef0735d8d2b08379be7cb2d88ba3ae2e501`
- Scratch arrays and full output: `/Users/wind/Documents/Codex/2026-11-02/zhe/work/topk_two_prior_vector_ed5luvb6`
- Independent maximum absolute errors: `{'metric': 8.881784197001252e-16, 'summary': 3.552713678800501e-15, 'ece_bin': 4.440892098500626e-16}`
- Checks: `{'client_variant_metrics': 1400, 'summary_values': 168, 'ece_bin_values': 25200, 'deployed_vector_reconstructions': 80}`
- 40 prior-vector + 40 zero-vector numerical fits, profiler-verified saved TRAIN inputs only.
- FP32 deployed states independently reconstructed with FP64 score application.
- NLL/loss alias, class-summed Brier, ECE15 bins, macro/weighted summaries, pooled ECE, q10 checked.
- Exact external PLAN and registration mandatory; wrong identities, incomplete/reordered clients, incomplete methods, non-DONE, existing output rejected.
- No gates/decision, CNN import/forward/backward, real image bytes, actual saved arrays or formal training.
- Frozen evaluator/solver/reader, main protocol, TopK original/prospective sources unchanged.
- External separate-family review: REVIEW_UNAVAILABLE.

Real-array invocation for root (not executed by this fixture):

```sh
/opt/anaconda3/envs/Tinyproto/bin/python work/calibrated_topk_reference/evaluate_saved_prior_vector.py \
  --input-root <complete_topk_saved_arrays> --out <fresh_prior_vector_output> \
  --protocol-plan work/predictive_pfl_confirm/PLAN.md \
  --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json
```
