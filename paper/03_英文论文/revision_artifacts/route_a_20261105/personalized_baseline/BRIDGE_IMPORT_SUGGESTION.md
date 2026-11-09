# Suggested final-evidence bridge import

Prepared from completed, frozen pFedMe scores. This note does **not** execute the import; root will run bridge mutations sequentially.

`result_pfedme.json` is the compact final-evidence record; `result_all_metrics.csv` keeps all six seeds and three-seed mean/sample SD metrics. Both filenames satisfy the bridge's result/metrics contract. Their numerical sources and 43-input model lock are pinned by SHA-256. This is **final** evidence and must not be imported as development evidence for candidate selection.

```sh
/Users/wind/Desktop/Codex/paper/06_ScientistTwo/upstream/Open-ScientistTwo/.venv/bin/python \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/bridge.py import-experiment \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/campaigns/fl_heterogeneity_20261105-v1 \
  --id pfedme-final-sixrun-20261105 \
  --file /private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/personalized_baseline/result_pfedme.json \
  --file /private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/personalized_baseline/result_all_metrics.csv \
  --run-receipt /private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/personalized_baseline/matrix_execution/run_20261105T151515412446Z.json \
  --device mixed \
  --scope final \
  --note 'Actual six-run pFedMe MPS training completed; 43-input model lock preceded CPU official-TEST scoring of raw final personalized theta. Independent exported-logit replay PASS, max discrepancy 1.47793e-12. Author optimizer with matched task adapter, outer lr .005 to .03, K5 inner gradients; not full author-code reproduction or state of the art. Final evidence is retained for comparison and not candidate selection. Import stores existing bytes and performs no training or scoring.'

/Users/wind/Desktop/Codex/paper/06_ScientistTwo/upstream/Open-ScientistTwo/.venv/bin/python \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/bridge.py check \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/campaigns/fl_heterogeneity_20261105-v1
```

The existing run receipt covers MPS training and the model-lock command; its status explicitly says it does not score TEST. Actual completed TEST evidence is provided by the imported result JSON/source hashes, and independently checked in `INDEPENDENT_SCORE_VERIFICATION.json`. The bridge import records bytes and provenance; it does not rerun or scientifically certify the experiment. Avoid re-running the import after its immutable ID has been used. Store the real bridge receipts from root's actual commands rather than writing a planned import as completed.
