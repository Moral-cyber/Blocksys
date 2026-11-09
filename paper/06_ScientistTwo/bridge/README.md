# Native agent / host-MPS evidence bridge

This executable adapter uses the installed Open-ScientistTwo Python package's
`Store`, `ResearchPolicy`, `profile_updates("paper-methodology")`,
`LimitationAssessment`, `Seeds`, and `Hypothesis`. It stores local campaign state
in `campaign.sqlite`, immutable content-addressed artifacts, stage events,
native-origin attestations, imported experiment receipts, and receipts for every
actual bridge command.

This is a **local native/host adapter, not an upstream live campaign**. The upstream
package is installed in `../upstream/Open-ScientistTwo/.venv` with Python 3.12.
At setup, Codex CLI reported `Not logged in`, and Docker was unavailable. The
original CLI/API/container live path remains disabled. This bridge never invokes
Codex CLI or an API, starts training, dispatches an agent automatically, creates
measurements, edits experiment artifacts, or issues an acceptance decision.

## Run

Use the installed upstream interpreter directly; no new dependency installation:

```sh
/Users/wind/Desktop/Codex/paper/06_ScientistTwo/upstream/Open-ScientistTwo/.venv/bin/python \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/bridge.py --help
```

All campaign directories must be below `bridge/campaigns`. Every command prints
JSON and returns 0 on success or 2 on rejection. `init` requires a new/empty
directory; records cannot be overwritten. A failed operation in an initialized
campaign also receives a real bridge execution receipt.

```sh
# Optional --context ID=/absolute/path freezes a local 06_ScientistTwo text input.
../upstream/Open-ScientistTwo/.venv/bin/python bridge.py init \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/campaigns/fl-native \
  --name fl-native

../upstream/Open-ScientistTwo/.venv/bin/python bridge.py status \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/campaigns/fl-native

../upstream/Open-ScientistTwo/.venv/bin/python bridge.py check \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/campaigns/fl-native
```

The relative commands above run from this `bridge` directory. The frozen paper
profile retains the upstream public ceilings: 16 limitation rounds, 4 search
rounds, 2 engineering retries, 1 ablation refinement, 2 review rounds and 1 meta
refinement. The original paper does not disclose `N_seed`; upstream chooses five
as an implementation choice. This adapter explicitly overrides `seed_ideas` to
three by default (`init --seed-ideas 2` also supported), matching the actual local
idea budget. The deviation is stored in the campaign. Never fabricate extra
ideas to fill a batch. These are ceilings/configuration, not proof that any round
has happened. This small adapter implements role/evidence recording only;
the remaining upstream controller stages are not reimplemented.

## Record an actual native output

Ask a real Codex native agent to produce a schema-valid JSON output and its origin
receipt as files. `record-role` only imports those files; it has no response
generator or scripted fallback. Supported roles are `limitation`, `seeds` and
`hypothesis`. Output fields are the exact upstream schemas:

- `limitation`: `limitations`, `relevant`, `reason`.
- `seeds`: `ideas`, containing the locally configured number of real
  `Hypothesis` objects (three by default, with an explicit two-idea option at init).
- `hypothesis`: `title`, `hypothesis`, `novelty_score`, `novelty_scope`, `source_ids`,
  plus optional `evidence_ids` and `adapter_payload`.

The receipt must contain `backend: "Codex native collaboration"`, the actual
canonical `agent_id` (for example `/root/route_a_citations`), a unique native
`request_id`, timezone-aware `produced_at`, the actual task description, exact
output file `output_sha256`, `status: "completed"`, and
`capture: "written_by_native_agent"`. A receipt is an explicit provenance
attestation; this does not pretend that the app supplies a cryptographic agent
signature. The real output recorded in `native_outputs/` was written by the
bridge's implementing native agent, and is clearly scoped to bridge readiness.

```sh
../upstream/Open-ScientistTwo/.venv/bin/python bridge.py record-role \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/campaigns/fl-native \
  --id limitation-01 --role limitation --round 1 \
  --output /absolute/path/native-output.json \
  --receipt /absolute/path/native-receipt.json
```

Output bytes must match the receipt hash. Unknown `source_ids` or `evidence_ids`
are rejected: sources must have been frozen via `init --context ID=PATH`, and
experiment IDs must already have been imported. Duplicate request IDs or
limitation/seed role rounds are rejected. Schema validation does not establish
novelty or scientific quality.

Adaptive hypotheses can reference only evidence explicitly imported with scope
`development`; final and existing-analysis imports are retained for review, not
fed back into hypothesis selection.

## Import an existing real experiment

`import-experiment` only reads result/metrics JSON/CSV below these fixed roots:

```text
/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105
/Users/wind/Desktop/Codex/paper/03_英文论文/revision_artifacts/route_a_20261105
```

Symlink escapes and other roots are rejected. Importing stores the original bytes
without computing new metrics. It records source paths, SHA-256, file size,
mtime, import time, declared device, evidence scope and an optional existing run
receipt/log. Unfinished/failed result statuses and nonfinite data are rejected.
The original source is never edited. Run time, device and measurement validity
are not inferred from the import time or a declared device. This import receipt
is separate from an experiment execution receipt.

```sh
../upstream/Open-ScientistTwo/.venv/bin/python bridge.py import-experiment \
  /Users/wind/Desktop/Codex/paper/06_ScientistTwo/bridge/campaigns/fl-native \
  --id cifar-fedavg-s20261231 \
  --file /private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/baselines/runs/cifar/s20261231/fedavg_dense/result.json \
  --device mps --scope development \
  --note 'Existing revision baseline; this command imports bytes and does not rerun training.'
```

`--file` can repeat for matching result/metrics files. `--run-receipt` can point
to an existing JSON/log/text receipt inside the same fixed roots. Final-test
imports must use `--scope final`; this bridge does not reopen selection or claim
that it enforces an upstream held-out campaign. Host/MPS computation occurs in
the existing revision experiment scripts, outside the upstream container runtime.

## Integrity and limits

`SOURCE_LOCK.json` pins this bridge, its README, the upstream source contract
closure and installed package versions. Source changes require a deliberate
new lock and a new campaign. The CLI does not rewrite its lock. `check` verifies
source identity, registered artifacts, role hashes/schema and experiment snapshots;
it flags a still-present original experiment file if its bytes changed. If a
temporary original was moved after promotion, the stored snapshot remains
verifiable and the missing original is reported as a note.

This adapter does not estimate paid API usage or pretend to account for native
app subscription usage. Its own model-call count is always zero. Native agent
work is external to this process, with backend explicitly recorded as
`Codex native collaboration`. SQLite contains no model/API credentials.

The smoke campaign demonstrates actual `init`, native-role import, an existing
experiment import, `status`, and `check`; see `SMOKE_REPORT.json`. It validates
the adapter mechanics, not the scientific quality of the FL paper or reproduction
of ScientistTwo's original results.
