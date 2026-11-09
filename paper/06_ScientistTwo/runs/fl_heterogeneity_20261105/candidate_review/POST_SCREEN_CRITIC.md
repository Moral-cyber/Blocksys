# Post-screen decision and provenance addendum

Written by `/root/route_a_citations` after reading the first-seed screen report.
The earlier `INDEPENDENT_CRITIC.md` and its source record are retained as the
pre-result review; this addendum does not rewrite its hypothesis or gate.

**Stop `count_geometry_v1` in this formulation.** The actual root-authorized
first-seed, two-task development report at
`/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/scientisttwo_screen/result_subset.json`
records `STOP_CANDIDATE`. I inspected its saved summaries and paired arithmetic;
the numerical experiment/replay was performed by the separate statistics agent.

| Contrast, client-macro development NLL | CIFAR, seed20261231 | Fashion, seed20261241 |
|---|---:|---:|
| Geometry minus original prior λ=10 | −0.001191945 | +0.000619258 |
| Geometry minus precision-permuted control | −0.000938851 | +0.001037785 |
| Geometry minus simple count-ridge | +0.000082483 | −0.000014091 |
| Geometry minus isotropic prior λ=1 | −0.000021494 | +0.002411601 |

Fashion violates both the predeclared improvement and alignment conditions.
The ordinary scalar-ridge rival is also better there. Count-ridge and exact
geometry are close but not numerically identical at the declared `1e-7` tolerance;
do not falsely report equivalence. These outcomes do not justify expanding to
all three seeds, tuning smoothing/normalizers, selecting the favorable task, or
claiming a new uncertainty mechanism. A first-seed stop bounds investment; it
does not prove failure on every possible seed or invalidate the original
isotropic PR-LFL predictor.

There is a further competing explanation from the idea generator: lower
precision for zero/low-count classes lets those biases move away from the
smoothed anchor. Cross-entropy can suppress absent classes when the stationary
development set also lacks them. Even if NLL had improved consistently, that
would not establish robustness to newly appearing classes or prior drift. New
partitions with similar absence patterns would not remove this boundary. No
drift-improvement claim is supported or proposed.

The original critic's source record intentionally stores the earlier reviewed
bytes of `limitations_and_ideas.md`, SHA-256
`c167eed1d3c7f2129dc97ab60702acae6e4c2779f885bdc995850dccbe8125ea`.
The idea agent subsequently appended a procedural skill acknowledgement and
reported no mechanism or role-JSON change. The current note has SHA-256
`d6b36590dc41fd78e6e3edcb8285522b3a3572149db4e8609b78bfb20ae91125`.
The reviewed `roles/seeds.json` remains SHA-256
`e7682ed64c934437eb2076c1331750b234b7934153cde53a81331ea4adf010f9`.
These are versioned provenance facts, not a reason to alter the original record.

The unchanged research decision is to finish the already frozen comparative FL
baselines and preserve adverse candidate outcomes. There is no basis to promote
this screen into the protected manuscript results.
