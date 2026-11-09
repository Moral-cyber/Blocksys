# Final evidence identity audit

Status: **PASS_ALL_EVIDENCE_IDENTITIES**. Checked 1274 identity references in 18 blocks, spanning 677 distinct current evidence files. The audit script has a separate checksum. This is a byte-identity check only; no experiment or score was recomputed.

| Evidence block | Declared references checked | Status |
|---|---:|---|
| P0 paired statistics | 7 | PASS |
| P1 lambda | 250 | PASS |
| P1 frozen drift | 248 | PASS |
| FedAvg/FedProx frozen fits | 290 | PASS |
| FedAvg/FedProx training sources | 3 | PASS |
| FedAvg/FedProx score sources | 6 | PASS |
| Multi-partition frozen sources | 22 | PASS |
| Multi-partition aggregate inputs | 13 | PASS |
| pFedMe frozen models and inputs | 43 | PASS |
| pFedMe source inputs | 25 | PASS |
| CIFAR100 frozen sources | 18 | PASS |
| CIFAR100 TRAIN data | 1 | PASS |
| CIFAR100 source data checkpoint evaluation replay identities | 25 | PASS |
| ScientistTwo stopped-screen frozen inputs | 253 | PASS |
| Archived copies of original temporary P1 sources | 4 | PASS |
| Multi-partition per-run source data and final checkpoints | 41 | PASS |
| pFedMe matrix launcher scalar lock | 3 | PASS |
| CIFAR100 per-run locked source data and scored checkpoints | 22 | PASS |

The report includes P0 statistics; all250lambda/248drift identities; dense FedAvg/FedProx fits and source locks; multi-partition sources, source data, aggregate inputs and checkpoints; the43pFedMe model/input lock plus source/launcher locks; CIFAR100 source/data/scored checkpoints; and all253stopped-screen frozen identities. Existing report files are not overwritten.

## Path relocation boundary

The original P1 result files retain `/private/tmp/pr_lfl_route_a/p1/...` paths. Those source files still exist and match; their copies inside the staged manuscript revision artifacts also match the same original hashes. The main experiment sources use current Desktop paper paths or staged revision-artifact paths. Final publication changes paths, not content; this audit records explicit prefix aliases and only permits a missing recorded path to resolve to an identical-byte relocated file. It never hides an existing mismatch by choosing a convenient copy. Root must separately verify the final delivery copy after publishing it.

The complete JSON records every expected and current SHA-256, checked path, record-file SHA-256 and alias decision. This audit does not certify scientific novelty, experimental independence, or that a file unavailable before this check was previously correct.
