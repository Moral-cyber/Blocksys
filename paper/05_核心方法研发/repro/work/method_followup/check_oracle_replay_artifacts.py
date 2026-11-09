"""Finite independent arithmetic replay of the six v2 additions; no gradient run."""
from pathlib import Path
import json
import math

import torch

import replay_oracle_probe as probe


if __name__ == "__main__":
    torch.set_num_threads(1)
    parent_dir = probe.HERE / "oracle_replays/cpu_topk20_tau90"
    output = probe.HERE / "oracle_replays/cpu_topk20_tau90_v2"
    result = json.loads((output / "result.json").read_text())
    assert result["parent_result_sha256"] == probe.sha(parent_dir / "result.json")
    for path, digest in result["executed_dependency_sha256"].items():
        assert probe.sha(path) == digest, path
    assert probe.sha(output / "new_oracles_and_supports.pt") == result["new_oracles_and_supports_sha256"]
    state = torch.load(Path(result["source_probe"]) / "shared_state.pt", weights_only=False)
    old = torch.load(parent_dir / "oracle_and_supports.pt", weights_only=False)
    new = torch.load(output / "new_oracles_and_supports.pt", weights_only=False)
    us = [d+r for d, r in zip(state["deltas"], state["residuals"])]
    weights = state["weights"]
    checked = []
    for row in result["rows"]:
        if row["variant"] not in result["new_variants"]:
            continue
        indices = new["support_indices"][row["variant"]]
        reference = new["current_dense_delta"] if row["reference"] == "current_dense_delta_oracle" else new["negative_gradient_at_BASE"]
        sent_values = []
        for i, (u, idx, core) in enumerate(zip(us, indices, old["shared_core_indices"])):
            assert idx.numel() == idx.unique().numel() == result["shared_K"]
            assert bool(torch.isin(core, idx).all())
            assert probe.tensor_sha(idx) == row["support_sha256"][i]
            sent = torch.zeros_like(u)
            sent[idx] = u[idx]
            sent_values.append(sent)
            k = result["shared_K"]
            top_energy = float(torch.topk(u.double().square(), k).values.sum())
            assert float(sent.double().square().sum()) >= result["tau"]*top_energy - 1e-10
            raw_score = u.double()*reference.double()
            score = raw_score if row["score_kind"] == "signed_product" else raw_score.abs() if row["score_kind"] == "absolute_product" else u.double()*reference.double().sign()
            filler = idx[~torch.isin(idx, core)]
            omitted = torch.ones(u.numel(), dtype=torch.bool)
            omitted[idx] = False
            if filler.numel() and bool(omitted.any()):
                assert float(score[filler].min()) >= float(score[omitted].max()) - 1e-12
        sent = sum(w*s for w, s in zip(weights, sent_values))
        change = (sent-old["Q"]).double()
        assert math.isclose(float(change.norm()), row["selection_change_norm"], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(float(torch.dot(old["gradient_at_topk"], change)), row["gradient_dotdelta"], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(float(torch.dot(new["gradient_at_BASE"], change)), row["BASE_gradient_dotdelta"], rel_tol=1e-12, abs_tol=1e-12)
        checked.append(row["variant"])
    assert len(checked) == 6
    report = {"status": "PASS", "scope": "finite saved-artifact provenance, score-order, support, energy and gradient-dot replay; not an external integrity verdict",
        "new_rows_checked": checked, "full_train_gradient_recomputed": False,
        "official_test_accessed": False, "confirmation_results_accessed": False,
        "validation_gradient_used": False, "checker_sha256": probe.sha(__file__)}
    probe.runner.atomic_json(probe.HERE / "oracle_replay_artifact_check.json", report)
    print(json.dumps(report, indent=2))
