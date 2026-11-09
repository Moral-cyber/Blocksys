"""Independent finite DEVELOPMENT probe checks; never opens final-test data.

Reads only the two explicit development probe directories below and the source
development checkpoints they identify. Rechecks saved-state arithmetic and
three validation counterfactuals per residual mode. One client delta per source
is recomputed; this is not a new training run or an independent repetition.
"""
from pathlib import Path
import hashlib
import inspect
import json
import math

import torch

import run_accelerated as accelerated
import run_controls as controls
import run_followup as runner
import run_lagrangian as lagrangian


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def check(name):
    out = HERE / "checkpoint_probes" / name
    result = json.loads((out / "result.json").read_text())
    source = Path(result["source_run"])
    assert source.parent == HERE / "development_runs"
    config = json.loads((source / "config.json").read_text())
    assert config["test_final"] is False
    assert config.get("training_backend", "cpu") == "cpu"
    assert config["variant"] in ("topk_ef", "align90")
    source_result = json.loads((source / "result.json").read_text())
    assert source_result["status"] == result["status"] == "DONE"
    assert source_result["variant"] == result["source_variant"] == config["variant"]
    assert source_result["seed"] == result["source_seed"] == config["seed"]
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    state = torch.load(out / "shared_state.pt", map_location="cpu", weights_only=False)
    assert checkpoint["rows"] == source_result["rows"]
    assert [r["round"] for r in checkpoint["rows"]] == list(range(1, config["rounds"] + 1))
    signature_config = {k: v for k, v in config.items() if k not in ("out", "run_signature")}
    assert config["run_signature"] == hashlib.sha256(json.dumps(signature_config, sort_keys=True).encode()).hexdigest()
    for key, path in (("source_config_sha256", source / "config.json"),
                      ("source_checkpoint_sha256", source / "checkpoint.pt"),
                      ("shared_state_sha256", out / "shared_state.pt"),
                      ("probe_source_sha256", HERE / "probe_checkpoint.py")):
        assert result[key] == sha(path), key
    imported = ("run_accelerated.py", "run_controls.py", "run_followup.py",
                "run_lagrangian.py", "guarded_codec.py", "lagrangian_codec.py")
    for filename in imported:
        assert result["dependency_sha256"][filename] == sha(HERE / filename), filename
    controls.install()
    trainsets, validation, manifest = runner.prepare(config["dataset"], config["mixing"],
        config["split_seed"], config["validation_size"])
    assert manifest == config["data_manifest"] == result["data_manifest"]
    model = runner.SmallCNN()
    model.load_state_dict(checkpoint["model"])
    assert torch.equal(state["base"], runner.vectorize(model))
    assert torch.equal(state["reference"], checkpoint["previous"])
    assert len(state["residuals"]) == len(state["deltas"]) == len(trainsets) == 20
    assert all(torch.equal(a, b) for a, b in zip(state["residuals"], checkpoint["residuals"]))
    weights = [len(ds) / sum(map(len, trainsets)) for ds in trainsets]
    assert weights == state["weights"]
    close(sum(weights), 1.)
    next_seed = config["seed"] + config["rounds"] * 100
    delta = accelerated.train_local(model, trainsets[0], next_seed,
        config["lr"], config["local_epochs"], backend="cpu")
    assert torch.equal(delta, state["deltas"][0]), "client0 next-round delta changed"
    assert torch.equal(state["base"], runner.vectorize(model))
    originals = [x.clone() for x in state["deltas"] + state["residuals"] + [state["reference"]]]
    checked_metrics = 0
    encoded_rows = 0
    max_validation_loss_difference = 0.
    for mode in ("actual_EF", "zero_residual_control"):
        residuals = state["residuals"] if mode == "actual_EF" else [torch.zeros_like(r) for r in state["residuals"]]
        candidates = [d + r for d, r in zip(state["deltas"], residuals)]
        dense = sum(w * u for w, u in zip(weights, candidates))
        dense_norm2 = float(dense.double().square().sum())
        for row in (r for r in result["rows"] if r["residual_mode"] == mode):
            variant = row["variant"]
            if variant == "dense_candidate_reference":
                sent, error, independent = dense, 0., 0.
                assert row["round_analytical_bytes"] == len(trainsets) * (64 + 4 * state["base"].numel())
            else:
                encoded = [lagrangian.encode_variant(d, r, state["reference"], config["fraction"], variant, next_seed + i)
                    for i, (d, r) in enumerate(zip(state["deltas"], residuals))]
                k = max(1, int(state["base"].numel() * config["fraction"]))
                for u, (s, residual, diag) in zip(candidates, encoded):
                    assert torch.equal(u, s + residual)
                    assert diag["selected_coordinates"] == k
                    assert diag["encoded_bytes"] == 64 + 8 * k
                    assert torch.count_nonzero(s).item() <= k
                assert row["round_analytical_bytes"] == len(trainsets) * (64 + 8 * k)
                sent = sum(w * e[0] for w, e in zip(weights, encoded))
                error = float((dense - sent).double().square().sum())
                independent = sum(w*w * float((u-e[0]).double().square().sum())
                    for w, u, e in zip(weights, candidates, encoded))
                encoded_rows += 1
            close(row["aggregate_error_squared"], error)
            close(row["aggregate_relative_error"], (error / (dense_norm2 + 1e-30))**.5)
            close(row["independent_error_energy"], independent)
            close(row["error_cross_term"], error - independent)
            close(row["aggregate_norm"], float(sent.double().norm()))
            if variant in ("dense_candidate_reference", "topk_ef", "align90"):
                runner.assign_vector(model, state["base"] + sent)
                metrics = runner.metrics(model, validation)
                assert metrics == row["validation"], (name, mode, variant, metrics, row["validation"])
                max_validation_loss_difference = max(max_validation_loss_difference,
                    abs(metrics["loss"] - row["validation"]["loss"]))
                checked_metrics += 1
    for before, after in zip(originals, state["deltas"] + state["residuals"] + [state["reference"]]):
        assert torch.equal(before, after), "an encoder mutated shared state"
    return {"name": name, "status": "PASS", "source": str(source),
            "source_round": config["rounds"], "probe_round": config["rounds"]+1,
            "delta_client0_seed": next_seed, "client0_delta_bitwise_reproduced": True,
            "encoding_rows_recomputed": encoded_rows, "validation_rows_recomputed": checked_metrics,
            "maximum_validation_loss_difference": max_validation_loss_difference,
            "parameter_count": state["base"].numel(), "K": k,
            "sparse_round_analytical_bytes": len(trainsets) * (64 + 8*k),
            "current_external_helper_sha256": sha(inspect.getsourcefile(runner.SmallCNN)),
            "source_does_not_record_external_helper_or_npz_hashes": True,
            "unrelated_manifest_drift": [filename for filename, digest in result["dependency_sha256"].items()
                if filename not in imported and (HERE/filename).exists() and sha(HERE/filename) != digest]}


if __name__ == "__main__":
    torch.set_num_threads(2)
    rows = [check(name) for name in ("cpu_topk20", "cpu_align9020")]
    report = {"status": "PASS", "scope": "independent finite development-only arithmetic and partial validation replay",
              "official_test_data_or_results_accessed": False, "confirmation_runs_accessed": False,
              "training_scope": "one client0 delta from each of two saved development endpoints only",
              "checks": rows}
    runner.atomic_json(HERE / "checkpoint_probe_check.json", report)
    print(json.dumps(report, indent=2))
