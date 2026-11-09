"""Small deterministic checks for the controls adapter, not FL evidence."""

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_controls as controls


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(16)
    delta = torch.randn(101, generator=generator)
    residual = torch.randn(101, generator=generator) / 5
    reference = torch.randn(101, generator=generator)
    candidate = delta + residual
    checked = []
    for variant in controls.CONTROL_VARIANTS:
        if variant.startswith("regtopk"):
            controls._REG_CONTEXT = {"variant": variant, "weights": [1.], "previous": [None], "call_count": 0}
        state = torch.random.get_rng_state().clone()
        sent, rem, diag = controls.encode_variant(delta, residual, reference, .2, variant, 13)
        assert torch.equal(torch.random.get_rng_state(), state), variant
        assert torch.allclose(sent + rem, candidate, atol=1e-7, rtol=1e-7)
        assert diag["selected_coordinates"] == 20
        assert diag["encoded_bytes"] == 64 + 8 * 20
        assert int((sent != 0).sum()) == 20  # random candidate has no zero entries
        if variant in ("reverse90_ef", "shuffle90_ef"):
            assert diag["retained_topk_energy_ratio"] >= .9 - 1e-12
        if variant == "reverse90_ef":
            assert diag["alignment_gain"] <= 1e-10
            assert diag["guidance_alignment_gain"] >= -1e-10
        if variant.startswith("prox"):
            original = controls._ORIGINAL_ENCODE(delta, residual, reference, .2, "topk_ef", 13)
            assert torch.equal(sent, original[0])
        for g in (None, torch.zeros_like(reference)):
            if variant.startswith("regtopk"):
                controls._REG_CONTEXT = {"variant": variant, "weights": [1.], "previous": [None], "call_count": 0}
            fallback = controls.encode_variant(delta, residual, g, .2, variant, 13)
            topk = controls._ORIGINAL_ENCODE(delta, residual, g, .2, "topk_ef", 13)
            assert torch.equal(fallback[0], topk[0])
        checked.append(variant)

    u = torch.tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
    g = torch.tensor([-1., -1., -1., -1., 1., 1., 1., 1., 1., 1.])
    expected = torch.tensor([10., 9., 0., 0., 6., 5., 0., 0., 0., 0.])
    for v in ("sign50_ef", "fixed50_ef"):
        got = controls.encode_variant(u, torch.zeros_like(u), g, .4, v, 1)
        assert torch.equal(got[0], expected)
    # Small K, full K, and no positive products must still select exact budgets.
    for n in (1, 2, 9):
        for rho in (.01, .2, 1.):
            for v in controls.CONTROL_VARIANTS:
                if v.startswith("regtopk"):
                    controls._REG_CONTEXT = {"variant": v, "weights": [1.], "previous": [None], "call_count": 0}
                d = torch.arange(1, n+1).float()
                got = controls.encode_variant(d, torch.zeros_like(d), -torch.ones_like(d), rho, v, 9)
                k = max(1, int(n*rho))
                assert got[2]["selected_coordinates"] == k
                assert int((got[0] != 0).sum()) == k

    # RegTopK must suppress previously selected, cancelling coordinates even
    # when the previous aggregate is exactly zero (not fall back to Top-K).
    for variant in ("regtopk1_ef", "regtopk5_ef", "regtopk10_ef"):
        controls._REG_CONTEXT = {"variant": variant, "weights": [.5,.5],
                                 "previous": [None,None], "call_count": 0}
        first = []
        for d in (torch.tensor([10.,9.,1.,.5]),torch.tensor([-10.,-9.,1.,.5])):
            first.append(controls.encode_variant(d,torch.zeros_like(d),None,.5,variant,9))
        previous_global = .5 * (first[0][0] + first[1][0])
        assert torch.equal(previous_global, torch.zeros(4))
        for client,d in enumerate((torch.tensor([10.,9.,1.,.5]),torch.tensor([-10.,-9.,1.,.5]))):
            sent,rem,diag=controls.encode_variant(d,first[client][1],previous_global,.5,variant,9)
            assert torch.equal(sent,torch.tensor([0.,0.,2.,1.]))
            assert diag["regtopk_client_index"]==client
            assert diag["regtopk_first_round"] is False
        assert controls._REG_CONTEXT["call_count"]==4
    controls._REG_CONTEXT = None

    model = nn.Linear(4, 10)
    dataset = TensorDataset(torch.randn(130, 4, generator=generator),
                            torch.randint(0, 10, (130,), generator=generator))
    controls._ACTIVE_MU = 0.
    baseline = controls._ORIGINAL_TRAIN(model, dataset, 53, .05, 1)
    assert torch.equal(baseline, controls.train_local(model, dataset, 53, .05, 1))
    for mu in (.01, .1):
        controls._ACTIVE_MU = mu
        actual = controls.train_local(model, dataset, 53, .05, 1)
        torch.manual_seed(53)
        local = copy.deepcopy(model)
        anchors = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.SGD(local.parameters(), lr=.05)
        loader = DataLoader(dataset, batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(53))
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(local(x), y)
            loss = loss + mu / 2 * sum((p-anchor).square().sum() for p, anchor in zip(local.parameters(), anchors))
            loss.backward()
            opt.step()
        expected_delta = controls.runner.vectorize(local) - controls.runner.vectorize(model)
        assert torch.allclose(actual, expected_delta, atol=2e-7, rtol=2e-6)
    controls._ACTIVE_MU = 0.

    # Exercise original one_run on a tiny linear model: checks integration and
    # source hash participation without touching real datasets or long runs.
    with tempfile.TemporaryDirectory(prefix="control_adapter_check_only_") as d:
        out = Path(d) / "prox001_ef_fixture"
        out.mkdir()
        params = argparse.Namespace(mode="train", dataset="temporary fixture only",
            mixing=0., split_seed=17, validation_size=16, seeds="17",
            variants=["prox001_ef"], rounds=1, fraction=.2, lr=.03,
            local_epochs=1, threads=1, test_final=False, probe_rounds=[1], out=str(d))
        original_model = controls.runner.SmallCNN
        controls.runner.SmallCNN = lambda: nn.Linear(4, 10)
        controls.install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                controls.one_run(params, [dataset], dataset, {"test_used_for_selection": False},
                                 "prox001_ef", 17, out)
            cfg = json.loads((out / "config.json").read_text())
            res = json.loads((out / "result.json").read_text())
            assert res["status"] == "DONE"
            assert cfg["fedprox_mu"] == .01 and cfg["control_variant"] == "prox001_ef"
            assert cfg["adapter_source_sha256"] == hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest()
            expected_signature = hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k not in ("out", "run_signature")}, sort_keys=True).encode()).hexdigest()
            assert expected_signature == cfg["run_signature"]
            changed = dict(cfg); changed["adapter_source_sha256"] = "different"
            changed_signature = hashlib.sha256(json.dumps({k:v for k,v in changed.items() if k not in ("out", "run_signature")}, sort_keys=True).encode()).hexdigest()
            assert changed_signature != cfg["run_signature"]
            regout=Path(d)/"regtopk1_fixture";regout.mkdir()
            regparams=copy.copy(params);regparams.rounds=2;regparams.variants=["regtopk1_ef"]
            with contextlib.redirect_stdout(io.StringIO()):
                controls.one_run(regparams,[dataset],dataset,{"test_used_for_selection": False},
                                 "regtopk1_ef",17,regout)
            regcfg=json.loads((regout/"config.json").read_text())
            regres=json.loads((regout/"result.json").read_text())
            assert regcfg["regtopk_mu"]==1 and regcfg["stateful_resume_supported"] is False
            assert regres["client_rows"][0]["regtopk_first_round"] is True
            assert regres["client_rows"][1]["regtopk_first_round"] is False
            regres["status"]="IN_PROGRESS"
            (regout/"result.json").write_text(json.dumps(regres))
            try:
                controls.one_run(regparams,[dataset],dataset,{"test_used_for_selection":False},"regtopk1_ef",17,regout)
            except RuntimeError as e:
                assert "cannot resume" in str(e)
            else:
                raise AssertionError("RegTopK interrupted-run resume must be rejected")
        finally:
            controls.runner.SmallCNN = original_model

    for argv in ([], ["--variants", "prox001_ef", "--mode", "snapshot"],
                 ["--variants","regtopk1_ef","--mode","snapshot"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                controls.main(argv)
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("expected explicit-variant or snapshot-objective error")

    report = {"status": "PASS", "variants_checked": checked,
        "checks": ["same K and value/index bytes", "candidate/residual conservation",
                   "RNG isolation", "missing/zero reference fallback", "sign/fixed quota supports",
                   "reverse/shuffle energy bounds", "mu=0 baseline identity",
                   "FedProx gradient equals explicit proximal objective over 3 mini-batches",
                   "tiny one_run integration and adapter hash in run_signature",
                   "RegTopK per-client history and exact cancellation behavior",
                   "RegTopK tiny integration, adaptation metadata and interrupted-resume rejection",
                   "explicit variants and FedProx/RegTopK snapshot rejection"],
        "scope": "Small deterministic implementation checks only, no learning advantage or literature reproduction claim."}
    Path(__file__).with_name("controls_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
