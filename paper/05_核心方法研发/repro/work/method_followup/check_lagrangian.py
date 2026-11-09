"""Property and tiny adapter checks, not evidence of learning advantage."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import TensorDataset

from guarded_codec import encode as topk_encode, pack_encoding
from lagrangian_codec import encode
import run_lagrangian as adapter


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261104)
    cases = 0
    max_iterations = 0
    fallback_count = 0
    for n in (1, 2, 7, 64, 1025):
        for rho in (.01, .2, .5, 1.):
            d = torch.randn(n, generator=generator)
            r = torch.randn(n, generator=generator)/4
            references = (torch.randn(n, generator=generator), -d, d,
                          torch.ones_like(d), torch.zeros_like(d), None)
            for g in references:
                plain = topk_encode(d, r, fraction=rho, reference=g)
                top_energy = float(plain.sent.double().square().sum())
                for tau in (.5, .9, .99, 1.):
                    for steps in (8, 12):
                        state = torch.random.get_rng_state().clone()
                        result = encode(d, r, fraction=rho, reference=g, tau=tau, bisection_steps=steps)
                        assert torch.equal(torch.random.get_rng_state(), state)
                        k = max(1, int(n*rho))
                        assert result.indices.numel() == k
                        assert result.indices.unique().numel() == k
                        assert bool((result.indices[1:] >= result.indices[:-1]).all())
                        assert torch.equal(result.sent[result.indices], (d+r)[result.indices])
                        assert torch.allclose(result.sent+result.residual,d+r,atol=1e-7,rtol=1e-7)
                        assert len(pack_encoding(result))==8*k
                        assert result.diagnostics["encoded_bytes"]==64+8*k
                        energy=float(result.sent.double().square().sum())
                        assert energy >= tau*top_energy, (n,rho,tau,result.diagnostics)
                        if g is not None:
                            value=float(torch.dot(result.sent.double(),g.double()))
                            baseline=float(torch.dot(plain.sent.double(),g.double()))
                            assert value >= baseline
                        if tau==1 or g is None or not bool(g.any()):
                            assert torch.equal(result.indices,plain.indices)
                            assert torch.equal(result.sent,plain.sent)
                        assert result.diagnostics["solver_iters"] <= 1+24+steps
                        max_iterations=max(max_iterations,result.diagnostics["solver_iters"])
                        fallback_count+=int(result.diagnostics["solver_status"].startswith("fallback"))
                        cases+=1

    for d,g in (
        (torch.zeros(4),torch.ones(4)),
        (torch.tensor([1.,0.,0.,0.]),torch.tensor([0.,1.,1.,1.])),
        (torch.ones(8),torch.arange(-4.,4.)),
        (torch.tensor([1e20,1e-20,1.,3.]),torch.tensor([-1e20,1e20,1e-20,-2.])),
        (torch.tensor([1e-20,2e-20,3e-20,4e-20]),torch.tensor([1e-20,-1e-20,1e-20,-1e-20])),
    ):
        result=encode(d,fraction=.5,reference=g,tau=.99)
        plain=topk_encode(d,fraction=.5,reference=g)
        assert float(result.sent.double().square().sum()) >= .99*float(plain.sent.double().square().sum())
        assert float(torch.dot(result.sent.double(),g.double())) >= float(torch.dot(plain.sent.double(),g.double()))

    infeasible=encode(torch.tensor([100.,1.]),fraction=.5,reference=torch.tensor([-.01,1.]),tau=.9,max_expansions=1)
    assert infeasible.diagnostics["solver_status"]=="fallback_no_feasible_high"

    # Discrete unsupported solution: feasible scalarization need not recover the
    # original constrained optimum.  This explicitly checks the stated limit.
    d=torch.tensor([10.**.5,6.**.5,1.])
    g=torch.tensor([0.,4./(6.**.5),10.])
    approximate=encode(d,fraction=.34,reference=g,tau=.59)
    assert approximate.indices.tolist()==[0]
    exact_feasible=torch.tensor([0.,d[1],0.])
    assert float(exact_feasible.double().square().sum()) >= .59*float(d[0].double().square())
    assert float(torch.dot(exact_feasible.double(),g.double())) > approximate.diagnostics["alignment"]

    # Tiny independent training/snapshot integration with no real data access.
    adapter.install()
    original_model=adapter.runner.SmallCNN
    adapter.runner.SmallCNN=lambda:nn.Linear(4,10)
    dataset=TensorDataset(torch.randn(70,4,generator=generator),torch.randint(0,10,(70,),generator=generator))
    try:
        with tempfile.TemporaryDirectory(prefix="lagrangian_adapter_check_only_") as tmp:
            params=argparse.Namespace(mode="train",dataset="temporary fixture only",mixing=0.,split_seed=19,
                validation_size=70,seeds="19",variants=["lagrange90_ef"],rounds=2,fraction=.2,lr=.03,
                local_epochs=1,threads=1,test_final=False,probe_rounds=[1,2],out=tmp)
            out=Path(tmp)/"train";out.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.one_run(params,[dataset],dataset,{"test_used_for_selection":False},"lagrange90_ef",19,out)
            cfg=json.loads((out/"config.json").read_text())
            result=json.loads((out/"result.json").read_text())
            assert result["status"]=="DONE"
            assert len(cfg["lagrangian_codec_source_sha256"])==64
            assert len(cfg["lagrangian_adapter_source_sha256"])==64
            assert cfg["selection_control"]=="normalized Lagrangian feasible support search"
            assert cfg["lagrangian_tau"]==.9
            assert result["client_rows"][0]["solver_status"]=="fallback_missing_reference"
            assert "solver_iters" in result["client_rows"][1]
            params.mode="snapshot";params.variants=["topk_ef","lagrange90_ef","lagrange99_ef"]
            out=Path(tmp)/"snapshot";out.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.snapshot_run(params,[dataset],dataset,{"test_used_for_selection":False},19,out)
            cfg=json.loads((out/"config.json").read_text())
            result=json.loads((out/"result.json").read_text())
            assert result["status"]=="DONE" and len(result["rows"])==6
            assert "lagrangian_codec_source_sha256" in cfg and "run_signature" in cfg
    finally:
        adapter.runner.SmallCNN=original_model

    report={"status":"PASS","random_property_cases":cases,"fallback_cases_in_grid":fallback_count,
        "max_observed_solver_iters":max_iterations,
        "checks":["exact K and sorted pair payload", "unmodified values and residual conservation",
                  "actual reconstructed energy floor", "actual reference alignment nondecrease",
                  "missing/zero/product-zero and tau-one fallbacks", "finite expansion/bisection budgets",
                  "forced failure fallback", "discrete suboptimality counterexample",
                  "tiny train and snapshot adapter integration; new source hashes in config"],
        "scope":"Implementation properties only; not exact constrained optimality, convergence, novelty or learning benefit."}
    Path(__file__).with_name("lagrangian_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
