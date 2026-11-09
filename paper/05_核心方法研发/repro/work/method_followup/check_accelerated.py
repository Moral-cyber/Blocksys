"""Small CPU/MPS objective and adapter checks; no dataset image loading.

Use --require-mps for a check that must exercise GPU rather than skip it.
The parent can run this with an MPS-enabled approved execution context.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_accelerated as accelerated
import summarize_accelerated as summaries


def explicit_mps(model,dataset,seed,lr,epochs,mu):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    local=copy.deepcopy(model).to("mps")
    local.train()
    anchors=[p.detach().clone() for p in local.parameters()]
    optimizer=torch.optim.SGD(local.parameters(),lr=lr)
    loader=DataLoader(dataset,batch_size=64,shuffle=True,
                      generator=torch.Generator(device="cpu").manual_seed(seed))
    for _ in range(epochs):
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            objective=nn.functional.cross_entropy(local(x.to("mps")),y.to("mps"))
            objective=objective+mu/2*sum((p-a).square().sum() for p,a in zip(local.parameters(),anchors))
            objective.backward()
            optimizer.step()
    return torch.cat([p.detach().flatten() for p in local.parameters()]).cpu()-accelerated.runner.vectorize(model)


def integration_checks(dataset,backend):
    runner=accelerated.runner
    controls=accelerated.controls
    original_model,original_prepare=runner.SmallCNN,runner.prepare
    original_implementation=accelerated._ORIGINAL_RUNNER_MAIN
    original_train=runner.train_local
    runner.SmallCNN=lambda:nn.Linear(4,10)
    count={"local":0}
    def tiny_main():
        args=argparse.Namespace(mode="train",dataset="temporary fixture only",mixing=0.,split_seed=31,
            validation_size=len(dataset),seeds="31",variants=["prox01_ef","lagrange90_ef"],rounds=2,
            fraction=.2,lr=.03,local_epochs=1,threads=1,test_final=False,probe_rounds=[1,2],out="temporary fixture")
        original_local=runner.train_local
        def checked(model,*rest):
            old={k:v.detach().clone() for k,v in model.state_dict().items()}
            delta=original_local(model,*rest)
            assert delta.device.type=="cpu" and bool(torch.isfinite(delta).all())
            assert all(torch.equal(old[k],v) for k,v in model.state_dict().items())
            count["local"]+=1
            return delta
        runner.train_local=checked
        try:
            with tempfile.TemporaryDirectory(prefix="accelerated_integration_check_only_") as tmp:
                for variant in args.variants:
                    out=Path(tmp)/variant;out.mkdir()
                    with contextlib.redirect_stdout(io.StringIO()):
                        runner.one_run(args,[dataset],dataset,{"test_used_for_selection":False},variant,31,out)
                    config=json.loads((out/"config.json").read_text())
                    result=json.loads((out/"result.json").read_text())
                    assert result["status"]=="DONE"
                    assert config["training_backend"]==backend
                    assert config["accelerated_adapter_source_sha256"]
                    assert config["accelerated_dependency_versions"]["torch"]==str(torch.__version__)
                    if variant=="prox01_ef":
                        assert config["fedprox_mu"]==.1
                    checkpoint=torch.load(out/"checkpoint.pt",weights_only=False)
                    assert all(v.device.type=="cpu" for v in checkpoint["model"].values())
                    assert all(v.device.type=="cpu" for v in checkpoint["residuals"])
                    assert checkpoint["previous"].device.type=="cpu"
        finally:
            runner.train_local=original_local
    accelerated._ORIGINAL_RUNNER_MAIN=tiny_main
    try:
        accelerated.lagrangian.install()
        accelerated.late_main(backend,"temporary integration fixture")
        assert count["local"]==4
    finally:
        runner.SmallCNN,runner.prepare=original_model,original_prepare
        runner.train_local=original_train
        accelerated._ORIGINAL_RUNNER_MAIN=original_implementation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-mps",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    has_mps=bool(torch.backends.mps.is_available())
    if args.require_mps and not has_mps:
        raise RuntimeError("--require-mps requested, but MPS is unavailable; this check cannot pass without GPU execution")
    generator=torch.Generator().manual_seed(20261106)
    torch.manual_seed(21)
    model=nn.Linear(4,10)
    dataset=TensorDataset(torch.randn(130,4,generator=generator),torch.randint(0,10,(130,),generator=generator))
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    saved_mu=accelerated.controls._ACTIVE_MU
    cpu_checks=[]
    gpu_checks=[]
    try:
        for mu in (0.,.01,.1):
            accelerated.controls._ACTIVE_MU=mu
            expected=accelerated.controls.train_local(model,dataset,59,.03,1)
            actual=accelerated.train_local(model,dataset,59,.03,1,backend="cpu")
            assert torch.equal(expected,actual)
            cpu_checks.append({"mu":mu,"bitwise_passthrough":True})
            if has_mps:
                delta=accelerated.train_local(model,dataset,59,.03,1,backend="mps")
                repeat=accelerated.train_local(model,dataset,59,.03,1,backend="mps")
                explicit=explicit_mps(model,dataset,59,.03,1,mu)
                assert delta.device.type=="cpu" and bool(torch.isfinite(delta).all())
                assert torch.allclose(delta,repeat,atol=2e-6,rtol=2e-5)
                assert torch.allclose(delta,explicit,atol=2e-6,rtol=2e-5)
                assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
                gpu_checks.append({"mu":mu,"explicit_objective_maxabs_difference":float((delta-explicit).abs().max()),
                    "repeat_maxabs_difference":float((delta-repeat).abs().max()),
                    "repeat_bitwise_equal":bool(torch.equal(delta,repeat)),"delta_cpu_finite":True,"global_unchanged":True})
        assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
        integration_checks(dataset,"cpu")
        if has_mps:
            integration_checks(dataset,"mps")
    finally:
        accelerated.controls._ACTIVE_MU=saved_mu

    cfg={"dataset":"fixture","rounds":2,"lr":.03,"local_epochs":1,
         "training_backend":"cpu","accelerated_adapter_source_sha256":"same",
         "accelerated_dependency_versions":{"torch":"fixture version"},
         "accelerated_runtime_identity":{"fixture":"runtime"}}
    ids=[]
    for backend in ("cpu","mps"):
        ids.append(summaries.settings({**cfg,"training_backend":backend})[0])
    assert ids[0]!=ids[1]
    assert summaries.settings(cfg)[0]!=summaries.settings({**cfg,"accelerated_dependency_versions":{"torch":"other"}})[0]
    assert summaries.settings(cfg)[0]!=summaries.settings({**cfg,"accelerated_adapter_source_sha256":"other"})[0]
    report={"status":"PASS" if has_mps else "CPU_PASS_MPS_UNAVAILABLE",
        "mps_available":has_mps,"cpu_checks":cpu_checks,"mps_checks":gpu_checks,
        "adapter_integration_backends":["cpu","mps"] if has_mps else ["cpu"],
        "summary_backend_dependency_source_isolation":True,
        "scope":"Small local-objective and adapter checks; no real data training or learning-benefit claim."}
    output=Path(__file__).with_name("accelerated_check.json" if has_mps else "accelerated_check_cpu.json")
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
