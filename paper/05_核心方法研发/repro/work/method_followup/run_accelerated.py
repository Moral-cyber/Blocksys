"""Independent local-training CPU/MPS adapter for frozen sparse-FL methods.

Only local model training can use MPS. Global models, evaluation, compression,
aggregation, residuals and checkpoints remain on CPU. This is a new numerical
backend and must not be silently pooled with historical CPU experiments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import run_controls as controls
import run_lagrangian as lagrangian
import run_followup as runner


HERE = Path(__file__).resolve().parent
_ORIGINAL_RUNNER_MAIN = runner.main


def train_local_mps(model, dataset, seed, lr, epochs):
    """Train a private copy on MPS, honoring controls._ACTIVE_MU dynamically."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend was requested but is unavailable in this process; no CPU fallback was made")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Global model parameters must remain on CPU")
    if any(b.device.type != "cpu" for b in model.buffers()):
        raise ValueError("Global model buffers must remain on CPU")
    mu = float(controls._ACTIVE_MU)
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("FedProx coefficient must be finite and nonnegative")
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    base = runner.vectorize(model).detach().clone()
    local = copy.deepcopy(model).to("mps")
    local.train()
    anchors = [p.detach().clone() for p in local.parameters()] if mu else None
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator(device="cpu").manual_seed(seed))
    opt = torch.optim.SGD(local.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to("mps"), y.to("mps")
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(local(x), y).backward()
            if mu:
                with torch.no_grad():
                    for p, anchor in zip(local.parameters(), anchors):
                        if p.grad is not None:
                            p.grad.add_(p-anchor, alpha=mu)
            opt.step()
    local_vector = torch.cat([p.detach().flatten() for p in local.parameters()]).cpu()
    delta = local_vector-base
    if delta.device.type != "cpu" or not bool(torch.isfinite(delta).all()):
        raise RuntimeError("Local MPS training produced a non-finite or non-CPU delta")
    return delta


def train_local(model, dataset, seed, lr, epochs, *, backend):
    if backend == "cpu":
        return controls.train_local(model, dataset, seed, lr, epochs)
    if backend == "mps":
        return train_local_mps(model, dataset, seed, lr, epochs)
    raise ValueError(f"unknown backend {backend!r}")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata(backend, route):
    model_source = inspect.getsourcefile(runner.SmallCNN)
    prepare_source = inspect.getsourcefile(runner.prepare)
    local_source = inspect.getsourcefile(controls.train_local)
    names = ("run_accelerated.py", "run_followup.py", "run_controls.py",
             "run_lagrangian.py", "guarded_codec.py", "lagrangian_codec.py")
    identities = {name: _sha(HERE/name) for name in names}
    for name in ("run_partitions.py", "run_fashion.py", "launch.py"):
        if (HERE/name).exists():
            identities[name] = _sha(HERE/name)
    return {
        "training_backend": backend,
        "accelerated_adapter_name": "run_accelerated.py",
        "accelerated_adapter_version": 1,
        "accelerated_adapter_source_sha256": identities["run_accelerated.py"],
        "accelerated_route": route,
        "accelerated_source_sha256": identities,
        "accelerated_model_source_sha256": _sha(model_source),
        "accelerated_loader_source_sha256": _sha(inspect.getsourcefile(runner.load_client)),
        "accelerated_prepare_source_sha256": _sha(prepare_source),
        "accelerated_original_local_source_sha256": _sha(local_source),
        "accelerated_test_dispatch_module_source_sha256": _sha(Path(runner.PROJECT)/"tools"/"sparse_update_benchmark.py"),
        "accelerated_official_test_policy": "inherited lazy --test-final factory; Fashion dispatches only to Fashion test",
        "accelerated_dependency_versions": {"torch": str(torch.__version__),
            "torchvision": _version("torchvision"), "numpy": str(np.__version__),
            "python": platform.python_version(), "torch_git_version": torch.version.git_version,
            "torch_build_configuration_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest()},
        "accelerated_runtime_identity": {"platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "PYTORCH_MPS_FAST_MATH": os.environ.get("PYTORCH_MPS_FAST_MATH"),
            "PYTORCH_MPS_PREFER_METAL": os.environ.get("PYTORCH_MPS_PREFER_METAL")},
        "accelerated_device_policy": "local private copy and minibatches on requested backend; global model/evaluation/codecs/residuals/aggregate/checkpoint CPU",
        "accelerated_batch_order": "CPU torch.Generator seeded by per-client training seed, batch64 shuffle",
        "accelerated_seed_policy": "torch.manual_seed plus torch.mps.manual_seed for MPS; no cross-backend equality claim",
        "accelerated_numeric_scope": "same mathematical local objective; CPU/MPS floating-point trajectories are separate experimental settings",
        "accelerated_project_root": str(Path(runner.PROJECT).resolve()),
    }


def metadata_args(args, metadata):
    params = copy.copy(args)
    for key,value in metadata.items():
        setattr(params,key,value)
    if str(args.dataset).startswith("Cifar10Clean"):
        source_config_path=Path(runner.PROJECT)/"dataset"/args.dataset/"config.json"
        if not source_config_path.is_file():
            raise FileNotFoundError(f"Legacy NPZ source configuration is missing: {source_config_path}")
        source=json.loads(source_config_path.read_text(encoding="utf-8"))
        params.accelerated_npz_source_config_sha256=_sha(source_config_path)
        params.accelerated_npz_dirichlet_alpha=source.get("alpha")
        params.accelerated_npz_dirichlet_generation_seed=source.get("seed")
        params.accelerated_npz_source_config_path=str(source_config_path.resolve())
        params.accelerated_npz_mixing_reassignment_seed=args.split_seed
        params.accelerated_npz_seed_interpretation=("fixed NPZ Dirichlet generation seed comes from source config.seed; "
            "CLI --split-seed controls only the optional mixing reassignment, not original NPZ generation")
    return params


def late_main(backend, route):
    """Called after the selected entry installs all of its own overrides."""
    original_train = runner.train_local
    original_one, original_snapshot = runner.one_run, runner.snapshot_run
    metadata = runtime_metadata(backend, route)

    def patched_local(model, dataset, seed, lr, epochs):
        return train_local(model,dataset,seed,lr,epochs,backend=backend)

    def one_run(args,trainsets,validation,manifest,variant,seed,out):
        return original_one(metadata_args(args,metadata),trainsets,validation,manifest,variant,seed,out)

    def snapshot(args,trainsets,validation,manifest,seed,out):
        return original_snapshot(metadata_args(args,metadata),trainsets,validation,manifest,seed,out)

    runner.train_local = patched_local
    runner.one_run,runner.snapshot_run = one_run,snapshot
    try:
        return _ORIGINAL_RUNNER_MAIN()
    finally:
        runner.train_local = original_train
        runner.one_run,runner.snapshot_run = original_one,original_snapshot


def _fashion_bind_project(fashion):
    """Correct only metadata paths in this process for portable launcher use."""
    original = fashion.source_and_override_metadata
    project = Path(runner.PROJECT).resolve()

    def metadata():
        # Replicate the frozen function's path-dependent values without first
        # trying nonexistent historical sibling paths in an exported package.
        source_names = ("run_fashion.py","run_followup.py","run_controls.py",
                        "run_lagrangian.py","guarded_codec.py","lagrangian_codec.py")
        return {
            "fashion_adapter_name":"run_fashion.py","fashion_adapter_version":1,
            "fashion_source_sha256":{name:_sha(HERE/name) for name in source_names},
            "fashion_model_source_sha256":_sha(inspect.getsourcefile(runner.SmallCNN)),
            "fashion_test_dispatch_module_sha256":_sha(project/"tools"/"sparse_update_benchmark.py"),
            "fashion_runtime_overrides":["runner.prepare=run_fashion.prepare",
                "runner.one_run=Fashion metadata wrapper around run_lagrangian.one_run",
                "runner.snapshot_run=Fashion metadata wrapper around run_lagrangian.snapshot_run",
                "sparse_update_benchmark.load_global_testset=run_fashion.dispatch_global_testset",
                "source_and_override_metadata paths bound by run_accelerated to selected project"],
            "fashion_backbone":"unchanged SmallCNN, 136874 parameters, shared fixed cross-task backbone",
            "fashion_architecture_claim":"comparison protocol only; not an optimized FashionMNIST architecture",
            "fashion_preprocessing":"uint8 gray28 -> zero-pad2 to gray32 -> repeat3 -> FP32 -> (x/255-.5)/.5",
            "fashion_official_test_gate":"explicit --test-final only; never evaluated at import/prepare",
        }
    fashion.source_and_override_metadata = metadata
    return original


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # The inherited training interface is options-only. Accept one optional
    # separator while still inspecting dataset/backend/entry on either side.
    if "--" in argv:
        at=argv.index("--")
        argv=argv[:at]+argv[at+1:]
    parser = argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument("--backend",choices=("cpu","mps"),default="mps")
    parser.add_argument("--entry",choices=("auto","run_controls","run_lagrangian","run_partitions","run_fashion"),default="auto")
    parser.add_argument("--dataset")
    known,forwarded = parser.parse_known_args(argv)
    if known.dataset is None:
        known.dataset={"run_fashion":"FashionIndexA0p1","run_partitions":"Cifar10IndexA0p1"}.get(known.entry,"Cifar10CleanA0p1")
    if "--help" not in argv and "-h" not in argv:
        if known.backend=="mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--backend mps is unavailable in this process. Use an MPS-enabled execution context or explicitly choose --backend cpu; no fallback occurred.")
    forwarded.extend(("--dataset",known.dataset))
    if known.dataset.startswith("Fashion"):
        automatic_entry="run_fashion"
    elif known.dataset.startswith("Cifar10Index"):
        automatic_entry="run_partitions"
    elif known.dataset.startswith("Cifar10Clean"):
        automatic_entry="run_lagrangian"
    else:
        raise ValueError(f"Unsupported dataset {known.dataset!r}; expected Cifar10Clean*, Cifar10Index*, or FashionIndex*")
    entry=automatic_entry if known.entry=="auto" else known.entry
    compatible=(entry==automatic_entry or
                (automatic_entry=="run_lagrangian" and entry=="run_controls"))
    if not compatible:
        raise ValueError(f"Entry {entry} is incompatible with dataset {known.dataset}; expected {automatic_entry}")
    route={"run_fashion":"FashionIndex -> run_fashion -> run_lagrangian",
           "run_partitions":"Cifar10Index -> run_partitions.prepare + run_lagrangian",
           "run_lagrangian":"Cifar10Clean -> run_lagrangian",
           "run_controls":"Cifar10Clean -> run_controls"}[entry]
    previous_main = runner.main
    previous_prepare = runner.prepare
    previous_raw = None
    previous_fashion_metadata = None
    fashion = partitions = None
    runner.main = lambda: late_main(known.backend,route)
    try:
        if entry=="run_fashion":
            import run_fashion as fashion
            previous_fashion_metadata = _fashion_bind_project(fashion)
            return fashion.main(forwarded)
        if entry=="run_partitions":
            import run_partitions as partitions
            previous_raw = partitions.RAW
            partitions.RAW = Path(runner.PROJECT)/"dataset"/"Cifar10"/"rawdata"
            partitions.load_official_train.cache_clear()
            runner.prepare = partitions.prepare
            return lagrangian.main(forwarded)
        return controls.main(forwarded) if entry=="run_controls" else lagrangian.main(forwarded)
    finally:
        runner.main,runner.prepare = previous_main,previous_prepare
        if partitions is not None:
            partitions.RAW = previous_raw
            partitions.load_official_train.cache_clear()
        if fashion is not None and previous_fashion_metadata is not None:
            fashion.source_and_override_metadata = previous_fashion_metadata


if __name__=="__main__":
    main()
