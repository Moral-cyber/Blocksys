"""Portable launcher for the frozen follow-up experiment adapters.

Resolves an explicitly supplied FL project, FL_PROJECT_ROOT, or a bounded
relative export layout.  It never searches the disk or changes frozen sources.
--dry-run validates paths and prints routing information without importing ML
modules, reading images, creating outputs, or starting training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ENTRIES = ("run_controls", "run_lagrangian", "run_partitions", "run_accelerated")


def _project_error(path):
    missing = []
    if not (path / "tools" / "fedavg_update_diagnostic.py").is_file():
        missing.append("tools/fedavg_update_diagnostic.py")
    if not (path / "dataset").is_dir():
        missing.append("dataset/ directory")
    return missing


def resolve_project(explicit=None, *, environment=None, here=None):
    """Resolve one project root, with authoritative explicit/env selections.

    A supplied invalid path raises an error instead of silently falling back.
    The only automatic candidates are known sibling/export relative layouts.
    """
    environment = os.environ if environment is None else environment
    here = HERE if here is None else Path(here).expanduser().resolve()
    if explicit is not None:
        chosen, source = Path(explicit).expanduser().resolve(), "--project-root"
    elif environment.get("FL_PROJECT_ROOT"):
        chosen, source = Path(environment["FL_PROJECT_ROOT"]).expanduser().resolve(), "FL_PROJECT_ROOT"
    else:
        candidates = [
            here.parent / "tinyproto_diag",           # current work/ layout
            here.parent.parent / "02_实验工程",       # paper/new_material/scripts/
            here.parent / "02_实验工程",              # paper/new_material/
        ]
        # Resolve and de-duplicate only these fixed paths; no glob or scanning.
        candidates = list(dict.fromkeys(p.resolve() for p in candidates))
        for candidate in candidates:
            if not _project_error(candidate):
                return candidate, "known_relative_layout"
        listed = "\n".join("  - " + str(p) for p in candidates)
        raise ValueError(
            "Cannot locate the FL project in the supported relative layouts.\n"
            "Supply --project-root or FL_PROJECT_ROOT pointing to the folder containing tools/ and dataset/.\n"
            "Checked only:\n" + listed)
    missing = _project_error(chosen)
    if missing:
        raise ValueError(f"Invalid project root from {source}: {chosen}\n"
                         f"Missing: {', '.join(missing)}.\n"
                         "Choose the experiment root containing tools/ and dataset/; no fallback was attempted.")
    return chosen, source


def _put_first(path):
    text = str(path)
    sys.path[:] = [p for p in sys.path if p != text]
    sys.path.insert(0, text)


def bind_project(project):
    """Import the requested data helper first, then bind the local runner.

    The helper's ROOT and the actual load_client function globals are checked;
    changing runner.PROJECT alone after a wrong helper import is insufficient.
    """
    tools = project / "tools"
    expected_helper = (tools / "fedavg_update_diagnostic.py").resolve()
    cached = sys.modules.get("fedavg_update_diagnostic")
    if cached is not None:
        cached_path = Path(getattr(cached, "__file__", "")).resolve()
        if cached_path != expected_helper:
            raise RuntimeError("A different fedavg_update_diagnostic is already imported. "
                               "Run launch.py in a fresh Python process instead of reusing that module. "
                               f"Loaded: {cached_path}; requested: {expected_helper}")
    _put_first(tools)
    helper = importlib.import_module("fedavg_update_diagnostic")
    if Path(helper.__file__).resolve() != expected_helper:
        raise RuntimeError(f"Imported helper from unexpected path: {helper.__file__}")
    if Path(helper.ROOT).resolve() != project:
        raise RuntimeError(f"Requested project {project}, but helper ROOT is {helper.ROOT}")

    # The data helper is now cached at the right identity. Import the runner
    # from this export folder, even if other experiment folders contain files
    # with the same adapter names.
    _put_first(HERE)
    runner = importlib.import_module("run_followup")
    if Path(runner.__file__).resolve() != (HERE / "run_followup.py").resolve():
        raise RuntimeError(f"Imported run_followup from unexpected path: {runner.__file__}")
    runner.PROJECT = project
    loader_root = Path(runner.load_client.__globals__["ROOT"]).resolve()
    if loader_root != project or runner.load_client is not helper.load_client:
        raise RuntimeError(f"Runner load_client is bound to the wrong project: {loader_root}")
    if runner.SmallCNN is not helper.SmallCNN:
        raise RuntimeError("Runner model class does not come from the selected project's helper")
    # Original runner inserts its historical default tools path. Re-establish
    # the selected tools priority for optional official-test helper imports.
    _put_first(tools)
    existing_benchmark = sys.modules.get("sparse_update_benchmark")
    if existing_benchmark is not None and Path(existing_benchmark.__file__).resolve() != (tools / "sparse_update_benchmark.py").resolve():
        raise RuntimeError("A sparse_update_benchmark from a different project is already loaded; use a fresh process")
    return runner, helper


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", help="FL project folder containing tools/ and dataset/")
    parser.add_argument("--entry", choices=ENTRIES, default="run_controls")
    parser.add_argument("--dry-run", action="store_true", help="resolve paths and route only; no imports/training/output writes")
    args, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    try:
        project, source = resolve_project(args.project_root)
    except ValueError as exc:
        parser.error(str(exc))
    entry_path = HERE / (args.entry + ".py")
    if not entry_path.is_file():
        parser.error(f"Entry script is missing from the export folder: {entry_path}")
    info = {"dry_run": args.dry_run, "entry": args.entry,
            "entry_source": str(entry_path), "project_root": str(project),
            "project_resolution": source,
            "helper_source": str(project / "tools" / "fedavg_update_diagnostic.py"),
            "dataset_directory": str(project / "dataset"),
            "forwarded_args": forwarded}
    if args.dry_run:
        info["loader_root_verification"] = "not imported in dry-run; checked before real execution"
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        return info
    runner, helper = bind_project(project)
    # run_partitions must be imported only after runner.PROJECT is bound.
    # Some frozen versions still compute RAW from HERE; bind that global in
    # this process as well without changing any source/checksum.
    _put_first(HERE)
    entry = importlib.import_module(args.entry)
    if Path(entry.__file__).resolve() != entry_path.resolve():
        raise RuntimeError(f"Imported {args.entry} from unexpected path: {entry.__file__}")
    _put_first(project / "tools")
    if args.entry == "run_partitions":
        actual_raw = (project / "dataset" / "Cifar10" / "rawdata").resolve()
        previous_raw = Path(entry.RAW).resolve()
        entry.RAW = actual_raw
        if hasattr(entry.load_official_train, "cache_clear"):
            entry.load_official_train.cache_clear()
        info["partition_raw_directory"] = str(entry.RAW)
        info["partition_raw_rebound"] = previous_raw != actual_raw
        if entry.RAW != actual_raw:
            raise RuntimeError("Could not bind index-partition source data to the selected project")
    info["verified_loader_root"] = str(Path(helper.ROOT).resolve())
    info["helper_source_sha256"] = hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()
    info["launcher_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
    return entry.main(forwarded)


if __name__ == "__main__":
    main()
