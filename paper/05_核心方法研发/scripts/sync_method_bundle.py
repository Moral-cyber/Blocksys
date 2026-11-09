"""Publish completed research artifacts, retaining prior desktop revisions."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/method_followup"
SOURCE = ROOT / "work/method_followup"
COMPONENTS = ("partial_fusion", "receiver_feedback", "label_head", "normalized_sparse",
              "shadow_training", "next_design", "bottleneck_reanchor", "label_calibration",
              "mixed_precision", "head_control", "control_reference", "exposure_prior", "balanced_quant", "prior_personalization", "prior_bias",
              "calibrated_lowbit_confirm", "calibration_baselines", "calibrated_fashion",
              "prior_vector", "predictive_pfl_confirm", "calibrated_topk_reference", "fedrod_linear_reference",
              "predictive_final_test", "predictive_manuscript", "blocksys_template",
              "full_payload_integrity", "durable_integrity", "durable_integrity_v2")
IGNORED = {"__pycache__", ".DS_Store", ".git"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_tree(source, target):
    shutil.copytree(source, target, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.tmp", ".git",
            "*_private_test_keys", "local_test_keys.json"))


def file_category(path):
    """Keep every top-level source, and fail before copying unknown types."""
    if path.name == ".gitignore":
        return "design"
    suffix = path.suffix.lower()
    if suffix == ".json" and (path.stem.lower().startswith(("search_", "github_"))
            or path.stem.lower().endswith(("_openalex", "_crossref", "_metadata", "_search"))):
        return "literature"
    category = {".py": "scripts", ".md": "design", ".json": "checks",
        ".jsonl": "checks", ".csv": "design", ".tsv": "design", ".log": "results/logs",
        ".pdf": "literature", ".txt": "literature", ".png": "literature",
        ".jpg": "literature", ".jpeg": "literature", ".svg": "literature",
        ".webp": "literature", ".html": "literature", ".htm": "literature",
        ".xml": "literature", ".bib": "literature", ".yaml": "design",
        ".yml": "design", ".toml": "design", ".tex": "design"}.get(suffix)
    if category is None:
        raise ValueError(f"unclassified source file; add an explicit category before staging: {path}")
    return category


def is_result(name):
    return name.startswith(("dev", "diagnostic", "mixing", "results", "check_independent",
                            "evaluation", "confirm_", "final_eval", "adaptation_", "assessment_",
                            "independent_", "quality_", "fashion_", "prospective_",
                            "final_", "test_", "export_", "official_", "train_states_",
                            "reference_preflight", "bcts_", "tables"))


def stage(include_durable_results=False):
    # Preflight all top-level sources before any copies. New file types cannot
    # disappear through a suffix whitelist; durable results are opt-in only.
    planned, deferred = [], []
    for path in sorted(SOURCE.iterdir()):
        if path.name in IGNORED or path.name.endswith(".tmp"):
            continue
        if path.name.endswith("_private_test_keys") or path.name == "local_test_keys.json":
            deferred.append({"source": str(path.relative_to(ROOT)),
                "reason": "random synthetic fixture keys excluded; test runner regenerates them locally"})
            continue
        if path.is_file():
            planned.append((path, OUTPUT / file_category(path) / path.name))
        elif path.is_dir():
            category = "scripts" if path.name == "datasets" else "results"
            planned.append((path, OUTPUT / category / path.name))
    for name in COMPONENTS:
        source = ROOT / "work" / name
        if not source.exists():
            continue
        for item in sorted(source.iterdir()):
            if item.name in IGNORED or item.name.endswith(".tmp"):
                continue
            if item.name.endswith("_private_test_keys") or item.name == "local_test_keys.json":
                deferred.append({"source": str(item.relative_to(ROOT)),
                    "reason": "random synthetic fixture keys excluded; test runner regenerates them locally"})
                continue
            if item.is_file():
                planned.append((item, OUTPUT / file_category(item) / name / item.name))
            elif item.is_dir():
                result_directory = is_result(item.name)
                if name.startswith("durable_integrity") and result_directory and not include_durable_results:
                    deferred.append({"source": str(item.relative_to(ROOT)),
                        "reason": "durable-integrity results may still be writing; not traversed or read"})
                    continue
                category = "results" if result_directory else "literature"
                planned.append((item, OUTPUT / category / name / item.name))
    for category in ("scripts", "design", "checks", "results", "literature"):
        (OUTPUT / category).mkdir(parents=True, exist_ok=True)
    for source, target in planned:
        if source.is_dir():
            copy_tree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(SOURCE / "README.md", OUTPUT / "scripts/README.md")
    copy_tree(ROOT / "work/method_design_review", OUTPUT / "design/independent_review")
    copy_tree(ROOT / "work/literature_followup", OUTPUT / "literature/source_archive")
    if (ROOT / "work/fednova_check").exists():
        copy_tree(ROOT / "work/fednova_check", OUTPUT / "literature/fednova_check")
    records = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.name == "local_test_keys.json" or any(part.endswith("_private_test_keys") for part in path.parts):
            continue
        if not include_durable_results and any(path.is_relative_to(OUTPUT / "results" / name)
                for name in ("durable_integrity", "durable_integrity_v2")):
            continue
        if path.is_file() and path.name not in ("bundle_manifest.json", "desktop_sync_receipt.json"):
            records.append({"path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"created_at": datetime.now().astimezone().isoformat(),
        "phase": "fixed prior-regularized vector method; completed evidence and manuscripts collected; consult current research status for exact scope",
        "scope": "scripts, completed raw component experiments, explicit checks, theory, literature evidence and development decisions; caches excluded",
        "official_test_policy": "final official TEST follows fixed model/state/protocol locks; historical exploration and development evidence remain separately labeled",
        "source_inventory": [{"source": str(source.relative_to(ROOT)),
            "target": str(target.relative_to(OUTPUT)), "kind": "directory" if source.is_dir() else "file"}
            for source, target in planned],
        "deferred_sources": deferred,
        "fixture_key_policy": "No synthetic private test keys in manifest or published bundle; test runners generate fresh local keys. Original work evidence retained.",
        "durable_results_included": include_durable_results,
        "records": records, "files": len(records), "bytes": sum(x["bytes"] for x in records)}
    (OUTPUT / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("created_at", "files", "bytes", "phase")}, ensure_ascii=False))


def publish():
    destination = Path("/Users/wind/Desktop/Codex/paper/06_核心方法研发")
    if not destination.is_dir():
        raise FileNotFoundError("existing paper/06_核心方法研发 must exist; refusing to invent another paper root")
    manifest = json.loads((OUTPUT / "bundle_manifest.json").read_text())
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    copied, identical, archived = 0, 0, []
    for row in manifest["records"] + [{"path": "bundle_manifest.json", "sha256": digest(OUTPUT / "bundle_manifest.json")}]:
        source, target = OUTPUT / row["path"], destination / row["path"]
        if digest(source) != row["sha256"]:
            raise RuntimeError(f"staged source changed: {source}")
        if target.exists():
            if digest(target) == row["sha256"]:
                identical += 1
                continue
            backup = destination / "_previous_revisions" / stamp / row["path"]
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            archived.append(str(backup.relative_to(destination)))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != row["sha256"]:
            raise RuntimeError(f"copy verification failed: {target}")
        copied += 1
    receipt = {"time": datetime.now().astimezone().isoformat(), "destination": str(destination),
        "manifest_sha256": digest(OUTPUT / "bundle_manifest.json"), "verified_files": len(manifest["records"]) + 1,
        "copied": copied, "identical": identical, "prior_revisions_preserved": archived,
        "other_desktop_directories_changed": False}
    for path in (OUTPUT / "desktop_sync_receipt.json", destination / "desktop_sync_receipt.json"):
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in receipt.items() if k != "prior_revisions_preserved"}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "publish"))
    parser.add_argument("--include-durable-results", action="store_true",
        help="stage completed durable-integrity results only after their writer has finished")
    args = parser.parse_args()
    if args.mode == "publish" and args.include_durable_results:
        parser.error("--include-durable-results applies to stage only")
    stage(include_durable_results=args.include_durable_results) if args.mode == "stage" else publish()
