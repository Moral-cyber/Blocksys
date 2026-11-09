"""Local evidence bridge; never invokes a model, trains, or invents measurements."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import platform
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent
ROOT = BRIDGE.parent
UPSTREAM = ROOT / "upstream/Open-ScientistTwo"
sys.path.insert(0, str(UPSTREAM / "src"))

from pydantic import Field, field_validator  # noqa: E402
from scientisttwo.live.models import Hypothesis, LimitationAssessment, Seeds  # noqa: E402
from scientisttwo.models import Record  # noqa: E402
from scientisttwo.research.policy import ResearchPolicy, profile_updates  # noqa: E402
from scientisttwo.store import IntegrityError, Store, file_digest  # noqa: E402

BACKEND = "Codex native collaboration"
EXPERIMENT_ROOTS = (
    Path("/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105"),
    Path("/Users/wind/Desktop/Codex/paper/03_英文论文/revision_artifacts/route_a_20261105"),
)
SCHEMAS = {"limitation": LimitationAssessment, "seeds": Seeds, "hypothesis": Hypothesis}
STAGES = {"limitation": "limitation", "seeds": "search", "hypothesis": "search"}
IDENTIFIER = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")


def now():
    return datetime.now(UTC).isoformat()


def strict_json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON token: {value}")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(data, parse_constant=invalid, object_pairs_hook=pairs)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def identifier(value):
    if not IDENTIFIER.fullmatch(value):
        raise ValueError("IDs must use 1-100 ASCII letters, digits, hyphens or underscores")
    return value


class NativeReceipt(Record):
    backend: str
    agent_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    produced_at: datetime
    task: str = Field(min_length=10)
    output_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: str
    capture: str

    @field_validator("backend")
    @classmethod
    def native_backend(cls, value):
        if value != BACKEND:
            raise ValueError(f"Only actual {BACKEND} outputs may be imported")
        return value

    @field_validator("agent_id")
    @classmethod
    def native_id(cls, value):
        if not re.fullmatch(r"/root(?:/[a-zA-Z0-9_-]+)*", value):
            raise ValueError("Supply the actual native agent's canonical task path")
        return value

    @field_validator("status")
    @classmethod
    def completed(cls, value):
        if value != "completed":
            raise ValueError("Only completed native outputs may be imported")
        return value

    @field_validator("capture")
    @classmethod
    def captured(cls, value):
        if value != "written_by_native_agent":
            raise ValueError("The producing native agent must write the output and receipt")
        return value

    @field_validator("produced_at")
    @classmethod
    def timestamp(cls, value):
        if value.tzinfo is None:
            raise ValueError("produced_at requires a time zone")
        if value.timestamp() > time.time() + 60:
            raise ValueError("produced_at cannot be in the future")
        return value


def source_identity():
    lock = read_json(BRIDGE / "SOURCE_LOCK.json")
    errors = []
    if platform.python_version_tuple()[:2] != tuple(lock["python_major_minor"]):
        errors.append("Python major/minor differs from source lock")
    for package, expected in lock["packages"].items():
        if version(package) != expected:
            errors.append(f"Dependency version changed: {package}")
    for entry in lock["files"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            errors.append(f"Missing/invalid source: {entry['path']}")
        elif file_digest(path) != entry["sha256"]:
            errors.append(f"Source changed: {entry['path']}")
    if errors:
        raise IntegrityError("; ".join(errors))
    return lock


def campaign_path(value):
    path = Path(value).expanduser().resolve()
    base = (BRIDGE / "campaigns").resolve()
    if not path.is_relative_to(base) or path == base or ".locks" in path.relative_to(base).parts:
        raise ValueError("Campaigns must be inside bridge/campaigns")
    return path


def open_campaign(path):
    if not (path / "campaign.sqlite").is_file():
        raise ValueError("Campaign does not exist; use init first")
    store = Store(path)
    if store.get("bridge_manifest") is None:
        store.close()
        raise ValueError("Not a native bridge campaign")
    return store


def bind_source(store):
    current = source_identity()
    frozen = store.json(store.get("bridge_manifest"))["source_lock"]
    if current != frozen:
        raise IntegrityError("Campaign source lock differs; start a new campaign")


def index(store, name):
    return store.get("bridge_" + name, {})


def append_entry(store, name, key, receipt):
    values = index(store, name)
    if key in values:
        raise ValueError(f"Immutable {name} ID already exists: {key}")
    ref = store.put_json(receipt, "bridge-" + name)
    values[key] = ref
    store.set("bridge_" + name, values)
    return ref


def init(store, args):
    if store.get("bridge_manifest") is not None:
        raise ValueError("Campaign already initialized; it cannot be reset")
    policy = profile_updates("paper-methodology")
    # Upstream chooses five; the source paper does not disclose N_seed. Never
    # pad a real native batch with invented ideas to match that implementation.
    policy["workflow"] = policy["workflow"].model_copy(update={"seed_ideas": args.seed_ideas})
    # Validate the exact upstream policy before serializing it.
    policy["workflow"] = ResearchPolicy.model_validate(policy["workflow"]).model_dump(mode="json")
    contexts = {}
    for value in args.context:
        key, separator, filename = value.partition("=")
        identifier(key)
        if not separator or key in contexts:
            raise ValueError("Context must be a unique ID=absolute-path")
        path = Path(filename).resolve()
        if not path.is_relative_to(ROOT.resolve()) or path.suffix not in {".md", ".txt", ".json"}:
            raise ValueError("Context must be a text document inside 06_ScientistTwo")
        if path.stat().st_size > 2_000_000:
            raise ValueError("Context exceeds 2 MB")
        contexts[key] = {"source_path": str(path), "sha256": file_digest(path),
                         "artifact_ref": store.put_file(path, "bridge-context")}
    manifest = {
        "schema_version": 1, "name": args.name, "created_at": now(),
        "implementation": "local native/host evidence bridge, not an upstream live campaign",
        "actual_backend": BACKEND, "model_transport": "native app agents; external/manual dispatch",
        "execution_adapter": "host experiment artifacts; MPS/CPU, not Docker/Podman",
        "upstream_live_enabled": False, "model_calls_made_by_bridge": 0,
        "policy": policy, "source_lock": source_identity(), "contexts": contexts,
        "experiment_roots": [str(path) for path in EXPERIMENT_ROOTS],
        "deviations": [
            "Uses native agent outputs instead of upstream response-only Codex CLI/API transports.",
            "Imports independently executed host/MPS/CPU runs; it does not run upstream containers.",
            "Uses upstream Store, paper profile and three role schemas, not ResearchController.",
            f"Local seed_ideas={args.seed_ideas} overrides upstream implementation's five; the original paper's N_seed is undisclosed. No fabricated padding.",
            "No automatic search, promotion, ablation, writing, peer-review or acceptance verdict.",
            "Native origin is an agent attestation plus byte hash, not a cryptographic app signature.",
        ],
    }
    ref = store.put_json(manifest, "bridge-manifest")
    store.set("bridge_manifest", ref)
    store.event("bridge-init", "initialized", "Initialized native/host evidence bridge")
    return {"manifest_ref": ref, "status": "initialized", "policy": policy}


def record_role(store, args):
    bind_source(store)
    key = identifier(args.id)
    if key in index(store, "roles"):
        raise ValueError("Role ID already recorded; use a fresh ID")
    path, receipt_path = Path(args.output).resolve(), Path(args.receipt).resolve()
    raw, native_raw = path.read_bytes(), receipt_path.read_bytes()
    receipt = NativeReceipt.model_validate(strict_json(native_raw))
    if hashlib.sha256(raw).hexdigest() != receipt.output_sha256:
        raise IntegrityError("Native output does not match receipt SHA-256")
    output = SCHEMAS[args.role].model_validate(strict_json(raw))
    manifest = store.json(store.get("bridge_manifest"))
    stage = STAGES[args.role]
    limit_key = "limitation_rounds" if stage == "limitation" else "search_rounds"
    if args.round < 1 or args.round > manifest["policy"][limit_key]:
        raise ValueError(f"Round outside frozen {stage} ceiling")
    hypotheses = output.ideas if args.role == "seeds" else [output] if args.role == "hypothesis" else []
    if args.role == "seeds" and len(hypotheses) != manifest["policy"]["workflow"]["seed_ideas"]:
        raise ValueError("Seed batch must match the locally frozen seed_ideas; never pad with fabricated ideas")
    for hypothesis in hypotheses:
        unknown = set(hypothesis.source_ids) - set(manifest["contexts"])
        if unknown:
            raise ValueError(f"Unregistered source IDs: {sorted(unknown)}")
        unknown = set(hypothesis.evidence_ids) - set(index(store, "experiments"))
        if unknown:
            raise ValueError(f"Unregistered experiment IDs: {sorted(unknown)}")
        for experiment_id in hypothesis.evidence_ids:
            evidence = store.json(index(store, "experiments")[experiment_id])
            if evidence["scope"] != "development":
                raise ValueError("Adaptive hypotheses may reference development evidence only")
    for existing_ref in index(store, "roles").values():
        existing = store.json(existing_ref)
        if existing["role"] == args.role and existing["round"] == args.round and args.role != "hypothesis":
            raise ValueError("Role/round already recorded; use the next permitted round")
        if existing["native_receipt"]["request_id"] == receipt.request_id:
            raise ValueError("Native request already recorded")
    content_ref = store.put_json(output.model_dump(mode="json"), "native-role-output")
    raw_ref = store.put(raw, "json", "native-role-raw")
    native_ref = store.put(native_raw, "json", "native-origin-attestation")
    record = {
        "id": key, "role": args.role, "stage": stage, "round": args.round,
        "imported_at": now(), "actual_backend": BACKEND,
        "native_receipt": receipt.model_dump(mode="json"),
        "native_receipt_ref": native_ref, "raw_output_ref": raw_ref,
        "validated_output_ref": content_ref, "output_sha256": receipt.output_sha256,
        "validation_scope": "Schema and supplied reference IDs; no scientific-quality certification",
        "model_calls_made_by_bridge": 0,
    }
    ref = append_entry(store, "roles", key, record)
    store.event("native-" + key, stage, f"Imported actual native {args.role} output, round {args.round}")
    return {"status": "recorded", "id": key, "record_ref": ref, "actual_backend": BACKEND}


def allowed_experiment_file(filename, measurement=True):
    path = Path(filename).resolve(strict=True)
    if not path.is_file() or not any(path.is_relative_to(root.resolve()) for root in EXPERIMENT_ROOTS):
        raise ValueError("Experiment input must be inside a fixed revision_artifacts root")
    if path.stat().st_size > 50_000_000:
        raise ValueError("Experiment evidence exceeds 50 MB; import a result/metrics summary")
    if measurement:
        if path.suffix not in {".json", ".csv"} or not re.fullmatch(r"(?:result|metrics)(?:[_-][\w.-]+)?\.(?:json|csv)", path.name):
            raise ValueError("Measurement files must be result/metrics JSON or CSV")
    elif path.suffix not in {".json", ".log", ".txt"}:
        raise ValueError("Optional existing run receipt must be JSON/log/text")
    return path


def numeric_count(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Nonfinite experiment measurement")
        return 1
    if isinstance(value, dict):
        return sum(numeric_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(numeric_count(item) for item in value)
    return 0


def snapshot_source(store, path, kind):
    before = path.stat()
    ref = store.put_file(path, kind)
    after = path.stat()
    sha = file_digest(path)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or ref.split("/")[1].split(".")[0] != sha:
        raise IntegrityError("Experiment source changed during import; wait for completion")
    return {"source_path": str(path), "source_sha256": sha, "source_size": after.st_size,
            "source_mtime_ns": after.st_mtime_ns, "artifact_ref": ref}


def import_experiment(store, args):
    bind_source(store)
    key = identifier(args.id)
    if key in index(store, "experiments"):
        raise ValueError("Experiment ID already imported; imports are immutable")
    paths = [allowed_experiment_file(value) for value in args.file]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate experiment input")
    files, counts = [], []
    for path in paths:
        snapshot = snapshot_source(store, path, "existing-experiment-result")
        data = store.read(snapshot["artifact_ref"])
        if path.suffix == ".json":
            value = strict_json(data)
            if isinstance(value, dict) and str(value.get("status", "")).upper() in {"RUNNING", "PENDING", "FAILED", "ERROR"}:
                raise ValueError("Cannot import an unfinished/failed experiment as a result")
            count = numeric_count(value)
        else:
            rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
            count = 0
            for row in rows:
                for cell in row.values():
                    try:
                        number = float(cell)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(number):
                        raise ValueError("Nonfinite CSV measurement")
                    count += 1
        if count == 0:
            raise ValueError("Result/metrics file contains no numeric observations")
        files.append(snapshot)
        counts.append(count)
    run_receipt = None
    if args.run_receipt:
        run_receipt = snapshot_source(store, allowed_experiment_file(args.run_receipt, False), "existing-run-receipt")
    record = {
        "id": key, "stage": "experiment_evidence", "imported_at": now(),
        "kind": "existing_host_run_import", "files": files, "numeric_leaf_counts": counts,
        "existing_run_receipt": run_receipt, "declared_execution_device": args.device,
        "device_evidence_scope": "Operator declaration; consult the original run/config/log for verification",
        "execution_performed_by_bridge": False, "measurements_generated_by_bridge": 0,
        "scope": args.scope, "note": args.note,
        "verification_scope": "Source bytes/hash and parseability; importing does not rerun or independently verify measurements",
    }
    ref = append_entry(store, "experiments", key, record)
    store.event("experiment-" + key, "experiment_evidence", "Imported immutable result from an existing host run")
    return {"status": "imported", "id": key, "receipt_ref": ref, "files": len(files), "new_training": 0}


def status(store, args):
    manifest = store.json(store.get("bridge_manifest"))
    return {
        "name": manifest["name"], "implementation": manifest["implementation"],
        "actual_backend": BACKEND, "upstream_live_enabled": False,
        "model_calls_made_by_bridge": 0, "policy": manifest["policy"],
        "roles": {key: {field: store.json(ref)[field] for field in ("role", "stage", "round", "actual_backend")}
                  for key, ref in index(store, "roles").items()},
        "experiments": list(index(store, "experiments")),
        "bridge_run_receipts": len(index(store, "runs")),
        "events": [dict(row) for row in store.db.execute("SELECT * FROM events ORDER BY time")],
        "deviations": manifest["deviations"],
    }


def check(store, args):
    errors, notes = store.verify(), []
    try:
        bind_source(store)
    except (ValueError, OSError, IntegrityError) as error:
        errors.append(str(error))
    manifest = store.json(store.get("bridge_manifest"))
    ResearchPolicy.model_validate(manifest["policy"]["workflow"])
    for key, ref in index(store, "roles").items():
        item = store.json(ref)
        receipt = NativeReceipt.model_validate(item["native_receipt"])
        if NativeReceipt.model_validate(strict_json(store.read(item["native_receipt_ref"]))) != receipt:
            errors.append(f"Native receipt snapshot mismatch: {key}")
        raw = store.read(item["raw_output_ref"])
        if hashlib.sha256(raw).hexdigest() != receipt.output_sha256:
            errors.append(f"Role raw hash mismatch: {key}")
        output = SCHEMAS[item["role"]].model_validate(strict_json(raw))
        if output.model_dump(mode="json") != store.json(item["validated_output_ref"]):
            errors.append(f"Role validated payload mismatch: {key}")
    for key, ref in index(store, "experiments").items():
        item = store.json(ref)
        sources = item["files"] + ([item["existing_run_receipt"]] if item["existing_run_receipt"] else [])
        for source in sources:
            path = Path(source["source_path"])
            if not path.exists():
                notes.append(f"Original moved/missing; stored snapshot remains verified: {key}: {path}")
            elif file_digest(path) != source["source_sha256"]:
                errors.append(f"Original experiment source changed since import: {key}: {path}")
            if hashlib.sha256(store.read(source["artifact_ref"])).hexdigest() != source["source_sha256"]:
                errors.append(f"Experiment snapshot mismatch: {key}")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "notes": notes,
            "artifacts": store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0],
            "roles": len(index(store, "roles")), "experiments": len(index(store, "experiments")),
            "scope": "Bridge/source/artifact integrity; scientific claims and upstream live completion are not certified"}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "record-role", "import-experiment", "check"):
        command = sub.add_parser(name)
        command.add_argument("campaign", help="Absolute path inside bridge/campaigns")
        if name == "init":
            command.add_argument("--name", default="fl-native-host")
            command.add_argument("--context", action="append", default=[], metavar="ID=PATH")
            command.add_argument("--seed-ideas", type=int, choices=(2, 3), default=3,
                                 help="Real native batch size; local adaptation, never fabricated padding")
        elif name == "record-role":
            command.add_argument("--id", required=True)
            command.add_argument("--role", choices=SCHEMAS, required=True)
            command.add_argument("--round", type=int, default=1)
            command.add_argument("--output", required=True)
            command.add_argument("--receipt", required=True)
        elif name == "import-experiment":
            command.add_argument("--id", required=True)
            command.add_argument("--file", action="append", required=True)
            command.add_argument("--run-receipt")
            command.add_argument("--device", choices=("mps", "cpu", "cuda", "mixed", "unspecified"), default="unspecified")
            command.add_argument("--scope", choices=("development", "final", "existing_analysis"), required=True)
            command.add_argument("--note", required=True)
    return root


def main():
    args = parser().parse_args()
    started = now()
    tick = time.monotonic()
    store = None
    lock_file = None
    code = 0
    result = None
    try:
        path = campaign_path(args.campaign)
        locks = BRIDGE / "campaigns/.locks"
        locks.mkdir(parents=True, exist_ok=True)
        lock_file = (locks / (hashlib.sha256(str(path).encode()).hexdigest() + ".lock")).open("a")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if args.command == "init":
            source_identity()
            if path.exists() and any(path.iterdir()):
                raise ValueError("init requires an empty/nonexistent campaign directory")
            store = Store(path)
        else:
            store = open_campaign(path)
        action = {"init": init, "status": status, "record-role": record_role,
                  "import-experiment": import_experiment, "check": check}[args.command]
        result = action(store, args)
        if result.get("status") == "FAIL":
            code = 2
    except (ValueError, OSError, IntegrityError, KeyError) as error:
        code = 2
        result = {"status": "ERROR", "error": str(error)}
    finally:
        if store is not None:
            if store.get("bridge_manifest") is not None:
                run = {"schema_version": 1, "command": args.command, "argv": sys.argv[1:],
                       "started_at": started, "completed_at": now(),
                       "elapsed_seconds": time.monotonic() - tick, "exit_code": code,
                       "host": {"system": platform.system(), "machine": platform.machine(),
                                "python": platform.python_version(), "executable": sys.executable},
                       "actual_backend": BACKEND, "kind": "actual_bridge_command_execution",
                       "model_calls": 0, "training_runs_started": 0,
                       "result": result}
                ref = append_entry(store, "runs", uuid.uuid4().hex, run)
                result = {**result, "bridge_run_receipt_ref": ref}
            store.close()
        if lock_file is not None:
            lock_file.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
