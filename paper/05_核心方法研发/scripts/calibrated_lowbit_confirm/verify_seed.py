"""Reuse the pinned independent verifier for a specified fresh training seed.

The original verifier stays unchanged. Only its three expected training-seed
integer literals and one per-seed disclosure string are specialized in memory.
Every mathematical formula, tolerance and performance gate remains unchanged.
No author execution module, original image or TEST is loaded.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import types


BASE_SHA = "f5db4270a7b6b9d280fdeba68e165467e62c0e1ac556724f0fab94c5c9237b1c"
ORIGINAL_SEED = 20261211


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=(20261212, 20261213), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    base = here / "verify_independent.py"
    base_bytes, wrapper_bytes = base.read_bytes(), Path(__file__).read_bytes()
    if sha(base_bytes) != BASE_SHA:
        raise ValueError("original independent verifier differs from pinned source")
    evaluation = json.loads((args.evaluation_dir / "result.json").read_text())
    if evaluation["status"] != "DONE" or evaluation["training_seed"] != args.seed:
        raise ValueError("completed evaluation with the explicit expected seed required")
    for method in ("ce_uniform3", "fedlc_uniform3", "fedlc_dense"):
        config = json.loads((args.model_dir / method / "config.json").read_text())
        if config["seed"] != args.seed:
            raise ValueError("all three configs must match the explicit expected seed")
    if args.output_dir.exists():
        raise ValueError("fresh independent output directory required")

    class SpecializeSeed(ast.NodeTransformer):
        integer_changes = 0
        disclosure_changes = 0

        def visit_Constant(self, node):
            if type(node.value) is int and node.value == ORIGINAL_SEED:
                node.value = args.seed
                self.integer_changes += 1
            elif isinstance(node.value, str) and "trainingseed20261211" in node.value:
                node.value = node.value.replace("trainingseed20261211", f"trainingseed{args.seed}")
                self.disclosure_changes += 1
            return node

    tree = ast.parse(base_bytes, filename=str(base))
    transform = SpecializeSeed()
    tree = transform.visit(tree)
    if (transform.integer_changes, transform.disclosure_changes) != (3, 1):
        raise ValueError("expected exactly three seed assertions and one disclosure specialization")
    ast.fix_missing_locations(tree)
    specialized_source = ast.unparse(tree) + "\n"
    module = types.ModuleType(f"_pinned_independent_verifier_seed{args.seed}")
    module.__file__ = str(base)
    exec(compile(tree, str(base), "exec"), module.__dict__)
    result = module.verify(here, args.model_dir.resolve(), args.evaluation_dir.resolve())
    if base.read_bytes() != base_bytes or Path(__file__).read_bytes() != wrapper_bytes:
        raise ValueError("verifier or wrapper source changed during verification")
    result["training_seed"] = args.seed
    result["partition_seed"] = 20261203
    result["verifier_identity"] = {
        "base_path": str(base),
        "base_sha256": BASE_SHA,
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": sha(wrapper_bytes),
        "specialized_source_sha256": sha(specialized_source.encode()),
        "expected_seed_integer_replacements": transform.integer_changes,
        "disclosure_replacements": transform.disclosure_changes,
        "math_and_gate_changes": 0,
        "base_and_wrapper_unchanged": True,
    }
    result["limits"].append("This is one independently checked fresh-seed replicate on the same partition. Aggregate conclusions require every prescribed seed and stronger calibration baselines; this per-seed FOLLOW_UP alone does not finish core research.")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "specialized_verifier.py").write_text(specialized_source)
    (args.output_dir / "LOCAL_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    report = module.make_report(result).replace("# 新分区确认独立核查", f"# 新分区确认独立核查：seed{args.seed}", 1)
    report += "原已认证verifier保持不变；wrapper只替换3处预期seed断言及1处披露文字，公式、容差和gate不变。\n"
    (args.output_dir / "LOCAL_CHECK.md").write_text(report)
    print(json.dumps({key: result[key] for key in ("status", "training_seed", "artifact_check", "checks_passed", "checks_total", "decision", "gates", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
