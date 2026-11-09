"""Backend-isolated adapter to summarize_followup's validation-only reports.

Uses all original DONE/seed/pairing rules, additionally separating numerical
backends and acceleration source/runtime/model/dependency identities. Historical
CPU runs without accelerated metadata remain in a distinct legacy group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import summarize_followup as base


_ORIGINAL_SETTINGS = base.settings


def settings(config):
    _, relevant = _ORIGINAL_SETTINGS(config)
    backend = config.get("training_backend", "legacy_cpu_no_accelerator_metadata")
    relevant["training_backend"] = backend
    for key in ("accelerated_adapter_source_sha256", "accelerated_adapter_version",
                "accelerated_route", "accelerated_source_sha256",
                "accelerated_model_source_sha256", "accelerated_loader_source_sha256",
                "accelerated_prepare_source_sha256", "accelerated_original_local_source_sha256",
                "accelerated_test_dispatch_module_source_sha256", "accelerated_official_test_policy",
                "accelerated_dependency_versions", "accelerated_runtime_identity",
                "accelerated_batch_order", "accelerated_seed_policy",
                "accelerated_npz_source_config_sha256", "accelerated_npz_dirichlet_alpha",
                "accelerated_npz_dirichlet_generation_seed", "accelerated_npz_mixing_reassignment_seed",
                "lagrangian_codec_source_sha256", "lagrangian_adapter_source_sha256",
                "fashion_source_sha256", "fashion_model_source_sha256"):
        relevant[key] = config.get(key)
    raw = json.dumps(relevant,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12], relevant


def summarize(input_path,output_path):
    previous=base.settings
    base.settings=settings
    try:
        metadata=base.summarize(Path(input_path),Path(output_path))
    finally:
        base.settings=previous
    metadata["accelerated_summary_policy"]="backend/source/runtime/model/dependency identities isolated; legacy CPU separate"
    metadata["accelerated_summarizer_source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (Path(output_path)/"summary_metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    report=Path(output_path)/"summary.md"
    content=report.read_text(encoding="utf-8")
    paragraph=("\n本汇总额外隔离 `training_backend`、加速源码、模型/数据工具源码、软件版本及运行身份。"
               "CPU 与 MPS 不混池，不做跨后端的同种子配对；缺少加速元数据的历史 CPU 另分组。"
               "这些分组身份可在 `summary_metadata.json` 的 settings 中核对。\n")
    first,rest=content.split("\n",1)
    report.write_text(first+"\n"+paragraph+rest,encoding="utf-8")
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not args.input.is_dir():
        p.error("--input must be an existing runs directory")
    info=summarize(args.input,args.output)
    print(json.dumps({k:v for k,v in info.items() if k not in ("settings","issues")},indent=2))


if __name__=="__main__":
    main()
