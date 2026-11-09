"""Summarize validation-only development runs without pseudo-replication.

The independent repetition is the training seed, never an individual round.
DONE training trajectories, unfinished runs, and snapshot counterfactuals are
kept separate.  Official-test fields are deliberately not accessed or reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TRAIN_METRICS = (
    "final_validation_accuracy", "final_validation_loss",
    "last5_validation_accuracy_mean", "last5_validation_loss_mean",
    "cumulative_bytes", "mean_filler_fraction", "mean_aggregate_relative_error",
)
SNAPSHOT_METRICS = (
    "validation_accuracy", "validation_loss", "aggregate_relative_error",
    "aggregate_error_squared", "mean_filler_fraction",
    "delta_accuracy_pp_vs_topk", "delta_loss_vs_topk",
    "delta_aggregate_relative_error_vs_topk", "delta_aggregate_error_squared_vs_topk",
)
PAIR_METRICS = (
    "final_accuracy_difference_pp", "final_loss_difference",
    "last5_accuracy_difference_pp", "last5_loss_difference",
    "mean_aggregate_relative_error_difference",
)


def number(value: Any) -> float | None:
    if isinstance(value, (bool, str)):
        return None
    try:
        out = float(value)
    except (ValueError, TypeError):
        return None
    return out if math.isfinite(out) else None


def average(values) -> float | None:
    numbers = [v for x in values if (v := number(x)) is not None]
    return float(np.mean(numbers)) if numbers else None


def sample_stats(values):
    xs = [v for x in values if (v := number(x)) is not None]
    return {
        "n": len(xs),
        "mean": float(np.mean(xs)) if xs else None,
        "sample_sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None,
    }


def csv_write(path: Path, rows: list[dict], prefix=()):
    fields = list(prefix) + sorted({k for row in rows for k in row} - set(prefix))
    with path.open("w", newline="", encoding="utf-8") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def settings(config):
    """Prevent pooling incompatible designs merely sharing method/mix/rho.

    Variant lists and output locations do not alter one trajectory.  Seed is
    intentionally excluded because it is the replication axis.  Data/source
    identities, rounds and optimization settings are deliberately retained.
    """
    manifest = config.get("data_manifest", {})
    names = ("dataset", "rounds", "lr", "local_epochs", "split_seed",
             "validation_size", "parameter_count", "codec_source_sha256",
             "runner_source_sha256", "adapter_source_sha256", "adapter_version")
    relevant = {key: config.get(key) for key in names}
    for key in ("source", "split_seed", "train_samples", "client_sizes",
                "validation_ids_sha256", "validation_samples",
                "assignment_label_sha256"):
        relevant["data_" + key] = manifest.get(key)
    raw = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12], relevant


def aggregate(rows, group_keys, metrics):
    buckets = {}
    for row in rows:
        key = tuple(row.get(x) for x in group_keys)
        buckets.setdefault(key, []).append(row)
    results = []
    def sort_value(value):
        if value is None:
            return (0, "")
        if isinstance(value, (int, float)):
            return (1, value)
        return (2, str(value))
    for key, bucket in sorted(buckets.items(), key=lambda x: tuple(sort_value(v) for v in x[0])):
        out = dict(zip(group_keys, key))
        out["n_seeds"] = len({row["seed"] for row in bucket})
        out["seeds"] = ",".join(map(str, sorted({row["seed"] for row in bucket})))
        for metric in metrics:
            stat = sample_stats(row.get(metric) for row in bucket)
            for name, value in stat.items():
                out[metric + "_" + name] = value
        results.append(out)
    return results


def summarize(input_path: Path, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    if (input_path / "config.json").exists() or (input_path / "result.json").exists():
        paths = [input_path / "result.json"]
    else:
        paths = sorted(set(input_path.rglob("result.json")) |
                       {p.with_name("result.json") for p in input_path.rglob("config.json")})
    runs, snapshots, incomplete, issues, setting_records = [], [], [], [], {}
    for result_path in paths:
        run_path = result_path.parent
        base = {"run_id": run_path.name, "path": str(run_path.resolve())}
        try:
            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            result = (json.loads(result_path.read_text(encoding="utf-8"))
                      if result_path.exists() else {"status": "NO_RESULT_YET", "rows": []})
        except (OSError, ValueError) as e:
            incomplete.append({**base, "status": "UNREADABLE", "detail": str(e)})
            issues.append({**base, "issue": "Cannot read config/result", "detail": str(e)})
            continue
        setting_id, setting_detail = settings(config)
        setting_records[setting_id] = setting_detail
        mode = config.get("mode", "train")
        seed = config.get("seed", result.get("seed"))
        if not isinstance(seed, int) or isinstance(seed, bool):
            issues.append({**base, "issue": "Missing or noninteger seed; excluded"})
            incomplete.append({**base, "status": "INVALID", "detail": "missing seed"})
            continue
        common = {**base, "mode": mode, "seed": seed,
                  "mix": config.get("mixing", config.get("data_manifest", {}).get("mixing")),
                  "rho": config.get("fraction"), "settings_id": setting_id,
                  "requested_rounds": config.get("rounds")}
        rows = result.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        status = result.get("status", "UNKNOWN")
        if status != "DONE":
            observed = max((row.get("round", 0) for row in rows), default=0)
            incomplete.append({**common, "status": status, "observed_round": observed,
                               "observed_rows": len(rows)})
            continue
        if config.get("data_manifest", {}).get("test_used_for_selection") is True:
            issues.append({**base, "issue": "Manifest says official test used for selection; excluded"})
            incomplete.append({**common, "status": "EXCLUDED_TEST_SELECTION"})
            continue
        if not rows:
            issues.append({**base, "issue": "DONE run has no rows; excluded"})
            incomplete.append({**common, "status": "INVALID_EMPTY_DONE"})
            continue

        if mode == "snapshot":
            seen = set()
            for row in rows:
                key = (row.get("round"), row.get("variant"))
                if key in seen:
                    raise ValueError(f"Duplicate snapshot row at {run_path}: {key}")
                seen.add(key)
                val = row.get("validation", {})
                acc, loss = number(val.get("accuracy")), number(val.get("loss"))
                if acc is None or loss is None:
                    issues.append({**base, "issue": f"Invalid snapshot validation metrics {key}; row excluded"})
                    continue
                snapshots.append({**common, "method": row.get("variant"),
                    "round": row.get("round"), "status": "DONE",
                    "validation_accuracy": acc, "validation_loss": loss,
                    **{key: number(row.get(key)) for key in (
                        "aggregate_relative_error", "aggregate_error_squared",
                        "error_cross_term", "mean_filler_fraction",
                        "mean_retained_topk_energy_ratio", "mean_alignment_gain")}})
        elif mode == "train":
            rounds = [row.get("round") for row in rows]
            expected = config.get("rounds")
            if expected is not None and rounds != list(range(1, int(expected) + 1)):
                issues.append({**base, "issue": "DONE rows are not a complete ordered trajectory; excluded"})
                incomplete.append({**common, "status": "INVALID_INCOMPLETE_DONE",
                                   "observed_rows": len(rows)})
                continue
            last = rows[-1]
            val = last.get("validation", {})
            acc, loss = number(val.get("accuracy")), number(val.get("loss"))
            if acc is None or loss is None or not 0 <= acc <= 1:
                issues.append({**base, "issue": "Invalid final validation accuracy/loss; excluded"})
                incomplete.append({**common, "status": "INVALID_METRICS"})
                continue
            tail = rows[-5:]
            runs.append({**common, "status": "DONE", "method": config.get("variant", result.get("variant")),
                "completed_rounds": len(rows), "final_round": last.get("round"),
                "final_validation_accuracy": acc, "final_validation_loss": loss,
                "last5_round_count": len(tail),
                "last5_validation_accuracy_mean": average(r.get("validation", {}).get("accuracy") for r in tail),
                "last5_validation_loss_mean": average(r.get("validation", {}).get("loss") for r in tail),
                "cumulative_bytes": last.get("cumulative_bytes"),
                "mean_filler_fraction": average(r.get("mean_filler_fraction") for r in rows),
                "mean_aggregate_relative_error": average(r.get("aggregate_relative_error") for r in rows),
                "mean_aggregate_error_squared": average(r.get("aggregate_error_squared") for r in rows),
                "mean_error_cross_term": average(r.get("error_cross_term") for r in rows),
                "total_recorded_elapsed_seconds": sum(number(r.get("elapsed_seconds")) or 0 for r in rows),
            })
        else:
            issues.append({**base, "issue": f"Unknown mode {mode}; excluded"})

    # Do not quietly count duplicate seed/config trajectories as replications.
    train_keys = ("settings_id", "mix", "rho", "method", "seed")
    snapshot_keys = train_keys + ("round",)
    for data, keys in ((runs, train_keys), (snapshots, snapshot_keys)):
        buckets = {}
        for row in data:
            key = tuple(row.get(k) for k in keys)
            buckets.setdefault(key, []).append(row)
        duplicates = {key for key, bucket in buckets.items() if len(bucket) > 1}
        if duplicates:
            for key in duplicates:
                issues.append({"issue": "Duplicate seed/config excluded from statistics",
                               "key": str(key), "paths": [r["path"] for r in buckets[key]]})
            data[:] = [r for r in data if tuple(r.get(k) for k in keys) not in duplicates]

    baseline_keys = ("settings_id", "mix", "rho", "seed")
    baseline = {tuple(r[k] for k in baseline_keys): r for r in runs if r["method"] == "topk_ef"}
    paired = []
    for row in runs:
        if row["method"] == "topk_ef":
            continue
        reference = baseline.get(tuple(row[k] for k in baseline_keys))
        if reference is None:
            continue
        paired.append({**{key: row[key] for key in baseline_keys}, "method": row["method"],
            "run_id": row["run_id"], "baseline_run_id": reference["run_id"],
            "final_accuracy_difference_pp": 100 * (row["final_validation_accuracy"] - reference["final_validation_accuracy"]),
            "final_loss_difference": row["final_validation_loss"] - reference["final_validation_loss"],
            "last5_accuracy_difference_pp": 100 * (row["last5_validation_accuracy_mean"] - reference["last5_validation_accuracy_mean"]),
            "last5_loss_difference": row["last5_validation_loss_mean"] - reference["last5_validation_loss_mean"],
            "mean_aggregate_relative_error_difference": row["mean_aggregate_relative_error"] - reference["mean_aggregate_relative_error"],
        })

    snap_base_keys = baseline_keys + ("round",)
    snap_base = {tuple(r[k] for k in snap_base_keys): r for r in snapshots if r["method"] == "topk_ef"}
    for row in snapshots:
        reference = snap_base.get(tuple(row[k] for k in snap_base_keys))
        for field in ("delta_accuracy_pp_vs_topk", "delta_loss_vs_topk",
                      "delta_aggregate_relative_error_vs_topk", "delta_aggregate_error_squared_vs_topk"):
            row[field] = None
        if reference is not None:
            row["delta_accuracy_pp_vs_topk"] = 100 * (row["validation_accuracy"] - reference["validation_accuracy"])
            row["delta_loss_vs_topk"] = row["validation_loss"] - reference["validation_loss"]
            for metric in ("aggregate_relative_error", "aggregate_error_squared"):
                if row.get(metric) is not None and reference.get(metric) is not None:
                    row["delta_" + metric + "_vs_topk"] = row[metric] - reference[metric]

    groups = aggregate(runs, ("settings_id", "method", "mix", "rho"), TRAIN_METRICS)
    paired_groups = aggregate(paired, ("settings_id", "method", "mix", "rho"), PAIR_METRICS)
    snap_groups = aggregate(snapshots, ("settings_id", "method", "mix", "rho", "round"), SNAPSHOT_METRICS)
    csv_write(output_path / "runs.csv", runs, ("run_id", "method", "mix", "rho", "seed", "status"))
    csv_write(output_path / "incomplete_runs.csv", incomplete, ("run_id", "status"))
    csv_write(output_path / "groups.csv", groups, ("settings_id", "method", "mix", "rho", "n_seeds"))
    csv_write(output_path / "paired_vs_topk.csv", paired, ("method", "mix", "rho", "seed"))
    csv_write(output_path / "paired_summary.csv", paired_groups, ("settings_id", "method", "mix", "rho", "n_seeds"))
    csv_write(output_path / "snapshots.csv", snapshots, ("method", "mix", "rho", "seed", "round"))
    csv_write(output_path / "snapshot_groups.csv", snap_groups, ("settings_id", "method", "mix", "rho", "round", "n_seeds"))
    metadata = {"input": str(input_path.resolve()), "official_test_fields_used": False,
                "repetition_unit": "training seed; rounds are repeated observations",
                "settings": setting_records, "issues": issues,
                "done_training_runs": len(runs), "incomplete_or_excluded_runs": len(incomplete),
                "done_snapshot_rows": len(snapshots)}
    (output_path / "summary_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    write_report(output_path / "summary.md", groups, paired_groups, snap_groups, incomplete, metadata)
    return metadata


def formatted(row, metric, scale=1, digits=4):
    mean, sd = row.get(metric + "_mean"), row.get(metric + "_sample_sd")
    if mean is None:
        return "—"
    main = f"{scale * mean:.{digits}f}"
    return main + (f" ± {scale * sd:.{digits}f}" if sd is not None else "（单种子）")


def write_report(path, groups, pairs, snapshots, incomplete, metadata):
    lines = ["# 方法补充实验：验证集汇总", "",
        "本汇总只读取验证指标进行开发分析；官方测试字段不被访问或用于选择。原始 JSON 保留在各运行目录，CSV 保留原始精度。", "",
        "独立重复单位为训练随机种子。最后五轮均值先在每个运行内计算，再在种子之间汇总；五个轮次不作为五次独立实验。仅 DONE 且轨迹完整的运行进入训练统计。单种子不计算样本标准差。", "",
        f"已完成训练运行：{metadata['done_training_runs']}；未完成或排除运行：{metadata['incomplete_or_excluded_runs']}；已完成 snapshot 行：{metadata['done_snapshot_rows']}。", "",
        "## 完整训练轨迹", "",
        "准确率以百分数表示；loss 为原始交叉熵，字节为上传表示的累计核算值，aggregate error 为每轮相对聚合误差的运行内均值。不同 settings_id 不混合统计。", "",
        "| 设置 | 方法 | mix | rho | 种子数 | 末轮准确率 % | 末轮 loss | 最后五轮准确率 % | 最后五轮 loss | 累计 bytes | 平均 filler | 平均 aggregate error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        vals = [r["settings_id"], r["method"], r["mix"], r["rho"], r["n_seeds"],
            formatted(r, "final_validation_accuracy", 100, 3), formatted(r, "final_validation_loss"),
            formatted(r, "last5_validation_accuracy_mean", 100, 3), formatted(r, "last5_validation_loss_mean"),
            formatted(r, "cumulative_bytes", 1, 0), formatted(r, "mean_filler_fraction"),
            formatted(r, "mean_aggregate_relative_error")]
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    if not groups:
        lines.append("\n暂无可用于正式汇总的 DONE 训练轨迹。")
    lines += ["", "## 同种子配对：相对 Top-K EF", "",
        "每个差值均为候选方法减去同设置、同训练种子的 topk_ef。准确率正差值有利；loss 和误差负差值有利。未匹配种子不参与。", "",
        "| 设置 | 方法 | mix | rho | 配对种子数 | 末轮准确率差 pp | 末轮 loss 差 | 最后五轮准确率差 pp | 最后五轮 loss 差 | 平均 aggregate error 差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pairs:
        vals = [r["settings_id"], r["method"], r["mix"], r["rho"], r["n_seeds"]] + [formatted(r, m) for m in PAIR_METRICS]
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines += ["", "## 相同密集轨迹状态的 snapshot", "",
        "这些是相同模型状态与相同客户端更新上的一步反事实，残差初值为零；不能当成独立训练结果。只按相同轮次跨种子统计，不将不同轮次混合成重复。", "",
        "| 设置 | 方法 | mix | rho | 轮次 | 种子数 | 验证准确率 % | loss | 准确率差 pp | loss 差 | aggregate error 差 | filler |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in snapshots:
        vals = [r["settings_id"], r["method"], r["mix"], r["rho"], r["round"], r["n_seeds"],
            formatted(r, "validation_accuracy", 100, 3), formatted(r, "validation_loss"),
            formatted(r, "delta_accuracy_pp_vs_topk"), formatted(r, "delta_loss_vs_topk"),
            formatted(r, "delta_aggregate_relative_error_vs_topk"), formatted(r, "mean_filler_fraction")]
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines += ["", "## 运行中与排除项", "",
        "| 运行 | 状态 | 已观测轮次 | 请求轮次 |", "|---|---|---:|---:|"]
    for r in incomplete:
        lines.append(f"| {r['run_id']} | {r['status']} | {r.get('observed_round', '—')} | {r.get('requested_rounds', '—')} |")
    lines += ["", "## 核对记录", "",
        "- 分组中的样本标准差使用 ddof=1。",
        "- 累计 bytes 是 runner 的编码模型核算值，不是网络测量或完整链上开销。",
        "- mix 改变客户端训练图像分配，rho 是保留坐标比例；设置、数据和代码身份见 summary_metadata.json。",
        "- 缺失/不兼容基线不生成配对差；重复同种子同配置运行被排除以避免伪重复。",
        "- 本分析器不会基于官方测试集选择候选方法或超参数。", ""]
    if metadata["issues"]:
        lines.append("检测到需核查事项，详见 summary_metadata.json：")
        for issue in metadata["issues"]:
            lines.append("- " + issue["issue"] + ("：" + issue["run_id"] if issue.get("run_id") else ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.input.is_dir():
        p.error("--input must be an existing runs directory")
    info = summarize(args.input, args.output)
    print(json.dumps({k: v for k, v in info.items() if k not in ("settings", "issues")}, indent=2))
    print(f"Wrote {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
