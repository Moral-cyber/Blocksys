"""Summarize sparse_update_benchmark CSV into paper-facing tables."""

from pathlib import Path
import argparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/sparse_pilot_5r.csv")
    parser.add_argument("--output", default="results/sparse_pilot_5r_summary.md")
    parser.add_argument("--expected-rounds", type=int, default=None,
                        help="Require every dataset/seed/method group to contain this many rounds")
    parser.add_argument("--expected-seeds", type=int, default=None,
                        help="Require every dataset/alpha/method group to contain this many seeds")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    df = pd.read_csv(source)
    global_rows = df[df["client_id"].astype(str) == "__global__"].copy()
    required = {"global_test_samples", "global_test_source", "train_samples", "client_test_samples"}
    missing = required - set(global_rows.columns)
    if missing:
        raise ValueError(f"benchmark file lacks provenance fields: {sorted(missing)}")
    if set(global_rows["global_test_samples"].dropna().astype(int)) != {10000}:
        raise ValueError("global accuracy must be evaluated on exactly 10,000 official CIFAR-10 test samples")
    if set(global_rows["global_test_source"].dropna().astype(str)) != {"official CIFAR-10 test split"}:
        raise ValueError("global accuracy source is not the official CIFAR-10 test split")
    if global_rows.empty:
        raise ValueError("no __global__ rows found in benchmark output")
    required = {"dataset", "seed", "method", "round", "overall_acc"}
    missing = required.difference(global_rows.columns)
    if missing:
        raise ValueError(f"benchmark output is missing columns: {sorted(missing)}")
    # A final row from an interrupted run is not a valid final result.  Fail
    # early instead of silently turning a partial trajectory into a paper table.
    group_sizes = global_rows.groupby(["dataset", "seed", "method"])["round"].nunique()
    if args.expected_rounds is not None:
        bad = group_sizes[group_sizes != args.expected_rounds]
        if not bad.empty:
            raise ValueError(f"incomplete round groups (expected {args.expected_rounds}): {bad.to_dict()}")
    duplicate_rounds = global_rows.groupby(["dataset", "seed", "method", "round"]).size()
    if (duplicate_rounds > 1).any():
        raise ValueError("duplicate global rows for at least one dataset/seed/method/round")
    if args.expected_seeds is not None:
        seed_counts = global_rows.groupby(["dataset", "alpha", "method"])["seed"].nunique()
        bad = seed_counts[seed_counts != args.expected_seeds]
        if not bad.empty:
            raise ValueError(f"unexpected seed counts (expected {args.expected_seeds}): {bad.to_dict()}")
    final = (
        global_rows.sort_values("round")
        .groupby(["dataset", "seed", "method"], as_index=False)
        .tail(1)
    )
    metrics = [
        "overall_acc",
        "client_acc_std",
        "client_acc_min",
        "cumulative_bytes",
        "compression_ratio_vs_dense",
        "mean_residual_norm",
    ]
    group_cols = ["dataset"]
    if "alpha" in final.columns:
        group_cols.append("alpha")
    group_cols.append("method")
    summary = final.groupby(group_cols, as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = [
        *group_cols,
        *[f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")],
    ]

    lines = [
        "# Sparse Update Benchmark Summary",
        "",
        f"Source: `{source}`",
        "",
        "Final-round mean (standard deviation across seeds):",
        "",
        "| Dataset | Alpha | Method | Accuracy | Client acc. std | Min client acc. | Cumulative bytes | Compression ratio | Residual norm |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        row_dict = row._asdict()
        lines.append(
            f"| {row_dict['dataset']} | {row_dict.get('alpha', '')} | {row_dict['method']} | "
            f"{row_dict['overall_acc_mean']:.4f} ({row_dict['overall_acc_std']:.4f}) | "
            f"{row_dict['client_acc_std_mean']:.4f} ({row_dict['client_acc_std_std']:.4f}) | "
            f"{row_dict['client_acc_min_mean']:.4f} ({row_dict['client_acc_min_std']:.4f}) | "
            f"{row_dict['cumulative_bytes_mean']:.0f} ({row_dict['cumulative_bytes_std']:.0f}) | "
            f"{row_dict['compression_ratio_vs_dense_mean']:.3f} ({row_dict['compression_ratio_vs_dense_std']:.3f}) | "
            f"{row_dict['mean_residual_norm_mean']:.4f} ({row_dict['mean_residual_norm_std']:.4f}) |"
        )

    lines.extend([
        "",
        "Interpretation guardrail: this summary reports measured outcomes only; it does not establish that novelty scoring protects minority classes or improves heterogeneous FL in general.",
        "",
    ])
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
