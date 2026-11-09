"""Summarize the FedAvg update-compression pilot."""

from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results" / "fedavg_update_diagnostic.json").read_text())
summary = []
for alpha in (0.1, 0.5, 1.0):
    for method in ("dense", "topk", "topk_ef"):
        group = [r for r in rows if r["alpha"] == alpha and r["method"] == method]
        global_rows = [r for r in group if r["client_id"] == "__global__"]
        client_rows = [r for r in group if r["client_id"] != "__global__"]
        last_by_seed = {}
        for row in global_rows:
            last_by_seed[row["seed"]] = row
        last = list(last_by_seed.values())
        final_acc = np.asarray([row["overall_acc"] for row in last], dtype=float)
        final_std = np.asarray([row["client_acc_std"] for row in last], dtype=float)
        final_min = np.asarray([row["client_acc_min"] for row in last], dtype=float)
        counts = np.asarray([r["client_samples"] for r in client_rows], dtype=float)
        retention = np.asarray([r["sent_norm_ratio"] for r in client_rows], dtype=float)
        cosine = np.asarray([r["update_sent_cosine"] for r in client_rows], dtype=float)
        summary.append({
            "alpha": alpha,
            "method": method,
            "seeds": len(last),
            "final_accuracy_mean": float(final_acc.mean()),
            "final_accuracy_std": float(final_acc.std()),
            "final_client_acc_std_mean": float(final_std.mean()),
            "final_client_acc_min_mean": float(final_min.mean()),
            "cumulative_bytes": float(np.mean([row["cumulative_bytes"] for row in last])),
            "dense_cumulative_bytes": float(np.mean([row.get("dense_bytes", 0) * 5 for row in last])),
            "mean_sent_norm_ratio": float(retention.mean()),
            "mean_update_sent_cosine": float(cosine.mean()),
            "sample_retention_corr": float(np.corrcoef(np.log1p(counts), retention)[0, 1]),
            "sample_cosine_corr": float(np.corrcoef(np.log1p(counts), cosine)[0, 1]),
        })

out = ROOT / "results" / "fedavg_update_diagnostic_summary.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print(out)
for row in summary:
    print(
        f"alpha={row['alpha']:<3} method={row['method']:<7} "
        f"acc={row['final_accuracy_mean']:.3f} client_std={row['final_client_acc_std_mean']:.3f} "
        f"retain={row['mean_sent_norm_ratio']:.3f} "
        f"corr(samples,retain)={row['sample_retention_corr']:+.3f} "
        f"bytes={row['cumulative_bytes']}"
    )
