"""Summarize the pilot's class-level selection bias diagnostics."""

from pathlib import Path
import csv
import json
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ("selected_norm", "selected_delta", "selected_random")


def pearson(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def alpha_from_name(name):
    return {"diag_a01": 0.1, "diag_a05": 0.5, "diag_a1": 1.0}[name]


def main():
    summary = []
    for path in sorted(RESULTS.glob("diag_a*_selection.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        run_name = path.stem.replace("_selection", "")
        alpha = alpha_from_name(run_name)
        for method in METHODS:
            rows = []
            for record in run["records"]:
                for client in record["clients"]:
                    selected = set(client[method])
                    for label in client["available_labels"]:
                        count = client["label_counts"][str(label)]
                        rows.append({
                            "count": count,
                            "log_count": math.log1p(count),
                            "selected": int(label in selected),
                            "norm": client["prototype_norm"][str(label)],
                        })
            if not rows:
                continue
            counts = np.asarray([r["count"] for r in rows], dtype=float)
            selected = np.asarray([r["selected"] for r in rows], dtype=float)
            norms = np.asarray([r["norm"] for r in rows], dtype=float)
            median = float(np.median(counts))
            rare = selected[counts <= median]
            common = selected[counts > median]
            summary.append({
                "run": run_name,
                "alpha": alpha,
                "method": method.replace("selected_", ""),
                "occurrences": int(len(rows)),
                "selection_rate": float(selected.mean()),
                "rare_selection_rate": float(rare.mean()) if len(rare) else float("nan"),
                "common_selection_rate": float(common.mean()) if len(common) else float("nan"),
                "rare_minus_common": float(rare.mean() - common.mean()) if len(rare) and len(common) else float("nan"),
                "count_selection_corr": pearson(np.log1p(counts), selected),
                "norm_selected_mean": float(norms[selected == 1].mean()) if np.any(selected == 1) else float("nan"),
                "norm_unselected_mean": float(norms[selected == 0].mean()) if np.any(selected == 0) else float("nan"),
            })

    csv_path = RESULTS / "diagnostic_selection_summary.csv"
    if summary:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    (RESULTS / "diagnostic_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(csv_path)
    for row in summary:
        print(
            f"alpha={row['alpha']:<3} method={row['method']:<6} "
            f"select={row['selection_rate']:.3f} "
            f"rare-common={row['rare_minus_common']:+.3f} "
            f"corr(log count, select)={row['count_selection_corr']:+.3f}"
        )


if __name__ == "__main__":
    main()
