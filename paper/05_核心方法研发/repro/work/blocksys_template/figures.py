"""Render manuscript figures at the supplied svproc template's 122 mm width.

Reads completed frozen result JSON only; it never trains or scores a model.
Usage: python work/blocksys_template/figures.py --output PATH
Import: from figures import generate; generate(output_folder)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/plot_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator

WIDTH_MM = 122.0
TASKS = {
    "cifar": ("CIFAR-10", (20261231, 20261232, 20261233)),
    "fashion": ("FashionMNIST", (20261241, 20261242, 20261243)),
}
VARIANTS = (
    ("zero_anchor_vector", "Zero vector", "#0072B2", "s"),
    ("prioranchored_bcts", "Prior BCTS", "#B04A00", "^"),
    ("prioranchored_vector", "Prior vector", "#007A5E", "o"),
)
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7.7,
    "axes.labelsize": 7.7,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_sources():
    rows, source_ids, plot_data = {}, {}, []
    for task, (_, seeds) in TASKS.items():
        aggregate = ROOT / f"work/predictive_final_test/official_{task}/result.json"
        agg = json.loads(aggregate.read_text())
        if agg.get("status") != "DONE" or tuple(agg["fixed_three_seeds"]) != seeds:
            raise ValueError(f"Frozen three-seed result required: {aggregate}")
        source_ids[rel(aggregate)] = sha(aggregate)
        rows[task] = []
        for seed in seeds:
            path = ROOT / f"work/predictive_final_test/official_{task}/fedlc_uniform3_s{seed}/result.json"
            row = json.loads(path.read_text())
            if (row.get("status") != "DONE" or row["seed"] != seed
                    or row["method"] != "fedlc_uniform3" or row["rounds"] != 20
                    or row["assignment_hash"] != agg["assignment_hash"]):
                raise ValueError(f"Unexpected completed-result identity: {path}")
            source_ids[rel(path)] = sha(path)
            rows[task].append(row)
        for variant, label, _, _ in VARIANTS:
            for metric in ("macro_accuracy", "macro_nll"):
                a = np.array([row["summaries"][variant][metric] for row in rows[task]], dtype=float)
                if a.shape != (3,) or not np.isfinite(a).all():
                    raise ValueError("Exactly three finite, unfiltered observations required")
                reference = agg["summaries"]["fedlc_uniform3"]["summaries"][variant][metric]
                if not np.isclose(a.mean(), reference["mean"], atol=1e-12, rtol=0):
                    raise ValueError("Mean differs from frozen aggregate")
                if not np.isclose(a.std(ddof=1), reference["sample_sd"], atol=1e-12, rtol=0):
                    raise ValueError("Sample SD differs from frozen aggregate")
            for row in rows[task]:
                plot_data.append({
                    "task": task, "seed": row["seed"], "variant": variant,
                    "label": label,
                    "macro_accuracy_percent": 100 * row["summaries"][variant]["macro_accuracy"],
                    "macro_nll": row["summaries"][variant]["macro_nll"],
                })
    return rows, source_ids, plot_data


def export(fig, stem: Path):
    """Preserve exact physical size: intentionally no tight bounding box."""
    result = []
    for extension in ("pdf", "svg", "png"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=600, bbox_inches=None, facecolor="white", transparent=False)
        result.append({"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
    return result


def prediction(rows, output):
    height_mm = 65.0
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.112, right=0.985, bottom=0.25, top=0.82, wspace=0.48)
    limits = {}
    for ax, (task, (task_label, _)) in zip(axes, TASKS.items()):
        xs, ys = [], []
        for variant, _, color, marker in VARIANTS:
            accuracy = 100 * np.array([r["summaries"][variant]["macro_accuracy"] for r in rows[task]])
            nll = np.array([r["summaries"][variant]["macro_nll"] for r in rows[task]])
            xs.extend(accuracy.tolist() + [accuracy.mean() - accuracy.std(ddof=1), accuracy.mean() + accuracy.std(ddof=1)])
            ys.extend(nll.tolist() + [nll.mean() - nll.std(ddof=1), nll.mean() + nll.std(ddof=1)])
            ax.errorbar(accuracy.mean(), nll.mean(),
                        xerr=accuracy.std(ddof=1), yerr=nll.std(ddof=1),
                        color=color, fmt=marker, markersize=5.8, capsize=2.3,
                        elinewidth=0.8, markeredgewidth=0.8, zorder=3)
            # No jitter or seed filtering: both coordinates remain their raw values.
            ax.scatter(accuracy, nll, edgecolors=color, facecolors="white",
                       marker=marker, s=12, linewidths=0.65, zorder=4)
        xpad, ypad = 0.12 * (max(xs) - min(xs)), 0.13 * (max(ys) - min(ys))
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
        ax.set_title(task_label, pad=6, fontweight="bold")
        ax.set_xlabel("Macro accuracy (%)", labelpad=3)
        ax.set_ylabel("Macro NLL", labelpad=3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.tick_params(length=2.4, width=0.6, pad=2)
        ax.grid(color="#D8DDE1", linewidth=0.45, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        limits[task] = {"accuracy_percent": list(ax.get_xlim()), "nll": list(ax.get_ylim())}
    fig.text(0.5, 0.955, "Open: three seeds; filled + bars: mean ± sample SD",
             ha="center", va="top", fontsize=7.2)
    legend_handles = [Line2D([], [], marker=marker, color=color, linestyle="None",
                            markersize=5.2, label=label)
                      for _, label, color, marker in VARIANTS]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.25)
    fig.canvas.draw()
    output_ids = export(fig, output / "prediction")
    plt.close(fig)
    return {"width_mm": WIDTH_MM, "height_mm": height_mm,
            "axis_limits": limits, "outputs": output_ids}


def workflow(output):
    height_mm = 104.0
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, height_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH_MM), ylim=(0, height_mm))
    ax.axis("off")
    ink = "#202B33"
    nodes = []
    ax.add_patch(FancyBboxPatch((1, 56), 120, 47, boxstyle="round,pad=0,rounding_size=1.1",
                               linewidth=0.75, edgecolor="#315E7C", facecolor="white"))
    ax.add_patch(FancyBboxPatch((1, 1), 120, 52, boxstyle="round,pad=0,rounding_size=1.1",
                               linewidth=0.75, linestyle=(0, (4, 2.5)), edgecolor="#755A32", facecolor="white"))
    ax.text(4, 99, "A  Learning and local adaptation", fontsize=9, weight="bold", color=ink,
            ha="left", va="center")
    ax.text(4, 94, "Reliable synchronous delivery; 20 full-participation rounds", fontsize=7.3,
            color=ink, ha="left", va="center")
    ax.text(4, 49, "B  Separate local protocol simulation", fontsize=9, weight="bold", color=ink,
            ha="left", va="center")
    ax.text(4, 44, "Existing payloads replayed with constructed updates", fontsize=7.3,
            color=ink, ha="left", va="center")

    def node(x, y, text, lane="a", size=7.8):
        w, h = 52, 12
        edge = "#315E7C" if lane == "a" else "#755A32"
        fill = "#F1F6FA" if lane == "a" else "#FBF7F0"
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                              linewidth=0.65, edgecolor=edge, facecolor=fill)
        ax.add_patch(patch)
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=ink,
                    fontsize=size, linespacing=1.18)
        nodes.append((patch, t, text))

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                    linewidth=0.8, color=ink, shrinkA=1, shrinkB=1))

    node(4, 78, "1  Local TRAIN\nCount-adjusted local SGD")
    node(66, 78, "2  uniform3 + FP32 EF\nDecode actual bytes\nSample-weighted aggregation", size=7.6)
    node(66, 59, "3  Freeze shared CNN\nTRAIN logits + prior-ridge fit")
    node(4, 59, "4  Local prediction\n20 FP32 values (80 B)")
    arrow((56, 84), (66, 84))
    arrow((92, 78), (92, 71))
    arrow((66, 65), (56, 65))
    node(4, 27, "1  Persist pending envelope\nCandidate EF residual\nRetry/query identical bytes", "b", size=7.6)
    node(66, 27, "2  Authenticate context\nHMAC-SM3; strict decode", "b")
    node(66, 8, "3  Durable incorporation\nSQLite transaction\nMatching incorporated receipt", "b", size=7.5)
    node(4, 8, "4  Commit EF residual\nOnly after matching receipt", "b")
    arrow((56, 33), (66, 33))
    arrow((92, 27), (92, 20))
    arrow((66, 14), (56, 14))
    ax.text(61, 3.6, "No real ledger or authenticated end-to-end FL run.", fontsize=7.1,
            color=ink, ha="center", va="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for patch, text_artist, text in nodes:
        p = patch.get_window_extent(renderer)
        t = text_artist.get_window_extent(renderer)
        if t.x0 < p.x0 or t.x1 > p.x1 or t.y0 < p.y0 or t.y1 > p.y1:
            overflow.append(text)
    if overflow:
        raise ValueError(f"Node text exceeds its final-size box: {overflow}")
    output_ids = export(fig, output / "workflow")
    plt.close(fig)
    return {"width_mm": WIDTH_MM, "height_mm": height_mm,
            "node_text_overflow": overflow, "outputs": output_ids}


def generate(output_folder):
    """Write figures, plotted values, bilingual alt text, and provenance."""
    output = Path(output_folder).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, source_ids, plot_data = load_sources()
    workflow_sources = [
        "work/predictive_manuscript/METHOD_EN.md",
        "work/durable_integrity_v2/protocol.py",
        "work/durable_integrity_v2/REPORT.md",
        "work/full_payload_integrity/run_integration.py",
    ]
    workflow_ids = {s: sha(ROOT / s) for s in workflow_sources}
    with plt.rc_context(STYLE):
        workflow_info = workflow(output)
        prediction_info = prediction(rows, output)
    (output / "prediction.data.json").write_text(json.dumps(plot_data, indent=2) + "\n")
    (output / "workflow.alt.md").write_text(
        "# Workflow text alternative\n\n"
        "English: Two separate panels. A shows reliable-delivery learning: count-adjusted "
        "local SGD, uniform3 blocks with FP32 error feedback, strict decoding of actual "
        "bytes, and sample-weighted aggregation over 20 rounds. The final shared CNN is "
        "frozen before a local TRAIN-only prior-ridge affine fit and prediction with "
        "twenty FP32 values. B shows an independently tested local protocol simulation: "
        "persist the pending envelope and candidate residual; retry or query identical "
        "bytes; authenticate context and strictly decode; incorporate durably in SQLite; "
        "commit the client's residual only after a matching incorporated receipt. "
        "No execution arrow joins A and B. There is no real ledger or authenticated "
        "end-to-end federated training experiment.\n\n"
        "中文：上下两个面板分别标明学习与独立协议模拟。A为可靠同步传输下的学习："
        "计数调整的本地SGD、uniform3分块及FP32误差反馈、真实字节严格解码、按样本数"
        "加权聚合，共20轮；冻结最终共享CNN后，仅以本地TRAIN拟合先验锚定的岭正则"
        "仿射变换，用20个FP32值预测。B为独立本地协议模拟：持久保存待决封装与候选"
        "残差，重试或查询复用相同字节，认证上下文并严格解码，以SQLite事务持久纳入；"
        "客户端仅在获得匹配的incorporated回执后提交残差。两面板没有执行连接箭头；"
        "没有真实链或经认证的端到端联邦训练实验。\n"
    )
    (output / "prediction.alt.md").write_text(
        "# Prediction text alternative\n\n"
        "English: Two scatter panels show client-macro accuracy (%) horizontally and "
        "client-macro NLL vertically for CIFAR-10 and FashionMNIST. Each compares "
        "zero-anchor vector, same-prior BCTS, and prior-anchor vector on the same "
        "FedLC-inspired uniform3 shared models. Open symbols show all three fixed "
        "training seeds without jitter; larger filled symbols and horizontal/vertical "
        "bars show the mean and sample SD (ddof=1). Color and marker shape jointly "
        "identify methods. Axis ranges differ by task and include every raw point and "
        "mean ± SD. Prior-anchor vector has lower mean NLL than zero-anchor vector on "
        "both tasks; same-prior BCTS is close. These panels do not display ECE or Brier. "
        "Exact plotted values accompany the artwork in prediction.data.json.\n\n"
        "中文：两个散点面板分别展示CIFAR-10与FashionMNIST，横轴为客户端宏平均准确率"
        "（%），纵轴为客户端宏平均NLL。比较相同FedLC-inspired uniform3共享模型上的"
        "零锚向量、同先验BCTS、先验锚向量。空心标记显示全部三个固定训练种子且无抖动；"
        "较大的实心标记及水平/垂直误差条表示均值与样本标准差（ddof=1）。颜色与形状"
        "共同标识方法。各任务使用不同轴范围，并包含所有原始点与均值±SD。先验锚向量"
        "在两个任务的平均NLL均低于零锚向量，同先验BCTS与其接近；本图不展示ECE或Brier。"
        "完整绘图数值见prediction.data.json。\n"
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "destination": "User-supplied BlockSys svproc template; final figure width 122 mm",
        "venue_rule_scope": "Width follows supplied class; current official 2027 rules not independently established here",
        "script": {"path": rel(Path(__file__)), "sha256": sha(Path(__file__))},
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                     "skill": "/Users/wind/.codex/skills/scientific-visualization/SKILL.md"},
        "common_export": {"png_dpi": 600, "opaque_white": True, "font": "DejaVu Sans",
                          "minimum_text_pt": 7.1, "pdf_fonttype": 42,
                          "svg_text": "Editable text; font not embedded",
                          "physical_size_preserved": "bbox_inches=None"},
        "workflow": {**workflow_info, "source_sha256": workflow_ids,
                     "evidence_boundary": "Learning and local protocol are separate; no deployed chain, authenticated end-to-end training, or distributed atomicity claim"},
        "prediction": {**prediction_info, "source_sha256": source_ids,
                       "unit": "Exactly three predefined training seeds per task; one fixed client assignment per task",
                       "transformations": "macro_accuracy * 100; no NLL transformation, smoothing, jitter, filtering, or seed selection",
                       "uncertainty": "Arithmetic mean and sample SD, ddof=1; both independently matched to frozen aggregate JSON",
                       "raw_points": len(plot_data), "encoding": "Each raw point is an open shape; means are larger filled shapes; horizontal and vertical sample-SD bars",
                       "data_file": "prediction.data.json"},
    }
    (output / "FIGURE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Replace any previous per-figure record so old 180 mm provenance cannot
    # remain alongside the newly generated 122 mm artwork.
    for name in ("workflow", "prediction"):
        individual = {
            "created_utc": provenance["created_utc"],
            "figure": name, "destination": provenance["destination"],
            "script": provenance["script"], "software": provenance["software"],
            "export": provenance["common_export"], **provenance[name],
        }
        (output / f"{name}.provenance.json").write_text(json.dumps(individual, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "figure_preview")
    args = parser.parse_args()
    manifest = generate(args.output)
    print(json.dumps({"output": str(args.output.resolve()),
                      "prediction_raw_points": manifest["prediction"]["raw_points"],
                      "workflow_text_overflow": manifest["workflow"]["node_text_overflow"]}, indent=2))
