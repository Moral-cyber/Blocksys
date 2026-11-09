"""Draw the process-only manuscript workflow at exactly 180 x 90 mm.

No experiment arrays, results, external assets, network services or image models
are used. Run with the project's Python environment; outputs stay beside this
script. The lower lane is a separate local protocol simulation, not an integrated
learning experiment or a blockchain deployment.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import tempfile
import xml.etree.ElementTree as ET

_mpl_config = tempfile.TemporaryDirectory(prefix="workflow_mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl_config.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WIDTH_MM, HEIGHT_MM = 180.0, 90.0
DPI = 600

COLORS = {
    "ink": "#192D3A", "arrow": "#405563", "blue": "#276A91",
    "blue_fill": "#EDF5FA", "teal": "#276B60", "teal_fill": "#EDF6F3",
    "sim": "#59636B", "sim_fill": "#FAFBFC", "boundary": "#89949B",
}

TOP = [
    {"x": 6, "w": 25, "text": "Local TRAIN\ncount-adjusted\nSGD", "kind": "blue"},
    {"x": 34, "w": 26, "text": "uniform3 + EF\nencode / decode", "kind": "blue"},
    {"x": 63, "w": 25, "text": "Sample-weighted\naggregation", "kind": "blue"},
    {"x": 91, "w": 23, "text": "Final shared\nmodel · frozen", "kind": "blue"},
    {"x": 117, "w": 32, "text": "Local TRAIN fit\nprior-ridge vector", "kind": "teal"},
    {"x": 152, "w": 22, "text": "Personal\nprobabilities", "kind": "teal"},
]
BOTTOM = [
    {"x": 6, "w": 18, "text": "Same wire\nbytes", "size": 7.0},
    {"x": 25.5, "w": 28, "text": "Canonical context\n+ HMAC-SM3", "size": 6.9},
    {"x": 55, "w": 20, "text": "REGISTERED", "size": 6.6},
    {"x": 76.5, "w": 22, "text": "ACCEPTED\nstrict decode", "size": 6.7},
    {"x": 100, "w": 29, "text": "INCORPORATED\nSQLite transaction", "size": 6.7},
    {"x": 130.5, "w": 22, "text": "Matching\nreceipt", "size": 7.0},
    {"x": 154, "w": 20, "text": "Commit\nEF residual", "size": 7.0},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrow(ax, start, end, *, color=None, style="-", zorder=2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=6.8, linewidth=.8, linestyle=style,
        color=color or COLORS["arrow"], shrinkA=0, shrinkB=0, zorder=zorder))


def node(ax, record, y, height, *, simulation=False):
    kind = "sim" if simulation else record["kind"]
    edge = COLORS[kind]
    fill = "#FFFFFF" if simulation else COLORS[kind + "_fill"]
    patch = FancyBboxPatch((record["x"], y), record["w"], height,
        boxstyle="round,pad=0,rounding_size=1.25", facecolor=fill,
        edgecolor=edge, linewidth=.85, zorder=3)
    ax.add_patch(patch)
    text = ax.text(record["x"] + record["w"] / 2, y + height / 2,
        record["text"], ha="center", va="center", color=COLORS["ink"],
        fontsize=record.get("size", 7.1), linespacing=1.35, zorder=4)
    return patch, text


def contrast(hex_a, hex_b):
    def luminance(value):
        rgb = [int(value[i:i+2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
        return sum(a*b for a, b in zip(linear, (.2126, .7152, .0722)))
    a, b = sorted((luminance(hex_a), luminance(hex_b)))
    return (b + .05) / (a + .05)


def main():
    style = {"font.family": "DejaVu Sans", "font.size": 7.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "predictive-workflow-v1", "figure.facecolor": "white",
        "savefig.facecolor": "white", "text.color": COLORS["ink"]}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, WIDTH_MM), ylim=(0, HEIGHT_MM))
        ax.set_axis_off()

        ax.text(6, 84, "a", fontsize=11, fontweight="bold", va="center")
        ax.text(12, 84, "Measured learning and local prediction", fontsize=9.3,
                fontweight="bold", va="center")

        # The feedback loop belongs only to the measured synchronous FL route.
        loop = MplPath([(75.5, 72), (75.5, 77), (18.5, 77), (18.5, 72)],
                       [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
        ax.add_patch(FancyArrowPatch(path=loop, arrowstyle="-|>", mutation_scale=6.8,
            linewidth=.8, edgecolor=COLORS["blue"], facecolor=COLORS["blue"], zorder=2))
        ax.text(47, 78.6, "Repeat FL rounds", fontsize=6.7, ha="center", color=COLORS["blue"])

        boxes = [node(ax, item, 57, 15) for item in TOP]
        for left, right in zip(TOP, TOP[1:]):
            arrow(ax, (left["x"] + left["w"] + .25, 64.5), (right["x"] - .25, 64.5))
        ax.text(47, 52.7, "Decoded updates enter aggregation", fontsize=6.8,
                ha="center", color=COLORS["blue"])
        ax.text(133, 52.7, "Fit once after the final model is frozen", fontsize=6.8,
                ha="center", color=COLORS["teal"])

        # No connecting arrow between lanes: the lower workflow was tested alone.
        boundary = FancyBboxPatch((4, 4), 172, 42,
            boxstyle="round,pad=0,rounding_size=1.5", facecolor=COLORS["sim_fill"],
            edgecolor=COLORS["boundary"], linewidth=.85, linestyle=(0, (4, 3)), zorder=0)
        ax.add_patch(boundary)
        ax.text(6, 41.6, "b", fontsize=11, fontweight="bold", va="center")
        ax.text(12, 41.6, "Local protocol simulation · evaluated separately", fontsize=9.0,
                fontweight="bold", va="center")
        ax.text(6, 36.9, "No end-to-end learning integration or real-chain evaluation", fontsize=7.0,
                color=COLORS["sim"], va="center")
        boxes.extend(node(ax, item, 21, 12.5, simulation=True) for item in BOTTOM)
        for left, right in zip(BOTTOM, BOTTOM[1:]):
            arrow(ax, (left["x"] + left["w"] + .15, 27.25), (right["x"] - .15, 27.25))

        # Retry applies to an unresolved pending upload, before matching receipt.
        retry = MplPath([(87.5, 21), (87.5, 16), (15, 16), (15, 21)],
                        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
        ax.add_patch(FancyArrowPatch(path=retry, arrowstyle="-|>", mutation_scale=6.8,
            linewidth=.8, linestyle=(0, (3, 2)), edgecolor=COLORS["sim"],
            facecolor=COLORS["sim"], zorder=2))
        ax.text(49.8, 11.6, "Retry: same bytes; no re-encoding", fontsize=6.7,
                color=COLORS["sim"], ha="center")
        ax.text(136, 14.7, "Keep pending until the", fontsize=6.6,
                color=COLORS["sim"], ha="center")
        ax.text(136, 10.6, "incorporated receipt matches", fontsize=6.6,
                color=COLORS["sim"], ha="center")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overflow = []
        for patch, text in boxes:
            if not patch.get_window_extent(renderer).contains(*text.get_window_extent(renderer).get_points()[0]):
                overflow.append(text.get_text())
            if not patch.get_window_extent(renderer).contains(*text.get_window_extent(renderer).get_points()[1]):
                overflow.append(text.get_text())
        if overflow:
            raise RuntimeError("Node text outside its box: " + repr(overflow))

        metadata = {"Title": "Measured learning route and separate local protocol simulation",
            "Author": "Research workflow diagram", "Creator": "Matplotlib; generate_workflow.py",
            "Subject": "Process-only schematic. No quantitative results or end-to-end/real-chain claim."}
        fig.savefig(HERE / "workflow.pdf", format="pdf", bbox_inches=None, metadata=metadata)
        fig.savefig(HERE / "workflow.svg", format="svg", bbox_inches=None,
            metadata={"Title": metadata["Title"], "Description": metadata["Subject"],
                      "Creator": "Matplotlib; generate_workflow.py"})
        fig.savefig(HERE / "workflow.png", format="png", dpi=DPI, bbox_inches=None,
            metadata={"Title": metadata["Title"], "Description": metadata["Subject"]})
        plt.close(fig)

    # Make the physical SVG dimensions explicit; the vector viewBox is preserved.
    svg_path = HERE / "workflow.svg"
    svg = svg_path.read_text()
    svg = svg.replace('width="510.23622pt" height="255.11811pt"', 'width="180mm" height="90mm"', 1)
    svg_path.write_text(svg)
    svg_root = ET.fromstring(svg)
    page = PdfReader(HERE / "workflow.pdf").pages[0]
    media = page.mediabox
    pdf_mm = [float(media.width) * 25.4 / 72, float(media.height) * 25.4 / 72]
    if max(abs(actual - desired) for actual, desired in zip(pdf_mm, (WIDTH_MM, HEIGHT_MM))) > 1e-6:
        raise RuntimeError("Unexpected PDF dimensions")
    with Image.open(HERE / "workflow.png") as image:
        png = {"pixels": list(image.size), "dpi": list(image.info.get("dpi", ())),
               "mode": image.mode, "fully_opaque": image.getextrema()[-1] == (255, 255) if image.mode == "RGBA" else True}
    if not png["fully_opaque"]:
        raise RuntimeError("Unexpected transparent pixels")

    source_paths = [ROOT / rel for rel in (
        "work/predictive_manuscript/METHOD_EN.md", "work/predictive_pfl_confirm/PLAN.md",
        "work/durable_integrity_v2/protocol.py", "work/durable_integrity_v2/REPORT.md",
        "work/full_payload_integrity/run_integration.py")]
    resources = page.get("/Resources", {})
    if hasattr(resources, "get_object"):
        resources = resources.get_object()
    xobjects = resources.get("/XObject", {})
    if hasattr(xobjects, "get_object"):
        xobjects = xobjects.get_object()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "figure": "workflow",
        "scope": "process-only schematic, no results, no model training, no TEST access",
        "evidence_boundary": "Upper lane: measured FL learning route. Lower lane: separately tested local protocol simulation; not end-to-end learning integration, deployed chain, network performance or distributed atomicity evidence.",
        "content_source": "Root's explicit workflow request, cross-checked against the listed manuscript/plan/protocol sources. No experiment metrics or result arrays were read for this figure.",
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "script": {"path": "generate_workflow.py", "sha256": sha(__file__)},
        "design": {"width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "png_dpi": DPI,
            "font": "DejaVu Sans", "svg_text": "editable text; font is not embedded", "pdf_fonts": "embedded TrueType via Type42",
            "top_lane": TOP, "separate_simulation_lane": BOTTOM,
            "color_redundancy": "panel labels, separate lane, dashed boundary, direct text labels; no lane-to-lane execution arrow",
            "retry": "Unresolved pending upload queries/resends the same stored envelope; no re-encoding. EF commits only after matching incorporated receipt."},
        "inspection": {"pdf_size_mm": pdf_mm, "svg_size": [svg_root.attrib["width"], svg_root.attrib["height"]],
            "png": png, "node_text_overflow": overflow,
            "text_contrast_white": contrast(COLORS["ink"], "#FFFFFF"),
            "minimum_node_text_contrast": min(contrast(COLORS["ink"], value) for value in ("#FFFFFF", COLORS["blue_fill"], COLORS["teal_fill"])),
            "pdf_has_raster_images": any(str(obj.get_object().get("/Subtype")) == "/Image"
                for obj in xobjects.values()),
            "publisher_requirements": "Not supplied; provisional manuscript artwork, no journal compliance claim."},
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
            "skill_guidance": "/Users/wind/.codex/skills/scientific-visualization/SKILL.md",
            "skill_use": "Process/evidence separation, redundant lane styling, exact-size vector export, provenance and alt text. Skill guidance is software provenance, not experimental evidence."},
        "outputs": [{"path": "workflow." + ext, "bytes": (HERE / ("workflow." + ext)).stat().st_size,
                     "sha256": sha(HERE / ("workflow." + ext))} for ext in ("pdf", "svg", "png")],
    }
    (HERE / "workflow.provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "pdf_size_mm": pdf_mm,
        "svg_size": provenance["inspection"]["svg_size"], "png": png,
        "node_text_overflow": overflow, "outputs": provenance["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
