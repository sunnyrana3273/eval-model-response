"""Figures for the report and dashboard (matplotlib, Agg backend).

Colours follow a validated categorical order (blue, orange, aqua) with the majority
baseline as a neutral dashed reference; text stays in ink colours, grids recessive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COLORS = {"behavioral": "#2a78d6", "probe": "#eb6834", "combined": "#1baf7a", "majority": "#8a8984"}
LABELS = {"behavioral": "Behavioral", "probe": "Activation probe", "combined": "Combined", "majority": "Majority"}
CONDITION_COLORS = {"monitor": "#2a78d6", "random": "#eb6834", "always": "#1baf7a", "none": "#8a8984"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def _style(name: str) -> dict:
    return {"color": COLORS.get(name, INK_2), "label": LABELS.get(name, name),
            "linestyle": "--" if name == "majority" else "-", "linewidth": 1.5 if name == "majority" else 2}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_pr(y: np.ndarray, scores: dict[str, np.ndarray], horizon: int, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    for name, p in scores.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, **_style(name))
        if name != "majority":  # a constant score has no PR curve; its precision is the prevalence line
            prec, rec, _ = precision_recall_curve(y, p)
            a2.plot(rec, prec, **_style(name))
    a1.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    a2.axhline(y.mean(), **{**_style("majority"), "label": f"Majority (prevalence {y.mean():.2f})"})
    a2.legend(loc="upper right")
    a1.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC (horizon {horizon})")
    a2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall (horizon {horizon})", ylim=(0, 1.02))
    a1.legend(loc="lower right")
    _save(fig, path)


def calibration(curves: dict[str, dict], horizon: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for name, c in curves.items():
        pts = [(m, f) for m, f, n in zip(c["mean_predicted"], c["fraction_positive"], c["counts"]) if n]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", markersize=5, **_style(name))
    ax.set(xlabel="Mean predicted probability", ylabel="Observed failure frequency",
           title=f"Calibration (horizon {horizon})", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    _save(fig, path)


def by_steps_to_onset(curves: dict[str, list[dict]], horizon: int, path: Path, min_pos: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, rows in curves.items():
        rows = [r for r in rows if r["n_pos"] >= min_pos]  # sparse tails are noise, not trend
        if rows:
            ax.plot([r["k"] for r in rows], [r["auroc"] for r in rows], marker="o", markersize=5, **_style(name))
    ax.axhline(0.5, color=GRID, linewidth=1)
    ax.set(xlabel="Decision steps before failure onset", ylabel="AUROC vs. clean-trajectory states",
           title=f"Early warning (horizon-{horizon} monitors)", ylim=(0.3, 1.02))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.text(0.01, 0.01, f"points with ≥{min_pos} pre-failure states", transform=ax.transAxes, fontsize=8, color=INK_2)
    ax.legend(loc="upper right")
    _save(fig, path)


def layer_sweep(rows: list[dict], horizon: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    layers = [r["layer"] for r in rows]
    ax.plot(layers, [r["val_auroc"] for r in rows], marker="o", markersize=5, color=COLORS["probe"],
            label="Validation (used for selection)")
    if all("test_auroc" in r for r in rows):
        ax.plot(layers, [r["test_auroc"] for r in rows], marker="o", markersize=5, color=COLORS["behavioral"],
                label="Test (descriptive only)")
    ax.axhline(0.5, color=GRID, linewidth=1)
    ax.set(xlabel="Model layer", ylabel="Probe AUROC", title=f"Probe AUROC by layer (horizon {horizon})")
    ax.legend(loc="lower right")
    _save(fig, path)


def by_horizon(results: dict, monitors: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    hs = sorted(int(h) for h in results)
    for i, name in enumerate(monitors):
        pts = [(h, results[str(h)][name]["auroc_ci"]) for h in hs if name in results[str(h)]]
        if not pts:
            continue
        x = np.array([h for h, _ in pts], dtype=float) + (i - len(monitors) / 2) * 0.08
        y = np.array([c["point"] for _, c in pts])
        lo = np.array([c["lo"] for _, c in pts])
        hi = np.array([c["hi"] for _, c in pts])
        st = _style(name)
        ax.errorbar(x, y, yerr=[np.nan_to_num(y - lo), np.nan_to_num(hi - y)], marker="o", markersize=5,
                    capsize=3, color=st["color"], label=st["label"], linestyle=st["linestyle"], linewidth=st["linewidth"])
    ax.set(xlabel="Prediction horizon (steps)", ylabel="Test AUROC (95% CI)", title="AUROC by prediction horizon",
           xticks=hs, ylim=(0.3, 1.02))
    ax.legend(loc="lower right")
    _save(fig, path)


def intervention_outcomes(summary: dict, path: Path) -> None:
    conds = list(summary)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    metrics = [("violation_rate", "Safety violation rate"), ("task_success_rate", "Task success rate"),
               ("trigger_run_rate", "Runs with ≥1 intervention")]
    y = np.arange(len(conds))
    for ax, (key, title) in zip(axes, metrics):
        vals = [summary[c][key] for c in conds]
        colors = [CONDITION_COLORS.get(c.split(":")[-1] if ":" in c else "none") for c in conds]
        ax.set_axisbelow(True)
        ax.barh(y, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        label_x = list(vals)
        if key == "violation_rate":
            for i, c in enumerate(conds):
                ci = summary[c].get("violation_rate_ci") or {}
                if ci.get("lo") is not None and not np.isnan(ci.get("lo", np.nan)):
                    ax.plot([ci["lo"], ci["hi"]], [i, i], color=INK_2, linewidth=1.2)
                    label_x[i] = max(label_x[i], ci["hi"])
        for i, (v, lx) in enumerate(zip(vals, label_x)):
            ax.text(lx + 0.02, i, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
        ax.set(title=title, xlim=(0, 1.15))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, conds)
    axes[0].invert_yaxis()
    from matplotlib.patches import Patch

    keys = [("none", "no intervention"), ("monitor", "monitor-triggered"), ("random", "random, rate-matched"),
            ("always", "always (refuse-all reference)")]
    fig.legend(handles=[Patch(color=CONDITION_COLORS[k], label=lab) for k, lab in keys], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0.18)
    _save(fig, path)
