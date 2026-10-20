"""Rebuild the dataset figures in figures/png and figures/svg.

    python figures/make_figures.py                 # all four
    python figures/make_figures.py --only gallery  # one

These describe the DATASET, not any learning method: what a task looks like,
how the four design axes are spread, how hard the tasks are, and where the
reference policies land. Numbers come from the frozen .npz and from
reference/, so a figure cannot drift from the data it claims to show.

Needs plotly and kaleido (pip install -r requirements.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from gridworld50 import load_family

ROOT = Path(__file__).resolve().parent.parent
PNG, SVG = Path(__file__).parent / "png", Path(__file__).parent / "svg"

# ── Academic palette ────────────────────────────────────────────────────────
AC = {
    "blue": "#2563EB", "amber": "#D97706", "green": "#059669", "red": "#DC2626",
    "violet": "#7C3AED", "teal": "#0891B2", "rose": "#BE185D", "sienna": "#92400E",
    "bg": "#FFFFFF", "surface": "#F8F9FA", "border": "#DEE2E6",
    "axis": "#495057", "grid": "#E9ECEF",
    "text_primary": "#212529", "text_muted": "#6C757D", "text_faint": "#ADB5BD",
}
FONT_UI = "Inter, Helvetica, Arial, sans-serif"
FONT_TITLE = "Source Serif 4, Georgia, serif"
FONT_MONO = "JetBrains Mono, Menlo, Consolas, monospace"


def install_template() -> None:
    """White ground, Tufte spine, faint horizontal grid, shared palette."""
    pio.templates["academic"] = go.layout.Template(layout=dict(
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        colorway=[AC["blue"], AC["amber"], AC["green"], AC["red"],
                  AC["violet"], AC["teal"], AC["rose"], AC["sienna"]],
        font=dict(family=FONT_UI, size=12, color=AC["text_primary"]),
        title=dict(font=dict(family=FONT_TITLE, size=16,
                             color=AC["text_primary"]), x=0.0, xanchor="left"),
        margin=dict(l=70, r=120, t=58, b=58),
        showlegend=False,
        xaxis=dict(showline=True, linecolor=AC["axis"], linewidth=1.2, mirror=False,
                   ticks="outside", tickcolor=AC["axis"], ticklen=4, tickwidth=1.0,
                   tickfont=dict(family=FONT_MONO, size=11, color=AC["text_muted"]),
                   title=dict(font=dict(family=FONT_UI, size=13,
                                        color=AC["text_primary"])),
                   showgrid=False, zeroline=False),
        yaxis=dict(showline=True, linecolor=AC["axis"], linewidth=1.2, mirror=False,
                   ticks="outside", tickcolor=AC["axis"], ticklen=4, tickwidth=1.0,
                   tickfont=dict(family=FONT_MONO, size=11, color=AC["text_muted"]),
                   title=dict(font=dict(family=FONT_UI, size=13,
                                        color=AC["text_primary"])),
                   showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                   zeroline=False),
    ))
    pio.templates.default = "academic"


def export(fig: go.Figure, stem: str, width: int, height: int) -> None:
    """Write PNG at 300 dpi and SVG from the same figure object."""
    PNG.mkdir(parents=True, exist_ok=True)
    SVG.mkdir(parents=True, exist_ok=True)
    # 300 dpi against a 96 dpi CSS pixel basis.
    fig.write_image(PNG / f"{stem}.png", width=width, height=height, scale=300 / 96)
    fig.write_image(SVG / f"{stem}.svg", width=width, height=height)
    print(f"  wrote png/{stem}.png and svg/{stem}.svg")


def load_reference() -> dict[str, list]:
    """Read the shipped reference numbers from reference/."""
    out = {}
    for name in ("floor_ceiling", "specialist_learnability", "joint_sep_matched"):
        path = ROOT / "reference" / f"{name}.json"
        out[name] = json.loads(path.read_text()) if path.exists() else []
    return out


# ── Figure 1: what a task looks like ───────────────────────────────────────

def figure_gallery(family, n: int = 8) -> None:
    """A sample of layouts: walls, the goal, and the cells a spawn may use."""
    # Spread the sample across the five path-length bands rather than taking the
    # first n, so the gallery shows the difficulty range instead of one corner.
    by_band: dict[tuple, list[int]] = {}
    for i, t in enumerate(family.tasks):
        by_band.setdefault(tuple(t.spec.params["start_band"]), []).append(i)
    picks: list[int] = []
    while len(picks) < n:
        before = len(picks)
        for band in sorted(by_band):
            pool = [i for i in by_band[band] if i not in picks]
            if pool and len(picks) < n:
                picks.append(pool[0])
        if len(picks) == before:
            break
    picks = sorted(picks)

    cols = 4
    rows = (len(picks) + cols - 1) // cols
    titles = []
    for i in picks:
        t = family.tasks[i]
        lo, hi = t.spec.params["start_band"]
        titles.append(f"task {i + 1} &nbsp; d {lo}-{hi} &nbsp; "
                      f"slip {t.slip} &nbsp; wall {t.spec.params['obstacle_density']:.0%}")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.035, vertical_spacing=0.085)

    for k, task_id in enumerate(picks):
        task = family.tasks[task_id]
        size = task.size
        # 0 free, 1 spawnable, 2 wall, 3 goal. A four-level ordinal scale, so
        # lightness carries the meaning and no hue cycles.
        layer = np.zeros((size, size), dtype=np.int8)
        starts = task.starts.numpy()
        layer[starts[:, 0], starts[:, 1]] = 1
        layer[task.blocked.numpy()] = 2
        layer[task.goal[0], task.goal[1]] = 3
        r, c = k // cols + 1, k % cols + 1
        fig.add_trace(go.Heatmap(
            z=layer, colorscale=[
                [0.00, AC["bg"]],      [0.249, AC["bg"]],
                [0.25, "#BFD3F7"],     [0.499, "#BFD3F7"],   # spawn band, blue 25%
                [0.50, AC["axis"]],    [0.749, AC["axis"]],   # wall
                [0.75, AC["red"]],     [1.00, AC["red"]],     # goal
            ],
            zmin=0, zmax=3, showscale=False, hoverinfo="skip",
        ), row=r, col=c)
        # The goal is one cell in 2500; a ring makes it findable at print size.
        fig.add_trace(go.Scatter(
            x=[task.goal[1]], y=[task.goal[0]], mode="markers",
            marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(
                width=1.6, color=AC["red"])), hoverinfo="skip",
        ), row=r, col=c)
        fig.update_xaxes(showticklabels=False, showline=False, ticks="",
                         showgrid=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, showline=False, ticks="",
                         showgrid=False, scaleanchor=f"x{k + 1 if k else ''}",
                         autorange="reversed", row=r, col=c)

    for note in fig.layout.annotations:
        note.font.update(family=FONT_UI, size=10.5, color=AC["text_muted"])

    fig.update_layout(
        title=dict(text="GridWorld-50, sample task layouts"),
        margin=dict(l=20, r=20, t=96, b=58), height=240 * rows + 150, width=1000,
    )
    fig.add_annotation(
        x=0, y=-0.055, xref="paper", yref="paper", showarrow=False,
        xanchor="left", align="left",
        text="<span style='color:#495057'>■</span> wall &nbsp;&nbsp;"
             "<span style='color:#BFD3F7'>■</span> spawnable, inside the band "
             "&nbsp;&nbsp;<span style='color:#DC2626'>■</span> goal &nbsp;&nbsp;"
             "<span style='color:#ADB5BD'>□</span> free but outside the band",
        font=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
    )
    export(fig, "task_gallery", 1000, 240 * rows + 150)


# ── Figure 2: how the four axes are spread ─────────────────────────────────

def figure_axes(family) -> None:
    """The four design axes over the fifty tasks, each against arrival order."""
    ids = np.arange(1, len(family.tasks) + 1)
    density = np.array([t.spec.params["obstacle_density"] for t in family.tasks])
    slip = np.array([t.slip for t in family.tasks])
    band_mid = np.array([np.mean(t.spec.params["start_band"]) for t in family.tasks])
    view = np.array([t.view_radius for t in family.tasks])

    panels = [
        ("Wall density", density, AC["blue"], ".0%", "fraction of cells blocked"),
        ("Slip", slip, AC["amber"], ".2f", "chance the action is replaced"),
        ("Spawn distance band", band_mid, AC["green"], ".0f", "band midpoint, steps"),
        ("View radius", view, AC["violet"], ".0f", "egocentric window, cells"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels],
                        horizontal_spacing=0.10, vertical_spacing=0.16)
    for k, (name, values, color, fmt, ylab) in enumerate(panels):
        r, c = k // 2 + 1, k % 2 + 1
        fig.add_trace(go.Scatter(
            x=ids, y=values, mode="markers",
            marker=dict(size=7, color=color, line=dict(width=1, color=AC["bg"])),
            hoverinfo="skip",
        ), row=r, col=c)
        fig.update_xaxes(title_text="task, arrival order" if r == 2 else None,
                         dtick=10, row=r, col=c)
        lo, hi = float(np.min(values)), float(np.max(values))
        pad = (hi - lo) * 0.18 or 0.5
        fig.update_yaxes(title_text=ylab, tickformat=fmt,
                         range=[lo - pad, hi + pad],
                         dtick=1 if name == "View radius" else None,
                         row=r, col=c)

    for note in fig.layout.annotations:
        note.font.update(family=FONT_UI, size=13, color=AC["text_primary"])

    fig.update_layout(
        title=dict(text="The four design axes vary independently"),
        height=640, width=980, margin=dict(l=76, r=40, t=86, b=62),
    )
    export(fig, "axis_coverage", 980, 640)


# ── Figure 3: the difficulty axis ──────────────────────────────────────────

def figure_path_lengths(family) -> None:
    """Per-task spawn-distance spread, which is what grades difficulty."""
    order = sorted(range(len(family.tasks)),
                   key=lambda i: np.mean(family.tasks[i].spec.params["start_band"]))
    xs, med, lo_w, hi_w, p95s = [], [], [], [], []
    for rank, i in enumerate(order, start=1):
        task = family.tasks[i]
        starts = task.starts.numpy()
        d = task.goal_dist.numpy()[starts[:, 0], starts[:, 1]]
        xs.append(rank)
        med.append(float(np.median(d)))
        lo_w.append(float(np.percentile(d, 5)))
        hi_w.append(float(np.percentile(d, 95)))
        p95s.append(float(np.percentile(d, 95)))

    fig = go.Figure()
    # Spread first, so the median marker sits on top of it.
    for x, lo, hi in zip(xs, lo_w, hi_w):
        fig.add_trace(go.Scatter(
            x=[x, x], y=[lo, hi], mode="lines",
            line=dict(color="#BFD3F7", width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=xs, y=med, mode="markers",
        marker=dict(size=6, color=AC["blue"], line=dict(width=1, color=AC["bg"])),
        hoverinfo="skip"))

    horizon = 160
    fig.add_hline(y=horizon * 0.5, line=dict(color=AC["red"], width=1.2, dash="dash"))
    fig.add_annotation(
        x=len(xs), y=horizon * 0.5, text=f"solvability limit, {horizon // 2} steps",
        showarrow=False, xanchor="right", yanchor="bottom", yshift=4,
        font=dict(family=FONT_UI, size=11, color=AC["red"]),
        bgcolor="rgba(255,255,255,0.82)", borderpad=2)
    worst = max(p95s)
    fig.add_annotation(
        x=xs[int(np.argmax(p95s))], y=worst,
        text=f"worst 95th percentile, {worst:.0f} steps",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=-54, ay=34, xanchor="right",
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        bgcolor="rgba(255,255,255,0.82)", borderpad=3)

    fig.update_layout(
        title=dict(text="Every task is solvable in half the episode budget"),
        xaxis=dict(title="task, ordered by spawn-distance band", dtick=10),
        yaxis=dict(title="shortest path from spawn to goal (steps)",
                   range=[0, max(horizon * 0.5, worst) * 1.20]),
        height=520, width=940, margin=dict(l=78, r=40, t=62, b=62),
    )
    export(fig, "path_length_profile", 940, 520)


# ── Figure 4: where the reference policies land ────────────────────────────

def figure_reference(family, ref: dict[str, list]) -> None:
    """Random floor, shortest-path oracle, single-task specialist, joint policy."""
    floor_ceiling = ref["floor_ceiling"]
    specialists = {r["task"]: r["score"] for r in ref["specialist_learnability"]}
    joint = ref["joint_sep_matched"][0]["scores"] if ref["joint_sep_matched"] else []

    ids = [r["task"] for r in floor_ceiling]
    rand = [r["random"]["mean"] for r in floor_ceiling]
    orac = [r["oracle"]["mean"] for r in floor_ceiling]
    spec = [specialists.get(i, np.nan) for i in ids]
    jnt = [joint[i - 1] if i - 1 < len(joint) else np.nan for i in ids]

    series = [
        ("Oracle", orac, AC["green"], 6.5),
        ("Joint", jnt, AC["amber"], 5.5),
        ("Specialist", spec, AC["blue"], 5.5),
        ("Random", rand, AC["text_faint"], 5.0),
    ]
    fig = go.Figure()
    for name, values, color, size in series:
        fig.add_trace(go.Scatter(
            x=ids, y=values, mode="markers", name=name,
            marker=dict(size=size, color=color, line=dict(width=0.8, color=AC["bg"])),
            hoverinfo="skip"))

    # Direct labels at each series' mean, pushed apart so near-equal means
    # (oracle and joint sit 0.01 apart) do not print on top of one another.
    labels = sorted(((float(np.nanmean(v)), n, c) for n, v, c, _ in series),
                    reverse=True)
    placed: list[float] = []
    for mean, name, color in labels:
        y = mean
        while any(abs(y - other) < 0.075 for other in placed):
            y -= 0.075
        placed.append(y)
        fig.add_annotation(
            x=ids[-1], y=y, xanchor="left", xshift=12,
            text=f"<b>{name}</b> <span style='font-family:{FONT_MONO}'>"
                 f"{mean:.2f}</span>",
            showarrow=False, font=dict(family=FONT_UI, size=11.5, color=color))

    fig.add_hline(y=0.6, line=dict(color=AC["text_faint"], width=1.0, dash="dot"))
    fig.add_annotation(
        x=1, y=0.6, text="success bar, 0.60", showarrow=False,
        xanchor="left", yanchor="bottom", yshift=3,
        font=dict(family=FONT_UI, size=10.5, color=AC["text_muted"]))

    fig.update_layout(
        title=dict(text="Reference scores per task, floor to ceiling"),
        xaxis=dict(title="task, arrival order", dtick=10,
                   range=[0, len(ids) + 1]),
        yaxis=dict(title="normalized episode score", range=[-0.08, 1.12],
                   dtick=0.2, tickformat=".1f"),
        height=520, width=1000, margin=dict(l=78, r=140, t=62, b=62),
    )
    export(fig, "reference_scores", 1000, 520)


FIGURES = {
    "gallery": lambda fam, ref: figure_gallery(fam),
    "axes": lambda fam, ref: figure_axes(fam),
    "paths": lambda fam, ref: figure_path_lengths(fam),
    "reference": figure_reference,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FIGURES), nargs="*",
                    help="build just these (default: all)")
    args = ap.parse_args()

    install_template()
    family = load_family()
    ref = load_reference()
    for name in (args.only or sorted(FIGURES)):
        print(f"{name}:")
        FIGURES[name](family, ref)


if __name__ == "__main__":
    main()
