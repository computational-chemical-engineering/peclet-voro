#!/usr/bin/env python3
"""Generate histogram report for per-cell Voronoi topology statistics.

Reads a CSV with columns:
  cell_id, vertices, edges, faces

Produces:
  - histogram PNGs for vertices, edges, and faces
  - a markdown report summarizing descriptive statistics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _describe(series: pd.Series) -> dict[str, float]:
    """Compute descriptive statistics for one topology count series.

    Args:
        series: Integer-valued pandas series.

    Returns:
        Dictionary with min/max/mean/std/median/percentiles.
    """
    arr = series.to_numpy(dtype=float)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _plot_hist(series: pd.Series, title: str, xlabel: str, out_file: Path, color: str) -> None:
    """Plot an integer histogram with unit-width bins centered on integers."""
    vals = series.to_numpy(dtype=int)
    vmin = int(vals.min())
    vmax = int(vals.max())
    bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cell count")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_xticks(np.arange(vmin, vmax + 1, max(1, (vmax - vmin) // 20 or 1)))
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def _mode_count(series: pd.Series) -> tuple[int, int]:
    """Return the modal integer value and its frequency."""
    vc = series.value_counts()
    mode_val = int(vc.index[0])
    mode_freq = int(vc.iloc[0])
    return mode_val, mode_freq


def main() -> None:
    """CLI entry point for histogram report generation."""
    ap = argparse.ArgumentParser(description="Generate Voronoi cell topology histogram report")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV with per-cell counts")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for report and figures")
    ap.add_argument("--n", type=int, required=True, help="Particle count used for the run")
    ap.add_argument("--seed", type=int, required=True, help="RNG seed used for the run")
    ap.add_argument("--report-name", type=str, default="CELL_TOPOLOGY_HIST_1E4_RANDOM.md")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ["vertices", "edges", "faces"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_vertices = "hist_vertices.png"
    fig_edges = "hist_edges.png"
    fig_faces = "hist_faces.png"

    _plot_hist(df["vertices"], "Voronoi Cell Vertices Histogram", "Vertices per cell", out_dir / fig_vertices, "#1f77b4")
    _plot_hist(df["edges"], "Voronoi Cell Edges Histogram", "Edges per cell", out_dir / fig_edges, "#ff7f0e")
    _plot_hist(df["faces"], "Voronoi Cell Faces Histogram", "Faces per cell", out_dir / fig_faces, "#2ca02c")

    stats_v = _describe(df["vertices"])
    stats_e = _describe(df["edges"])
    stats_f = _describe(df["faces"])

    mode_v, mode_v_n = _mode_count(df["vertices"])
    mode_e, mode_e_n = _mode_count(df["edges"])
    mode_f, mode_f_n = _mode_count(df["faces"])

    report = out_dir / args.report_name
    lines: list[str] = []
    lines.append("# Voronoi Cell Topology Statistics (Random Uniform, N=1e4)")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Particles: {args.n:,}")
    lines.append("- Distribution: random uniform in [0,1)^3")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Input CSV: {args.csv}")
    lines.append("")
    lines.append("## Histograms")
    lines.append("")
    lines.append(f"![Vertices histogram]({fig_vertices})")
    lines.append("")
    lines.append(f"![Edges histogram]({fig_edges})")
    lines.append("")
    lines.append(f"![Faces histogram]({fig_faces})")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Min | P05 | Median | P95 | Max | Mean | Std | Mode (count) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Vertices | {stats_v['min']:.0f} | {stats_v['p05']:.1f} | {stats_v['median']:.1f} | {stats_v['p95']:.1f} | {stats_v['max']:.0f} | {stats_v['mean']:.3f} | {stats_v['std']:.3f} | {mode_v} ({mode_v_n}) |"
    )
    lines.append(
        f"| Edges | {stats_e['min']:.0f} | {stats_e['p05']:.1f} | {stats_e['median']:.1f} | {stats_e['p95']:.1f} | {stats_e['max']:.0f} | {stats_e['mean']:.3f} | {stats_e['std']:.3f} | {mode_e} ({mode_e_n}) |"
    )
    lines.append(
        f"| Faces | {stats_f['min']:.0f} | {stats_f['p05']:.1f} | {stats_f['median']:.1f} | {stats_f['p95']:.1f} | {stats_f['max']:.0f} | {stats_f['mean']:.3f} | {stats_f['std']:.3f} | {mode_f} ({mode_f_n}) |"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Edge count is computed per cell via Euler relation for convex polyhedra: E = V + F - 2.")
    lines.append("- For periodic random Voronoi tessellations in 3D, face and edge counts track each other closely through this relation.")

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()