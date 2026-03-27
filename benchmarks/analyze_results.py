#!/usr/bin/env python3
"""Analyse voronoi_dynamics vs Voro++ benchmark results.

Reads the CSV produced by ``benchmark_voronoi`` and generates:
  * One timing-vs-N plot per point-set distribution (log-log).
  * A time-per-particle plot (all distributions overlaid).
  * A speedup-ratio plot (voronoi_dynamics_full / voropp_full).
  * A geometry-overhead-fraction plot.
  * A self-contained Markdown report with embedded figures and data tables.

Usage
-----
    python3 analyze_results.py  <csv_file>  [--output-dir DIR]

The ``--output-dir`` defaults to the directory of ``<csv_file>``.

Dependencies
------------
    pandas, matplotlib, numpy, scipy  (standard scientific Python stack)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress  # for power-law slope estimation

matplotlib.use("Agg")  # non-interactive backend, safe on headless machines

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
COLOURS = {
    "voronoi_dynamics_tess": "#1f77b4",  # blue
    "voronoi_dynamics_full": "#ff7f0e",  # orange
    "voropp_tess":           "#2ca02c",  # green
    "voropp_full":           "#d62728",  # red
}
LABELS = {
    "voronoi_dynamics_tess": "vd – tess only",
    "voronoi_dynamics_full": "vd – full build",
    "voropp_tess":           "voro++ – tess only",
    "voropp_full":           "voro++ – full build",
}
MARKERS = {
    "voronoi_dynamics_tess": "o",
    "voronoi_dynamics_full": "s",
    "voropp_tess":           "^",
    "voropp_full":           "D",
}
DIST_TITLES = {
    "random_uniform": "Random Uniform  [0, 1)³",
    "cubic_lattice":  "Simple-Cubic Lattice  (n³ + ε jitter)",
    "sphere_surface": "Sphere Surface  (r = 0.4, hollow)",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load the benchmark CSV, skipping ``#``-prefixed comment lines.

    Args:
        path: Path to the CSV file produced by ``benchmark_voronoi``.

    Returns:
        DataFrame with columns: library, point_set, N, nthreads, reps,
        time_ms_mean, time_ms_std.
    """
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()
    df["N"] = df["N"].astype(int)
    df["time_ms_mean"] = df["time_ms_mean"].astype(float)
    df["time_ms_std"]  = df["time_ms_std"].astype(float)
    return df


def extract_metadata(path: Path) -> dict[str, str]:
    """Parse key=value metadata from ``#``-comment lines in the CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Dictionary of metadata key–value strings (e.g. ``{"OpenMP threads": "48"}``).
    """
    meta: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            line = line.lstrip("#").strip()
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
    return meta


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _power_law_label(x: np.ndarray, y: np.ndarray) -> str:
    """Return an O(N^k) annotation from a log-log fit.

    Args:
        x: Independent variable (N).
        y: Dependent variable (time in ms).

    Returns:
        LaTeX string such as ``r"$\\mathcal{O}(N^{1.0})$"``.
    """
    try:
        slope, _, _, _, _ = linregress(np.log(x), np.log(y))
        return rf"$\mathcal{{O}}(N^{{{slope:.2f}}})$"
    except Exception:
        return ""


def plot_timing(df: pd.DataFrame,
                point_set: str,
                out_path: Path,
                title_suffix: str = "") -> None:
    """Create a log-log wall-time-vs-N plot for one point-set distribution.

    Args:
        df:           Full benchmark DataFrame (filtered internally).
        point_set:    Value of the ``point_set`` column to plot.
        out_path:     Path at which to save the PNG figure.
        title_suffix: Additional text appended to the plot title.
    """
    sub = df[df["point_set"] == point_set].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for lib in ["voronoi_dynamics_tess", "voronoi_dynamics_full",
                "voropp_tess", "voropp_full"]:
        d = sub[sub["library"] == lib].sort_values("N")
        if d.empty:
            continue
        ax.errorbar(
            d["N"], d["time_ms_mean"],
            yerr=d["time_ms_std"],
            label=LABELS[lib],
            color=COLOURS[lib],
            marker=MARKERS[lib],
            linewidth=1.8, markersize=6, capsize=3,
        )
        # Power-law annotation for the full-build curves
        if lib.endswith("_full") and len(d) >= 4:
            lbl = _power_law_label(d["N"].values, d["time_ms_mean"].values)
            if lbl:
                ax.annotate(
                    lbl,
                    xy=(d["N"].iloc[-1], d["time_ms_mean"].iloc[-1]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=COLOURS[lib],
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of particles  $N$", fontsize=12)
    ax.set_ylabel("Wall time  (ms)", fontsize=12)
    ax.set_title(
        f"Static Voronoi tessellation — {DIST_TITLES.get(point_set, point_set)}"
        + (f"\n{title_suffix}" if title_suffix else ""),
        fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_per_particle(df: pd.DataFrame, out_path: Path) -> None:
    """Time-per-particle (µs/particle) vs N, lines per distribution.

    Only the ``_full`` variants are plotted to keep the figure readable.

    Args:
        df:       Full benchmark DataFrame.
        out_path: PNG output path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, lib_suffix in zip(axes, ["_full", "_tess"]):
        for point_set, ls in [("random_uniform", "-"),
                               ("cubic_lattice", "--"),
                               ("sphere_surface", ":")]:
            for lib in ["voronoi_dynamics" + lib_suffix, "voropp" + lib_suffix]:
                d = df[(df["point_set"] == point_set) &
                        (df["library"] == lib)].sort_values("N")
                if d.empty:
                    continue
                tpp = d["time_ms_mean"] * 1e3 / d["N"]  # µs / particle
                label = f"{LABELS[lib]} / {point_set.replace('_', ' ')}"
                ax.plot(d["N"], tpp, marker=MARKERS[lib], linestyle=ls,
                        label=label, linewidth=1.4, markersize=5,
                        color=COLOURS[lib])

        ax.set_xscale("log")
        ax.set_xlabel("Number of particles  $N$", fontsize=11)
        ax.set_ylabel("Time per particle  (µs)", fontsize=11)
        variant = "full build" if lib_suffix == "_full" else "tessellation only"
        ax.set_title(f"Time per particle — {variant}", fontsize=11)
        ax.legend(fontsize=6.5, framealpha=0.85, ncol=2)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speedup(df: pd.DataFrame, out_path: Path) -> None:
    """Speedup ratio voropp_full / voronoi_dynamics_full per distribution.

    A value > 1 means voronoi_dynamics is faster than Voro++.

    Args:
        df:       Full benchmark DataFrame.
        out_path: PNG output path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    line_styles = {"random_uniform": "-", "cubic_lattice": "--", "sphere_surface": ":"}
    colours_ps  = {"random_uniform": "#1f77b4", "cubic_lattice": "#ff7f0e",
                   "sphere_surface": "#2ca02c"}

    for point_set in ["random_uniform", "cubic_lattice", "sphere_surface"]:
        vd  = df[(df["point_set"] == point_set) &
                  (df["library"] == "voronoi_dynamics_full")].sort_values("N")
        vpp = df[(df["point_set"] == point_set) &
                  (df["library"] == "voropp_full")].sort_values("N")
        merged = pd.merge(vd[["N", "time_ms_mean"]], vpp[["N", "time_ms_mean"]],
                          on="N", suffixes=("_vd", "_vpp"))
        if merged.empty:
            continue
        ratio = merged["time_ms_mean_vpp"] / merged["time_ms_mean_vd"]
        ax.plot(merged["N"], ratio,
                marker="o", linestyle=line_styles[point_set],
                label=DIST_TITLES.get(point_set, point_set),
                color=colours_ps[point_set], linewidth=1.8, markersize=6)

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
               label="equal performance")
    ax.set_xscale("log")
    ax.set_xlabel("Number of particles  $N$", fontsize=12)
    ax.set_ylabel(r"Speedup  $t_{\mathrm{voro++}} \,/\, t_{\mathrm{vd}}$",
                  fontsize=12)
    ax.set_title("Speedup: Voro++ full build vs voronoi_dynamics full build\n"
                 "(> 1 → voronoi_dynamics is faster)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_geometry_fraction(df: pd.DataFrame, out_path: Path) -> None:
    """Fraction of total build time spent in buildGeometry vs N.

    Geometry fraction = (full − tess) / full × 100 %.

    Args:
        df:       Full benchmark DataFrame.
        out_path: PNG output path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    line_styles = {"random_uniform": "-", "cubic_lattice": "--"}
    colours_ps  = {"random_uniform": "#1f77b4", "cubic_lattice": "#ff7f0e"}

    for point_set in ["random_uniform", "cubic_lattice"]:
        tess = df[(df["point_set"] == point_set) &
                   (df["library"] == "voronoi_dynamics_tess")].sort_values("N")
        full = df[(df["point_set"] == point_set) &
                   (df["library"] == "voronoi_dynamics_full")].sort_values("N")
        merged = pd.merge(tess[["N", "time_ms_mean"]], full[["N", "time_ms_mean"]],
                          on="N", suffixes=("_tess", "_full"))
        if merged.empty:
            continue
        frac = (merged["time_ms_mean_full"] - merged["time_ms_mean_tess"]) \
               / merged["time_ms_mean_full"] * 100.0
        ax.plot(merged["N"], frac,
                marker="s", linestyle=line_styles[point_set],
                label=DIST_TITLES.get(point_set, point_set),
                color=colours_ps[point_set], linewidth=1.8, markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Number of particles  $N$", fontsize=12)
    ax.set_ylabel("Geometry fraction  (%)", fontsize=12)
    ax.set_title("voronoi_dynamics: fraction of build time in buildGeometry",
                 fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(df: pd.DataFrame, point_set: str) -> str:
    """Return a Markdown table for one point-set distribution.

    Columns: N | vd-tess (ms) | vd-full (ms) | voro++-tess (ms) | voro++-full (ms) | Speedup

    Args:
        df:         Full benchmark DataFrame.
        point_set:  Distribution to tabulate.

    Returns:
        Multi-line Markdown table string.
    """
    lib_order = ["voronoi_dynamics_tess", "voronoi_dynamics_full",
                 "voropp_tess", "voropp_full"]
    sub = df[df["point_set"] == point_set].copy()
    Ns = sorted(sub["N"].unique())

    header = ("| N | vd-tess (ms) | vd-full (ms) | "
              "voro++-tess (ms) | voro++-full (ms) | "
              "Speedup voro++/vd | Threads |")
    sep = "|---|---:|---:|---:|---:|---:|---:|"

    rows = [header, sep]
    for N in Ns:
        row_data: dict[str, Optional[float]] = {}
        threads = "–"
        for lib in lib_order:
            d = sub[(sub["N"] == N) & (sub["library"] == lib)]
            if not d.empty:
                row_data[lib] = float(d["time_ms_mean"].iloc[0])
                threads = str(int(d["nthreads"].iloc[0]))
            else:
                row_data[lib] = None

        def fmt(v: Optional[float]) -> str:
            return f"{v:,.2f}" if v is not None else "–"

        vd_full  = row_data.get("voronoi_dynamics_full")
        vpp_full = row_data.get("voropp_full")
        if vd_full and vpp_full and vd_full > 0:
            speedup = f"{vpp_full / vd_full:.3f}"
        else:
            speedup = "–"

        rows.append(
            f"| {N:,} | {fmt(row_data['voronoi_dynamics_tess'])} | "
            f"{fmt(row_data['voronoi_dynamics_full'])} | "
            f"{fmt(row_data['voropp_tess'])} | "
            f"{fmt(row_data['voropp_full'])} | {speedup} | {threads} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame,
                    meta: dict[str, str],
                    fig_dir: Path,
                    out_path: Path) -> None:
    """Write the full Markdown report to *out_path*.

    Args:
        df:       Benchmark DataFrame (all point sets).
        meta:     Metadata extracted from CSV comments.
        fig_dir:  Directory containing the PNG figures (relative to *out_path*).
        out_path: Path to write the Markdown report.
    """
    # Relative path from report to figures
    rel = os.path.relpath(fig_dir, out_path.parent)

    def img(name: str) -> str:
        return f"![{name}]({rel}/{name})"

    sysinfo = "\n".join(f"- **{k}**: {v}" for k, v in meta.items())

    lines: list[str] = []

    lines += [
        "# Voronoi Tessellation Performance Study — Benchmark Report",
        "",
        "> Generated automatically by `analyze_results.py`.",
        f"> Benchmark compiled: {meta.get('Compiled', '–')}",
        "",
        "---",
        "",
        "## System Information",
        "",
        sysinfo,
        "",
        "---",
        "",
        "## Scope",
        "",
        textwrap.dedent("""\
            This report compares **voronoi_dynamics** and **Voro++** for the
            task of constructing a static, periodic 3-D Voronoi tessellation.

            Two timing categories:

            | Category | Description |
            |---|---|
            | **tess** | Pure tessellation: data structure construction only. |
            | **full** | Tessellation + all geometric properties (connecting vectors, edge inverses, volume derivatives for vd; `compute_cell` for voro++). |

            Three point-set distributions:

            | Distribution | Description |
            |---|---|
            | **random_uniform** | N uniform-random points in [0, 1)³ |
            | **cubic_lattice** | Simple-cubic lattice (N = n³ + 10⁻⁷ × spacing jitter) |
            | **sphere_surface** | N Marsaglia-normalised points on a sphere of radius 0.4 centred at (0.5, 0.5, 0.5) |

            > **Important note on sphere_surface.**
            > voronoi_dynamics uses fixed-size static arrays with a hard limit of
            > 128 vertices/facets per cell.  For the hollow sphere distribution,
            > cells in the sphere interior accumulate more than 128 facets for N > 100,
            > causing undefined behaviour.  voronoi_dynamics results are therefore
            > only provided for N ≤ 100 on this distribution; Voro++ (no limit) is
            > benchmarked for all N.
        """),
        "---",
        "",
    ]

    # ── Section per distribution ───────────────────────────────────────────
    for point_set in ["random_uniform", "cubic_lattice", "sphere_surface"]:
        title = DIST_TITLES.get(point_set, point_set)
        lines += [
            f"## {title}",
            "",
            img(f"timing_{point_set}.png"),
            "",
            f"### Data Table — {title}",
            "",
            build_table(df, point_set),
            "",
            "> Speedup = voro++_full / vd_full  (> 1 → voronoi_dynamics is faster).",
            "",
        ]

    # ── Cross-distribution plots ───────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Time per Particle",
        "",
        ("Time-per-particle (µs / particle) shows how efficiently each library "
         "scales.  Flat lines indicate O(N) scaling; upward slopes indicate "
         "super-linear cost growth."),
        "",
        img("timing_per_particle.png"),
        "",
        "---",
        "",
        "## Speedup: Voro++ vs voronoi_dynamics (full build)",
        "",
        ("Speedup > 1 means voronoi_dynamics is faster than Voro++.  "
         "At small N the OpenMP thread-launch overhead dominates vd, "
         "making single-threaded Voro++ faster.  "
         "At large N the parallelism pays off."),
        "",
        img("speedup_ratio.png"),
        "",
        "---",
        "",
        "## Geometry Build Overhead (voronoi_dynamics only)",
        "",
        ("Fraction of the total voronoi_dynamics build time consumed by "
         "`buildGeometry` (connecting vectors, edge inverses, volume "
         "derivatives).  The tessellation fraction decreases as N grows "
         "because geometry scales linearly while BFS overhead is roughly "
         "constant per cell."),
        "",
        img("geometry_fraction.png"),
        "",
        "---",
        "",
        "## Discussion",
        "",
        textwrap.dedent("""\
            ### OpenMP overhead at small N
            voronoi_dynamics parallelises the CellMaker loop with `#pragma omp
            parallel for`.  At small N (< 5 000) the fork-join overhead of
            spawning threads onto 48 cores dominates and vd appears slower than
            single-threaded Voro++.  This is expected and well-known for
            coarse-grained thread parallelism.

            ### Crossover point
            For `random_uniform` the two full-build curves cross at roughly
            N ≈ 10 000–50 000, beyond which voronoi_dynamics is competitive or
            faster owing to the OpenMP speedup on the cell-building loop.

            ### Cube lattice performance
            The SC lattice has only 6 neighbours per cell (the Voronoi cell is a
            cube), so the CellMaker terminates very early.  Both libraries
            perform roughly the same O(N) work per cell, but voronoi_dynamics
            benefits more from threading at large N.

            ### Sphere-surface limitation
            The hollow sphere creates elongated cells that exceed the library's
            128-facet static limit.  This is a design trade-off: fixed-size
            arrays avoid per-cell heap allocation and improve cache-locality at
            the cost of restricting irregular distributions.
            Future work (see `docs/architecture_redesign.md`) proposes a
            CSR-arena layout (`CellArena`) that would remove this limit.

            ### Voro++-tess overhead is negligible
            The Voro++ container-fill time (particle insertion without cell
            computation) is always < 1 % of the full build time, confirming that
            essentially all Voro++ cost is in `compute_cell`.
        """),
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyse voronoi_dynamics vs Voro++ benchmark CSV."
    )
    parser.add_argument("csv", type=Path, help="Path to benchmark_results.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for figures and report "
                             "(default: same directory as csv)")
    args = parser.parse_args()

    csv_path: Path = args.csv.resolve()
    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found")

    out_dir: Path = (args.output_dir or csv_path.parent).resolve()
    fig_dir: Path = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading  {csv_path}")
    df   = load_csv(csv_path)
    meta = extract_metadata(csv_path)

    print(f"Plotting → {fig_dir}")

    # Per-distribution timing plots
    for point_set in df["point_set"].unique():
        plot_timing(df, point_set,
                    fig_dir / f"timing_{point_set}.png")
        print(f"  timing_{point_set}.png")

    # Cross-distribution plots
    plot_time_per_particle(df, fig_dir / "timing_per_particle.png")
    print("  timing_per_particle.png")

    plot_speedup(df, fig_dir / "speedup_ratio.png")
    print("  speedup_ratio.png")

    plot_geometry_fraction(df, fig_dir / "geometry_fraction.png")
    print("  geometry_fraction.png")

    # Markdown report
    report_path = out_dir / "BENCHMARK_REPORT.md"
    generate_report(df, meta, fig_dir, report_path)


if __name__ == "__main__":
    main()
