#!/usr/bin/env python3
"""Generate report for fair basic-build benchmark with thread scaling."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["N"] = df["N"].astype(int)
    df["nthreads"] = df["nthreads"].astype(int)
    df["time_ms_mean"] = df["time_ms_mean"].astype(float)
    return df


def make_fair_single_thread_plot(df_noomp: pd.DataFrame, out_file: Path) -> pd.DataFrame:
    vd = df_noomp[df_noomp["library"] == "voronoi_dynamics_build_basic"].copy()
    vp = df_noomp[df_noomp["library"] == "voropp_compute_cell"].copy()
    merged = vd.merge(vp, on=["point_set", "N"], suffixes=("_vd", "_vpp")).sort_values(["point_set", "N"])
    merged["ratio_vd_over_vpp"] = merged["time_ms_mean_vd"] / merged["time_ms_mean_vpp"]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    styles = {
        ("random_uniform", "vd"): dict(color="#1f77b4", marker="o", ls="-", label="vd build(false) random"),
        ("random_uniform", "vpp"): dict(color="#1f77b4", marker="s", ls="--", label="voro++ compute_cell random"),
        ("cubic_lattice", "vd"): dict(color="#ff7f0e", marker="o", ls="-", label="vd build(false) lattice"),
        ("cubic_lattice", "vpp"): dict(color="#ff7f0e", marker="s", ls="--", label="voro++ compute_cell lattice"),
    }

    for ps in ["random_uniform", "cubic_lattice"]:
        s = merged[merged["point_set"] == ps]
        ax.plot(s["N"], s["time_ms_mean_vd"], linewidth=2, markersize=5, **styles[(ps, "vd")])
        ax.plot(s["N"], s["time_ms_mean_vpp"], linewidth=2, markersize=5, **styles[(ps, "vpp")])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_xlabel("Number of particles N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Fair single-thread comparison: vd build(false) vs voro++ compute_cell")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)

    return merged


def make_vd_scaling_plot(df_threads: pd.DataFrame, df_noomp: pd.DataFrame, out_file: Path) -> pd.DataFrame:
    vd = df_threads[df_threads["library"] == "voronoi_dynamics_build_basic"].copy()

    base = df_noomp[df_noomp["library"] == "voronoi_dynamics_build_basic"][
        ["point_set", "N", "time_ms_mean"]
    ].rename(columns={"time_ms_mean": "t_noomp"})
    merged = vd.merge(base, on=["point_set", "N"])
    merged["speedup_vs_noomp"] = merged["t_noomp"] / merged["time_ms_mean"]

    reps = {
        "random_uniform": [10000, 100000, 1000000],
        "cubic_lattice": [8000, 125000, 1000000],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, ps in zip(axes, ["random_uniform", "cubic_lattice"]):
        sps = merged[merged["point_set"] == ps]
        cmap = ["#1f77b4", "#2ca02c", "#d62728"]
        for color, n in zip(cmap, reps[ps]):
            sn = sps[sps["N"] == n].sort_values("nthreads")
            if sn.empty:
                continue
            ax.plot(
                sn["nthreads"],
                sn["speedup_vs_noomp"],
                marker="o",
                linewidth=2,
                color=color,
                label=f"N={n:,}",
            )
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_xlabel("Threads")
        ax.set_title(ps)
        ax.grid(True, ls="--", lw=0.4, alpha=0.7)
        ax.set_xticks(sorted(sps["nthreads"].unique()))
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Speedup vs no-OpenMP single-thread")
    fig.suptitle("voronoi_dynamics basic-build thread scaling")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)

    return merged


def make_markdown(
    fair: pd.DataFrame,
    scaling: pd.DataFrame,
    out_md: Path,
    fig_fair: str,
    fig_scaling: str,
) -> None:
    rnd = fair[fair["point_set"] == "random_uniform"].copy()
    lat = fair[fair["point_set"] == "cubic_lattice"].copy()
    thread_counts = sorted(scaling["nthreads"].unique().tolist())
    thread_counts_txt = ", ".join(str(int(t)) for t in thread_counts)

    rnd_slower = rnd[rnd["ratio_vd_over_vpp"] > 1]["N"].tolist()
    lat_slower = lat[lat["ratio_vd_over_vpp"] > 1]["N"].tolist()
    rnd_slower_txt = ", ".join(f"{int(n):,}" for n in rnd_slower) if rnd_slower else "none"
    lat_slower_txt = ", ".join(f"{int(n):,}" for n in lat_slower) if lat_slower else "none"

    slope_rows = []
    for ps in ["random_uniform", "cubic_lattice"]:
        s_vd = fair[fair["point_set"] == ps]
        x = np.log(s_vd["N"].to_numpy())
        y_vd = np.log(s_vd["time_ms_mean_vd"].to_numpy())
        y_vp = np.log(s_vd["time_ms_mean_vpp"].to_numpy())
        svd = np.polyfit(x, y_vd, 1)[0]
        svp = np.polyfit(x, y_vp, 1)[0]
        slope_rows.append((ps, svd, svp))

    scale_summary = []
    for ps in ["random_uniform", "cubic_lattice"]:
        s = scaling[(scaling["point_set"] == ps) & (scaling["N"] == scaling[scaling["point_set"] == ps]["N"].max())]
        s = s.sort_values("nthreads")
        if not s.empty:
            best = s.loc[s["time_ms_mean"].idxmin()]
            scale_summary.append((ps, int(best["nthreads"]), float(best["speedup_vs_noomp"])))

    lines: list[str] = []
    lines.append("# Basic-Build Benchmark Report (No Sphere)")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Fair comparison: `CellComplex::build(false)` vs `voro++ compute_cell`")
    lines.append("- Sphere case excluded")
    lines.append("- Single-thread no-OpenMP fair comparison")
    lines.append(f"- vd thread scaling at {thread_counts_txt} threads")
    lines.append("")
    lines.append("## Fair Single-thread Comparison")
    lines.append("")
    lines.append(f"![fair]({fig_fair})")
    lines.append("")
    lines.append(f"- Random uniform: vd is slower only at N = {rnd_slower_txt}; otherwise vd is faster.")
    lines.append(f"- Cubic lattice: vd is slower at N = {lat_slower_txt}.")
    lines.append("")
    lines.append("### Scaling exponents (log-log fit)")
    lines.append("")
    lines.append("| Dataset | vd build(false) slope | voro++ compute_cell slope |")
    lines.append("|---|---:|---:|")
    for ps, svd, svp in slope_rows:
        lines.append(f"| {ps} | {svd:.3f} | {svp:.3f} |")
    lines.append("")

    lines.append("### Fair tables")
    for ps, s in [("random_uniform", rnd), ("cubic_lattice", lat)]:
        lines.append("")
        lines.append(f"#### {ps}")
        lines.append("| N | vd build(false) (ms) | voro++ compute_cell (ms) | vd/voro++ |")
        lines.append("|---:|---:|---:|---:|")
        for _, r in s.iterrows():
            lines.append(
                f"| {int(r['N']):,} | {r['time_ms_mean_vd']:.3f} | {r['time_ms_mean_vpp']:.3f} | {r['ratio_vd_over_vpp']:.3f} |"
            )
    lines.append("")
    lines.append("## vd Thread Scaling")
    lines.append("")
    lines.append(f"![scaling]({fig_scaling})")
    lines.append("")
    lines.append("- Speedups are reported relative to no-OpenMP single-thread runtime.")
    for ps, best_t, best_s in scale_summary:
        lines.append(
            f"- {ps}: best observed runtime among tested thread counts occurs at {best_t} threads "
            f"(speedup vs no-OpenMP single-thread: {best_s:.2f}x)."
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noomp-csv", type=Path, required=True)
    ap.add_argument("--threads-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_noomp = read_csv(args.noomp_csv)

    thread_csvs = sorted(args.threads_dir.glob("benchmark_full_build_vd_t*.csv"))
    if not thread_csvs:
        raise RuntimeError("No thread sweep CSV files found.")
    df_threads = pd.concat([read_csv(p) for p in thread_csvs], ignore_index=True)

    fair = make_fair_single_thread_plot(df_noomp, args.out_dir / "full_build_fair_single_thread.png")
    scaling = make_vd_scaling_plot(df_threads, df_noomp, args.out_dir / "full_build_vd_thread_scaling.png")

    make_markdown(
        fair=fair,
        scaling=scaling,
        out_md=args.out_dir / "FULL_BUILD_THREADS_REPORT.md",
        fig_fair="full_build_fair_single_thread.png",
        fig_scaling="full_build_vd_thread_scaling.png",
    )


if __name__ == "__main__":
    main()
