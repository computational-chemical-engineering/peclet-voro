#!/usr/bin/env python3
"""Generate report for NbrList setup diagnostic benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["N"] = df["N"].astype(int)
    df["nthreads"] = df["nthreads"].astype(int)
    df["time_ms_mean"] = df["time_ms_mean"].astype(float)
    df["time_ms_std"] = df["time_ms_std"].astype(float)
    return df


def load_data(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("benchmark_nbrlist_setup_diagnostic_t*.csv"))
    if not csvs:
        raise RuntimeError("No diagnostic CSV files found.")
    return pd.concat([read_csv(p) for p in csvs], ignore_index=True)


def make_overhead_plot(df: pd.DataFrame, out_file: Path) -> None:
    datasets = [("random_uniform", 10000), ("cubic_lattice", 8000)]
    labels = {
        "omp_empty_region": "empty parallel region",
        "omp_three_barriers": "parallel region + 3 barriers",
    }
    colors = {
        "omp_empty_region": "#1f77b4",
        "omp_three_barriers": "#d62728",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for ax, (point_set, nval) in zip(axes, datasets):
        s = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        for bench in ["omp_empty_region", "omp_three_barriers"]:
            sb = s[s["benchmark"] == bench].sort_values("nthreads")
            ax.plot(sb["nthreads"], sb["time_ms_mean"], marker="o", linewidth=2, color=colors[bench], label=labels[bench])
        ax.set_title(f"{point_set}, N={nval:,}")
        ax.set_xlabel("Threads")
        ax.set_xticks(sorted(s["nthreads"].unique()))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("OpenMP overhead diagnostics")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def make_setup_plot(df: pd.DataFrame, out_file: Path) -> None:
    datasets = [("random_uniform", 10000), ("cubic_lattice", 8000)]
    labels = {
        "setup_index_only": "index",
        "setup_histogram_only": "histogram",
        "setup_scatter_only": "scatter",
        "setup_library": "library setup",
    }
    colors = {
        "setup_index_only": "#1f77b4",
        "setup_histogram_only": "#ff7f0e",
        "setup_scatter_only": "#2ca02c",
        "setup_library": "#d62728",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for ax, (point_set, nval) in zip(axes, datasets):
        s = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        for bench in ["setup_index_only", "setup_histogram_only", "setup_scatter_only", "setup_library"]:
            sb = s[s["benchmark"] == bench].sort_values("nthreads")
            ax.plot(sb["nthreads"], sb["time_ms_mean"], marker="o", linewidth=2, color=colors[bench], label=labels[bench])
        ax.set_title(f"{point_set}, N={nval:,}")
        ax.set_xlabel("Threads")
        ax.set_xticks(sorted(s["nthreads"].unique()))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("NbrList::setup diagnostic subphases")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def fmt_row(df: pd.DataFrame, point_set: str, nval: int, benchmark: str, nthreads: int) -> str:
    row = df[
        (df["point_set"] == point_set)
        & (df["N"] == nval)
        & (df["benchmark"] == benchmark)
        & (df["nthreads"] == nthreads)
    ]
    if row.empty:
        return "n/a"
    return f"{row.iloc[0]['time_ms_mean']:.3f}"


def make_markdown(df: pd.DataFrame, out_md: Path) -> None:
    cases = [("random_uniform", 10000), ("cubic_lattice", 8000)]
    lines: list[str] = []
    lines.append("# NbrList Setup Diagnostic Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Separate diagnostic benchmark for OpenMP overhead and NbrList setup subphases")
    lines.append("- Cases: `random_uniform N=10000` and `cubic_lattice N=8000`")
    lines.append("- Benchmarks: empty region, 3 barriers, index, histogram, scatter, decomposed setup, library setup")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("![overhead](nbrlist_setup_diagnostic_overhead.png)")
    lines.append("")
    lines.append("![setup](nbrlist_setup_diagnostic_setup.png)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    for point_set, nval in cases:
        s = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        empty_48 = s[(s["benchmark"] == "omp_empty_region") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        barriers_48 = s[(s["benchmark"] == "omp_three_barriers") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        index_48 = s[(s["benchmark"] == "setup_index_only") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        hist_48 = s[(s["benchmark"] == "setup_histogram_only") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        scatter_48 = s[(s["benchmark"] == "setup_scatter_only") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        lib_48 = s[(s["benchmark"] == "setup_library") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        lib_32 = s[(s["benchmark"] == "setup_library") & (s["nthreads"] == 32)]["time_ms_mean"].iloc[0]

        lines.append(
            f"- `{point_set} N={nval:,}`: library setup jumps from {lib_32:.3f} ms at 32 threads to {lib_48:.3f} ms at 48 threads."
        )
        lines.append(
            f"- `{point_set} N={nval:,}` at 48 threads: empty region {empty_48:.3f} ms, 3 barriers {barriers_48:.3f} ms, "
            f"index {index_48:.3f} ms, histogram {hist_48:.3f} ms, scatter {scatter_48:.3f} ms."
        )
    lines.append("")

    lines.append("## Tables")
    for point_set, nval in cases:
        lines.append("")
        lines.append(f"### {point_set}, N={nval:,}")
        lines.append("")
        lines.append("| Threads | empty | 3 barriers | index | histogram | scatter | decomposed | library |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        for nthreads in sorted(sub["nthreads"].unique()):
            lines.append(
                f"| {nthreads} | "
                f"{fmt_row(df, point_set, nval, 'omp_empty_region', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'omp_three_barriers', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'setup_index_only', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'setup_histogram_only', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'setup_scatter_only', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'setup_full_decomposed', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'setup_library', nthreads)} |"
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.results_dir)
    make_overhead_plot(df, args.results_dir / "nbrlist_setup_diagnostic_overhead.png")
    make_setup_plot(df, args.results_dir / "nbrlist_setup_diagnostic_setup.png")
    make_markdown(df, args.results_dir / "NBRLIST_SETUP_DIAGNOSTIC_REPORT.md")


if __name__ == "__main__":
    main()
