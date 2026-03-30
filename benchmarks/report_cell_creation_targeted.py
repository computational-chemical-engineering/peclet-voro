#!/usr/bin/env python3
"""Generate a focused report for the low-N cell-creation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BENCH_ORDER = [
    "vd_setup_only",
    "vd_cells_dynamic64",
    "vd_cells_static",
    "vd_total_dynamic64",
    "vd_total_static",
]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["N"] = df["N"].astype(int)
    df["nthreads"] = df["nthreads"].astype(int)
    df["time_ms_mean"] = df["time_ms_mean"].astype(float)
    df["time_ms_std"] = df["time_ms_std"].astype(float)
    return df


def combined_data(noomp_csv: Path, threads_dir: Path) -> pd.DataFrame:
    df_noomp = read_csv(noomp_csv)
    thread_csvs = sorted(threads_dir.glob("benchmark_cell_creation_targeted_t*.csv"))
    if not thread_csvs:
        raise RuntimeError("No targeted thread sweep CSV files found.")
    df_threads = pd.concat([read_csv(p) for p in thread_csvs], ignore_index=True)
    return pd.concat([df_noomp, df_threads], ignore_index=True)


def make_stage_plot(df: pd.DataFrame, out_file: Path) -> None:
    datasets = [
        ("random_uniform", 10000),
        ("cubic_lattice", 8000),
    ]
    labels = {
        "vd_setup_only": "NbrList::setup",
        "vd_cells_dynamic64": "cells dynamic(64)",
        "vd_cells_static": "cells static",
    }
    colors = {
        "vd_setup_only": "#1f77b4",
        "vd_cells_dynamic64": "#d62728",
        "vd_cells_static": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for ax, (point_set, nval) in zip(axes, datasets):
        s = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        for bench in ["vd_setup_only", "vd_cells_dynamic64", "vd_cells_static"]:
            sb = s[s["benchmark"] == bench].sort_values("nthreads")
            ax.plot(
                sb["nthreads"],
                sb["time_ms_mean"],
                marker="o",
                linewidth=2,
                color=colors[bench],
                label=labels[bench],
            )
        ax.set_title(f"{point_set}, N={nval:,}")
        ax.set_xlabel("Threads")
        ax.set_xticks(sorted(s["nthreads"].unique()))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Targeted low-N breakdown: setup vs cell loop")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def make_total_plot(df: pd.DataFrame, out_file: Path) -> None:
    datasets = [
        ("random_uniform", 10000),
        ("cubic_lattice", 8000),
    ]
    labels = {
        "vd_total_dynamic64": "total dynamic(64)",
        "vd_total_static": "total static",
        "vd_total_dynamic64_second_of_two": "2nd-of-2 dynamic(64)",
        "vd_total_static_second_of_two": "2nd-of-2 static",
        "vd_total_dynamic64_persistent_second_of_two": "persistent 2nd-of-2 dynamic(64)",
        "vd_total_static_persistent_second_of_two": "persistent 2nd-of-2 static",
    }
    colors = {
        "vd_total_dynamic64": "#d62728",
        "vd_total_static": "#2ca02c",
        "vd_total_dynamic64_second_of_two": "#9467bd",
        "vd_total_static_second_of_two": "#8c564b",
        "vd_total_dynamic64_persistent_second_of_two": "#ff7f0e",
        "vd_total_static_persistent_second_of_two": "#17becf",
    }
    styles = {
        "vd_total_dynamic64": "-",
        "vd_total_static": "-",
        "vd_total_dynamic64_second_of_two": "--",
        "vd_total_static_second_of_two": "--",
        "vd_total_dynamic64_persistent_second_of_two": ":",
        "vd_total_static_persistent_second_of_two": ":",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for ax, (point_set, nval) in zip(axes, datasets):
        s = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        for bench in [
            "vd_total_dynamic64",
            "vd_total_static",
            "vd_total_dynamic64_second_of_two",
            "vd_total_static_second_of_two",
            "vd_total_dynamic64_persistent_second_of_two",
            "vd_total_static_persistent_second_of_two",
        ]:
            sb = s[s["benchmark"] == bench].sort_values("nthreads")
            ax.plot(
                sb["nthreads"],
                sb["time_ms_mean"],
                marker="o",
                linewidth=2,
                color=colors[bench],
                linestyle=styles[bench],
                label=labels[bench],
            )
        ax.set_title(f"{point_set}, N={nval:,}")
        ax.set_xlabel("Threads")
        ax.set_xticks(sorted(s["nthreads"].unique()))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Targeted low-N total runtime vs threads")
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
    cases = [
        ("random_uniform", 10000),
        ("cubic_lattice", 8000),
    ]

    lines: list[str] = []
    lines.append("# Targeted Cell-Creation Benchmark Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Separate diagnostic benchmark for low-N OpenMP behavior")
    lines.append("- Cases: `random_uniform N=10000` and `cubic_lattice N=8000`")
    lines.append("- Includes `NbrList::setup`, cell loop only, and total runtime")
    lines.append("- Compares `schedule(dynamic, 64)` against `schedule(static)`")
    lines.append("- Includes explicit `24`-thread measurements")
    lines.append("- Includes ordinary back-to-back calls and a true persistent-team second-of-two measurement")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("![stages](cell_creation_targeted_stage_breakdown.png)")
    lines.append("")
    lines.append("![totals](cell_creation_targeted_total_vs_threads.png)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    for point_set, nval in cases:
        s = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        total_dyn_32 = s[(s["benchmark"] == "vd_total_dynamic64") & (s["nthreads"] == 32)]["time_ms_mean"].iloc[0]
        total_dyn_48 = s[(s["benchmark"] == "vd_total_dynamic64") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        total_static_32 = s[(s["benchmark"] == "vd_total_static") & (s["nthreads"] == 32)]["time_ms_mean"].iloc[0]
        total_static_48 = s[(s["benchmark"] == "vd_total_static") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        total_dyn_second_48 = s[
            (s["benchmark"] == "vd_total_dynamic64_second_of_two") & (s["nthreads"] == 48)
        ]["time_ms_mean"].iloc[0]
        total_static_second_48 = s[
            (s["benchmark"] == "vd_total_static_second_of_two") & (s["nthreads"] == 48)
        ]["time_ms_mean"].iloc[0]
        total_dyn_persistent_second_48 = s[
            (s["benchmark"] == "vd_total_dynamic64_persistent_second_of_two") & (s["nthreads"] == 48)
        ]["time_ms_mean"].iloc[0]
        total_static_persistent_second_48 = s[
            (s["benchmark"] == "vd_total_static_persistent_second_of_two") & (s["nthreads"] == 48)
        ]["time_ms_mean"].iloc[0]
        setup_48 = s[(s["benchmark"] == "vd_setup_only") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        cells_dyn_48 = s[(s["benchmark"] == "vd_cells_dynamic64") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        cells_static_48 = s[(s["benchmark"] == "vd_cells_static") & (s["nthreads"] == 48)]["time_ms_mean"].iloc[0]
        best_total = s[s["benchmark"] == "vd_total_static"].sort_values("time_ms_mean").iloc[0]

        lines.append(
            f"- `{point_set} N={nval:,}`: `dynamic(64)` gets worse from 32 to 48 threads "
            f"({total_dyn_32:.3f} ms -> {total_dyn_48:.3f} ms), while `static` stays much lower "
            f"({total_static_32:.3f} ms -> {total_static_48:.3f} ms)."
        )
        lines.append(
            f"- `{point_set} N={nval:,}` at 48 threads: setup is {setup_48:.3f} ms, "
            f"cell loop is {cells_dyn_48:.3f} ms with `dynamic(64)` and {cells_static_48:.3f} ms with `static`."
        )
        lines.append(
            f"- `{point_set} N={nval:,}` at 48 threads: second-of-two total is "
            f"{total_dyn_second_48:.3f} ms with `dynamic(64)` and {total_static_second_48:.3f} ms with `static`."
        )
        lines.append(
            f"- `{point_set} N={nval:,}` at 48 threads with a persistent team: second-of-two total is "
            f"{total_dyn_persistent_second_48:.3f} ms with `dynamic(64)` and "
            f"{total_static_persistent_second_48:.3f} ms with `static`."
        )
        lines.append(
            f"- `{point_set} N={nval:,}`: best measured `total_static` point is "
            f"{int(best_total['nthreads'])} threads at {best_total['time_ms_mean']:.3f} ms."
        )
    lines.append("")

    lines.append("## Tables")
    for point_set, nval in cases:
        lines.append("")
        lines.append(f"### {point_set}, N={nval:,}")
        lines.append("")
        lines.append("| Threads | setup | cells dynamic(64) | cells static | total dynamic(64) | total static | 2nd-of-2 dynamic(64) | 2nd-of-2 static | persistent 2nd-of-2 dynamic(64) | persistent 2nd-of-2 static |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = df[(df["point_set"] == point_set) & (df["N"] == nval)]
        for nthreads in sorted(sub["nthreads"].unique()):
            lines.append(
                f"| {nthreads} | "
                f"{fmt_row(df, point_set, nval, 'vd_setup_only', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_cells_dynamic64', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_cells_static', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_total_dynamic64', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_total_static', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_total_dynamic64_second_of_two', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_total_static_second_of_two', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_total_dynamic64_persistent_second_of_two', nthreads)} | "
                f"{fmt_row(df, point_set, nval, 'vd_total_static_persistent_second_of_two', nthreads)} |"
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noomp-csv", type=Path, required=True)
    ap.add_argument("--threads-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = combined_data(args.noomp_csv, args.threads_dir)

    make_stage_plot(df, args.out_dir / "cell_creation_targeted_stage_breakdown.png")
    make_total_plot(df, args.out_dir / "cell_creation_targeted_total_vs_threads.png")
    make_markdown(df, args.out_dir / "CELL_CREATION_TARGETED_REPORT.md")


if __name__ == "__main__":
    main()
