#!/usr/bin/env python3
"""Generate a markdown report for large-N update thread-scaling benchmarks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["fullBuild", "updateNbrListSweep", "updateNbrListSweepAsync"]
METHOD_LABELS = {
    "fullBuild": "fullBuild",
    "updateNbrListSweep": "updateNbrListSweep",
    "updateNbrListSweepAsync": "updateNbrListSweepAsync",
}
METHOD_COLORS = {
    "fullBuild": "#1f77b4",
    "updateNbrListSweep": "#2ca02c",
    "updateNbrListSweepAsync": "#ff7f0e",
}


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    int_cols = [
        "N",
        "steps",
        "worker_count",
        "initial_non_convex_at_build",
        "step",
        "non_convex_before",
        "rebuild_candidates",
        "local_rebuild_cells",
        "full_rebuild_cells",
        "repair_iterations",
        "repair_proposals_total",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)
    float_cols = ["dt", "velocity_sigma", "update_ms", "non_convex_fraction_before"]
    for col in float_cols:
        df[col] = df[col].astype(float)
    return df


def combined_data(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("benchmark_update_thread_scaling_*.csv"))
    if not csvs:
        raise RuntimeError("No benchmark_update_thread_scaling_*.csv files found.")
    return pd.concat([read_csv(p) for p in csvs], ignore_index=True)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["N", "worker_count", "method", "dt"], as_index=False)
    summary = g.agg(
        steps=("steps", "first"),
        velocity_sigma=("velocity_sigma", "first"),
        initial_non_convex_at_build=("initial_non_convex_at_build", "first"),
        step1_non_convex_before=("non_convex_before", "first"),
        step1_non_convex_fraction_before=("non_convex_fraction_before", "first"),
        mean_update_ms=("update_ms", "mean"),
        mean_non_convex_before=("non_convex_before", "mean"),
        mean_non_convex_fraction_before=("non_convex_fraction_before", "mean"),
        mean_rebuild_candidates=("rebuild_candidates", "mean"),
        mean_local_rebuild_cells=("local_rebuild_cells", "mean"),
        mean_full_rebuild_cells=("full_rebuild_cells", "mean"),
        mean_repair_iterations=("repair_iterations", "mean"),
        mean_repair_proposals_total=("repair_proposals_total", "mean"),
    )
    return summary.sort_values(["N", "worker_count", "method", "dt"]).reset_index(drop=True)


def reference_fraction(summary: pd.DataFrame, n: int, threads: int) -> pd.DataFrame:
    ref = (
        summary[(summary["N"] == n) & (summary["worker_count"] == threads)]
        .groupby("dt", as_index=False)["mean_non_convex_fraction_before"]
        .mean()
        .sort_values("dt")
    )
    return ref


def estimate_crossover(summary: pd.DataFrame, n: int, threads: int, method_a: str, method_b: str) -> dict:
    ga = summary[(summary["N"] == n) & (summary["worker_count"] == threads) & (summary["method"] == method_a)]
    gb = summary[(summary["N"] == n) & (summary["worker_count"] == threads) & (summary["method"] == method_b)]
    if ga.empty or gb.empty:
        return {"status": "missing"}

    merged = ga[["dt", "mean_update_ms"]].merge(
        gb[["dt", "mean_update_ms"]], on="dt", suffixes=("_a", "_b")
    ).sort_values("dt")
    frac = reference_fraction(summary, n, threads)
    merged = merged.merge(frac, on="dt", how="left")
    if merged.empty:
        return {"status": "missing"}

    dts = merged["dt"].to_numpy()
    diffs = (merged["mean_update_ms_a"] - merged["mean_update_ms_b"]).to_numpy()
    fracs = merged["mean_non_convex_fraction_before"].to_numpy()

    for i in range(len(dts)):
        if diffs[i] == 0.0:
            return {
                "status": "cross",
                "dt": float(dts[i]),
                "non_convex_fraction": float(fracs[i]),
                "faster_low_dt": method_b if i == 0 else (method_a if diffs[i - 1] < 0 else method_b),
                "faster_high_dt": method_a if i + 1 == len(dts) else (method_a if diffs[min(i + 1, len(dts) - 1)] < 0 else method_b),
            }

    for i in range(len(dts) - 1):
        if diffs[i] == 0.0:
            continue
        if diffs[i] * diffs[i + 1] < 0.0:
            alpha = abs(diffs[i]) / (abs(diffs[i]) + abs(diffs[i + 1]))
            log_dt = math.log10(dts[i]) + alpha * (math.log10(dts[i + 1]) - math.log10(dts[i]))
            frac_interp = fracs[i] + alpha * (fracs[i + 1] - fracs[i])
            return {
                "status": "cross",
                "dt": 10.0 ** log_dt,
                "non_convex_fraction": float(frac_interp),
                "faster_low_dt": method_a if diffs[i] < 0 else method_b,
                "faster_high_dt": method_a if diffs[i + 1] < 0 else method_b,
            }

    always = method_a if np.all(diffs < 0.0) else method_b
    return {"status": "none", "always_faster": always}


def make_fastest_method_heatmap(summary: pd.DataFrame, n: int, out_file: Path) -> None:
    sub = summary[summary["N"] == n]
    threads = sorted(sub["worker_count"].unique())
    dts = sorted(sub["dt"].unique())
    matrix = np.full((len(threads), len(dts)), -1, dtype=int)
    method_to_idx = {m: i for i, m in enumerate(METHOD_ORDER)}
    for r, threads_i in enumerate(threads):
        for c, dt in enumerate(dts):
            rows = sub[(sub["worker_count"] == threads_i) & (sub["dt"] == dt)]
            fastest = rows.sort_values("mean_update_ms").iloc[0]["method"]
            matrix[r, c] = method_to_idx[fastest]

    cmap = matplotlib.colors.ListedColormap([METHOD_COLORS[m] for m in METHOD_ORDER])
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=-0.5, vmax=len(METHOD_ORDER) - 0.5)
    ax.set_xticks(range(len(dts)))
    ax.set_xticklabels([f"{dt:.0e}" if dt < 1.0 else f"{dt:g}" for dt in dts], rotation=45, ha="right")
    ax.set_yticks(range(len(threads)))
    ax.set_yticklabels([str(t) for t in threads])
    ax.set_xlabel("dt")
    ax.set_ylabel("Threads")
    ax.set_title(f"Fastest Method by dt and Thread Count (N={n:,})")
    legend_handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=6, label=METHOD_LABELS[m]) for m in METHOD_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def make_speedup_heatmaps(summary: pd.DataFrame, n: int, out_file: Path) -> None:
    sub = summary[summary["N"] == n]
    threads = sorted(sub["worker_count"].unique())
    dts = sorted(sub["dt"].unique())
    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(15.2, 3.8), sharey=True)
    if len(METHOD_ORDER) == 1:
        axes = [axes]

    for ax, method in zip(axes, METHOD_ORDER):
        method_rows = sub[sub["method"] == method]
        pivot = method_rows.pivot(index="worker_count", columns="dt", values="mean_update_ms").reindex(
            index=threads, columns=dts
        )
        base = pivot.loc[1]
        speedup = base / pivot
        im = ax.imshow(speedup.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xticks(range(len(dts)))
        ax.set_xticklabels([f"{dt:.0e}" if dt < 1.0 else f"{dt:g}" for dt in dts], rotation=45, ha="right")
        ax.set_yticks(range(len(threads)))
        ax.set_yticklabels([str(t) for t in threads])
        ax.set_xlabel("dt")
        if ax is axes[0]:
            ax.set_ylabel("Threads")
        cbar = fig.colorbar(im, ax=ax, shrink=0.84)
        cbar.set_label("Speedup vs 1 thread")

    fig.suptitle(f"Parallel Speedup Heatmaps (N={n:,})")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def make_markdown(summary: pd.DataFrame, out_md: Path) -> None:
    ns = sorted(summary["N"].unique())
    threads = sorted(summary["worker_count"].unique())
    dts = sorted(summary["dt"].unique())
    lines: list[str] = []
    lines.append("# Update Thread-Scaling Study")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- System sizes: " + ", ".join(f"`N={n:,}`" for n in ns))
    lines.append("- Methods: `fullBuild`, `updateNbrListSweep` (no switch), `updateNbrListSweepAsync` (no switch)")
    lines.append("- Threads: " + ", ".join(f"`{t}`" for t in threads))
    lines.append("- Timestep range: " + ", ".join(f"`{dt:.0e}`" if dt < 1.0 else f"`{dt:g}`" for dt in dts))
    lines.append("- Velocity standard deviation uses `sigma = N^(-1/3)` in a unit periodic box")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for n in ns:
        lines.append(f"### N={n:,}")
        lines.append("")
        lines.append(f"![fastest](update_thread_scaling_fastest_N{n}.png)")
        lines.append("")
        lines.append(f"![speedup](update_thread_scaling_speedup_N{n}.png)")
        lines.append("")

    lines.append("## Crossover Estimates")
    lines.append("")
    for n in ns:
        lines.append(f"### N={n:,}")
        lines.append("")
        lines.append("| threads | pair | crossover dt | non-convex fraction | interpretation |")
        lines.append("|---:|:--|---:|---:|:--|")
        for threads_i in threads:
            for a, b in [
                ("updateNbrListSweep", "fullBuild"),
                ("updateNbrListSweepAsync", "fullBuild"),
                ("updateNbrListSweep", "updateNbrListSweepAsync"),
            ]:
                est = estimate_crossover(summary, n, threads_i, a, b)
                if est["status"] == "cross":
                    lines.append(
                        f"| {threads_i} | `{METHOD_LABELS[a]}` vs `{METHOD_LABELS[b]}` | "
                        f"{est['dt']:.3e} | {est['non_convex_fraction']:.3f} | "
                        f"`{METHOD_LABELS[est['faster_low_dt']]}` faster at smaller dt, "
                        f"`{METHOD_LABELS[est['faster_high_dt']]}` faster at larger dt |"
                    )
                elif est["status"] == "none":
                    lines.append(
                        f"| {threads_i} | `{METHOD_LABELS[a]}` vs `{METHOD_LABELS[b]}` | - | - | "
                        f"`{METHOD_LABELS[est['always_faster']]}` faster over sampled range |"
                    )
                else:
                    lines.append(
                        f"| {threads_i} | `{METHOD_LABELS[a]}` vs `{METHOD_LABELS[b]}` | - | - | missing |"
                    )
        lines.append("")

    lines.append("## Sampled Runtimes")
    lines.append("")
    lines.append("| N | threads | method | dt | mean update ms | mean non-convex fraction | mean repair iterations |")
    lines.append("|---:|---:|:--|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['N'])} | {int(row['worker_count'])} | {METHOD_LABELS[row['method']]} | "
            f"{row['dt']:.3e} | {fmt(row['mean_update_ms'])} | "
            f"{fmt(row['mean_non_convex_fraction_before'])} | {fmt(row['mean_repair_iterations'])} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    df = combined_data(args.results_dir)
    summary = make_summary(df)
    for n in sorted(summary["N"].unique()):
        make_fastest_method_heatmap(summary, n, args.results_dir / f"update_thread_scaling_fastest_N{n}.png")
        make_speedup_heatmaps(summary, n, args.results_dir / f"update_thread_scaling_speedup_N{n}.png")
    make_markdown(summary, args.results_dir / "UPDATE_THREAD_SCALING_REPORT.md")


if __name__ == "__main__":
    main()
