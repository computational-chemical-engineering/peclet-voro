#!/usr/bin/env python3
"""Generate a markdown report comparing incremental update strategies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "updateFast": "updateFast",
    "updateNbrListLocal": "updateNbrListLocal",
    "updateNbrListSweep": "updateNbrListSweep",
    "updateNbrListSweepAsync": "updateNbrListSweepAsync",
}

METHOD_COLORS = {
    "updateFast": "#1f77b4",
    "updateNbrListLocal": "#d62728",
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
        "non_convex_after",
        "rebuild_candidates",
        "local_rebuild_cells",
        "empty_after_local_rebuild",
        "full_rebuild_cells",
        "repair_iterations",
        "repair_proposals_total",
        "repair_target_groups_total",
        "repair_cells_changed_total",
        "repair_direct_attempts",
        "repair_direct_successes",
        "repair_indirect_candidates",
        "repair_batch_calls",
        "repair_batch_changes",
        "topology_changed_cells",
        "no_nbr_cells_after",
        "final_signature_mismatch_cells",
        "final_non_reciprocal_pairs",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)
    float_cols = [
        "dt",
        "velocity_sigma",
        "update_ms",
        "max_convex_violation_after",
        "final_max_abs_volume_diff",
        "final_max_rel_volume_diff",
        "final_total_volume_update",
        "final_total_volume_rebuild",
    ]
    for col in float_cols:
        df[col] = df[col].astype(float)
    return df


def combined_data(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("benchmark_update_strategy_dt*.csv"))
    if not csvs:
        raise RuntimeError("No benchmark_update_strategy_dt*.csv files found.")
    return pd.concat([read_csv(p) for p in csvs], ignore_index=True)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "dt"], as_index=False)
    summary = g.agg(
        N=("N", "first"),
        steps=("steps", "first"),
        velocity_sigma=("velocity_sigma", "first"),
        worker_count=("worker_count", "first"),
        initial_non_convex_at_build=("initial_non_convex_at_build", "first"),
        step1_non_convex_before=("non_convex_before", "first"),
        mean_update_ms=("update_ms", "mean"),
        mean_non_convex_before=("non_convex_before", "mean"),
        mean_non_convex_after=("non_convex_after", "mean"),
        mean_rebuild_candidates=("rebuild_candidates", "mean"),
        mean_local_rebuild_cells=("local_rebuild_cells", "mean"),
        mean_empty_after_local_rebuild=("empty_after_local_rebuild", "mean"),
        mean_full_rebuild_cells=("full_rebuild_cells", "mean"),
        mean_repair_iterations=("repair_iterations", "mean"),
        mean_repair_proposals_total=("repair_proposals_total", "mean"),
        mean_topology_changed_cells=("topology_changed_cells", "mean"),
        mean_no_nbr_cells_after=("no_nbr_cells_after", "mean"),
        max_convex_violation_after=("max_convex_violation_after", "max"),
        final_signature_mismatch_cells=("final_signature_mismatch_cells", "first"),
        final_non_reciprocal_pairs=("final_non_reciprocal_pairs", "first"),
        final_max_abs_volume_diff=("final_max_abs_volume_diff", "first"),
        final_max_rel_volume_diff=("final_max_rel_volume_diff", "first"),
        final_total_volume_update=("final_total_volume_update", "first"),
        final_total_volume_rebuild=("final_total_volume_rebuild", "first"),
    )
    return summary.sort_values(["method", "dt"]).reset_index(drop=True)


def plot_by_method(ax: plt.Axes, summary: pd.DataFrame, ycol: str, title: str, ylabel: str,
                   yscale: str | None = None) -> None:
    for method, group in summary.groupby("method", sort=False):
        ax.plot(
            group["dt"],
            group[ycol],
            marker="o",
            linewidth=2,
            color=METHOD_COLORS.get(method, None),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xscale("log")
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel("dt")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)


def make_workload_plot(summary: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True)
    plot_by_method(
        axes[0, 0], summary, "step1_non_convex_before", "Non-convex cells before first update", "Cells"
    )
    plot_by_method(axes[0, 1], summary, "mean_update_ms", "Mean runtime per step", "ms")
    plot_by_method(axes[1, 0], summary, "mean_local_rebuild_cells", "Mean local rebuild cells", "Cells")
    plot_by_method(axes[1, 1], summary, "mean_full_rebuild_cells", "Mean full rebuild cells", "Cells")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Incremental update workload vs timestep")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def make_quality_plot(summary: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), sharex=True)
    plot_by_method(
        axes[0], summary, "final_max_rel_volume_diff", "Final max relative cell-volume diff", "Relative diff", "log"
    )
    plot_by_method(
        axes[1], summary, "final_signature_mismatch_cells", "Final signature mismatch cells", "Cells"
    )
    total_vol_diff = summary.copy()
    total_vol_diff["final_total_vol_diff"] = (
        total_vol_diff["final_total_volume_update"] - total_vol_diff["final_total_volume_rebuild"]
    ).abs()
    plot_by_method(
        axes[2], total_vol_diff, "final_total_vol_diff", "Final total-volume difference", "|V_update - V_rebuild|", "log"
    )
    axes[0].legend(fontsize=8)
    fig.suptitle("Incremental update quality vs timestep")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def fmt_sci(v: float) -> str:
    return f"{v:.3e}"


def make_markdown(summary: pd.DataFrame, out_md: Path) -> None:
    lines: list[str] = []
    dt_values = sorted(summary["dt"].unique())
    dt_range = f"{dt_values[0]:.0e} .. {dt_values[-1]:.0e}"
    worker_values = sorted(summary["worker_count"].unique())
    lines.append("# Incremental Update Strategy Study")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- System: `N=10,000` random particles in a unit periodic box")
    lines.append("- Velocities: independent normal components with `sigma=1.0`")
    lines.append(f"- Timestep study over `dt = {dt_range}`")
    if len(worker_values) == 1:
        if worker_values[0] == 0:
            lines.append("- Execution mode: serial (`worker_count=0`)")
        else:
            lines.append(f"- Execution mode: persistent worker team with `worker_count={worker_values[0]}`")
    lines.append("- Methods: `updateFast`, `updateNbrListLocal`, `updateNbrListSweep`, and `updateNbrListSweepAsync`")
    lines.append("- Final state of each method is compared against a clean static rebuild")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("![workload](update_strategy_workload_vs_dt.png)")
    lines.append("")
    lines.append("![quality](update_strategy_quality_vs_dt.png)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    for method, group in summary.groupby("method", sort=False):
        fastest = group.sort_values("mean_update_ms").iloc[0]
        slowest = group.sort_values("mean_update_ms").iloc[-1]
        best_quality = group.sort_values("final_max_rel_volume_diff").iloc[0]
        worst_quality = group.sort_values("final_max_rel_volume_diff").iloc[-1]
        label = METHOD_LABELS.get(method, method)
        lines.append(
            f"- `{label}`: fastest at `dt={fastest['dt']:.1e}` with `{fmt(fastest['mean_update_ms'])} ms`; "
            f"slowest at `dt={slowest['dt']:.1e}` with `{fmt(slowest['mean_update_ms'])} ms`."
        )
        lines.append(
            f"- `{label}`: best final quality at `dt={best_quality['dt']:.1e}` with max relative volume "
            f"difference `{fmt_sci(best_quality['final_max_rel_volume_diff'])}`; worst at "
            f"`dt={worst_quality['dt']:.1e}` with `{fmt_sci(worst_quality['final_max_rel_volume_diff'])}`."
        )
    lines.append(
        "- `final_non_reciprocal_pairs` is reported for the updated tessellation alone. "
        "When `final_signature_mismatch_cells` is zero, that count is shared by the clean rebuild "
        "and is not evidence of a strategy mismatch."
    )
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append(
        "| method | dt | step-1 non-convex | mean update ms | mean local rebuild | "
        "mean full rebuild | mean iterations | final mismatch cells | final max rel vol diff | final total vol diff |"
    )
    lines.append("|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        total_vol_diff = abs(row["final_total_volume_update"] - row["final_total_volume_rebuild"])
        lines.append(
            f"| {METHOD_LABELS.get(row['method'], row['method'])} | "
            f"{row['dt']:.1e} | "
            f"{int(row['step1_non_convex_before'])} | "
            f"{fmt(row['mean_update_ms'])} | "
            f"{fmt(row['mean_local_rebuild_cells'])} | "
            f"{fmt(row['mean_full_rebuild_cells'])} | "
            f"{fmt(row['mean_repair_iterations'])} | "
            f"{int(row['final_signature_mismatch_cells'])} | "
            f"{fmt_sci(row['final_max_rel_volume_diff'])} | "
            f"{fmt_sci(total_vol_diff)} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    df = combined_data(args.results_dir)
    summary = make_summary(df)

    make_workload_plot(summary, args.results_dir / "update_strategy_workload_vs_dt.png")
    make_quality_plot(summary, args.results_dir / "update_strategy_quality_vs_dt.png")
    make_markdown(summary, args.results_dir / "UPDATE_STRATEGY_STUDY_REPORT.md")


if __name__ == "__main__":
    main()
