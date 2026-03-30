#!/usr/bin/env python3
"""Generate a markdown report for the updateFast timestep study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    int_cols = [
        "N",
        "steps",
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
    csvs = sorted(results_dir.glob("benchmark_update_fast_dt*.csv"))
    if not csvs:
        raise RuntimeError("No benchmark_update_fast_dt*.csv files found.")
    return pd.concat([read_csv(p) for p in csvs], ignore_index=True)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("dt", as_index=False)
    summary = g.agg(
        N=("N", "first"),
        steps=("steps", "first"),
        velocity_sigma=("velocity_sigma", "first"),
        initial_non_convex_at_build=("initial_non_convex_at_build", "first"),
        step1_non_convex_before=("non_convex_before", "first"),
        mean_update_ms=("update_ms", "mean"),
        max_update_ms=("update_ms", "max"),
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
    return summary.sort_values("dt").reset_index(drop=True)


def make_workload_plot(summary: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True)
    x = summary["dt"]

    axes[0, 0].plot(x, summary["step1_non_convex_before"], marker="o", linewidth=2, color="#d62728")
    axes[0, 0].set_ylabel("Cells")
    axes[0, 0].set_title("Non-convex cells before first update")

    axes[0, 1].plot(x, summary["mean_update_ms"], marker="o", linewidth=2, color="#1f77b4")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].set_title("Mean updateFast runtime per step")

    axes[1, 0].plot(x, summary["mean_local_rebuild_cells"], marker="o", linewidth=2, color="#2ca02c", label="local")
    axes[1, 0].plot(x, summary["mean_full_rebuild_cells"], marker="o", linewidth=2, color="#ff7f0e", label="fallback full")
    axes[1, 0].set_ylabel("Cells")
    axes[1, 0].set_title("Mean rebuilt cells per step")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, summary["mean_repair_iterations"], marker="o", linewidth=2, color="#9467bd")
    axes[1, 1].set_ylabel("Iterations")
    axes[1, 1].set_title("Mean wave iterations per step")

    for ax in axes.flat:
      ax.set_xscale("log")
      ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
      ax.set_xlabel("dt")

    fig.suptitle("updateFast workload vs timestep")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def make_quality_plot(summary: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), sharex=True)
    x = summary["dt"]

    axes[0].plot(x, summary["final_max_rel_volume_diff"], marker="o", linewidth=2, color="#d62728")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Final max relative cell-volume diff")
    axes[0].set_xlabel("dt")
    axes[0].set_ylabel("Relative diff")
    axes[0].grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

    axes[1].plot(x, summary["final_signature_mismatch_cells"], marker="o", linewidth=2, color="#1f77b4")
    axes[1].plot(x, summary["final_non_reciprocal_pairs"], marker="o", linewidth=2, color="#2ca02c")
    axes[1].set_xscale("log")
    axes[1].set_title("Final topology errors")
    axes[1].set_xlabel("dt")
    axes[1].set_ylabel("Count")
    axes[1].legend(["signature mismatch cells", "non-reciprocal pairs"], fontsize=8)
    axes[1].grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

    total_vol_diff = (summary["final_total_volume_update"] - summary["final_total_volume_rebuild"]).abs()
    axes[2].plot(x, total_vol_diff, marker="o", linewidth=2, color="#9467bd")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Final total-volume difference")
    axes[2].set_xlabel("dt")
    axes[2].set_ylabel("|V_update - V_rebuild|")
    axes[2].grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

    fig.suptitle("updateFast final tessellation quality vs timestep")
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def fmt_sci(v: float) -> str:
    return f"{v:.3e}"


def make_markdown(summary: pd.DataFrame, out_md: Path) -> None:
    best_quality = summary.sort_values("final_max_rel_volume_diff").iloc[0]
    worst_quality = summary.sort_values("final_max_rel_volume_diff").iloc[-1]
    fastest = summary.sort_values("mean_update_ms").iloc[0]
    heaviest = summary.sort_values("mean_update_ms").iloc[-1]

    lines: list[str] = []
    lines.append("# updateFast Timestep Study")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- System: `N=10,000` random particles in a unit periodic box")
    lines.append("- Velocities: independent normal components with `sigma=1.0`")
    lines.append("- Timestep study over `dt = 1e-4 .. 1e-1`")
    lines.append("- Update path: `CellComplex::updateFast()`")
    lines.append("- Final state is compared against a clean static rebuild")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("![workload](update_fast_workload_vs_dt.png)")
    lines.append("")
    lines.append("![quality](update_fast_final_quality_vs_dt.png)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Best final volume agreement was at `dt={best_quality['dt']:.1e}` with max relative per-cell "
        f"volume difference `{fmt_sci(best_quality['final_max_rel_volume_diff'])}`."
    )
    lines.append(
        f"- Worst final volume agreement was at `dt={worst_quality['dt']:.1e}` with max relative per-cell "
        f"volume difference `{fmt_sci(worst_quality['final_max_rel_volume_diff'])}`."
    )
    lines.append(
        f"- Fastest mean per-step runtime was at `dt={fastest['dt']:.1e}` with "
        f"`{fmt(fastest['mean_update_ms'])} ms`."
    )
    lines.append(
        f"- Slowest mean per-step runtime was at `dt={heaviest['dt']:.1e}` with "
        f"`{fmt(heaviest['mean_update_ms'])} ms`."
    )
    lines.append(
        "- `final_non_reciprocal_pairs` is reported for the updated tessellation alone. "
        "When `final_signature_mismatch_cells` is zero, that count is shared by the clean rebuild "
        "and is not evidence of an `updateFast` mismatch."
    )
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append("| dt | step-1 non-convex | mean update ms | mean local rebuild | mean full rebuild | mean wave iterations | final mismatch cells | final non-reciprocal pairs | final max rel vol diff | final total vol diff |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        total_vol_diff = abs(row["final_total_volume_update"] - row["final_total_volume_rebuild"])
        lines.append(
            f"| {row['dt']:.1e} | "
            f"{int(row['step1_non_convex_before'])} | "
            f"{fmt(row['mean_update_ms'])} | "
            f"{fmt(row['mean_local_rebuild_cells'])} | "
            f"{fmt(row['mean_full_rebuild_cells'])} | "
            f"{fmt(row['mean_repair_iterations'])} | "
            f"{int(row['final_signature_mismatch_cells'])} | "
            f"{int(row['final_non_reciprocal_pairs'])} | "
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

    make_workload_plot(summary, args.results_dir / "update_fast_workload_vs_dt.png")
    make_quality_plot(summary, args.results_dir / "update_fast_final_quality_vs_dt.png")
    make_markdown(summary, args.results_dir / "UPDATE_FAST_STUDY_REPORT.md")


if __name__ == "__main__":
    main()
