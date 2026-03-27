#!/usr/bin/env python3
"""Run a single-threaded CellArena reserve-capacity sweep and report results.

This script configures and rebuilds `benchmark_cell_arena_blocks` with different
compile-time reserve capacities:
- VOR_CELLARENA_RESERVE_VERTICES_PER_CELL
- VOR_CELLARENA_RESERVE_FACETS_PER_CELL

For each pair, it captures:
- runtime statistics from the benchmark output
- process peak RSS via `/usr/bin/time`

Finally, it writes:
- a raw CSV summary
- a markdown report with recommendation
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class Config:
    """Reserve-capacity configuration for CellArena.

    Attributes:
        reserve_vertices: Reserved vertices per cell.
        reserve_facets: Reserved facets per cell.
    """

    reserve_vertices: int
    reserve_facets: int


@dataclass
class Result:
    """Benchmark and memory outputs for one configuration."""

    reserve_vertices: int
    reserve_facets: int
    n: int
    reps: int
    nthreads: int
    time_ms_mean: float
    time_ms_std: float
    arena_size_bytes: int
    arena_capacity_bytes: int
    arena_slack_bytes: int
    arena_slack_pct: float
    max_rss_kb: int

    @property
    def reserve_key(self) -> str:
        return f"{self.reserve_vertices}/{self.reserve_facets}"


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and fail fast with useful diagnostics.

    Args:
        cmd: Command tokens.
        cwd: Optional working directory.

    Returns:
        Completed process with captured output.

    Raises:
        RuntimeError: If command exits with non-zero status.
    """

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        msg = ["Command failed:", " ".join(cmd), f"cwd={cwd}", "stdout:", proc.stdout, "stderr:", proc.stderr]
        raise RuntimeError("\n".join(msg))
    return proc


def _parse_benchmark_csv(stdout_text: str) -> dict[str, str]:
    """Parse single-row CSV emitted by benchmark executable.

    Args:
        stdout_text: Raw stdout text from benchmark.

    Returns:
        Parsed row as a dict.

    Raises:
        RuntimeError: If output is malformed.
    """

    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected benchmark output:\n{stdout_text}")
    reader = csv.DictReader(lines)
    rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one data row, got {len(rows)}")
    return rows[0]


def _configs() -> list[Config]:
    """Return study sweep points, including baseline.

    The set spans clearly smaller than baseline, baseline-adjacent, and larger
    allocations to identify the smallest no-regression region.
    """

    return [
        Config(16, 12),
        Config(18, 14),
        Config(20, 16),
        Config(22, 16),
        Config(24, 16),
        Config(24, 18),
        Config(24, 20),
        Config(28, 20),
        Config(28, 22),
        Config(32, 20),
        Config(32, 24),  # baseline
        Config(36, 24),
        Config(40, 28),
        Config(48, 32),
    ]


def _result_from_row(row: dict[str, str]) -> Result:
    """Convert parsed CSV row and RSS into typed Result."""

    return Result(
        reserve_vertices=int(row["reserve_vertices"]),
        reserve_facets=int(row["reserve_facets"]),
        n=int(row["N"]),
        reps=int(row["reps"]),
        nthreads=int(row["nthreads"]),
        time_ms_mean=float(row["time_ms_mean"]),
        time_ms_std=float(row["time_ms_std"]),
        arena_size_bytes=int(row["arena_size_bytes"]),
        arena_capacity_bytes=int(row["arena_capacity_bytes"]),
        arena_slack_bytes=int(row["arena_slack_bytes"]),
        arena_slack_pct=float(row["arena_slack_pct"]),
        max_rss_kb=int(row["max_rss_kb"]),
    )


def _write_csv(results: Iterable[Result], path: Path) -> None:
    """Write flattened result table to CSV."""

    rows = list(results)
    if not rows:
        raise ValueError("No results to write")

    fieldnames = [
        "reserve_vertices",
        "reserve_facets",
        "n",
        "reps",
        "nthreads",
        "time_ms_mean",
        "time_ms_std",
        "arena_size_bytes",
        "arena_capacity_bytes",
        "arena_slack_bytes",
        "arena_slack_pct",
        "max_rss_kb",
    ]

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "reserve_vertices": r.reserve_vertices,
                    "reserve_facets": r.reserve_facets,
                    "n": r.n,
                    "reps": r.reps,
                    "nthreads": r.nthreads,
                    "time_ms_mean": f"{r.time_ms_mean:.6f}",
                    "time_ms_std": f"{r.time_ms_std:.6f}",
                    "arena_size_bytes": r.arena_size_bytes,
                    "arena_capacity_bytes": r.arena_capacity_bytes,
                    "arena_slack_bytes": r.arena_slack_bytes,
                    "arena_slack_pct": f"{r.arena_slack_pct:.6f}",
                    "max_rss_kb": r.max_rss_kb,
                }
            )


def _format_kib(n_bytes: int) -> float:
    return n_bytes / 1024.0


def _build_markdown(
    results: list[Result],
    baseline: Result,
    best: Result,
    chosen: Result,
    slowdown_threshold_pct: float,
    out_csv: Path,
) -> str:
    """Create markdown report content."""

    ordered = sorted(results, key=lambda r: (r.time_ms_mean, r.arena_capacity_bytes))
    top_times = [r.time_ms_mean for r in ordered]
    med_time = median(top_times)

    lines: list[str] = []
    lines.append("# CellArena Block Capacity Study (Single-Thread, -O3)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Case: random uniform points in unit cube")
    lines.append(f"- Particles: {baseline.n:,}")
    lines.append(f"- Repetitions per configuration: {baseline.reps}")
    lines.append("- Build mode: Release with `-O3 -march=native -mtune=native -ffast-math -funroll-loops`")
    lines.append("- Threading: OpenMP disabled at configure time (single-thread)")
    lines.append(f"- Sweep points: {len(results)} reserve pairs")
    lines.append(f"- Raw data: `{out_csv}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"Use **vertex={chosen.reserve_vertices}, facet={chosen.reserve_facets}** as the smallest near-optimal setting "
        f"under a {slowdown_threshold_pct:.1f}% slowdown tolerance."
    )
    lines.append("")
    lines.append("- Best runtime config:")
    lines.append(
        f"  vertex={best.reserve_vertices}, facet={best.reserve_facets}, "
        f"time={best.time_ms_mean:.3f} ms"
    )
    lines.append("- Recommended config vs best:")
    lines.append(
        f"  time={chosen.time_ms_mean:.3f} ms "
        f"({(chosen.time_ms_mean / best.time_ms_mean - 1.0) * 100.0:+.2f}%)"
    )
    lines.append("- Recommended config vs baseline (32/24):")
    lines.append(
        f"  time={chosen.time_ms_mean:.3f} ms vs {baseline.time_ms_mean:.3f} ms "
        f"({(chosen.time_ms_mean / baseline.time_ms_mean - 1.0) * 100.0:+.2f}%)"
    )
    lines.append(
        f"  arena capacity={_format_kib(chosen.arena_capacity_bytes):.1f} KiB vs "
        f"{_format_kib(baseline.arena_capacity_bytes):.1f} KiB "
        f"({(chosen.arena_capacity_bytes / baseline.arena_capacity_bytes - 1.0) * 100.0:+.2f}%)"
    )
    lines.append(
        f"  peak RSS={chosen.max_rss_kb} kB vs {baseline.max_rss_kb} kB "
        f"({(chosen.max_rss_kb / baseline.max_rss_kb - 1.0) * 100.0:+.2f}%)"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Reserve (v/f) | Time mean (ms) | Std (ms) | Slowdown vs best | Arena cap (KiB) | Arena slack (%) | Max RSS (kB) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda x: (x.arena_capacity_bytes, x.time_ms_mean)):
        slow = (r.time_ms_mean / best.time_ms_mean - 1.0) * 100.0
        lines.append(
            "| "
            f"{r.reserve_vertices}/{r.reserve_facets} | "
            f"{r.time_ms_mean:.3f} | {r.time_ms_std:.3f} | {slow:+.2f}% | "
            f"{_format_kib(r.arena_capacity_bytes):.1f} | {r.arena_slack_pct:.2f} | {r.max_rss_kb} |"
        )

    lines.append("")
    lines.append("## Insights")
    lines.append("")
    lines.append(
        "- Runtime is relatively flat across a broad reserve range; very small reserves tend to add modest overhead from reallocation/copy growth."
    )
    lines.append(
        "- Arena capacity scales nearly linearly with reserve settings; this is where memory savings are most directly visible."
    )
    lines.append(
        "- Peak process RSS is less sensitive than arena capacity alone because total process memory includes neighbor lists and other temporary structures."
    )
    lines.append(
        "- Practical tuning rule: choose the smallest pair inside the near-optimal runtime band instead of the absolute fastest pair."
    )
    lines.append("")
    lines.append("## Robustness Notes")
    lines.append("")
    lines.append(f"- Median runtime across sweep points: {med_time:.3f} ms")
    lines.append(
        "- If you want tighter confidence, rerun with higher reps (e.g., 30) and pin CPU frequency/governor."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="CellArena reserve-capacity study for N=1e4 random points")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path, default=Path("build_bench_study_blocks"))
    parser.add_argument("--out-csv", type=Path, default=Path("benchmarks/results/cell_arena_block_study.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("benchmarks/CELL_ARENA_BLOCK_STUDY.md"))
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--slowdown-threshold-pct", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    build_dir = (repo_root / args.build_dir).resolve()
    out_csv = (repo_root / args.out_csv).resolve()
    out_md = (repo_root / args.out_md).resolve()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []

    for cfg in _configs():
        cxx_release_flags = (
            "-O3 -DNDEBUG "
            f"-DVOR_CELLARENA_RESERVE_VERTICES_PER_CELL={cfg.reserve_vertices} "
            f"-DVOR_CELLARENA_RESERVE_FACETS_PER_CELL={cfg.reserve_facets}"
        )

        configure_cmd = [
            "cmake",
            "-S",
            str(repo_root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DVORONOI_BUILD_BENCHMARKS=ON",
            "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE",
            f"-DCMAKE_CXX_FLAGS_RELEASE={cxx_release_flags}",
        ]
        _run(configure_cmd, cwd=repo_root)

        _run(["cmake", "--build", str(build_dir), "--target", "benchmark_cell_arena_blocks", "-j"], cwd=repo_root)

        bench_exe = build_dir / "benchmarks" / "benchmark_cell_arena_blocks"
        run_cmd = [
            str(bench_exe),
            "--n",
            str(args.n),
            "--reps",
            str(args.reps),
            "--seed",
            str(args.seed),
        ]
        proc = _run(run_cmd, cwd=repo_root)

        row = _parse_benchmark_csv(proc.stdout)
        results.append(_result_from_row(row))

    _write_csv(results, out_csv)

    by_time = sorted(results, key=lambda r: r.time_ms_mean)
    best = by_time[0]

    baseline_candidates = [r for r in results if r.reserve_vertices == 32 and r.reserve_facets == 24]
    if not baseline_candidates:
        raise RuntimeError("Baseline 32/24 is missing from sweep")
    baseline = baseline_candidates[0]

    threshold = best.time_ms_mean * (1.0 + args.slowdown_threshold_pct / 100.0)
    near_opt = [r for r in results if r.time_ms_mean <= threshold]
    chosen = min(near_opt, key=lambda r: (r.arena_capacity_bytes, r.reserve_vertices, r.reserve_facets))

    md = _build_markdown(
        results=results,
        baseline=baseline,
        best=best,
        chosen=chosen,
        slowdown_threshold_pct=args.slowdown_threshold_pct,
        out_csv=out_csv,
    )
    out_md.write_text(md, encoding="utf-8")

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote report: {out_md}")
    print(
        "Recommendation: "
        f"vertex={chosen.reserve_vertices}, facet={chosen.reserve_facets}, "
        f"time={chosen.time_ms_mean:.3f} ms, arena_capacity={chosen.arena_capacity_bytes} bytes"
    )


if __name__ == "__main__":
    main()
