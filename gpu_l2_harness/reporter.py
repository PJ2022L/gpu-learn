from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gpu_l2_harness.analyzer import analyze_results
from gpu_l2_harness.runner import ensure_run_dirs


def generate_report(run_id: str) -> Path:
    dirs = ensure_run_dirs(run_id)
    analysis_path = dirs["parsed"] / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    else:
        analysis = analyze_results(run_id)

    csv_path = dirs["parsed"] / "summary.csv"
    _write_summary_csv(csv_path, analysis.get("summary", []))
    _write_placeholder_figure_script(dirs["figures"])

    report_path = dirs["reports"] / "report.md"
    lines = [
        "# H800 L2 Persistent Set-Aside Report",
        "",
        f"Run ID: `{run_id}`",
        "",
        "## Method",
        "",
        "This harness uses CUDA L2 persisting cache set-aside plus a primed keeper buffer.",
        "Set-aside is not a hard physical L2 partition; final conclusions require NCU evidence.",
        "",
        "## Summary",
        "",
    ]
    summary = analysis.get("summary", [])
    if not summary:
        lines.append("No benchmark results are available yet. Dry-run completed the harness output skeleton.")
    else:
        lines.append(f"Analyzed {len(summary)} task results. See `results/{run_id}/parsed/summary.csv`.")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"Figure scripts are stored under `results/{run_id}/figures/`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "operator",
        "dtype",
        "status",
        "effective_l2_label",
        "latency_ms",
        "latency_ratio",
        "capacity_slope_ms_per_mb_removed",
        "classification",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_placeholder_figure_script(figures_dir: Path) -> None:
    fig_dir = figures_dir / "latency_vs_effective_l2"
    fig_dir.mkdir(parents=True, exist_ok=True)
    script = fig_dir / "plot.py"
    if not script.exists():
        script.write_text(
            "from pathlib import Path\n\n"
            "# Placeholder: real plotting reads ../../parsed/summary.csv once experiments exist.\n"
            "Path(__file__).with_name('README.txt').write_text('No plotted data yet.\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )

