"""Run deterministic RQ1-RQ5 experiments and write Markdown/CSV results."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

from src.evaluation import run_experiments


def write_results(rows: list[dict],
                  output_dir: str | Path = "results") -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "rq_metrics.csv"
    markdown_path = directory / "rq_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["rq", "variant", "metric", "value", "unit"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Deterministic RQ Evaluation Summary", "",
             "Synthetic seeded benchmarks for implementation validation. "
             "Results are not deployment measurements.", ""]
    for rq in ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5"):
        lines.extend([f"## {rq}", "", "| Variant | Metric | Value | Unit |",
                      "|---|---|---:|---|"])
        lines.extend(
            f"| {row['variant']} | {row['metric']} | {row['value']:.6g} | {row['unit']} |"
            for row in rows if row["rq"] == rq
        )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, csv_path


if __name__ == "__main__":
    start = perf_counter()
    summary, metrics = write_results(run_experiments())
    print(f"Wrote {summary} and {metrics} in {perf_counter() - start:.2f}s")
