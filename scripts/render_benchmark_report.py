#!/usr/bin/env python3
"""Render an optimizer benchmark JSON report as concise Markdown."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    lines = ["# Optimizer benchmark report", "", f"Suite: `{report.get('suite', 'unknown')}`", ""]
    gate = report.get("quality_gate")
    if gate:
        lines.append(
            f"Quality gate: **{'PASS' if gate.get('ok') else 'FAIL'}** (max regression {gate.get('max_regression', 0):.1%})"
        )
        lines.append("")
    comparisons = report.get("comparisons", {})
    rows = []
    for variant, metrics in comparisons.items():
        for metric, value in metrics.items():
            rows.append((variant, metric, value.get("speedup"), value.get("regression")))
    if rows:
        lines.extend(["| Variant | Metric | Speedup | Regression |", "|---|---|---:|---:|"])
        for variant, metric, speedup, regression in rows:
            lines.append(f"| `{variant}` | `{metric}` | {speedup:.3f}x | {regression:+.1%} |")
        lines.append("")
    variants = report.get("variants", {})
    provenance_rows = []
    for variant, payload in variants.items():
        compiler = payload.get("compiler")
        if compiler is None and payload.get("benchmark"):
            benchmarks = payload["benchmark"].get("benchmarks", [])
            compiler = benchmarks[0].get("compiler") if benchmarks else None
        if compiler is not None:
            provenance_rows.append(
                (variant, compiler.get("applied_passes", []), compiler.get("skipped_passes", []))
            )
    if provenance_rows:
        lines.extend(
            [
                "## Compiler provenance",
                "",
                "| Variant | Applied passes | Skipped passes |",
                "|---|---|---|",
            ]
        )
        for variant, applied, skipped in provenance_rows:
            applied_text = ", ".join(f"`{name}`" for name in applied) or "—"
            skipped_text = ", ".join(f"`{name}`" for name in skipped) or "—"
            lines.append(f"| `{variant}` | {applied_text} | {skipped_text} |")
        lines.append("")
    violations = gate.get("violations", []) if gate else []
    if violations:
        lines.extend(["## Violations", "", *[f"- `{item}`" for item in violations], ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "Speedup greater than 1.0x is an improvement. Results are machine- and "
            "workload-specific; correctness is evaluated before performance.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
