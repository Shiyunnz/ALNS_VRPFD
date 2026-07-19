#!/usr/bin/env python3
"""Export replay Table-6 CSV files to LaTeX tables for manuscript integration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert replay Table-6 CSVs (ND/UD/NDC) into LaTeX table snippets."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("sensitivity/results_new/scenario_replay"),
        help="Directory containing table6_instance25_r30_r40_r50_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated .tex files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="table6_replay_instance25_r30_r40_r50",
        help="Filename prefix for generated TeX files",
    )
    return parser.parse_args()


def _instance_parts(instance_name: str) -> tuple[str, str]:
    # Example: R_30_25_1 -> (30x30, 1)
    parts = instance_name.split("_")
    if len(parts) < 4:
        return instance_name, "-"
    region = parts[1]
    idx = parts[3]
    return f"{region}x{region}", idx


def _fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _format_row(
    row: pd.Series,
    region_cell: str,
    no_cell: str,
) -> str:
    region, idx = _instance_parts(str(row["Instance"]))
    gamma = int(row["gamma"])

    avg_cost = _fmt_float(float(row["AvgCost"]), 2)
    std_cost = _fmt_float(float(row["StdCost"]), 2)
    max_cost = _fmt_float(float(row["MaxCost"]), 2)
    min_cost = _fmt_float(float(row["MinCost"]), 2)
    avg_unserved = _fmt_float(float(row["AvgUnserved"]), 3)
    p_zero = _fmt_float(float(row["P(Unserved=0)%"]), 1)
    avg_no_takeoff = _fmt_float(float(row["AvgNoTakeoff"]), 3)
    avg_abort = _fmt_float(float(row["AvgAbortReturn"]), 3)

    return (
        f"{region_cell} & {no_cell} & {gamma} & {avg_cost} & {std_cost} & {max_cost} & {min_cost} "
        f"& {avg_unserved} & {p_zero} & {avg_no_takeoff} & {avg_abort} \\\\"
    )


def _build_table_tex(df: pd.DataFrame, distribution: str) -> str:
    dist_label = {"ND": "Normal", "UD": "Uniform", "NDC": "Normal-Clustered"}.get(
        distribution, distribution
    )
    caption = (
        f"Scenario replay statistics under {dist_label} demand distribution "
        f"(Instance25, regions 30/40/50)."
    )
    label = f"tab:replay_table6_{distribution.lower()}"

    rows: list[str] = []
    grouped = df.copy()
    grouped["region_sort"] = grouped["Instance"].str.extract(r"R_(\d+)_")[0].astype(int)
    grouped["idx_sort"] = grouped["Instance"].str.extract(r"R_\d+_\d+_(\d+)")[0].astype(int)
    grouped = grouped.sort_values(["region_sort", "idx_sort", "gamma"])

    for region in sorted(grouped["region_sort"].unique()):
        sub_region = grouped[grouped["region_sort"] == region]
        region_rowspan = len(sub_region)
        region_tex = f"${region}\\times{region}$"
        region_written = False
        for idx in sorted(sub_region["idx_sort"].unique()):
            sub_instance = sub_region[sub_region["idx_sort"] == idx].sort_values("gamma")
            no_rowspan = len(sub_instance)
            no_written = False
            for _, row in sub_instance.iterrows():
                region_cell = ""
                if not region_written:
                    region_cell = f"\\multirow{{{region_rowspan}}}{{*}}{{{region_tex}}}"
                    region_written = True
                no_cell = ""
                if not no_written:
                    no_cell = f"\\multirow{{{no_rowspan}}}{{*}}{{{idx}}}"
                    no_written = True
                rows.append(_format_row(row, region_cell, no_cell))
            if idx != sorted(sub_region["idx_sort"].unique())[-1]:
                rows.append("\\addlinespace[1pt]")
        rows.append("\\midrule")

    if rows and rows[-1] == "\\midrule":
        rows.pop()

    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{cc ccccccccc}\n"
        "\\toprule\n"
        "Region & No. & $\\Gamma$ & AvgCost & StdCost & MaxCost & MinCost & AvgUnserved & "
        "$P(U=0)$\\% & AvgNoTakeoff & AvgAbortReturn \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


def _convert_one(csv_path: Path, out_path: Path, distribution: str) -> None:
    df = pd.read_csv(csv_path)
    required = {
        "Instance",
        "gamma",
        "AvgCost",
        "StdCost",
        "MaxCost",
        "MinCost",
        "AvgUnserved",
        "P(Unserved=0)%",
        "AvgNoTakeoff",
        "AvgAbortReturn",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    out_path.write_text(_build_table_tex(df, distribution), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for dist in ("ND", "UD", "NDC"):
        src = args.csv_dir / f"table6_instance25_r30_r40_r50_{dist}.csv"
        if not src.exists():
            raise FileNotFoundError(f"Missing input CSV: {src}")
        dst = args.output_dir / f"{args.prefix}_{dist.lower()}.tex"
        _convert_one(src, dst, dist)
        generated.append(dst)

    combined = args.output_dir / f"{args.prefix}_all.tex"
    combined.write_text(
        "\n".join(
            [
                "% Auto-generated include file",
                f"\\input{{results/{generated[0].name}}}",
                f"\\input{{results/{generated[1].name}}}",
                f"\\input{{results/{generated[2].name}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for path in generated + [combined]:
        print(path)


if __name__ == "__main__":
    main()
