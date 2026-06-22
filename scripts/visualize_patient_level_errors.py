"""Visualize patient-level dose MAE comparisons.

Example:
    python scripts/visualize_patient_level_errors.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot patient-level dose MAE comparison figures.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "docs" / "patient_level_dose_errors.csv",
        help="CSV file created by evaluate_patient_level_errors.py.",
    )
    parser.add_argument(
        "--models",
        nargs=2,
        default=["baseline_full_50", "baseline_full_100"],
        help="Two model names to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "patient_level_errors",
        help="Directory for figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output PNG resolution.")
    return parser.parse_args()


def save_figure(fig, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {output_path}")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Patient-level error CSV not found: {args.input}")

    first_model, second_model = args.models
    first_col = f"{first_model}_dose_mae"
    second_col = f"{second_model}_dose_mae"
    diff_col = f"{second_model}_minus_{first_model}"

    df = pd.read_csv(args.input)
    required_cols = {"patient", first_col, second_col, diff_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {args.input}: {sorted(missing_cols)}")

    df = df.dropna(subset=[first_col, second_col, diff_col]).copy()
    df[first_col] = df[first_col].astype(float)
    df[second_col] = df[second_col].astype(float)
    df[diff_col] = df[diff_col].astype(float)
    df = df.sort_values(diff_col)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df[first_col], df[second_col], alpha=0.8)
    axis_min = min(df[first_col].min(), df[second_col].min())
    axis_max = max(df[first_col].max(), df[second_col].max())
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(f"{first_model} patient dose MAE")
    ax.set_ylabel(f"{second_model} patient dose MAE")
    ax.set_title("Patient-level dose MAE comparison")
    ax.grid(alpha=0.25)
    save_figure(fig, args.output_dir / f"{first_model}_vs_{second_model}_patient_mae_scatter.png", args.dpi)

    colors = ["#2ca25f" if value < 0 else "#de2d26" for value in df[diff_col]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["patient"], df[diff_col], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Validation patient")
    ax.set_ylabel(f"{second_model} - {first_model} dose MAE")
    ax.set_title("Patient-level dose MAE difference")
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, args.output_dir / f"{first_model}_vs_{second_model}_patient_mae_difference.png", args.dpi)

    improved = int((df[diff_col] < 0).sum())
    worsened = int((df[diff_col] > 0).sum())
    print(f"{second_model} improved over {first_model} for {improved}/{len(df)} patients.")
    print(f"{second_model} worsened versus {first_model} for {worsened}/{len(df)} patients.")
    print("Largest improvements:")
    print(df[["patient", diff_col]].head(5).to_string(index=False))
    print("Largest degradations:")
    print(df[["patient", diff_col]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
