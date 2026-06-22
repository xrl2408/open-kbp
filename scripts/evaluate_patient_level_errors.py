"""Evaluate patient-level dose MAE for existing validation predictions.

Example:
    python scripts/evaluate_patient_level_errors.py --models baseline_full_50 baseline_full_100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from provided_code.data_loader import DataLoader
from provided_code.utils import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate patient-level dose MAE for validation predictions.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline_full_50", "baseline_full_100"],
        help="Model result folders under results/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "patient_level_dose_errors.csv",
        help="CSV path for patient-level dose errors.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing model result folders.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=PROJECT_ROOT / "provided-data" / "validation-pats",
        help="Validation patient directory.",
    )
    return parser.parse_args()


def load_reference_patient(patient_path: Path):
    loader = DataLoader([patient_path], batch_size=1)
    loader.set_mode("evaluation")
    return loader.get_patients([patient_path.stem])


def load_predicted_dose(prediction_path: Path, patient_id: str) -> np.ndarray:
    loader = DataLoader([prediction_path], batch_size=1)
    loader.set_mode("predicted_dose")
    batch = loader.get_patients([patient_id])
    return batch.predicted_dose.flatten()


def patient_dose_mae(reference_batch, predicted_dose: np.ndarray) -> float:
    reference_dose = reference_batch.dose.flatten()
    possible_dose_mask = reference_batch.possible_dose_mask.flatten()
    return float(np.sum(np.abs(reference_dose - predicted_dose)) / max(np.sum(possible_dose_mask), 1))


def main() -> None:
    args = parse_args()
    validation_paths = sorted(get_paths(args.validation_dir), key=lambda path: path.stem)
    if not validation_paths:
        raise FileNotFoundError(f"No validation patients found in {args.validation_dir}")

    rows = []
    for patient_path in validation_paths:
        patient_id = patient_path.stem
        reference_batch = load_reference_patient(patient_path)
        row = {"patient": patient_id}

        for model in args.models:
            prediction_path = args.results_dir / model / "validation-predictions" / f"{patient_id}.csv"
            if not prediction_path.is_file():
                row[f"{model}_dose_mae"] = ""
                continue

            predicted_dose = load_predicted_dose(prediction_path, patient_id)
            row[f"{model}_dose_mae"] = f"{patient_dose_mae(reference_batch, predicted_dose):.6f}"

        if len(args.models) == 2:
            first_key = f"{args.models[0]}_dose_mae"
            second_key = f"{args.models[1]}_dose_mae"
            if row[first_key] and row[second_key]:
                diff = float(row[second_key]) - float(row[first_key])
                row[f"{args.models[1]}_minus_{args.models[0]}"] = f"{diff:.6f}"
            else:
                row[f"{args.models[1]}_minus_{args.models[0]}"] = ""

        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote patient-level dose errors to {args.output}")
    for model in args.models:
        values = [float(row[f"{model}_dose_mae"]) for row in rows if row[f"{model}_dose_mae"]]
        if values:
            print(f"{model}: n={len(values)}, mean={np.mean(values):.3f}, min={np.min(values):.3f}, max={np.max(values):.3f}")


if __name__ == "__main__":
    main()
