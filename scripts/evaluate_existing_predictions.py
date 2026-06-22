"""Evaluate existing OpenKBP validation prediction folders.

Example:
    python scripts/evaluate_existing_predictions.py \
        --models baseline_full_20 baseline_full_50 baseline_full_100 baseline_full_150
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator
from provided_code.utils import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing validation dose predictions.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model result folders under results/. Defaults to all folders with validation-predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "experiment_summary.csv",
        help="CSV path for the experiment summary.",
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


def discover_models(results_dir: Path) -> list[str]:
    models = []
    if not results_dir.exists():
        return models
    for child in sorted(results_dir.iterdir()):
        if (child / "validation-predictions").is_dir():
            models.append(child.name)
    return models


def infer_epoch(model_name: str) -> str:
    marker = "baseline_full_"
    if model_name.startswith(marker):
        return model_name.removeprefix(marker)
    return ""


def evaluate_model(model_name: str, results_dir: Path, validation_paths_by_id: dict[str, Path]) -> dict[str, str]:
    prediction_dir = results_dir / model_name / "validation-predictions"
    prediction_paths = get_paths(prediction_dir, extension="csv")
    prediction_paths = sorted(prediction_paths, key=lambda path: path.stem)

    matched_prediction_paths = [path for path in prediction_paths if path.stem in validation_paths_by_id]
    matched_reference_paths = [validation_paths_by_id[path.stem] for path in matched_prediction_paths]

    row = {
        "model": model_name,
        "epoch": infer_epoch(model_name),
        "training_data": "full training set" if model_name.startswith("baseline_full_") else "early/dry-run setup",
        "validation_patients": str(len(matched_prediction_paths)),
        "dose_score": "",
        "dvh_score": "",
        "notes": "",
    }

    if not matched_prediction_paths:
        row["notes"] = "No validation prediction CSV files found."
        return row

    evaluator = DoseEvaluator(DataLoader(matched_reference_paths), DataLoader(matched_prediction_paths))
    evaluator.evaluate()
    dose_score, dvh_score = evaluator.get_scores()

    row["dose_score"] = f"{dose_score:.3f}"
    row["dvh_score"] = f"{dvh_score:.3f}"
    if len(matched_prediction_paths) == 40 and model_name.startswith("baseline_full_"):
        row["notes"] = "Full training set; full 40-patient validation set."
    elif len(matched_prediction_paths) < 40:
        row["notes"] = "Partial validation predictions only."
    return row


def main() -> None:
    args = parse_args()
    validation_paths = get_paths(args.validation_dir)
    validation_paths_by_id = {path.stem: path for path in validation_paths}
    models = args.models if args.models is not None else discover_models(args.results_dir)

    if not models:
        raise FileNotFoundError(f"No validation prediction folders found in {args.results_dir}")

    rows = [evaluate_model(model, args.results_dir, validation_paths_by_id) for model in models]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "epoch", "training_data", "validation_patients", "dose_score", "dvh_score", "notes"]
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote experiment summary to {args.output}")
    for row in rows:
        print(
            f"{row['model']}: n={row['validation_patients']}, "
            f"dose={row['dose_score'] or 'NA'}, dvh={row['dvh_score'] or 'NA'}"
        )


if __name__ == "__main__":
    main()

