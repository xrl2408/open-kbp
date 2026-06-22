"""Visualize true dose, predicted dose, and absolute error for one patient.

Example:
    python scripts/visualize_prediction_error.py --model baseline_full_100 --patient pt_224
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from provided_code.data_loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dose prediction error visualization for one patient.")
    parser.add_argument("--model", default="baseline_full_100", help="Model folder under results/.")
    parser.add_argument("--patient", default="pt_224", help="Patient ID, for example pt_224.")
    parser.add_argument("--roi", default="PTV70", help="ROI mask to overlay on CT.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "prediction_error",
        help="Directory for PNG output.",
    )
    parser.add_argument("--dpi", type=int, default=90, help="Output PNG resolution.")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=PROJECT_ROOT / "provided-data" / "validation-pats",
        help="Validation patient directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing model result folders.",
    )
    return parser.parse_args()


def choose_slice(error_volume: np.ndarray, fallback_volume: np.ndarray) -> int:
    """Choose the axial slice with the largest prediction error."""
    error_by_slice = np.sum(error_volume, axis=(0, 1))
    if np.max(error_by_slice) > 0:
        return int(np.argmax(error_by_slice))

    nonzero_counts = np.count_nonzero(fallback_volume, axis=(0, 1))
    if np.max(nonzero_counts) > 0:
        return int(np.argmax(nonzero_counts))
    return fallback_volume.shape[2] // 2


def load_reference(patient_dir: Path, patient_id: str):
    loader = DataLoader([patient_dir], batch_size=1)
    loader.set_mode("training_model")
    return loader.get_patients([patient_id])


def load_prediction(prediction_path: Path, patient_id: str):
    loader = DataLoader([prediction_path], batch_size=1)
    loader.set_mode("predicted_dose")
    return loader.get_patients([patient_id])


def main() -> None:
    args = parse_args()
    patient_dir = args.validation_dir / args.patient
    prediction_path = args.results_dir / args.model / "validation-predictions" / f"{args.patient}.csv"

    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Validation patient directory not found: {patient_dir}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")

    reference_batch = load_reference(patient_dir, args.patient)
    prediction_batch = load_prediction(prediction_path, args.patient)

    ct = np.squeeze(reference_batch.ct[0])
    true_dose = np.squeeze(reference_batch.dose[0])
    pred_dose = np.squeeze(prediction_batch.predicted_dose[0])
    possible_dose_mask = np.squeeze(reference_batch.possible_dose_mask[0])
    error = np.abs(true_dose - pred_dose) * possible_dose_mask

    roi_names = reference_batch.structure_mask_names
    if args.roi not in roi_names:
        raise ValueError(f"ROI {args.roi!r} not found. Available ROIs: {', '.join(roi_names)}")
    roi_idx = roi_names.index(args.roi)
    roi_mask = reference_batch.structure_masks[0, :, :, :, roi_idx]

    slice_idx = choose_slice(error, true_dose)

    ct_slice = ct[:, :, slice_idx]
    true_slice = true_dose[:, :, slice_idx]
    pred_slice = pred_dose[:, :, slice_idx]
    error_slice = error[:, :, slice_idx]
    roi_slice = roi_mask[:, :, slice_idx]

    dose_vmax = max(float(np.max(true_dose)), float(np.max(pred_dose)), 1.0)
    error_vmax = max(float(np.percentile(error[error > 0], 99)) if np.any(error > 0) else 1.0, 1.0)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))

    axes[0].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("CT")

    axes[1].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(roi_slice.T, cmap="Reds", alpha=0.45, origin="lower")
    axes[1].set_title(f"CT + {args.roi}")

    true_image = axes[2].imshow(true_slice.T, cmap="magma", origin="lower", vmin=0, vmax=dose_vmax)
    axes[2].set_title("True dose")
    fig.colorbar(true_image, ax=axes[2], fraction=0.046, pad=0.04)

    pred_image = axes[3].imshow(pred_slice.T, cmap="magma", origin="lower", vmin=0, vmax=dose_vmax)
    axes[3].set_title("Predicted dose")
    fig.colorbar(pred_image, ax=axes[3], fraction=0.046, pad=0.04)

    error_image = axes[4].imshow(error_slice.T, cmap="inferno", origin="lower", vmin=0, vmax=error_vmax)
    axes[4].set_title("Absolute error")
    fig.colorbar(error_image, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    mean_abs_error = np.sum(error) / max(np.sum(possible_dose_mask), 1)
    fig.suptitle(
        f"{args.model} {args.patient}, axial slice {slice_idx}, "
        f"patient dose MAE {mean_abs_error:.3f}",
        fontsize=14,
    )
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.model}_{args.patient}_error.png"
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote visualization to {output_path}")


if __name__ == "__main__":
    main()
