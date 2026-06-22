"""Visualize true dose, predicted dose, and absolute error for one patient.

Examples:
    python scripts/visualize_prediction_error.py --model baseline_full_100 --patient pt_224
    python scripts/visualize_prediction_error.py --compare-models baseline_full_50 baseline_full_100 --patient pt_224
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
    parser.add_argument(
        "--compare-models",
        nargs="+",
        default=None,
        help="Model folders under results/ to compare on shared min/max error slices.",
    )
    parser.add_argument("--patient", default="pt_224", help="Patient ID, for example pt_224.")
    parser.add_argument("--roi", default="PTV70", help="ROI mask to overlay on CT.")
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Fixed axial slice index to visualize. Overrides --slice-mode when set.",
    )
    parser.add_argument(
        "--slice-mode",
        default="max-error",
        choices=["max-error", "max-true-dose", "max-roi"],
        help=(
            "How to choose the axial slice when --slice-index is not set. "
            "Use max-true-dose or max-roi for model-to-model comparisons on the same patient."
        ),
    )
    parser.add_argument(
        "--comparison-slice-modes",
        nargs="+",
        default=["min-error", "max-error"],
        choices=["min-error", "max-error"],
        help="Comparison figures to generate when --compare-models is set.",
    )
    parser.add_argument(
        "--min-error-dose-threshold",
        type=float,
        default=0.1,
        help=(
            "For min-error comparison slices, ignore slices whose true-dose sum is below this "
            "fraction of the patient's maximum per-slice true-dose sum."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "prediction_error",
        help="Directory for visualization output.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution for raster formats.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "svg", "pdf"],
        help="Output formats to write. SVG/PDF keep text vectorized but embed image panels as rasters.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[22.0, 5.2],
        metavar=("WIDTH", "HEIGHT"),
        help="Matplotlib figure size in inches.",
    )
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


def choose_slice(
    error_volume: np.ndarray,
    true_dose_volume: np.ndarray,
    roi_volume: np.ndarray,
    slice_mode: str,
) -> int:
    """Choose an axial slice for visualization."""
    if slice_mode == "max-true-dose":
        dose_by_slice = np.sum(true_dose_volume, axis=(0, 1))
        if np.max(dose_by_slice) > 0:
            return int(np.argmax(dose_by_slice))

    if slice_mode == "max-roi":
        roi_by_slice = np.sum(roi_volume, axis=(0, 1))
        if np.max(roi_by_slice) > 0:
            return int(np.argmax(roi_by_slice))

    error_by_slice = np.sum(error_volume, axis=(0, 1))
    if np.max(error_by_slice) > 0:
        return int(np.argmax(error_by_slice))

    nonzero_counts = np.count_nonzero(true_dose_volume, axis=(0, 1))
    if np.max(nonzero_counts) > 0:
        return int(np.argmax(nonzero_counts))
    return true_dose_volume.shape[2] // 2


def load_reference(patient_dir: Path, patient_id: str):
    loader = DataLoader([patient_dir], batch_size=1)
    loader.set_mode("training_model")
    return loader.get_patients([patient_id])


def load_prediction(prediction_path: Path, patient_id: str):
    loader = DataLoader([prediction_path], batch_size=1)
    loader.set_mode("predicted_dose")
    return loader.get_patients([patient_id])


def validate_patient_path(patient_dir: Path) -> None:
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Validation patient directory not found: {patient_dir}")


def validate_prediction_path(prediction_path: Path) -> None:
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")


def get_roi_mask(reference_batch, roi_name: str) -> np.ndarray:
    roi_names = reference_batch.structure_mask_names
    if roi_name not in roi_names:
        raise ValueError(f"ROI {roi_name!r} not found. Available ROIs: {', '.join(roi_names)}")
    roi_idx = roi_names.index(roi_name)
    return reference_batch.structure_masks[0, :, :, :, roi_idx]


def choose_comparison_slice(
    errors_by_model: dict[str, np.ndarray],
    true_dose: np.ndarray,
    possible_dose_mask: np.ndarray,
    slice_mode: str,
    min_error_dose_threshold: float,
) -> tuple[int, float]:
    """Choose a slice using mean per-voxel error averaged across comparison models."""
    combined_error = np.mean(np.stack(list(errors_by_model.values()), axis=0), axis=0)
    error_by_slice = np.sum(combined_error, axis=(0, 1))
    possible_voxels_by_slice = np.sum(possible_dose_mask, axis=(0, 1))
    slice_mae = error_by_slice / np.maximum(possible_voxels_by_slice, 1)

    true_dose_by_slice = np.sum(true_dose, axis=(0, 1))
    min_true_dose_sum = min_error_dose_threshold * max(float(np.max(true_dose_by_slice)), 1.0)
    candidate_slices = np.where((possible_voxels_by_slice > 0) & (true_dose_by_slice >= min_true_dose_sum))[0]
    if len(candidate_slices) == 0:
        candidate_slices = np.where(possible_voxels_by_slice > 0)[0]
    if len(candidate_slices) == 0:
        candidate_slices = np.arange(true_dose.shape[2])

    candidate_mae = slice_mae[candidate_slices]
    if slice_mode == "min-error":
        selected_idx = int(candidate_slices[np.argmin(candidate_mae)])
    elif slice_mode == "max-error":
        selected_idx = int(candidate_slices[np.argmax(candidate_mae)])
    else:
        raise ValueError(f"Unsupported comparison slice mode: {slice_mode}")
    return selected_idx, float(slice_mae[selected_idx])


def save_figure(fig, output_dir: Path, output_stem: str, formats: list[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for output_format in formats:
        output_path = output_dir / f"{output_stem}.{output_format}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        output_paths.append(output_path)
    plt.close(fig)
    print("Wrote visualizations:")
    for output_path in output_paths:
        print(f"  {output_path}")


def make_comparison_visualizations(args, reference_batch) -> None:
    ct = np.squeeze(reference_batch.ct[0])
    true_dose = np.squeeze(reference_batch.dose[0])
    possible_dose_mask = np.squeeze(reference_batch.possible_dose_mask[0])
    roi_mask = get_roi_mask(reference_batch, args.roi)

    predictions_by_model = {}
    errors_by_model = {}
    for model in args.compare_models:
        prediction_path = args.results_dir / model / "validation-predictions" / f"{args.patient}.csv"
        validate_prediction_path(prediction_path)
        prediction_batch = load_prediction(prediction_path, args.patient)
        pred_dose = np.squeeze(prediction_batch.predicted_dose[0])
        predictions_by_model[model] = pred_dose
        errors_by_model[model] = np.abs(true_dose - pred_dose) * possible_dose_mask

    dose_vmax = max([float(np.max(true_dose)), 1.0] + [float(np.max(pred)) for pred in predictions_by_model.values()])
    nonzero_errors = [error[error > 0] for error in errors_by_model.values() if np.any(error > 0)]
    all_errors = np.concatenate(nonzero_errors) if nonzero_errors else np.array([])
    error_vmax = max(float(np.percentile(all_errors, 99)) if len(all_errors) else 1.0, 1.0)

    for comparison_mode in args.comparison_slice_modes:
        slice_idx, slice_mae = choose_comparison_slice(
            errors_by_model,
            true_dose,
            possible_dose_mask,
            comparison_mode,
            args.min_error_dose_threshold,
        )

        n_models = len(args.compare_models)
        n_cols = 3 + 2 * n_models
        fig, axes = plt.subplots(1, n_cols, figsize=tuple(args.figsize))

        axes[0].imshow(ct[:, :, slice_idx].T, cmap="gray", origin="lower")
        axes[0].set_title("CT")

        axes[1].imshow(ct[:, :, slice_idx].T, cmap="gray", origin="lower")
        axes[1].imshow(roi_mask[:, :, slice_idx].T, cmap="Reds", alpha=0.45, origin="lower")
        axes[1].set_title(f"CT + {args.roi}")

        true_image = axes[2].imshow(true_dose[:, :, slice_idx].T, cmap="magma", origin="lower", vmin=0, vmax=dose_vmax)
        axes[2].set_title("True dose")
        fig.colorbar(true_image, ax=axes[2], fraction=0.046, pad=0.04)

        axis_idx = 3
        for model in args.compare_models:
            pred_image = axes[axis_idx].imshow(
                predictions_by_model[model][:, :, slice_idx].T,
                cmap="magma",
                origin="lower",
                vmin=0,
                vmax=dose_vmax,
            )
            axes[axis_idx].set_title(f"{model}\npredicted")
            fig.colorbar(pred_image, ax=axes[axis_idx], fraction=0.046, pad=0.04)
            axis_idx += 1

            error_image = axes[axis_idx].imshow(
                errors_by_model[model][:, :, slice_idx].T,
                cmap="inferno",
                origin="lower",
                vmin=0,
                vmax=error_vmax,
            )
            axes[axis_idx].set_title(f"{model}\nabsolute error")
            fig.colorbar(error_image, ax=axes[axis_idx], fraction=0.046, pad=0.04)
            axis_idx += 1

        for ax in axes:
            ax.axis("off")

        model_label = "_vs_".join(args.compare_models)
        fig.suptitle(
            f"{args.patient}, {comparison_mode.replace('-', ' ')} slice {slice_idx}, "
            f"combined slice MAE {slice_mae:.3f}",
            fontsize=14,
        )
        fig.tight_layout()
        save_figure(
            fig,
            args.output_dir,
            f"{model_label}_{args.patient}_{comparison_mode}_comparison",
            args.formats,
            args.dpi,
        )


def main() -> None:
    args = parse_args()
    patient_dir = args.validation_dir / args.patient
    validate_patient_path(patient_dir)

    reference_batch = load_reference(patient_dir, args.patient)

    if args.compare_models:
        make_comparison_visualizations(args, reference_batch)
        return

    prediction_path = args.results_dir / args.model / "validation-predictions" / f"{args.patient}.csv"
    validate_prediction_path(prediction_path)
    prediction_batch = load_prediction(prediction_path, args.patient)

    ct = np.squeeze(reference_batch.ct[0])
    true_dose = np.squeeze(reference_batch.dose[0])
    pred_dose = np.squeeze(prediction_batch.predicted_dose[0])
    possible_dose_mask = np.squeeze(reference_batch.possible_dose_mask[0])
    error = np.abs(true_dose - pred_dose) * possible_dose_mask

    roi_mask = get_roi_mask(reference_batch, args.roi)

    if args.slice_index is not None:
        if args.slice_index < 0 or args.slice_index >= ct.shape[2]:
            raise ValueError(f"Slice index must be between 0 and {ct.shape[2] - 1}: {args.slice_index}")
        slice_idx = args.slice_index
    else:
        slice_idx = choose_slice(error, true_dose, roi_mask, args.slice_mode)

    ct_slice = ct[:, :, slice_idx]
    true_slice = true_dose[:, :, slice_idx]
    pred_slice = pred_dose[:, :, slice_idx]
    error_slice = error[:, :, slice_idx]
    roi_slice = roi_mask[:, :, slice_idx]

    dose_vmax = max(float(np.max(true_dose)), float(np.max(pred_dose)), 1.0)
    error_vmax = max(float(np.percentile(error[error > 0], 99)) if np.any(error > 0) else 1.0, 1.0)

    fig, axes = plt.subplots(1, 5, figsize=tuple(args.figsize))

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

    save_figure(fig, args.output_dir, f"{args.model}_{args.patient}_error", args.formats, args.dpi)


if __name__ == "__main__":
    main()
