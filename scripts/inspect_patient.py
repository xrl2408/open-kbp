from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from provided_code import DataLoader


def choose_slice(volume):
    """Choose the axial slice with the largest non-zero area."""
    nonzero_counts = np.count_nonzero(volume, axis=(0, 1))
    if np.max(nonzero_counts) == 0:
        return volume.shape[2] // 2
    return int(np.argmax(nonzero_counts))


def main():
    project_root = Path(__file__).resolve().parents[1]
    patient_dir = project_root / "provided-data" / "train-pats" / "pt_1"
    output_dir = project_root / "outputs" / "inspect_patient"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    loader = DataLoader([patient_dir], batch_size=1)
    loader.set_mode("training_model")
    batch = loader.get_patients(["pt_1"])

    ct = np.squeeze(batch.ct[0])
    dose = np.squeeze(batch.dose[0])
    possible_dose_mask = np.squeeze(batch.possible_dose_mask[0])
    masks = batch.structure_masks[0]
    roi_names = batch.structure_mask_names
    voxel_dimensions = np.squeeze(batch.voxel_dimensions[0])

    print("Patient: pt_1")
    print(f"CT shape: {ct.shape}")
    print(f"Dose shape: {dose.shape}")
    print(f"Possible dose mask shape: {possible_dose_mask.shape}")
    print(f"Structure masks shape: {masks.shape}")
    print(f"Voxel dimensions: {voxel_dimensions}")
    print()
    print("ROI voxel counts:")
    for idx, roi_name in enumerate(roi_names):
        roi_mask = masks[:, :, :, idx]
        print(f"  {roi_name:12s}: {int(np.count_nonzero(roi_mask))}")

    dose_slice_idx = choose_slice(dose)
    ptv70_idx = roi_names.index("PTV70")
    ptv70_mask = masks[:, :, :, ptv70_idx]
    mask_slice_idx = choose_slice(ptv70_mask)

    slice_idx = dose_slice_idx if np.count_nonzero(dose[:, :, dose_slice_idx]) > 0 else mask_slice_idx

    ct_slice = ct[:, :, slice_idx]
    dose_slice = dose[:, :, slice_idx]
    ptv70_slice = ptv70_mask[:, :, slice_idx]
    possible_slice = possible_dose_mask[:, :, slice_idx]

    print()
    print(f"Selected axial slice index: {slice_idx}")
    print(f"CT min/max: {ct.min():.3f} / {ct.max():.3f}")
    print(f"Dose min/max: {dose.min():.3f} / {dose.max():.3f}")
    print(f"Possible dose voxels: {int(np.count_nonzero(possible_dose_mask))}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("CT")

    axes[1].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(ptv70_slice.T, cmap="Reds", alpha=0.45, origin="lower")
    axes[1].set_title("CT + PTV70")

    dose_image = axes[2].imshow(dose_slice.T, cmap="magma", origin="lower")
    axes[2].set_title("Dose")
    fig.colorbar(dose_image, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(possible_slice.T, cmap="Blues", origin="lower")
    axes[3].set_title("Possible Dose Mask")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"OpenKBP pt_1, axial slice {slice_idx}", fontsize=14)
    fig.tight_layout()

    output_path = output_dir / "pt_1_overview.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print()
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
