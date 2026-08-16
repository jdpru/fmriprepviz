"""Derive white and pial surface masks from FreeSurfer's ribbon.mgz.

The masks are filled binary volumes. Their boundaries — the surfaces we
actually want to draw — are produced by contouring them at level 0.5 at
render time, so no boundary extraction happens here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

RIBBON_RELATIVE_PATH = Path("mri") / "ribbon.mgz"
WHITE_LABELS = (2, 41)
GRAY_LABELS = (3, 42)


class FreeSurferNotFoundError(RuntimeError):
    """No FreeSurfer subject directory containing a ribbon could be used."""


@dataclass(frozen=True)
class SurfaceMasks:
    """Filled binary masks whose contours are the white and pial surfaces."""

    white: nib.Nifti1Image
    pial: nib.Nifti1Image


def resolve_fs_dir(derivatives_root, subject: str, fs_dir=None) -> Path:
    """Find the FreeSurfer subject directory holding ribbon.mgz."""
    candidates = _fs_dir_candidates(derivatives_root, subject, fs_dir)
    for candidate in candidates:
        if (candidate / RIBBON_RELATIVE_PATH).is_file():
            return candidate
    tried = "\n  ".join(str(c / RIBBON_RELATIVE_PATH) for c in candidates)
    raise FreeSurferNotFoundError(
        "Could not find a FreeSurfer ribbon.mgz for "
        f"sub-{subject}. Tried:\n  {tried}\n"
        "Pass --fs-dir to point at the subject's FreeSurfer directory."
    )


def _fs_dir_candidates(derivatives_root, subject: str, fs_dir) -> list[Path]:
    if fs_dir is not None:
        return [Path(fs_dir)]
    candidates = [
        Path(derivatives_root) / "sourcedata" / "freesurfer" / f"sub-{subject}"
    ]
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if subjects_dir:
        candidates.append(Path(subjects_dir) / f"sub-{subject}")
    return candidates


def load_surface_masks(fs_dir) -> SurfaceMasks:
    """Build filled white and pial masks from a FreeSurfer ribbon."""
    ribbon_path = Path(fs_dir) / RIBBON_RELATIVE_PATH
    if not ribbon_path.is_file():
        raise FreeSurferNotFoundError(f"Expected a ribbon at {ribbon_path}")
    ribbon = nib.load(ribbon_path)
    data = np.asarray(ribbon.dataobj)
    return SurfaceMasks(
        white=_mask_image(np.isin(data, WHITE_LABELS), ribbon.affine),
        pial=_mask_image(np.isin(data, WHITE_LABELS + GRAY_LABELS), ribbon.affine),
    )


def _mask_image(mask: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(mask.astype(np.uint8), affine)
