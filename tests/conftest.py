"""Synthetic fMRIPrep derivatives tree shared by the whole test suite.

The ribbon is built from concentric spheres of known radius, so tests can
assert that derived surfaces land at the correct anatomical location rather
than merely checking that an image was produced.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

RIBBON_SHAPE = (128, 128, 128)
RIBBON_VOXEL_SIZE = 1.0
RIBBON_ORIGIN = -64.0

BOLDREF_SHAPE = (48, 48, 48)
BOLDREF_VOXEL_SIZE = 3.0
BOLDREF_ORIGIN = -72.0

WHITE_RADIUS_MM = 30.0
PIAL_RADIUS_MM = 40.0

# (session, task, run) for each space-T1w boldref in the synthetic tree.
# run-10 is present so ordering tests can catch naive string sorting.
BOLDREF_ENTITIES = [
    ("01", "rest", "1"),
    ("01", "rest", "2"),
    ("01", "rest", "10"),
    ("02", "rest", "1"),
    ("02", "nback", "1"),
]

MNI_BOLDREF_NAME = (
    "sub-01_ses-01_task-rest_run-1"
    "_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz"
)


def _affine(voxel_size: float, origin: float) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = np.diag([voxel_size] * 3)
    affine[:3, 3] = origin
    return affine


def _world_grids(shape, voxel_size, origin):
    axes = [np.arange(n) * voxel_size + origin for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def _world_radius(shape, voxel_size, origin):
    grids = _world_grids(shape, voxel_size, origin)
    return np.sqrt(sum(g**2 for g in grids)), grids


def _ribbon_data() -> np.ndarray:
    """Concentric spheres labelled with real FreeSurfer ribbon values."""
    radius, grids = _world_radius(RIBBON_SHAPE, RIBBON_VOXEL_SIZE, RIBBON_ORIGIN)
    left = grids[0] < 0
    white = radius < WHITE_RADIUS_MM
    gray = (radius >= WHITE_RADIUS_MM) & (radius < PIAL_RADIUS_MM)

    data = np.zeros(RIBBON_SHAPE, dtype=np.uint8)
    data[white & left] = 2
    data[white & ~left] = 41
    data[gray & left] = 3
    data[gray & ~left] = 42
    return data


def _boldref_data(seed: int) -> np.ndarray:
    radius, _ = _world_radius(BOLDREF_SHAPE, BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
    rng = np.random.default_rng(seed)
    data = np.where(radius < PIAL_RADIUS_MM, 1000.0, 0.0)
    return (data + rng.normal(0.0, 20.0, BOLDREF_SHAPE)).astype(np.float32)


def _write_boldrefs(root: Path) -> None:
    for index, (ses, task, run) in enumerate(BOLDREF_ENTITIES):
        func = root / "sub-01" / f"ses-{ses}" / "func"
        func.mkdir(parents=True, exist_ok=True)
        name = (
            f"sub-01_ses-{ses}_task-{task}_run-{run}"
            "_space-T1w_boldref.nii.gz"
        )
        image = nib.Nifti1Image(
            _boldref_data(index), _affine(BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
        )
        image.to_filename(func / name)


def _write_mni_boldref(root: Path) -> None:
    """A non-T1w boldref, so error messages have another space to report."""
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        _boldref_data(99), _affine(BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
    )
    image.to_filename(func / MNI_BOLDREF_NAME)


def _write_ribbon(root: Path) -> None:
    mri = root / "sourcedata" / "freesurfer" / "sub-01" / "mri"
    mri.mkdir(parents=True, exist_ok=True)
    ribbon = nib.MGHImage(
        _ribbon_data(), _affine(RIBBON_VOXEL_SIZE, RIBBON_ORIGIN)
    )
    ribbon.to_filename(mri / "ribbon.mgz")


@pytest.fixture(scope="session")
def derivatives(tmp_path_factory) -> Path:
    """A complete synthetic derivatives tree: boldrefs plus a FreeSurfer dir."""
    root = tmp_path_factory.mktemp("derivatives")
    _write_boldrefs(root)
    _write_mni_boldref(root)
    _write_ribbon(root)
    return root


@pytest.fixture(scope="session")
def derivatives_without_t1w(tmp_path_factory) -> Path:
    """A tree fMRIPrep produced without T1w among its output spaces."""
    root = tmp_path_factory.mktemp("mni_only")
    _write_mni_boldref(root)
    _write_ribbon(root)
    return root
