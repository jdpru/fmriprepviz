import numpy as np
import pytest

from conftest import PIAL_RADIUS_MM, RIBBON_VOXEL_SIZE, WHITE_RADIUS_MM
from fmriprepviz.ribbon import (
    FreeSurferNotFoundError,
    load_surface_masks,
    resolve_fs_dir,
)

TOLERANCE_MM = 2 * RIBBON_VOXEL_SIZE


def _outer_radius(mask_img):
    ijk = np.argwhere(np.asarray(mask_img.dataobj) > 0)
    homogeneous = np.column_stack([ijk, np.ones(len(ijk))])
    world = (mask_img.affine @ homogeneous.T).T[:, :3]
    return np.linalg.norm(world, axis=1).max()


def test_resolves_the_freesurfer_dir_inside_sourcedata(derivatives):
    resolved = resolve_fs_dir(derivatives, "01")
    assert resolved == derivatives / "sourcedata" / "freesurfer" / "sub-01"


def test_explicit_fs_dir_takes_precedence(derivatives):
    explicit = derivatives / "sourcedata" / "freesurfer" / "sub-01"
    assert resolve_fs_dir(derivatives, "99", fs_dir=explicit) == explicit


def test_falls_back_to_subjects_dir(derivatives, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SUBJECTS_DIR", str(derivatives / "sourcedata" / "freesurfer")
    )
    empty_root = tmp_path / "no_sourcedata"
    empty_root.mkdir()
    resolved = resolve_fs_dir(empty_root, "01")
    assert resolved == derivatives / "sourcedata" / "freesurfer" / "sub-01"


def test_unresolvable_fs_dir_lists_every_path_tried(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        resolve_fs_dir(tmp_path, "01")
    message = str(excinfo.value)
    assert "sourcedata" in message
    assert "ribbon.mgz" in message


def test_white_mask_covers_only_white_matter_labels(surface_masks):
    assert _outer_radius(surface_masks.white) < WHITE_RADIUS_MM


def test_white_mask_reaches_the_white_surface(surface_masks):
    assert _outer_radius(surface_masks.white) > WHITE_RADIUS_MM - TOLERANCE_MM


def test_pial_mask_reaches_the_pial_surface(surface_masks):
    outer = _outer_radius(surface_masks.pial)
    assert outer < PIAL_RADIUS_MM
    assert outer > PIAL_RADIUS_MM - TOLERANCE_MM


def test_pial_mask_is_filled_and_contains_the_white_mask(surface_masks):
    white = np.asarray(surface_masks.white.dataobj) > 0
    pial = np.asarray(surface_masks.pial.dataobj) > 0
    assert np.all(pial[white])
    assert pial.sum() > white.sum()


def test_masks_keep_the_ribbon_affine(derivatives, surface_masks):
    import nibabel as nib

    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    assert np.allclose(surface_masks.white.affine, ribbon.affine)
    assert np.allclose(surface_masks.pial.affine, ribbon.affine)


def test_missing_ribbon_is_reported(tmp_path):
    fs_dir = tmp_path / "sub-01"
    (fs_dir / "mri").mkdir(parents=True)
    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        load_surface_masks(fs_dir)
    assert "ribbon.mgz" in str(excinfo.value)
