import io

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from fmriprepviz.cuts import select_cuts
from fmriprepviz.discovery import find_boldrefs
from fmriprepviz.render import FRAME_SIZE, NoMaskOverlapError, render_frame
from fmriprepviz.ribbon import SurfaceMasks

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def cuts(surface_masks):
    return select_cuts(surface_masks.pial, n_cuts=5)


@pytest.fixture(scope="module")
def boldrefs(derivatives):
    return find_boldrefs(derivatives, "01")


def _as_array(png_bytes):
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))


def test_returns_png_bytes(derivatives, surface_masks, cuts, boldrefs):
    png = render_frame(
        nib.load(boldrefs[0].path), surface_masks, cuts, boldrefs[0].label
    )
    assert png.startswith(PNG_MAGIC)


def test_frame_has_the_fixed_size(surface_masks, cuts, boldrefs):
    png = render_frame(
        nib.load(boldrefs[0].path), surface_masks, cuts, boldrefs[0].label
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE


def test_every_frame_has_identical_dimensions(surface_masks, cuts, boldrefs):
    """GIF and MP4 assembly require it, so it is checked rather than assumed."""
    sizes = {
        Image.open(
            io.BytesIO(render_frame(nib.load(ref.path), surface_masks, cuts, ref.label))
        ).size
        for ref in boldrefs[:3]
    }
    assert sizes == {FRAME_SIZE}


def test_rendering_is_deterministic(surface_masks, cuts, boldrefs):
    first = render_frame(
        nib.load(boldrefs[0].path), surface_masks, cuts, boldrefs[0].label
    )
    second = render_frame(
        nib.load(boldrefs[0].path), surface_masks, cuts, boldrefs[0].label
    )
    assert np.array_equal(_as_array(first), _as_array(second))


def test_frame_contains_red_contour_pixels(surface_masks, cuts, boldrefs):
    """The background is grayscale, so any strongly red pixel is a contour."""
    pixels = _as_array(
        render_frame(nib.load(boldrefs[0].path), surface_masks, cuts, boldrefs[0].label)
    ).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    reddish = (red > 120) & (red - green > 60) & (red - blue > 60)
    assert reddish.sum() > 200


def test_upsampling_disabled_still_renders(surface_masks, cuts, boldrefs):
    png = render_frame(
        nib.load(boldrefs[0].path),
        surface_masks,
        cuts,
        boldrefs[0].label,
        upsample=False,
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE


def test_one_row_per_display_mode(surface_masks, boldrefs):
    """Fewer orientations must still produce a correctly sized frame."""
    single = select_cuts(surface_masks.pial, n_cuts=5, display_modes=("z",))
    png = render_frame(
        nib.load(boldrefs[0].path), surface_masks, single, boldrefs[0].label
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE


def test_non_overlapping_mask_is_reported(surface_masks, cuts, boldrefs):
    """A pial mask that shares no world-space voxels with the display grid
    must raise a clear error rather than silently falling back to
    whole-volume intensity percentiles (which reintroduces the exact
    cross-session flicker the windowing is meant to prevent).

    upsample=False is required to reach this specific check: with the
    default upsample=True, nilearn's own resample_to_img already raises
    its own BoundingBoxError while placing the BOLD reference onto the
    (badly shifted) mask grid, before this code path is ever reached.
    """
    shifted_affine = surface_masks.pial.affine.copy()
    shifted_affine[:3, 3] += 1000.0
    shifted_pial = nib.Nifti1Image(
        np.asarray(surface_masks.pial.dataobj), shifted_affine
    )
    shifted_masks = SurfaceMasks(white=surface_masks.white, pial=shifted_pial)
    with pytest.raises(NoMaskOverlapError):
        render_frame(
            nib.load(boldrefs[0].path),
            shifted_masks,
            cuts,
            boldrefs[0].label,
            upsample=False,
        )


def test_label_text_is_drawn_in_the_frame(surface_masks, cuts, boldrefs):
    """A regression that dropped, mis-coloured or mis-positioned the label
    would pass every other test, yet the label is the only thing telling a
    viewer which session a frame shows.
    """
    boldref = nib.load(boldrefs[0].path)
    first = _as_array(render_frame(boldref, surface_masks, cuts, "label-one"))
    second = _as_array(render_frame(boldref, surface_masks, cuts, "label-two"))
    top_rows = int(round(FRAME_SIZE[1] * 0.05))
    first_strip = first[:top_rows]
    second_strip = second[:top_rows]
    assert not np.array_equal(first_strip, second_strip)
    assert first_strip.std() > 0
    assert second_strip.std() > 0
