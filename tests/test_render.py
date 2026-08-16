import io

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from fmriprepviz.cuts import select_cuts
from fmriprepviz.discovery import find_boldrefs
from fmriprepviz.render import FRAME_SIZE, render_frame

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
