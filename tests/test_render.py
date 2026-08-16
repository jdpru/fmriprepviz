import io

import nibabel as nib
import numpy as np
import pytest
from nilearn.image.resampling import BoundingBoxError
from PIL import Image

from conftest import BOLDREF_VOXEL_SIZE, PIAL_RADIUS_MM
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


def _reddish_mask(png_bytes):
    """Boolean pixel mask of the drawn red contours.

    Same criterion as test_frame_contains_red_contour_pixels: the
    background is grayscale, so any strongly red pixel is a contour.
    """
    pixels = _as_array(png_bytes).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return (red > 120) & (red - green > 60) & (red - blue > 60)


def _reddish_bbox(png_bytes):
    """(x_min, x_max, y_min, y_max) of the drawn red contours.

    A handful of anti-aliased border pixels legitimately flicker in or out
    of the `_reddish_mask` threshold between two otherwise-identical renders
    whenever the underlying grayscale windowing differs even slightly (here,
    because the two boldrefs' independent noise draws give the masked
    intensity percentiles slightly different values) -- this is unrelated to
    contour placement, so a full pixel-mask equality check would be too
    strict. The bounding box is robust to that noise while still catching
    the actual defect: a display grid that is not pinned across frames
    shifts and rescales the whole contour, moving this box by many pixels,
    not flickering a handful of edge pixels.
    """
    ys, xs = np.where(_reddish_mask(png_bytes))
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))


def _sphere_boldref(shape, voxel_size=3.0, radius_mm=PIAL_RADIUS_MM, seed=0):
    """A synthetic boldref: a bright sphere of `radius_mm` centred at the
    world origin, on a grid of the given shape/voxel size. Two boldrefs
    built with the same radius_mm but different shape/voxel_size share the
    same anatomy but differ in field of view -- exactly the case that
    exposed the --no-upsample bug.
    """
    origin = -(shape[0] * voxel_size) / 2.0
    axes = [np.arange(n) * voxel_size + origin for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt(sum(g**2 for g in grids))
    signal = np.where(radius < radius_mm, 1000.0, 0.0)
    rng = np.random.default_rng(seed)
    data = (signal + rng.normal(0.0, 20.0, shape)).astype(np.float32)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([voxel_size] * 3)
    affine[:3, 3] = origin
    return nib.Nifti1Image(data, affine)


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


def _bright_disc_mask(png_bytes):
    """Boolean pixel mask of the bright (grayscale, non-contour) disc."""
    pixels = _as_array(png_bytes).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    grayscale = (np.abs(red - green) < 10) & (np.abs(green - blue) < 10)
    return grayscale & (red > 100)


def test_contours_land_on_the_anatomy_not_merely_somewhere(surface_masks, boldrefs):
    """Guards the join between mask geometry (verified in test_ribbon.py)
    and red-pixel presence (verified by test_frame_contains_red_contour_pixels
    above): neither of those checks would catch contours drawn at the wrong
    scale, offset, orientation, or on the wrong row, since both pass as long
    as red pixels exist *somewhere* with the right mask *shape*. A left-right
    flip or a half-voxel offset in the join between mask and boldref would
    ship looking correct to both and only be visible here.

    The synthetic fixture makes this tractable: the boldref's bright disc
    and the pial mask share a 40 mm radius (PIAL_RADIUS_MM), so on a single
    axial cut through the sphere's centre (z=0, its widest cross-section),
    the outer red ring must sit right on the bright disc's edge. This
    compares their positions along the frame's central row.

    Tolerance: one BOLD voxel's width, converted to pixels using the
    disc's own measured width (rather than a hard-coded pixel count) --
    the coarsest unit at which the underlying BOLD data can resolve the
    sphere's true edge in the first place, so agreement any tighter than
    that would be over-claiming precision the input data does not have.
    """
    single_axial_cut = {"z": [0.0]}
    png = render_frame(
        nib.load(boldrefs[0].path), surface_masks, single_axial_cut, "label"
    )
    pixels = _as_array(png)
    central_row = pixels.shape[0] // 2

    red_xs = np.where(_reddish_mask(png)[central_row])[0]
    bright_xs = np.where(_bright_disc_mask(png)[central_row])[0]
    assert red_xs.size > 0, "no red contour pixels on the central row"
    assert bright_xs.size > 0, "no bright disc pixels on the central row"

    disc_width_px = bright_xs.max() - bright_xs.min()
    px_per_mm = disc_width_px / (2.0 * PIAL_RADIUS_MM)
    tolerance_px = BOLDREF_VOXEL_SIZE * px_per_mm

    assert abs(red_xs.min() - bright_xs.min()) <= tolerance_px, (
        f"left contour edge {red_xs.min()} vs disc edge {bright_xs.min()}, "
        f"tolerance {tolerance_px:.1f}px"
    )
    assert abs(red_xs.max() - bright_xs.max()) <= tolerance_px, (
        f"right contour edge {red_xs.max()} vs disc edge {bright_xs.max()}, "
        f"tolerance {tolerance_px:.1f}px"
    )


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
    """A pial mask that shares no voxels with the display grid must raise a
    clear error rather than silently falling back to whole-volume intensity
    percentiles (which reintroduces the exact cross-session flicker the
    windowing is meant to prevent).

    Since Fix 1 (--no-upsample now resamples onto the mask grid too, with
    nearest-neighbour interpolation, to keep the display pinned across
    frames -- see test_display_pinned_across_differing_boldref_fov below),
    a *mismatched-affine* mask no longer reaches this guard at all: nilearn's
    own resample_to_img raises its own BoundingBoxError while placing the
    BOLD reference onto the (badly shifted) mask grid, under both upsample
    values (see test_shifted_mask_now_raises_bounding_box_error_either_way
    below). What this guard still protects against is a mask that shares the
    display grid exactly but happens to contain zero foreground voxels
    (e.g. if SurfaceMasks were ever constructed directly with an empty
    mask, bypassing ribbon.load_surface_masks's own guard against that).
    """
    empty_pial = nib.Nifti1Image(
        np.zeros_like(np.asarray(surface_masks.pial.dataobj)),
        surface_masks.pial.affine,
    )
    empty_masks = SurfaceMasks(white=surface_masks.white, pial=empty_pial)
    with pytest.raises(NoMaskOverlapError):
        render_frame(
            nib.load(boldrefs[0].path),
            empty_masks,
            cuts,
            boldrefs[0].label,
            upsample=False,
        )


def test_shifted_mask_now_raises_bounding_box_error_either_way(
    surface_masks, cuts, boldrefs
):
    """Documents a direct consequence of Fix 1: because --no-upsample now
    always resamples the BOLD reference onto the mask's own grid (only the
    interpolation mode changes), a totally mismatched mask affine trips
    nilearn's BoundingBoxError under upsample=False exactly as it already
    did under upsample=True, rather than reaching this module's own
    NoMaskOverlapError guard (see test_non_overlapping_mask_is_reported
    above for what still reaches that guard). test_cli.py's
    test_no_upsample_reports_a_mismatched_fs_dir_as_no_overlap confirms the
    CLI still reports the identical wording either way.
    """
    shifted_affine = surface_masks.pial.affine.copy()
    shifted_affine[:3, 3] += 1000.0
    shifted_pial = nib.Nifti1Image(
        np.asarray(surface_masks.pial.dataobj), shifted_affine
    )
    shifted_masks = SurfaceMasks(white=surface_masks.white, pial=shifted_pial)
    with pytest.raises(BoundingBoxError):
        render_frame(
            nib.load(boldrefs[0].path),
            shifted_masks,
            cuts,
            boldrefs[0].label,
            upsample=False,
        )


def test_display_pinned_across_differing_boldref_fov(surface_masks, cuts):
    """This is the direct expression of the tool's central invariant: masks
    and cuts are computed once and reused for every frame, so anything that
    appears to move between frames must be misregistration, never a display
    artefact. Two boldrefs here are identical except for field of view (a
    48^3 grid at 3 mm vs. a 100^3 grid at 3 mm), both containing a bright
    sphere of the same radius as the pial mask. If the display grid were not
    pinned, nilearn would set each axis's limits from the bounds of what is
    actually drawn on it, so the same anatomy would appear at a different
    zoom level -- moving the red contours on screen -- purely because of a
    FOV difference between sessions that has nothing to do with
    registration. This must hold under both upsample=True (already correct
    before this fix) and upsample=False (the bug Fix 1 corrects: previously
    it left the BOLD on its own native grid, letting the display zoom to
    that grid's own bounds).
    """
    small_fov = _sphere_boldref((48, 48, 48), voxel_size=3.0)
    large_fov = _sphere_boldref((100, 100, 100), voxel_size=3.0)

    for upsample in (True, False):
        small_png = render_frame(
            small_fov, surface_masks, cuts, "small-fov", upsample=upsample
        )
        large_png = render_frame(
            large_fov, surface_masks, cuts, "large-fov", upsample=upsample
        )
        small_bbox = _reddish_bbox(small_png)
        large_bbox = _reddish_bbox(large_png)
        assert small_bbox == large_bbox, (
            f"contour bounding box moved between differing FOVs with "
            f"upsample={upsample}: {small_bbox} vs {large_bbox}"
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


def test_no_colorbar_is_drawn(surface_masks, cuts, boldrefs):
    """nilearn's plot_anat defaults to colorbar=True, adding a per-row
    colorbar whose labelled range is recomputed from each frame's own
    masked intensity window. That per-frame windowing is correct and
    deliberate, but this tool's premise is a display that never changes on
    its own: a number that ticks between frames for reasons unrelated to
    registration trains the eye on a meaningless signal, so no colorbar may
    be drawn at all.

    A right-edge strip is the natural place to look, but the true right
    edge of the frame is not where nilearn puts things: inspecting an
    actual rendered frame shows the colorbar's tick text and gradient bar
    sit around columns 997-1077 of 1200, with a ~120px fully-black margin
    beyond that to the frame's true edge -- so the rightmost 2% (or even
    10%) of columns is black whether or not the bug is present, and would
    not detect it. It is also necessary to clear the coronal/axial "R"
    orientation annotation, which sits just to the left of the colorbar
    today and shifts right (to end around column 1040) once the colorbar
    is removed, as the freed width is given back to the anatomical slices.
    The rightmost 12.5% of columns (the last 150 of 1200) was chosen by
    rendering both the current and fixed output and checking column
    contents directly: it reliably contains the colorbar in the current,
    buggy output, and reliably stays outside both the colorbar and the
    (shifted) "R" annotation once the colorbar is gone.
    """
    pixels = _as_array(
        render_frame(nib.load(boldrefs[0].path), surface_masks, cuts, boldrefs[0].label)
    )
    strip_width = int(round(FRAME_SIZE[0] * 0.125))
    right_strip = pixels[:, -strip_width:]
    assert np.array_equal(right_strip, np.zeros_like(right_strip))
