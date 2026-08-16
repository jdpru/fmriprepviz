import os
import shutil

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from fmriprepviz.cli import build_parser, main
from fmriprepviz.render import NO_OVERLAP_MESSAGE


def test_parser_defaults():
    args = build_parser().parse_args(["/derivs", "--subject", "01"])
    assert args.subject == "01"
    assert args.output.name == "qc.gif"
    assert args.fps == 2
    assert args.n_cuts == 7
    assert args.display_mode == "xyz"
    assert args.no_html is False
    assert args.no_upsample is False


def test_parser_accepts_multiple_sessions():
    args = build_parser().parse_args(
        ["/derivs", "--subject", "01", "--ses", "01", "02"]
    )
    assert args.ses == ["01", "02"]


def test_end_to_end_writes_a_gif_and_a_viewer(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )
    assert exit_code == 0
    assert output.is_file()
    assert output.with_suffix(".html").is_file()


def test_gif_has_one_frame_per_boldref(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    main([str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"])
    with Image.open(output) as gif:
        assert gif.n_frames == 5


def test_filters_reduce_the_frame_count(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--n-cuts", "3", "--ses", "02",
        ]
    )
    with Image.open(output) as gif:
        assert gif.n_frames == 2


def test_no_html_skips_the_viewer(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--n-cuts", "3", "--no-html",
        ]
    )
    assert not output.with_suffix(".html").exists()


def test_single_frame_warns_but_still_writes(derivatives, tmp_path, capsys):
    output = tmp_path / "qc.gif"
    main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--n-cuts", "3", "--ses", "02", "--task", "nback",
        ]
    )
    assert "one frame" in capsys.readouterr().err
    assert output.is_file()


def test_missing_boldrefs_exits_non_zero_with_a_message(
    derivatives_without_t1w, tmp_path, capsys
):
    exit_code = main(
        [str(derivatives_without_t1w), "--subject", "01", "-o", str(tmp_path / "qc.gif")]
    )
    assert exit_code == 1
    assert "MNI152NLin2009cAsym" in capsys.readouterr().err


def test_missing_freesurfer_exits_non_zero_with_a_message(
    derivatives, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fs-dir", str(tmp_path / "nowhere"),
        ]
    )
    assert exit_code == 1
    assert "ribbon.mgz" in capsys.readouterr().err


def test_cut_coords_requires_a_single_orientation(derivatives, tmp_path, capsys):
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--cut-coords", "-20", "0", "20",
        ]
    )
    assert exit_code == 1
    assert "--display-mode" in capsys.readouterr().err


def test_cut_coords_are_used_when_one_orientation_is_given(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--display-mode", "z", "--cut-coords", "-20", "0", "20",
        ]
    )
    assert exit_code == 0
    assert output.is_file()


def test_non_positive_fps_exits_non_zero_with_a_message(derivatives, tmp_path, capsys):
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fps", "0",
        ]
    )
    assert exit_code == 1
    assert "--fps" in capsys.readouterr().err


def _write_mismatched_fs_dir(tmp_path):
    """A FreeSurfer subject directory whose ribbon shares no world-space
    voxels with any boldref in the `derivatives` fixture, simulating
    --fs-dir pointing at the wrong subject.
    """
    fs_dir = tmp_path / "wrong-subject-fs"
    mri = fs_dir / "mri"
    mri.mkdir(parents=True)
    shape = (20, 20, 20)
    affine = np.eye(4)
    affine[:3, 3] = 5000.0  # far from any boldref in the `derivatives` fixture
    data = np.zeros(shape, dtype=np.uint8)
    data[5:10, 5:10, 5:10] = 2  # white matter label
    data[10:15, 10:15, 10:15] = 3  # gray matter label
    nib.MGHImage(data, affine).to_filename(mri / "ribbon.mgz")
    return fs_dir


def test_default_upsample_reports_a_mismatched_fs_dir_as_no_overlap(
    derivatives, tmp_path, capsys
):
    """With upsampling on (the default), a mismatched --fs-dir trips
    nilearn's own BoundingBoxError inside resample_to_img before
    render.render_frame's own NoMaskOverlapError guard is ever reached.
    The CLI must still report the same clear wording and exit non-zero,
    not let nilearn's internal exception surface as a traceback.
    """
    fs_dir = _write_mismatched_fs_dir(tmp_path)
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fs-dir", str(fs_dir),
        ]
    )
    assert exit_code == 1
    assert NO_OVERLAP_MESSAGE in capsys.readouterr().err


def test_no_upsample_reports_a_mismatched_fs_dir_as_no_overlap(
    derivatives, tmp_path, capsys
):
    """With --no-upsample, the mismatched --fs-dir trips the identical
    nilearn BoundingBoxError as the upsampled path above: since Fix 1,
    --no-upsample also resamples the BOLD reference onto the mask's grid
    (with nearest-neighbour interpolation, to keep the display pinned
    across frames), rather than leaving it on its own native grid. The CLI
    must report the identical wording either way, since from the user's
    point of view it is the same mistake.
    """
    fs_dir = _write_mismatched_fs_dir(tmp_path)
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fs-dir", str(fs_dir), "--no-upsample",
        ]
    )
    assert exit_code == 1
    assert NO_OVERLAP_MESSAGE in capsys.readouterr().err


def test_cuts_and_masks_are_computed_once_regardless_of_frame_count(
    derivatives, tmp_path, monkeypatch
):
    """The registration-check guarantee this tool makes rests on every frame
    being drawn with the *same* cuts and the *same* surface masks. If a
    future change derived cuts or masks per boldref instead of once from the
    ribbon, every frame would silently show slightly different anatomy and
    the movie would stop being a valid registration test -- while every
    other test in this suite, which only checks frame counts and file
    existence, would keep passing. This test guards that invariant directly
    by counting calls through a real (not stubbed) end-to-end run.
    """
    import fmriprepviz.cli as cli_module

    real_select_cuts = cli_module.select_cuts
    real_load_surface_masks = cli_module.load_surface_masks
    cuts_calls = []
    masks_calls = []

    def counting_select_cuts(*args, **kwargs):
        cuts_calls.append(1)
        return real_select_cuts(*args, **kwargs)

    def counting_load_surface_masks(*args, **kwargs):
        masks_calls.append(1)
        return real_load_surface_masks(*args, **kwargs)

    monkeypatch.setattr(cli_module, "select_cuts", counting_select_cuts)
    monkeypatch.setattr(cli_module, "load_surface_masks", counting_load_surface_masks)

    output = tmp_path / "qc.gif"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )

    assert exit_code == 0
    with Image.open(output) as gif:
        assert gif.n_frames == 5  # multiple frames, yet...
    assert len(cuts_calls) == 1  # ...cuts are computed once
    assert len(masks_calls) == 1  # ...masks are loaded once


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_mp4_output(derivatives, tmp_path):
    output = tmp_path / "qc.mp4"
    main([str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"])
    assert output.is_file()


def test_missing_output_directory_is_created(derivatives, tmp_path):
    """-o subdir/qc.gif is an ordinary thing to type; the parent directory
    not existing yet should not be treated as an error.
    """
    output = tmp_path / "does" / "not" / "exist" / "qc.gif"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )
    assert exit_code == 0
    assert output.is_file()


def test_unwritable_output_location_exits_non_zero_with_a_message(
    derivatives, tmp_path, capsys
):
    """An -o path under a read-only directory must be reported cleanly
    (exit 1, message on stderr), not as an uncaught OSError traceback.
    """
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o555)
    output = readonly_dir / "qc.gif"
    try:
        exit_code = main(
            [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
        )
        assert exit_code == 1
        assert capsys.readouterr().err.strip() != ""
    finally:
        readonly_dir.chmod(0o755)


def _write_corrupt_boldref_tree(tmp_path):
    """A minimal derivatives tree whose single space-T1w boldref is garbage
    bytes with a valid-looking name, simulating a truncated/corrupt file.
    """
    root = tmp_path / "corrupt_deriv"
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    name = "sub-01_ses-01_task-rest_run-1_space-T1w_boldref.nii.gz"
    (func / name).write_bytes(b"not a real nifti file, just garbage bytes")
    return root


def test_corrupt_boldref_exits_non_zero_with_a_message(derivatives, tmp_path, capsys):
    """A truncated or corrupt boldref .nii.gz must be reported cleanly, not
    as an uncaught nibabel ImageFileError traceback. Reuses the shared
    `derivatives` fixture's FreeSurfer directory via --fs-dir, since cuts
    and masks are derived from the ribbon alone and never touch the
    (corrupt) boldref before nib.load(ref.path) is reached.
    """
    corrupt_root = _write_corrupt_boldref_tree(tmp_path)
    fs_dir = derivatives / "sourcedata" / "freesurfer" / "sub-01"
    exit_code = main(
        [
            str(corrupt_root), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fs-dir", str(fs_dir), "--n-cuts", "3",
        ]
    )
    assert exit_code == 1
    assert capsys.readouterr().err.strip() != ""


def _write_stub_ffmpeg(bin_dir):
    """A stub `ffmpeg` that is found on PATH but always fails, simulating a
    minimal ffmpeg build without libx264 -- a common homebrew/conda case.
    """
    stub = bin_dir / "ffmpeg"
    stub.write_text("#!/bin/sh\necho 'stub ffmpeg: no encoder available' >&2\nexit 1\n")
    stub.chmod(0o755)
    return stub


def test_ffmpeg_encode_failure_exits_non_zero_with_a_message(
    derivatives, tmp_path, monkeypatch, capsys
):
    """movie.write_mp4 raises a plain RuntimeError with a helpful message
    when ffmpeg itself fails, but the CLI's catch tuple used to list only
    RuntimeError *subclasses* -- so this specific, deliberately-constructed
    helpful error was exactly the one that escaped as a traceback.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_ffmpeg(bin_dir)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))

    output = tmp_path / "qc.mp4"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )
    assert exit_code == 1
    assert "ffmpeg" in capsys.readouterr().err.lower()
