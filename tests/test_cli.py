import shutil

import pytest
from PIL import Image

from fmriprepviz.cli import build_parser, main


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_mp4_output(derivatives, tmp_path):
    output = tmp_path / "qc.mp4"
    main([str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"])
    assert output.is_file()
