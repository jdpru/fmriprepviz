import io
import shutil

import pytest
from PIL import Image

from fmriprepviz.movie import write_gif, write_movie

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _solid_png(color, size=(40, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="png")
    return buffer.getvalue()


@pytest.fixture
def frames():
    return [_solid_png(c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]


def test_gif_contains_every_frame(frames, tmp_path):
    path = tmp_path / "qc.gif"
    write_gif(frames, path)
    with Image.open(path) as gif:
        assert gif.n_frames == len(frames)


def test_gif_loops_forever(frames, tmp_path):
    path = tmp_path / "qc.gif"
    write_gif(frames, path)
    with Image.open(path) as gif:
        assert gif.info["loop"] == 0


def test_fps_sets_the_frame_duration(frames, tmp_path):
    path = tmp_path / "qc.gif"
    write_gif(frames, path, fps=4)
    with Image.open(path) as gif:
        assert gif.info["duration"] == 250


def test_write_movie_dispatches_on_the_suffix(frames, tmp_path):
    path = tmp_path / "qc.gif"
    write_movie(frames, path)
    assert path.is_file()


def test_unknown_suffix_is_rejected(frames, tmp_path):
    with pytest.raises(ValueError, match="avi"):
        write_movie(frames, tmp_path / "qc.avi")


def test_empty_frame_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        write_movie([], tmp_path / "qc.gif")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_mp4_is_written_when_ffmpeg_is_available(frames, tmp_path):
    path = tmp_path / "qc.mp4"
    write_movie(frames, path)
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.skipif(HAS_FFMPEG, reason="ffmpeg is installed")
def test_mp4_without_ffmpeg_explains_the_alternative(frames, tmp_path):
    from fmriprepviz.movie import FFmpegNotFoundError

    with pytest.raises(FFmpegNotFoundError, match="gif"):
        write_movie(frames, tmp_path / "qc.mp4")
