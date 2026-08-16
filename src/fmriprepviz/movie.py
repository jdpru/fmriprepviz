"""Assemble rendered PNG frames into a movie."""

from __future__ import annotations

import io
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

DEFAULT_FPS = 2


class FFmpegNotFoundError(RuntimeError):
    """MP4 output was requested but ffmpeg is not available."""


def write_movie(frames: Sequence[bytes], path, fps: int = DEFAULT_FPS) -> Path:
    """Write frames to `path`, choosing the format from its suffix."""
    path = Path(path)
    if not frames:
        raise ValueError("Cannot write a movie with no frames.")
    writers = {".gif": write_gif, ".mp4": write_mp4}
    suffix = path.suffix.lower()
    if suffix not in writers:
        raise ValueError(
            f"Unsupported movie format '{suffix}'; use one of "
            f"{', '.join(sorted(writers))}."
        )
    return writers[suffix](frames, path, fps=fps)


def write_gif(frames: Sequence[bytes], path, fps: int = DEFAULT_FPS) -> Path:
    path = Path(path)
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=_duration_ms(fps),
        loop=0,
    )
    return path


def write_mp4(frames: Sequence[bytes], path, fps: int = DEFAULT_FPS) -> Path:
    path = Path(path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegNotFoundError(
            "MP4 output needs ffmpeg on PATH, which was not found. "
            "Install ffmpeg, or write a gif instead."
        )
    command = [
        ffmpeg, "-y",
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    result = subprocess.run(
        command, input=b"".join(frames), capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to write the movie:\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
    return path


def _duration_ms(fps: int) -> int:
    return int(round(1000 / fps))
