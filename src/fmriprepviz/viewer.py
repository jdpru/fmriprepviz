"""Write a self-contained HTML page for stepping through QC frames."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path
from string import Template

DATA_URI_PREFIX = "data:image/png;base64,"
DEFAULT_FPS = 2

_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
body { background: #111; color: #eee; font-family: system-ui, sans-serif;
       margin: 0; padding: 1rem; }
#frame { display: block; max-width: 100%; margin: 0 auto; image-rendering: auto; }
#label { text-align: center; margin-top: .5rem; font-variant-numeric: tabular-nums; }
#controls { display: flex; gap: .5rem; align-items: center;
            justify-content: center; margin-top: .75rem; }
button { background: #333; color: #eee; border: 1px solid #555;
         border-radius: 4px; padding: .35rem .8rem; cursor: pointer; }
button:hover { background: #444; }
input[type="range"] { flex: 1 1 auto; max-width: 40rem; }
</style>
</head>
<body>
<img id="frame" alt="registration QC frame">
<div id="label"></div>
<div id="controls">
  <button id="prev" title="Previous frame (left arrow)">&#9664;</button>
  <button id="play" title="Play or pause">Play</button>
  <button id="next" title="Next frame (right arrow)">&#9654;</button>
  <input id="slider" type="range" min="0" value="0">
</div>
<script>
const FRAMES = $frames_json;
const LABELS = $labels_json;
const INTERVAL_MS = $interval_ms;

const img = document.getElementById("frame");
const label = document.getElementById("label");
const slider = document.getElementById("slider");
const playButton = document.getElementById("play");
let index = 0;
let timer = null;

slider.max = FRAMES.length - 1;

function show(i) {
  index = (i + FRAMES.length) % FRAMES.length;
  img.src = FRAMES[index];
  label.textContent = (index + 1) + " / " + FRAMES.length + "  " + LABELS[index];
  slider.value = index;
}

function stop() {
  clearInterval(timer);
  timer = null;
  playButton.textContent = "Play";
}

function play() {
  timer = setInterval(function () { show(index + 1); }, INTERVAL_MS);
  playButton.textContent = "Pause";
}

playButton.addEventListener("click", function () { timer ? stop() : play(); });
document.getElementById("prev").addEventListener("click", function () {
  stop(); show(index - 1);
});
document.getElementById("next").addEventListener("click", function () {
  stop(); show(index + 1);
});
slider.addEventListener("input", function () {
  stop(); show(Number(slider.value));
});
document.addEventListener("keydown", function (event) {
  if (event.key === "ArrowLeft") { stop(); show(index - 1); }
  if (event.key === "ArrowRight") { stop(); show(index + 1); }
});

show(0);
</script>
</body>
</html>
"""
)


def write_html(
    frames: Sequence[bytes],
    labels: Sequence[str],
    path,
    fps: int = DEFAULT_FPS,
    title: str = "fmriprepviz",
) -> Path:
    """Write one HTML file with every frame embedded and step controls."""
    if len(frames) != len(labels):
        raise ValueError(
            f"Got {len(frames)} frames but {len(labels)} labels; they must match."
        )
    document = _TEMPLATE.substitute(
        title=title,
        frames_json=json.dumps([_data_uri(frame) for frame in frames]),
        labels_json=json.dumps(list(labels)),
        interval_ms=int(round(1000 / fps)),
    )
    path = Path(path)
    path.write_text(document, encoding="utf-8")
    return path


def _data_uri(png_bytes: bytes) -> str:
    return DATA_URI_PREFIX + base64.b64encode(png_bytes).decode("ascii")
