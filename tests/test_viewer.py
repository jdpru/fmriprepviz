import base64
import io

import pytest
from PIL import Image

from fmriprepviz.viewer import DATA_URI_PREFIX, write_html


def _solid_png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(buffer, format="png")
    return buffer.getvalue()


@pytest.fixture
def frames():
    return [_solid_png(c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]


@pytest.fixture
def labels():
    return ["ses-01 task-rest run-1", "ses-02 task-rest run-1", "ses-03 task-rest run-1"]


def test_writes_a_single_file(frames, labels, tmp_path):
    path = write_html(frames, labels, tmp_path / "qc.html")
    assert path.is_file()


def test_embeds_every_frame(frames, labels, tmp_path):
    path = write_html(frames, labels, tmp_path / "qc.html")
    text = path.read_text()
    assert text.count(DATA_URI_PREFIX) == len(frames)
    # Verify each frame's distinct payload is present (not just count occurrences)
    for frame in frames:
        payload = base64.b64encode(frame).decode("ascii")
        assert payload in text


def test_includes_every_label(frames, labels, tmp_path):
    path = write_html(frames, labels, tmp_path / "qc.html")
    text = path.read_text()
    assert all(label in text for label in labels)


def test_has_step_controls(frames, labels, tmp_path):
    path = write_html(frames, labels, tmp_path / "qc.html")
    text = path.read_text()
    assert 'type="range"' in text
    assert 'id="prev"' in text
    assert 'id="next"' in text
    assert 'id="play"' in text


def test_references_no_external_assets(frames, labels, tmp_path):
    path = write_html(frames, labels, tmp_path / "qc.html")
    text = path.read_text()
    assert "http://" not in text
    assert "https://" not in text


def test_fps_sets_the_playback_interval(frames, labels, tmp_path):
    path = write_html(frames, labels, tmp_path / "qc.html", fps=4)
    assert "const INTERVAL_MS = 250;" in path.read_text()


def test_mismatched_labels_are_rejected(frames, tmp_path):
    with pytest.raises(ValueError, match="labels"):
        write_html(frames, ["only one"], tmp_path / "qc.html")


def test_hostile_inputs_are_escaped(frames, tmp_path):
    """Title and labels with HTML/script syntax must be safely escaped."""
    hostile_title = "</title><script>"
    hostile_label = "</script>"
    labels = [hostile_label, "normal", "frame"]

    path = write_html(frames, labels, tmp_path / "qc.html", title=hostile_title)
    text = path.read_text()

    # Raw hostile sequences should not appear unescaped in HTML markup
    assert "</title><script>" not in text
    # Ensure </script> in labels doesn't break out of script block prematurely
    # (the real closing script tag should come after all JS executes)
    script_idx = text.find("<script>")
    end_script_idx = text.rfind("</script>")
    assert script_idx < end_script_idx
    # The hostile label's </script> should be escaped in the JSON, so shouldn't
    # appear as a bare </script> before the real closing tag
    assert text.rfind("</script>", script_idx, end_script_idx) == -1

    # Verify frames and labels still round-trip correctly
    assert all(base64.b64encode(frame).decode("ascii") in text for frame in frames)
    assert hostile_label in text  # Should appear escaped in JSON string
    assert hostile_title in text  # Should appear escaped in title
