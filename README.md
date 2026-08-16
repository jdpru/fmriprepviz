# fmriprepviz

Build a flipbook movie from an fMRIPrep subject's `space-T1w` BOLD reference
images, with the FreeSurfer white and pial surfaces overlaid in red, to assess
BOLD-to-anatomical registration across sessions.

Every frame shows the same anatomical cuts and the same surface contours, so
anything that appears to move as the movie plays is misregistration.

## Requirements

- fMRIPrep run with `T1w` among its `--output-spaces`
- A FreeSurfer subject directory containing `mri/ribbon.mgz`
- `ffmpeg` on `PATH`, only if you want MP4 output

## Install

```bash
uv sync
```

## Use

```bash
uv run fmriprepviz /path/to/derivatives --subject 01 -o qc.gif
```

This writes `qc.gif` and a self-contained `qc.html` viewer with a frame slider,
prev/next buttons, play/pause and arrow-key stepping.

Useful options:

| Option | Purpose |
|---|---|
| `--ses 01 02` | restrict to these sessions |
| `--task rest` | restrict to this task |
| `--run 1` | restrict to this run |
| `--fs-dir PATH` | FreeSurfer subject directory, when it is not under `sourcedata/freesurfer` |
| `--n-cuts 7` | cuts per orientation |
| `--display-mode z` | orientations to show, letters from `x`, `y`, `z` |
| `--cut-coords -20 0 20` | explicit cuts in mm; needs a single `--display-mode` |
| `--fps 2` | playback speed |
| `--no-html` | skip the viewer |
| `--no-upsample` | show BOLD at native resolution |

`-o qc.mp4` produces MP4 instead of GIF.

## Scope

Only `space-T1w` boldrefs are supported. FreeSurfer surfaces exist solely in
the subject's native anatomical space, so template spaces such as
`MNI152NLin2009cAsym` would require warping the surfaces, which this tool does
not do.

## Development

```bash
uv run pytest
```
