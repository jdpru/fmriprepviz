# fmriprepviz Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a command-line tool that turns the `space-T1w` BOLD reference images of one fMRIPrep subject into a flipbook movie with the FreeSurfer white and pial surfaces overlaid in red, so cross-session registration quality can be judged by eye.

**Architecture:** A linear pipeline of six small, independently testable modules. `discovery` finds and orders the boldref files; `ribbon` turns FreeSurfer's `ribbon.mgz` into two filled binary masks; `cuts` picks a fixed set of anatomical cut coordinates once; `render` draws one frame as PNG bytes; `movie` and `viewer` package the frames. `cli` wires them together. The masks and cut coordinates are computed once and reused for every frame — that invariant is what makes the movie a valid registration test.

**Tech Stack:** Python 3.11+, nibabel, nilearn, matplotlib (Agg), numpy, Pillow; pytest for tests; uv for package management; hatchling for packaging.

**Spec:** `docs/superpowers/specs/2026-08-16-fmriprepviz-session-registration-movie-design.md`

## Global Constraints

These apply to every task. Each task's requirements implicitly include this section.

- Package management is `uv`. Every local command runs through `uv run`. Never call `pip` or a bare `python`.
- `src/fmriprepviz/__init__.py` must remain completely empty — no code, no imports, no docstring. This is non-negotiable.
- Test-driven development, strictly. The failing test is written and committed **before** the implementation that satisfies it. Never write implementation first, never skip watching the test fail.
- Never weaken a test to make it pass. A test only changes when a requirement changed or the test itself was wrong.
- Tests are functions, not classes. Persistent objects come from pytest fixtures.
- Only `space-T1w` boldrefs are supported. Other spaces are out of scope.
- Python `>=3.11`. Runtime dependencies exactly: `nibabel>=5.2`, `nilearn>=0.11`, `matplotlib>=3.8`, `numpy>=1.26`, `pillow>=10.0`. Dev dependency: `pytest>=8.0`.
- Matplotlib must use the `Agg` backend, set before `pyplot` is imported, so rendering works headless.
- Frames must all have identical pixel dimensions. Never pass `bbox_inches="tight"` to `savefig` — it produces variable-size output and breaks GIF and MP4 assembly.

---

### Task 1: Project scaffolding and synthetic test dataset

The whole test suite runs against a synthetic fMRIPrep derivatives tree whose geometry we control exactly. Because the ribbon is built from concentric spheres of known radius, later tasks can assert that contours land in the *right place*, not merely that an image was produced.

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/fmriprepviz/__init__.py` (empty)
- Create: `tests/conftest.py`
- Test: `tests/test_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces: pytest fixtures `derivatives` (session-scoped `Path` to a synthetic derivatives root) and `derivatives_without_t1w` (session-scoped `Path` to a tree containing only MNI-space boldrefs). Module constants `WHITE_RADIUS_MM = 30.0`, `PIAL_RADIUS_MM = 40.0`, `BOLDREF_ENTITIES`.

- [ ] **Step 1: Create the project files**

`pyproject.toml`:

```toml
[project]
name = "fmriprepviz"
version = "0.1.0"
description = "Cross-session BOLD registration QC movies from fMRIPrep output"
requires-python = ">=3.11"
dependencies = [
    "nibabel>=5.2",
    "nilearn>=0.11",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "pillow>=10.0",
]

[project.scripts]
fmriprepviz = "fmriprepviz.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fmriprepviz"]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:

```
__pycache__/
*.py[cod]
.venv/
dist/
build/
*.egg-info/
.pytest_cache/
*.gif
*.mp4
*.html
```

Create `src/fmriprepviz/__init__.py` as a genuinely empty file:

```bash
mkdir -p src/fmriprepviz tests
touch src/fmriprepviz/__init__.py
```

- [ ] **Step 2: Install the environment**

Run: `uv sync`
Expected: a `.venv` is created and the dependencies resolve. Note that the `[project.scripts]` entry point references `fmriprepviz.cli`, which does not exist yet; that is fine, `uv sync` does not import it.

- [ ] **Step 3: Write the synthetic dataset fixture**

Create `tests/conftest.py`:

```python
"""Synthetic fMRIPrep derivatives tree shared by the whole test suite.

The ribbon is built from concentric spheres of known radius, so tests can
assert that derived surfaces land at the correct anatomical location rather
than merely checking that an image was produced.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

RIBBON_SHAPE = (128, 128, 128)
RIBBON_VOXEL_SIZE = 1.0
RIBBON_ORIGIN = -64.0

BOLDREF_SHAPE = (48, 48, 48)
BOLDREF_VOXEL_SIZE = 3.0
BOLDREF_ORIGIN = -72.0

WHITE_RADIUS_MM = 30.0
PIAL_RADIUS_MM = 40.0

# (session, task, run) for each space-T1w boldref in the synthetic tree.
# run-10 is present so ordering tests can catch naive string sorting.
BOLDREF_ENTITIES = [
    ("01", "rest", "1"),
    ("01", "rest", "2"),
    ("01", "rest", "10"),
    ("02", "rest", "1"),
    ("02", "nback", "1"),
]

MNI_BOLDREF_NAME = (
    "sub-01_ses-01_task-rest_run-1"
    "_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz"
)


def _affine(voxel_size: float, origin: float) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = np.diag([voxel_size] * 3)
    affine[:3, 3] = origin
    return affine


def _world_grids(shape, voxel_size, origin):
    axes = [np.arange(n) * voxel_size + origin for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def _world_radius(shape, voxel_size, origin):
    grids = _world_grids(shape, voxel_size, origin)
    return np.sqrt(sum(g**2 for g in grids)), grids


def _ribbon_data() -> np.ndarray:
    """Concentric spheres labelled with real FreeSurfer ribbon values."""
    radius, grids = _world_radius(RIBBON_SHAPE, RIBBON_VOXEL_SIZE, RIBBON_ORIGIN)
    left = grids[0] < 0
    white = radius < WHITE_RADIUS_MM
    gray = (radius >= WHITE_RADIUS_MM) & (radius < PIAL_RADIUS_MM)

    data = np.zeros(RIBBON_SHAPE, dtype=np.uint8)
    data[white & left] = 2
    data[white & ~left] = 41
    data[gray & left] = 3
    data[gray & ~left] = 42
    return data


def _boldref_data(seed: int) -> np.ndarray:
    radius, _ = _world_radius(BOLDREF_SHAPE, BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
    rng = np.random.default_rng(seed)
    data = np.where(radius < PIAL_RADIUS_MM, 1000.0, 0.0)
    return (data + rng.normal(0.0, 20.0, BOLDREF_SHAPE)).astype(np.float32)


def _write_boldrefs(root: Path) -> None:
    for index, (ses, task, run) in enumerate(BOLDREF_ENTITIES):
        func = root / "sub-01" / f"ses-{ses}" / "func"
        func.mkdir(parents=True, exist_ok=True)
        name = (
            f"sub-01_ses-{ses}_task-{task}_run-{run}"
            "_space-T1w_boldref.nii.gz"
        )
        image = nib.Nifti1Image(
            _boldref_data(index), _affine(BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
        )
        image.to_filename(func / name)


def _write_mni_boldref(root: Path) -> None:
    """A non-T1w boldref, so error messages have another space to report."""
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        _boldref_data(99), _affine(BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
    )
    image.to_filename(func / MNI_BOLDREF_NAME)


def _write_ribbon(root: Path) -> None:
    mri = root / "sourcedata" / "freesurfer" / "sub-01" / "mri"
    mri.mkdir(parents=True, exist_ok=True)
    ribbon = nib.MGHImage(
        _ribbon_data(), _affine(RIBBON_VOXEL_SIZE, RIBBON_ORIGIN)
    )
    ribbon.to_filename(mri / "ribbon.mgz")


@pytest.fixture(scope="session")
def derivatives(tmp_path_factory) -> Path:
    """A complete synthetic derivatives tree: boldrefs plus a FreeSurfer dir."""
    root = tmp_path_factory.mktemp("derivatives")
    _write_boldrefs(root)
    _write_mni_boldref(root)
    _write_ribbon(root)
    return root


@pytest.fixture(scope="session")
def derivatives_without_t1w(tmp_path_factory) -> Path:
    """A tree fMRIPrep produced without T1w among its output spaces."""
    root = tmp_path_factory.mktemp("mni_only")
    _write_mni_boldref(root)
    _write_ribbon(root)
    return root
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_fixtures.py`:

```python
"""The synthetic dataset is the foundation of every other test, so its own
geometry is verified first."""

import nibabel as nib
import numpy as np

from conftest import (
    BOLDREF_ENTITIES,
    PIAL_RADIUS_MM,
    RIBBON_ORIGIN,
    RIBBON_VOXEL_SIZE,
    WHITE_RADIUS_MM,
)


def _voxel_radii(mask, affine):
    ijk = np.argwhere(mask)
    homogeneous = np.column_stack([ijk, np.ones(len(ijk))])
    world = (affine @ homogeneous.T).T[:, :3]
    return np.linalg.norm(world, axis=1)


def test_tree_contains_one_t1w_boldref_per_entity_set(derivatives):
    found = sorted((derivatives / "sub-01").glob("**/func/*_space-T1w_boldref.nii.gz"))
    assert len(found) == len(BOLDREF_ENTITIES)


def test_tree_contains_a_non_t1w_boldref(derivatives):
    found = list((derivatives / "sub-01").glob("**/func/*space-MNI*_boldref.nii.gz"))
    assert len(found) == 1


def test_ribbon_uses_real_freesurfer_labels(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    labels = set(np.unique(np.asarray(ribbon.dataobj)).tolist())
    assert labels == {0, 2, 3, 41, 42}


def test_white_labels_fill_a_sphere_of_the_expected_radius(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    data = np.asarray(ribbon.dataobj)
    radii = _voxel_radii(np.isin(data, (2, 41)), ribbon.affine)
    assert radii.max() < WHITE_RADIUS_MM
    assert radii.max() > WHITE_RADIUS_MM - 2 * RIBBON_VOXEL_SIZE


def test_gray_labels_extend_to_the_pial_radius(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    data = np.asarray(ribbon.dataobj)
    radii = _voxel_radii(np.isin(data, (3, 42)), ribbon.affine)
    assert radii.max() < PIAL_RADIUS_MM
    assert radii.max() > PIAL_RADIUS_MM - 2 * RIBBON_VOXEL_SIZE
    assert radii.min() < WHITE_RADIUS_MM + 2 * RIBBON_VOXEL_SIZE


def test_ribbon_origin_places_the_sphere_centre_at_the_world_origin(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    assert np.allclose(ribbon.affine[:3, 3], RIBBON_ORIGIN)
```

Note: importing from `conftest` directly requires `tests/` to be on `sys.path`, which pytest's default `rootdir`-based `sys.path` insertion provides because `tests/` contains no `__init__.py`. Do not add one.

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'conftest'`, or assertion failures, depending on what is present. The tests must fail before `conftest.py` is complete.

If you wrote `conftest.py` in Step 3 before the test, that is the correct order here: the fixture is test infrastructure, not implementation. The tests in Step 4 are what must be seen failing, so if they pass immediately, deliberately break one assertion, confirm it fails, then restore it.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: PASS, 6 tests.

If the radius assertions fail, the likely cause is that `nib.MGHImage` rewrote the affine. Print `ribbon.affine` and compare against `_affine(1.0, -64.0)`; MGH stores the affine through its own header fields, and a non-diagonal or shifted result means the header round-trip needs an explicit `nib.MGHImage(data, affine)` construction rather than mutating an existing header.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore uv.lock src/fmriprepviz/__init__.py tests/conftest.py tests/test_fixtures.py
git commit -m "test: add project scaffolding and synthetic fMRIPrep fixture"
```

---

### Task 2: Boldref discovery

**Files:**
- Create: `src/fmriprepviz/discovery.py`
- Test: `tests/test_discovery.py`

**Interfaces:**
- Consumes: fixtures `derivatives`, `derivatives_without_t1w` from Task 1.
- Produces:
  - `BoldRef` frozen dataclass with fields `path: Path`, `entities: dict[str, str]`, and property `label: str`
  - `find_boldrefs(derivatives_root, subject, ses=None, task=None, run=None) -> list[BoldRef]`
  - `parse_entities(name: str) -> dict[str, str]`
  - `NoBoldRefsError(RuntimeError)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_discovery.py`:

```python
import pytest

from fmriprepviz.discovery import NoBoldRefsError, find_boldrefs, parse_entities


def test_parses_bids_entities_from_a_filename():
    entities = parse_entities(
        "sub-01_ses-02_task-rest_run-10_space-T1w_boldref.nii.gz"
    )
    assert entities["sub"] == "01"
    assert entities["ses"] == "02"
    assert entities["task"] == "rest"
    assert entities["run"] == "10"
    assert entities["space"] == "T1w"


def test_finds_every_t1w_boldref(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert len(refs) == 5


def test_ignores_boldrefs_in_other_spaces(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert all(ref.entities["space"] == "T1w" for ref in refs)


def test_orders_by_session_then_task_then_run_numerically(derivatives):
    refs = find_boldrefs(derivatives, "01")
    ordered = [(r.entities["ses"], r.entities["task"], r.entities["run"]) for r in refs]
    assert ordered == [
        ("01", "rest", "1"),
        ("01", "rest", "2"),
        ("01", "rest", "10"),
        ("02", "nback", "1"),
        ("02", "rest", "1"),
    ]


def test_label_reads_as_human_readable_entities(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert refs[0].label == "ses-01 task-rest run-1"


def test_filters_by_session(derivatives):
    refs = find_boldrefs(derivatives, "01", ses=["02"])
    assert len(refs) == 2
    assert all(ref.entities["ses"] == "02" for ref in refs)


def test_filters_by_task(derivatives):
    refs = find_boldrefs(derivatives, "01", task=["nback"])
    assert len(refs) == 1


def test_filters_by_run(derivatives):
    refs = find_boldrefs(derivatives, "01", run=["1"])
    assert len(refs) == 3


def test_filters_combine(derivatives):
    refs = find_boldrefs(derivatives, "01", ses=["01"], run=["2"])
    assert len(refs) == 1
    assert refs[0].entities["run"] == "2"


def test_missing_t1w_output_names_the_spaces_that_are_present(
    derivatives_without_t1w,
):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives_without_t1w, "01")
    message = str(excinfo.value)
    assert "MNI152NLin2009cAsym" in message
    assert "T1w" in message


def test_unknown_subject_reports_the_directory_searched(derivatives):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives, "99")
    assert "sub-99" in str(excinfo.value)


def test_over_restrictive_filters_report_the_filters(derivatives):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives, "01", task=["nosuchtask"])
    assert "nosuchtask" in str(excinfo.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmriprepviz.discovery'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_discovery.py
git commit -m "test: specify boldref discovery, ordering and filtering"
```

- [ ] **Step 4: Write the implementation**

Create `src/fmriprepviz/discovery.py`:

```python
"""Locate and order the space-T1w BOLD reference images of one subject."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BOLDREF_SUFFIX = "_space-T1w_boldref.nii.gz"
ANY_BOLDREF_GLOB = "**/func/*_boldref.nii.gz"
LABEL_ENTITIES = ("ses", "task", "acq", "run")
SORT_ENTITIES = ("ses", "task", "acq", "run")

_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z0-9]+)-(?P<value>[a-zA-Z0-9]+)")


class NoBoldRefsError(RuntimeError):
    """No usable BOLD reference images were found."""


@dataclass(frozen=True)
class BoldRef:
    """One BOLD reference image and the BIDS entities of its filename."""

    path: Path
    entities: dict[str, str]

    @property
    def label(self) -> str:
        present = [k for k in LABEL_ENTITIES if k in self.entities]
        return " ".join(f"{k}-{self.entities[k]}" for k in present)


def parse_entities(name: str) -> dict[str, str]:
    """Extract BIDS key-value entities from a filename."""
    return {m["key"]: m["value"] for m in _ENTITY_RE.finditer(name)}


def find_boldrefs(
    derivatives_root,
    subject: str,
    ses: Sequence[str] | None = None,
    task: Sequence[str] | None = None,
    run: Sequence[str] | None = None,
) -> list[BoldRef]:
    """Return this subject's space-T1w boldrefs, ordered for playback."""
    subject_dir = Path(derivatives_root) / f"sub-{subject}"
    found = [
        BoldRef(path, parse_entities(path.name))
        for path in subject_dir.glob(f"**/func/*{BOLDREF_SUFFIX}")
    ]
    filters = {"ses": ses, "task": task, "run": run}
    kept = [ref for ref in found if _matches(ref, filters)]
    if not kept:
        raise NoBoldRefsError(_explain_empty(subject_dir, found, filters))
    return sorted(kept, key=_sort_key)


def _matches(ref: BoldRef, filters: dict[str, Sequence[str] | None]) -> bool:
    return all(
        wanted is None or ref.entities.get(key) in set(wanted)
        for key, wanted in filters.items()
    )


def _sort_key(ref: BoldRef) -> tuple:
    return tuple(_natural(ref.entities.get(key, "")) for key in SORT_ENTITIES)


def _natural(value: str) -> tuple[int, int, str]:
    """Sort numeric entity values numerically, so run-2 precedes run-10."""
    if value.isdigit():
        return (0, int(value), "")
    return (1, 0, value)


def _explain_empty(subject_dir: Path, found, filters) -> str:
    if found:
        active = {k: list(v) for k, v in filters.items() if v is not None}
        return (
            f"No space-T1w boldref under {subject_dir} matched the requested "
            f"filters {active}. Relax or remove them."
        )
    spaces = _spaces_present(subject_dir)
    if spaces:
        return (
            f"No space-T1w boldref found under {subject_dir}. The spaces "
            f"present are: {', '.join(spaces)}. Re-run fMRIPrep with T1w "
            "among its --output-spaces to produce the images this tool needs."
        )
    return (
        f"No boldref images found under {subject_dir}. Check the derivatives "
        "path and the subject label."
    )


def _spaces_present(subject_dir: Path) -> list[str]:
    spaces = {
        entities["space"]
        for path in subject_dir.glob(ANY_BOLDREF_GLOB)
        if "space" in (entities := parse_entities(path.name))
    }
    return sorted(spaces)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_discovery.py -v`
Expected: PASS, 12 tests.

If `test_parses_bids_entities_from_a_filename` fails on `space`, check that the regex is not consuming `T1w` partially — `[a-zA-Z0-9]+` is greedy and correct here. If ordering fails, print `_sort_key` for each ref; the usual cause is comparing tuples whose elements have mixed types.

- [ ] **Step 6: Commit**

```bash
git add src/fmriprepviz/discovery.py
git commit -m "feat: discover and order space-T1w boldrefs"
```

---

### Task 3: FreeSurfer ribbon to surface masks

FreeSurfer's `ribbon.mgz` labels left white matter 2, left cortical gray 3, right white matter 41, right cortical gray 42. The white surface is the boundary of the white-matter volume; the pial surface is the boundary of white plus gray. We produce *filled* binary masks and let nilearn's `add_contours` draw their boundaries at level 0.5 — the contour of a filled mask is exactly the surface we want, so no boundary extraction code is needed.

**Files:**
- Create: `src/fmriprepviz/ribbon.py`
- Modify: `tests/conftest.py` (append the `surface_masks` fixture)
- Test: `tests/test_ribbon.py`

**Interfaces:**
- Consumes: fixture `derivatives` from Task 1.
- Produces:
  - `SurfaceMasks` frozen dataclass with fields `white: nibabel.Nifti1Image`, `pial: nibabel.Nifti1Image`
  - `load_surface_masks(fs_dir) -> SurfaceMasks`
  - `resolve_fs_dir(derivatives_root, subject, fs_dir=None) -> Path`
  - `FreeSurferNotFoundError(RuntimeError)`
  - New session-scoped fixture `surface_masks` returning `SurfaceMasks` for the synthetic subject.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ribbon.py`:

```python
import numpy as np
import pytest

from conftest import PIAL_RADIUS_MM, RIBBON_VOXEL_SIZE, WHITE_RADIUS_MM
from fmriprepviz.ribbon import (
    FreeSurferNotFoundError,
    load_surface_masks,
    resolve_fs_dir,
)

TOLERANCE_MM = 2 * RIBBON_VOXEL_SIZE


def _outer_radius(mask_img):
    ijk = np.argwhere(np.asarray(mask_img.dataobj) > 0)
    homogeneous = np.column_stack([ijk, np.ones(len(ijk))])
    world = (mask_img.affine @ homogeneous.T).T[:, :3]
    return np.linalg.norm(world, axis=1).max()


def test_resolves_the_freesurfer_dir_inside_sourcedata(derivatives):
    resolved = resolve_fs_dir(derivatives, "01")
    assert resolved == derivatives / "sourcedata" / "freesurfer" / "sub-01"


def test_explicit_fs_dir_takes_precedence(derivatives):
    explicit = derivatives / "sourcedata" / "freesurfer" / "sub-01"
    assert resolve_fs_dir(derivatives, "99", fs_dir=explicit) == explicit


def test_falls_back_to_subjects_dir(derivatives, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SUBJECTS_DIR", str(derivatives / "sourcedata" / "freesurfer")
    )
    empty_root = tmp_path / "no_sourcedata"
    empty_root.mkdir()
    resolved = resolve_fs_dir(empty_root, "01")
    assert resolved == derivatives / "sourcedata" / "freesurfer" / "sub-01"


def test_unresolvable_fs_dir_lists_every_path_tried(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        resolve_fs_dir(tmp_path, "01")
    message = str(excinfo.value)
    assert "sourcedata" in message
    assert "ribbon.mgz" in message


def test_white_mask_covers_only_white_matter_labels(surface_masks):
    assert _outer_radius(surface_masks.white) < WHITE_RADIUS_MM


def test_white_mask_reaches_the_white_surface(surface_masks):
    assert _outer_radius(surface_masks.white) > WHITE_RADIUS_MM - TOLERANCE_MM


def test_pial_mask_reaches_the_pial_surface(surface_masks):
    outer = _outer_radius(surface_masks.pial)
    assert outer < PIAL_RADIUS_MM
    assert outer > PIAL_RADIUS_MM - TOLERANCE_MM


def test_pial_mask_is_filled_and_contains_the_white_mask(surface_masks):
    white = np.asarray(surface_masks.white.dataobj) > 0
    pial = np.asarray(surface_masks.pial.dataobj) > 0
    assert np.all(pial[white])
    assert pial.sum() > white.sum()


def test_masks_keep_the_ribbon_affine(derivatives, surface_masks):
    import nibabel as nib

    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    assert np.allclose(surface_masks.white.affine, ribbon.affine)
    assert np.allclose(surface_masks.pial.affine, ribbon.affine)


def test_missing_ribbon_is_reported(tmp_path):
    fs_dir = tmp_path / "sub-01"
    (fs_dir / "mri").mkdir(parents=True)
    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        load_surface_masks(fs_dir)
    assert "ribbon.mgz" in str(excinfo.value)
```

Append to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def surface_masks(derivatives):
    """White and pial masks derived from the synthetic ribbon."""
    from fmriprepviz.ribbon import load_surface_masks, resolve_fs_dir

    return load_surface_masks(resolve_fs_dir(derivatives, "01"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ribbon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmriprepviz.ribbon'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_ribbon.py tests/conftest.py
git commit -m "test: specify ribbon-derived white and pial surface masks"
```

- [ ] **Step 4: Write the implementation**

Create `src/fmriprepviz/ribbon.py`:

```python
"""Derive white and pial surface masks from FreeSurfer's ribbon.mgz.

The masks are filled binary volumes. Their boundaries — the surfaces we
actually want to draw — are produced by contouring them at level 0.5 at
render time, so no boundary extraction happens here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

RIBBON_RELATIVE_PATH = Path("mri") / "ribbon.mgz"
WHITE_LABELS = (2, 41)
GRAY_LABELS = (3, 42)


class FreeSurferNotFoundError(RuntimeError):
    """No FreeSurfer subject directory containing a ribbon could be used."""


@dataclass(frozen=True)
class SurfaceMasks:
    """Filled binary masks whose contours are the white and pial surfaces."""

    white: nib.Nifti1Image
    pial: nib.Nifti1Image


def resolve_fs_dir(derivatives_root, subject: str, fs_dir=None) -> Path:
    """Find the FreeSurfer subject directory holding ribbon.mgz."""
    candidates = _fs_dir_candidates(derivatives_root, subject, fs_dir)
    for candidate in candidates:
        if (candidate / RIBBON_RELATIVE_PATH).is_file():
            return candidate
    tried = "\n  ".join(str(c / RIBBON_RELATIVE_PATH) for c in candidates)
    raise FreeSurferNotFoundError(
        "Could not find a FreeSurfer ribbon.mgz for "
        f"sub-{subject}. Tried:\n  {tried}\n"
        "Pass --fs-dir to point at the subject's FreeSurfer directory."
    )


def _fs_dir_candidates(derivatives_root, subject: str, fs_dir) -> list[Path]:
    if fs_dir is not None:
        return [Path(fs_dir)]
    candidates = [
        Path(derivatives_root) / "sourcedata" / "freesurfer" / f"sub-{subject}"
    ]
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if subjects_dir:
        candidates.append(Path(subjects_dir) / f"sub-{subject}")
    return candidates


def load_surface_masks(fs_dir) -> SurfaceMasks:
    """Build filled white and pial masks from a FreeSurfer ribbon."""
    ribbon_path = Path(fs_dir) / RIBBON_RELATIVE_PATH
    if not ribbon_path.is_file():
        raise FreeSurferNotFoundError(f"Expected a ribbon at {ribbon_path}")
    ribbon = nib.load(ribbon_path)
    data = np.asarray(ribbon.dataobj)
    return SurfaceMasks(
        white=_mask_image(np.isin(data, WHITE_LABELS), ribbon.affine),
        pial=_mask_image(np.isin(data, WHITE_LABELS + GRAY_LABELS), ribbon.affine),
    )


def _mask_image(mask: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(mask.astype(np.uint8), affine)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ribbon.py -v`
Expected: PASS, 10 tests.

If `test_falls_back_to_subjects_dir` fails, confirm the explicit-`fs_dir` branch returns early and does not append the environment candidate.

- [ ] **Step 6: Commit**

```bash
git add src/fmriprepviz/ribbon.py
git commit -m "feat: derive white and pial masks from the FreeSurfer ribbon"
```

---

### Task 4: Cut coordinate selection

**Files:**
- Create: `src/fmriprepviz/cuts.py`
- Test: `tests/test_cuts.py`

**Interfaces:**
- Consumes: fixture `surface_masks` from Task 3.
- Produces: `select_cuts(mask_img, n_cuts=7, display_modes=("x", "y", "z")) -> dict[str, list[float]]`, module constants `DEFAULT_N_CUTS = 7`, `CENTRAL_FRACTION = 0.8`, `AXIS_INDEX`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cuts.py`:

```python
import numpy as np
import pytest

from conftest import PIAL_RADIUS_MM
from fmriprepviz.cuts import CENTRAL_FRACTION, select_cuts


def test_returns_one_entry_per_display_mode(surface_masks):
    cuts = select_cuts(surface_masks.pial)
    assert set(cuts) == {"x", "y", "z"}


def test_honours_the_requested_number_of_cuts(surface_masks):
    cuts = select_cuts(surface_masks.pial, n_cuts=5)
    assert all(len(values) == 5 for values in cuts.values())


def test_honours_the_requested_display_modes(surface_masks):
    cuts = select_cuts(surface_masks.pial, display_modes=("z",))
    assert set(cuts) == {"z"}


def test_cuts_stay_inside_the_mask_extent(surface_masks):
    cuts = select_cuts(surface_masks.pial)
    for values in cuts.values():
        assert min(values) > -PIAL_RADIUS_MM
        assert max(values) < PIAL_RADIUS_MM


def test_cuts_avoid_the_outer_edges_of_the_mask(surface_masks):
    """The outermost slices of a brain are nearly empty and waste frames."""
    cuts = select_cuts(surface_masks.pial)
    margin = (1 - CENTRAL_FRACTION) / 2 * (2 * PIAL_RADIUS_MM)
    for values in cuts.values():
        assert min(values) > -PIAL_RADIUS_MM + margin - 2.0


def test_cuts_of_a_centred_sphere_are_symmetric(surface_masks):
    cuts = select_cuts(surface_masks.pial)
    for values in cuts.values():
        assert pytest.approx(values[0], abs=2.0) == -values[-1]


def test_cuts_are_evenly_spaced(surface_masks):
    cuts = select_cuts(surface_masks.pial, n_cuts=7)
    for values in cuts.values():
        spacing = np.diff(values)
        assert np.allclose(spacing, spacing[0])


def test_selection_is_deterministic(surface_masks):
    assert select_cuts(surface_masks.pial) == select_cuts(surface_masks.pial)


def test_unknown_display_mode_is_rejected(surface_masks):
    with pytest.raises(ValueError, match="w"):
        select_cuts(surface_masks.pial, display_modes=("w",))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cuts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmriprepviz.cuts'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_cuts.py
git commit -m "test: specify fixed cut coordinate selection"
```

- [ ] **Step 4: Write the implementation**

Create `src/fmriprepviz/cuts.py`:

```python
"""Choose the anatomical locations shown identically in every frame.

The cuts derive from the pial mask alone, so they are the same for every
session. That is what makes the flipbook a registration test: anything that
moves between frames is the BOLD, not the display.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
DEFAULT_N_CUTS = 7
CENTRAL_FRACTION = 0.8


def select_cuts(
    mask_img,
    n_cuts: int = DEFAULT_N_CUTS,
    display_modes: Sequence[str] = ("x", "y", "z"),
) -> dict[str, list[float]]:
    """Evenly spaced cut coordinates across the central part of the mask."""
    unknown = [mode for mode in display_modes if mode not in AXIS_INDEX]
    if unknown:
        raise ValueError(
            f"Unknown display mode(s) {unknown}; expected any of "
            f"{sorted(AXIS_INDEX)}"
        )
    world = _world_coordinates(mask_img)
    return {
        mode: _even_cuts(world[:, AXIS_INDEX[mode]], n_cuts)
        for mode in display_modes
    }


def _world_coordinates(mask_img) -> np.ndarray:
    ijk = np.argwhere(np.asarray(mask_img.dataobj) > 0)
    homogeneous = np.column_stack([ijk, np.ones(len(ijk))])
    return (mask_img.affine @ homogeneous.T).T[:, :3]


def _even_cuts(values: np.ndarray, n_cuts: int) -> list[float]:
    low, high = float(values.min()), float(values.max())
    margin = (1.0 - CENTRAL_FRACTION) / 2.0 * (high - low)
    return [float(v) for v in np.linspace(low + margin, high - margin, n_cuts)]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cuts.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 6: Commit**

```bash
git add src/fmriprepviz/cuts.py
git commit -m "feat: select fixed cut coordinates from the pial mask"
```

---

### Task 5: Frame rendering

**Files:**
- Create: `src/fmriprepviz/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `SurfaceMasks` from Task 3, `select_cuts` output from Task 4, fixtures `derivatives` and `surface_masks`.
- Produces: `render_frame(boldref_img, masks, cuts, label, upsample=True) -> bytes` (PNG), module constants `FIGSIZE = (12.0, 6.0)`, `DPI = 100`, `FRAME_SIZE = (1200, 600)`, `CONTOUR_COLOR = "red"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmriprepviz.render'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_render.py
git commit -m "test: specify frame rendering with red surface contours"
```

- [ ] **Step 4: Write the implementation**

Create `src/fmriprepviz/render.py`:

```python
"""Render one BOLD reference image as a labelled QC frame."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nilearn import image, plotting  # noqa: E402

FIGSIZE = (12.0, 6.0)
DPI = 100
FRAME_SIZE = (int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI))
CONTOUR_COLOR = "red"
CONTOUR_LEVELS = [0.5]
CONTOUR_LINEWIDTH = 0.6
WINDOW_PERCENTILES = (2.0, 98.0)


def render_frame(boldref_img, masks, cuts, label: str, upsample: bool = True) -> bytes:
    """Draw one frame and return it as PNG bytes."""
    display_img = _prepare(boldref_img, masks.pial, upsample)
    vmin, vmax = _window(display_img, masks.pial)
    figure, axes = plt.subplots(
        len(cuts), 1, figsize=FIGSIZE, dpi=DPI, facecolor="black"
    )
    try:
        for axis, (mode, coordinates) in zip(np.atleast_1d(axes), cuts.items()):
            _draw_row(figure, axis, display_img, masks, mode, coordinates, vmin, vmax)
        figure.subplots_adjust(left=0, right=1, top=0.94, bottom=0.0, hspace=0.05)
        figure.text(0.01, 0.985, label, color="white", va="top", fontsize=11)
        return _to_png(figure)
    finally:
        plt.close(figure)


def _prepare(boldref_img, reference_img, upsample: bool):
    """Optionally resample the BOLD onto the 1 mm anatomical grid."""
    if not upsample:
        return boldref_img
    return image.resample_to_img(
        boldref_img,
        reference_img,
        interpolation="linear",
        force_resample=True,
        copy_header=True,
    )


def _window(display_img, mask_img) -> tuple[float, float]:
    """Robust intensity limits from inside the brain only.

    Sessions carry arbitrary scanner intensity units, so a shared global
    window would make the movie flicker. Masked robust scaling keeps
    brightness comparable without concealing anything.
    """
    mask = image.resample_to_img(
        mask_img,
        display_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    data = np.asarray(display_img.dataobj, dtype=float)
    inside = data[np.asarray(mask.dataobj) > 0]
    values = inside if inside.size else data.ravel()
    low, high = np.percentile(values, WINDOW_PERCENTILES)
    return float(low), float(high)


def _draw_row(figure, axis, display_img, masks, mode, coordinates, vmin, vmax):
    display = plotting.plot_anat(
        display_img,
        figure=figure,
        axes=axis,
        display_mode=mode,
        cut_coords=coordinates,
        vmin=vmin,
        vmax=vmax,
        black_bg=True,
        annotate=True,
        draw_cross=False,
    )
    for mask in (masks.white, masks.pial):
        display.add_contours(
            mask,
            levels=CONTOUR_LEVELS,
            colors=CONTOUR_COLOR,
            linewidths=CONTOUR_LINEWIDTH,
        )


def _to_png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, facecolor="black")
    return buffer.getvalue()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_render.py -v`
Expected: PASS, 7 tests.

Debugging notes if things fail:
- Wrong frame size almost always means `bbox_inches="tight"` crept into `savefig`. It must not be there.
- If `test_frame_contains_red_contour_pixels` fails, write a frame to disk with `Path("/tmp/frame.png").write_bytes(png)` and look at it. A frame with no red usually means the masks and the boldref do not overlap in world space — print both affines and compare.
- If nilearn raises about `force_resample` or `copy_header`, the installed nilearn is older than 0.11. Check `uv run python -c "import nilearn; print(nilearn.__version__)"` and fix the floor in `pyproject.toml`.
- `plt.subplots(1, 1)` returns a bare Axes rather than an array, which is why `np.atleast_1d` wraps it.

- [ ] **Step 6: Commit**

```bash
git add src/fmriprepviz/render.py
git commit -m "feat: render QC frames with red white and pial contours"
```

---

### Task 6: Movie assembly

**Files:**
- Create: `src/fmriprepviz/movie.py`
- Test: `tests/test_movie.py`

**Interfaces:**
- Consumes: PNG byte strings produced by `render_frame`.
- Produces: `write_movie(frames, path, fps=DEFAULT_FPS)`, `write_gif(frames, path, fps=DEFAULT_FPS)`, `write_mp4(frames, path, fps=DEFAULT_FPS)`, `FFmpegNotFoundError(RuntimeError)`, `DEFAULT_FPS = 2`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_movie.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_movie.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmriprepviz.movie'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_movie.py
git commit -m "test: specify GIF and MP4 assembly"
```

- [ ] **Step 4: Write the implementation**

Create `src/fmriprepviz/movie.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_movie.py -v`
Expected: PASS, with either the ffmpeg test or the no-ffmpeg test skipped depending on the machine.

Note that libx264 requires even pixel dimensions. The real frames are 1200x600, and the test frames 40x30, so both are safe.

- [ ] **Step 6: Commit**

```bash
git add src/fmriprepviz/movie.py
git commit -m "feat: assemble frames into GIF and MP4 movies"
```

---

### Task 7: Self-contained HTML viewer

**Files:**
- Create: `src/fmriprepviz/viewer.py`
- Test: `tests/test_viewer.py`

**Interfaces:**
- Consumes: PNG byte strings and their labels.
- Produces: `write_html(frames, labels, path, fps=2, title="fmriprepviz") -> Path`, `DATA_URI_PREFIX = "data:image/png;base64,"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_viewer.py`:

```python
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
    assert path.read_text().count(DATA_URI_PREFIX) == len(frames)


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_viewer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmriprepviz.viewer'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_viewer.py
git commit -m "test: specify the self-contained HTML frame viewer"
```

- [ ] **Step 4: Write the implementation**

Create `src/fmriprepviz/viewer.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_viewer.py -v`
Expected: PASS, 7 tests.

If `Template.substitute` raises `KeyError` or `ValueError`, some `$` slipped into the CSS or JavaScript. There should be exactly four: `$title`, `$frames_json`, `$labels_json`, `$interval_ms`. Any other literal `$` must be written `$$`.

- [ ] **Step 6: Commit**

```bash
git add src/fmriprepviz/viewer.py
git commit -m "feat: write a self-contained HTML frame viewer"
```

---

### Task 8: Command-line interface

**Files:**
- Create: `src/fmriprepviz/cli.py`
- Create: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2 through 7.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_mp4_output(derivatives, tmp_path):
    output = tmp_path / "qc.mp4"
    main([str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"])
    assert output.is_file()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fmriprepviz.cli'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: specify the fmriprepviz command-line interface"
```

- [ ] **Step 4: Write the implementation**

Create `src/fmriprepviz/cli.py`:

```python
"""Command-line entry point for fmriprepviz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib

from fmriprepviz.cuts import DEFAULT_N_CUTS, select_cuts
from fmriprepviz.discovery import NoBoldRefsError, find_boldrefs
from fmriprepviz.movie import DEFAULT_FPS, FFmpegNotFoundError, write_movie
from fmriprepviz.render import render_frame
from fmriprepviz.ribbon import (
    FreeSurferNotFoundError,
    load_surface_masks,
    resolve_fs_dir,
)
from fmriprepviz.viewer import write_html

DEFAULT_OUTPUT = Path("qc.gif")
DEFAULT_DISPLAY_MODE = "xyz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmriprepviz",
        description=(
            "Build a flipbook movie of a subject's space-T1w BOLD reference "
            "images with the FreeSurfer white and pial surfaces overlaid in "
            "red, for assessing registration across sessions."
        ),
    )
    parser.add_argument("derivatives", type=Path, help="fMRIPrep derivatives root")
    parser.add_argument("--subject", required=True, help="subject label, without 'sub-'")
    parser.add_argument("--ses", nargs="+", help="restrict to these session labels")
    parser.add_argument("--task", nargs="+", help="restrict to these task labels")
    parser.add_argument("--run", nargs="+", help="restrict to these run labels")
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help="output movie; .gif or .mp4 (default: qc.gif)",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS, help="frames per second"
    )
    parser.add_argument(
        "--n-cuts", type=int, default=DEFAULT_N_CUTS,
        help="number of cuts per orientation",
    )
    parser.add_argument(
        "--display-mode", default=DEFAULT_DISPLAY_MODE,
        help="orientations to show, as letters from x, y, z (default: xyz)",
    )
    parser.add_argument(
        "--cut-coords", nargs="+", type=float,
        help="explicit cut coordinates in mm; requires a single --display-mode",
    )
    parser.add_argument(
        "--fs-dir", type=Path,
        help="FreeSurfer subject directory containing mri/ribbon.mgz",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="skip the HTML viewer"
    )
    parser.add_argument(
        "--no-upsample", action="store_true",
        help="show BOLD at its native resolution instead of the 1 mm grid",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (NoBoldRefsError, FreeSurferNotFoundError, FFmpegNotFoundError,
            ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args) -> int:
    refs = find_boldrefs(
        args.derivatives, args.subject, ses=args.ses, task=args.task, run=args.run
    )
    if len(refs) == 1:
        print(
            "warning: only one frame was found, so the movie cannot show "
            "cross-session drift.",
            file=sys.stderr,
        )
    masks = load_surface_masks(
        resolve_fs_dir(args.derivatives, args.subject, args.fs_dir)
    )
    cuts = _resolve_cuts(args, masks)
    frames = _render_frames(refs, masks, cuts, upsample=not args.no_upsample)
    write_movie(frames, args.output, fps=args.fps)
    print(f"wrote {args.output}", file=sys.stderr)
    if not args.no_html:
        html_path = args.output.with_suffix(".html")
        write_html(frames, [ref.label for ref in refs], html_path, fps=args.fps)
        print(f"wrote {html_path}", file=sys.stderr)
    return 0


def _resolve_cuts(args, masks) -> dict[str, list[float]]:
    display_modes = tuple(args.display_mode)
    if args.cut_coords is None:
        return select_cuts(
            masks.pial, n_cuts=args.n_cuts, display_modes=display_modes
        )
    if len(display_modes) != 1:
        raise ValueError(
            "--cut-coords applies to one orientation only; pass a single "
            "letter to --display-mode, for example --display-mode z."
        )
    return {display_modes[0]: list(args.cut_coords)}


def _render_frames(refs, masks, cuts, upsample: bool) -> list[bytes]:
    frames = []
    for index, ref in enumerate(refs, start=1):
        label = f"{index}/{len(refs)}  {ref.label}"
        print(f"rendering {label}", file=sys.stderr)
        frames.append(
            render_frame(
                nib.load(ref.path), masks, cuts, label, upsample=upsample
            )
        )
    return frames
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, with the MP4 test skipped when ffmpeg is absent.

If `test_cut_coords_requires_a_single_orientation` fails with an uncaught exception, confirm `ValueError` is in the `except` tuple in `main`.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -v`
Expected: every test passes. Report the exact count.

- [ ] **Step 7: Write the README**

Create `README.md`:

````markdown
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
````

- [ ] **Step 8: Commit**

```bash
git add src/fmriprepviz/cli.py README.md
git commit -m "feat: add the fmriprepviz command-line interface"
```

---

### Task 9: Smoke run against real data

The synthetic fixtures verify contour *placement* analytically, but only real
data confirms the tool copes with a genuine fMRIPrep tree.

**Files:**
- None created. This task produces observations and, if needed, follow-up fixes.

**Interfaces:**
- Consumes: the finished CLI.
- Produces: a report to the user.

- [ ] **Step 1: Run against the real boldrefs**

The dataset at `/Users/poldrack/data_unsynced/network/bids/derivatives/fmriprep_25.2.4`
has five `space-T1w` boldrefs for `sub-s336` (ses-02, 04, 06, 08, 10, one
`task-stopSignal_run-1` each) but **no FreeSurfer output** anywhere on this
machine. Ask the user for the FreeSurfer subject directory before this step. If
it is still unavailable, run the discovery half only and record that the visual
check is outstanding:

```bash
uv run python -c "
from fmriprepviz.discovery import find_boldrefs
root = '/Users/poldrack/data_unsynced/network/bids/derivatives/fmriprep_25.2.4'
for ref in find_boldrefs(root, 's336'):
    print(ref.label, ref.path.name)
"
```

Expected: five lines, ses-02 through ses-10 in order.

- [ ] **Step 2: Run the full pipeline once a FreeSurfer directory is available**

```bash
uv run fmriprepviz \
  /Users/poldrack/data_unsynced/network/bids/derivatives/fmriprep_25.2.4 \
  --subject s336 --fs-dir /path/to/freesurfer/sub-s336 \
  -o /tmp/s336_qc.gif
```

- [ ] **Step 3: Inspect the result**

Open `/tmp/s336_qc.html`, step through the five frames, and confirm:
- the red contours follow the gray/white and gray/CSF boundaries
- the contours do not move between frames
- left and right are not flipped (compare against the `L`/`R` annotations)
- brightness is comparable across sessions

Report what you observe. Any discrepancy is a bug to investigate with the
systematic-debugging skill, not something to paper over.
