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
