"""Corpus manifest for dataset provenance and audit trail.

Records every source audio file's origin URL, license, collector, and
classification so the dataset is reproducible and auditable.  This matches
the discipline already used for ``ScenarioDefinition`` / ``dataset_manifest.json``
and directly feeds the dataset-composition section of the Phase G final report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Required fields for provenance validation
_REQUIRED_PROVENANCE_FIELDS = frozenset({
    "source_url",
    "license",
    "collector",
})


@dataclass
class CorpusEntry:
    """One audio file's provenance record.

    Attributes
    ----------
    file_path : str
        Relative path to the audio file within the corpus.
    source_url : str
        Origin URL (download page, dataset homepage, or freesound link).
    license : str
        SPDX identifier or full license name (e.g. ``"CC-BY-4.0"``).
    collector : str
        Person or tool that obtained this file.
    dataset_name : str
        Name of the parent dataset (e.g. ``"librispeech"``, ``"musan"``).
    noise_family : str | None
        Noise taxonomy category, if applicable (speech entries are None).
    speaker_id : str | None
        Speaker identity, if applicable (noise entries are None).
    duration_seconds : float | None
        Duration of the audio clip, if known.
    extra : dict
        Any additional metadata.
    """

    file_path: str
    source_url: str
    license: str
    collector: str
    dataset_name: str
    noise_family: str | None = None
    speaker_id: str | None = None
    duration_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusManifest:
    """Collection of provenance entries for the entire corpus.

    Attributes
    ----------
    entries : list[CorpusEntry]
        All source files in the corpus.
    version : int
        Schema version for forward compatibility.
    description : str
        Human-readable description of the manifest.
    """

    entries: list[CorpusEntry] = field(default_factory=list)
    version: int = 1
    description: str = "PS26052 ANC corpus provenance manifest"

    def add_entry(self, entry: CorpusEntry) -> None:
        """Add a single entry to the manifest."""
        self.entries.append(entry)

    def add_entries(self, entries: list[CorpusEntry]) -> None:
        """Add multiple entries to the manifest."""
        self.entries.extend(entries)

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    @property
    def datasets(self) -> set[str]:
        """Unique dataset names in the manifest."""
        return {e.dataset_name for e in self.entries}

    @property
    def noise_families(self) -> set[str]:
        """Unique noise families (excluding None)."""
        return {e.noise_family for e in self.entries if e.noise_family is not None}

    @property
    def speakers(self) -> set[str]:
        """Unique speaker IDs (excluding None)."""
        return {e.speaker_id for e in self.entries if e.speaker_id is not None}

    def entries_by_noise_family(self) -> dict[str, list[CorpusEntry]]:
        """Group entries by noise family."""
        groups: dict[str, list[CorpusEntry]] = {}
        for entry in self.entries:
            if entry.noise_family is not None:
                groups.setdefault(entry.noise_family, []).append(entry)
        return groups


def validate_manifest(manifest: CorpusManifest) -> list[str]:
    """Validate a manifest for completeness.

    Returns a list of validation error messages.  An empty list means the
    manifest is valid.
    """
    errors: list[str] = []

    if not manifest.entries:
        errors.append("Manifest has no entries.")
        return errors

    for i, entry in enumerate(manifest.entries):
        prefix = f"Entry {i} ({entry.file_path})"

        if not entry.file_path or not entry.file_path.strip():
            errors.append(f"{prefix}: file_path is empty.")

        for req_field in _REQUIRED_PROVENANCE_FIELDS:
            value = getattr(entry, req_field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"{prefix}: missing required field '{req_field}'.")

    return errors


def write_manifest(manifest: CorpusManifest, path: str | Path) -> Path:
    """Serialize a manifest to JSON.

    Parameters
    ----------
    manifest : CorpusManifest
        The manifest to write.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The written file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "format": "ps26052-corpus-manifest",
        "version": manifest.version,
        "description": manifest.description,
        "num_entries": manifest.num_entries,
        "datasets": sorted(manifest.datasets),
        "noise_families": sorted(manifest.noise_families),
        "speakers": sorted(manifest.speakers),
        "entries": [asdict(e) for e in manifest.entries],
    }

    out.write_text(
        json.dumps(data, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    return out


def load_manifest(path: str | Path) -> CorpusManifest:
    """Load a manifest from JSON.

    Parameters
    ----------
    path : str or Path
        Path to the manifest JSON file.

    Returns
    -------
    CorpusManifest
        The loaded manifest.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if data.get("format") != "ps26052-corpus-manifest":
        raise ValueError(f"Unsupported manifest format: {data.get('format')}")

    manifest = CorpusManifest(
        version=data.get("version", 1),
        description=data.get("description", ""),
    )

    for entry_dict in data.get("entries", []):
        manifest.add_entry(CorpusEntry(
            file_path=entry_dict["file_path"],
            source_url=entry_dict["source_url"],
            license=entry_dict["license"],
            collector=entry_dict["collector"],
            dataset_name=entry_dict["dataset_name"],
            noise_family=entry_dict.get("noise_family"),
            speaker_id=entry_dict.get("speaker_id"),
            duration_seconds=entry_dict.get("duration_seconds"),
            extra=entry_dict.get("extra", {}),
        ))

    return manifest
