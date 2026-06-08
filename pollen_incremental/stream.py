"""Stream manifest loader.

Reads the `stream_split.json` produced by `scripts/prepare_stream.py` and gives
the training/eval code a typed view of each session: which species are new,
where the data lives, which mapping CSV to feed extend_state(), etc.

This module deliberately has no torch dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Session:
    """One incremental session — base (S₀) or one of S₁…Sₖ."""

    session: int
    label: str
    case: str  # "base" | "new_leaf" | "new_branch"
    new_species: list[str]
    cumulative_species: list[str]
    new_species_ids: dict[str, int]    # species_name → species_id after this session
    species_before: list[str]           # what was visible BEFORE this session
    data_dir: Path                      # streams/session_{k}/  (contains train/val/test/)
    mapping_csv: Path                   # data/mapping_session_{k}.csv

    @property
    def is_base(self) -> bool:
        return self.session == 0

    @property
    def delta_count(self) -> int:
        return len(self.new_species)

    @property
    def cumulative_count(self) -> int:
        return len(self.cumulative_species)

    def train_dir(self) -> Path:
        return self.data_dir / "train"

    def val_dir(self) -> Path:
        return self.data_dir / "val"

    def test_dir(self) -> Path:
        return self.data_dir / "test"


@dataclass(frozen=True)
class Stream:
    """The full sequence of sessions."""

    src_root: Path
    src_csv: Path
    stream_root: Path
    sessions: list[Session]

    def __post_init__(self) -> None:
        if not self.sessions:
            raise ValueError("Stream must contain at least one session")
        if self.sessions[0].session != 0 or not self.sessions[0].is_base:
            raise ValueError("First session must be the base (session=0)")
        # Sessions must be consecutive starting at 0
        for i, s in enumerate(self.sessions):
            if s.session != i:
                raise ValueError(
                    f"Sessions must be consecutive [0..N]; got session id {s.session} at position {i}"
                )

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Session:
        return self.sessions[idx]

    def base(self) -> Session:
        return self.sessions[0]

    def incremental(self) -> list[Session]:
        """All sessions except the base, in order."""
        return self.sessions[1:]


def load_stream(manifest_path: Path | str, mapping_dir: Path | str | None = None) -> Stream:
    """Load the stream manifest written by `scripts/prepare_stream.py`.

    Args:
        manifest_path: path to `stream_split.json`.
        mapping_dir: directory containing `mapping_session_{k}.csv`.
            Defaults to the parent directory of the manifest (where
            prepare_stream.py writes both files).

    Returns:
        Stream object.

    Raises:
        FileNotFoundError: if manifest or any expected mapping CSV is missing.
        ValueError: if the manifest is malformed.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    for key in ("src_root", "src_csv", "stream_root", "sessions"):
        if key not in data:
            raise ValueError(f"Manifest missing required key: {key!r}")

    mapping_dir = Path(mapping_dir) if mapping_dir is not None else manifest_path.parent
    stream_root = Path(data["stream_root"])

    sessions: list[Session] = []
    for s in data["sessions"]:
        sid = int(s["session"])
        mapping_csv = mapping_dir / f"mapping_session_{sid}.csv"
        if not mapping_csv.is_file():
            raise FileNotFoundError(
                f"Mapping CSV for session {sid} not found: {mapping_csv}"
            )
        sessions.append(Session(
            session=sid,
            label=str(s["label"]),
            case=str(s["case"]),
            new_species=list(s["new_species"]),
            cumulative_species=list(s["cumulative_species"]),
            new_species_ids={k: int(v) for k, v in s["new_species_ids"].items()},
            species_before=list(s["species_before"]),
            data_dir=stream_root / f"session_{sid}",
            mapping_csv=mapping_csv,
        ))

    return Stream(
        src_root=Path(data["src_root"]),
        src_csv=Path(data["src_csv"]),
        stream_root=stream_root,
        sessions=sessions,
    )
