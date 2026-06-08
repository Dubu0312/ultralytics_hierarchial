"""Unit tests for pollen_incremental.stream.

Tests cover:
    1. Manifest round-trip (load a fake manifest, get Stream/Session objects back)
    2. Schema validation (missing keys, non-consecutive sessions, missing CSV)
    3. Session helpers (is_base, delta_count, paths)
    4. Stream helpers (base(), incremental())
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pollen_incremental.stream import Session, Stream, load_stream


# =========================
# Fixtures
# =========================


@pytest.fixture
def manifest_dict() -> dict:
    """Minimal but realistic manifest with 3 sessions (base + 2 incremental)."""
    return {
        "src_root": "/tmp/src",
        "src_csv": "/tmp/src/mapping.csv",
        "stream_root": "/tmp/streams",
        "sessions": [
            {
                "session": 0,
                "label": "base",
                "case": "base",
                "new_species": ["abel", "bide"],
                "cumulative_species": ["abel", "bide"],
                "new_species_ids": {"abel": 0, "bide": 1},
                "species_before": [],
                "cumulative_count": 2,
                "delta_count": 2,
            },
            {
                "session": 1,
                "label": "S1_cler",
                "case": "new_leaf",
                "new_species": ["cler"],
                "cumulative_species": ["abel", "bide", "cler"],
                "new_species_ids": {"cler": 2},
                "species_before": ["abel", "bide"],
                "cumulative_count": 3,
                "delta_count": 1,
            },
            {
                "session": 2,
                "label": "S2_duff",
                "case": "new_branch",
                "new_species": ["duff"],
                "cumulative_species": ["abel", "bide", "cler", "duff"],
                "new_species_ids": {"duff": 3},
                "species_before": ["abel", "bide", "cler"],
                "cumulative_count": 4,
                "delta_count": 1,
            },
        ],
    }


@pytest.fixture
def manifest_on_disk(tmp_path: Path, manifest_dict: dict) -> Path:
    """Write the manifest + dummy mapping CSVs to a tmp directory."""
    manifest_path = tmp_path / "stream_split.json"
    manifest_path.write_text(json.dumps(manifest_dict), encoding="utf-8")
    # Stream loader requires mapping_session_{k}.csv files to exist (content not checked).
    for sid in (0, 1, 2):
        (tmp_path / f"mapping_session_{sid}.csv").write_text("class_name,ordor,familia,genus\n")
    return manifest_path


# =========================
# Happy path
# =========================


def test_load_stream_basic(manifest_on_disk: Path) -> None:
    stream = load_stream(manifest_on_disk)
    assert len(stream) == 3
    assert stream.src_root == Path("/tmp/src")
    assert stream.stream_root == Path("/tmp/streams")


def test_session_fields(manifest_on_disk: Path) -> None:
    stream = load_stream(manifest_on_disk)
    s0, s1, s2 = stream[0], stream[1], stream[2]

    # Base session
    assert s0.session == 0
    assert s0.case == "base"
    assert s0.is_base is True
    assert s0.cumulative_count == 2
    assert s0.delta_count == 2

    # Incremental
    assert s1.is_base is False
    assert s1.new_species == ["cler"]
    assert s1.new_species_ids == {"cler": 2}
    assert s1.species_before == ["abel", "bide"]
    assert s1.case == "new_leaf"

    # Branch case
    assert s2.case == "new_branch"


def test_session_paths(manifest_on_disk: Path) -> None:
    stream = load_stream(manifest_on_disk)
    s1 = stream[1]
    assert s1.data_dir == Path("/tmp/streams/session_1")
    assert s1.train_dir() == Path("/tmp/streams/session_1/train")
    assert s1.val_dir() == Path("/tmp/streams/session_1/val")
    assert s1.test_dir() == Path("/tmp/streams/session_1/test")


def test_stream_base_and_incremental(manifest_on_disk: Path) -> None:
    stream = load_stream(manifest_on_disk)
    assert stream.base().session == 0
    inc = stream.incremental()
    assert len(inc) == 2
    assert [s.session for s in inc] == [1, 2]


def test_mapping_csv_paths_resolved(manifest_on_disk: Path) -> None:
    """mapping_csv on each Session should point at the correct CSV file."""
    stream = load_stream(manifest_on_disk)
    for s in stream.sessions:
        assert s.mapping_csv.name == f"mapping_session_{s.session}.csv"
        assert s.mapping_csv.is_file()


def test_load_stream_with_custom_mapping_dir(tmp_path: Path, manifest_dict: dict) -> None:
    """If mapping CSVs are in a different dir than the manifest, the caller can specify."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict), encoding="utf-8")

    csv_dir = tmp_path / "mappings"
    csv_dir.mkdir()
    for sid in (0, 1, 2):
        (csv_dir / f"mapping_session_{sid}.csv").write_text("class_name,ordor,familia,genus\n")

    stream = load_stream(manifest_path, mapping_dir=csv_dir)
    assert stream[1].mapping_csv == csv_dir / "mapping_session_1.csv"


# =========================
# Validation
# =========================


def test_load_stream_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        load_stream(tmp_path / "nonexistent.json")


def test_load_stream_rejects_missing_csv(tmp_path: Path, manifest_dict: dict) -> None:
    """Manifest references mapping_session_{k}.csv that doesn't exist."""
    manifest_path = tmp_path / "stream_split.json"
    manifest_path.write_text(json.dumps(manifest_dict), encoding="utf-8")
    # Only create CSV for S0 — S1 and S2 missing
    (tmp_path / "mapping_session_0.csv").write_text("class_name,ordor,familia,genus\n")
    with pytest.raises(FileNotFoundError, match="Mapping CSV.*session 1.*not found"):
        load_stream(manifest_path)


def test_load_stream_rejects_missing_key(tmp_path: Path) -> None:
    bad = {"src_root": "/tmp"}  # missing other required keys
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing required key"):
        load_stream(p)


def test_stream_rejects_non_consecutive_sessions() -> None:
    s0 = Session(
        session=0, label="base", case="base",
        new_species=["a"], cumulative_species=["a"],
        new_species_ids={"a": 0}, species_before=[],
        data_dir=Path("/x/0"), mapping_csv=Path("/x/0.csv"),
    )
    s2 = Session(
        session=2, label="bad", case="new_leaf",  # skipped session 1
        new_species=["b"], cumulative_species=["a", "b"],
        new_species_ids={"b": 1}, species_before=["a"],
        data_dir=Path("/x/2"), mapping_csv=Path("/x/2.csv"),
    )
    with pytest.raises(ValueError, match="consecutive"):
        Stream(
            src_root=Path("/x"), src_csv=Path("/x/m.csv"),
            stream_root=Path("/x"), sessions=[s0, s2],
        )


def test_stream_rejects_no_base() -> None:
    s1 = Session(
        session=1, label="x", case="new_leaf",
        new_species=["a"], cumulative_species=["a"],
        new_species_ids={"a": 0}, species_before=[],
        data_dir=Path("/x/1"), mapping_csv=Path("/x/1.csv"),
    )
    with pytest.raises(ValueError, match="base"):
        Stream(
            src_root=Path("/x"), src_csv=Path("/x/m.csv"),
            stream_root=Path("/x"), sessions=[s1],
        )


def test_stream_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one session"):
        Stream(
            src_root=Path("/x"), src_csv=Path("/x/m.csv"),
            stream_root=Path("/x"), sessions=[],
        )


# =========================
# Integration with real manifest written by prepare_stream.py (if available)
# =========================


def test_real_manifest_loads(tmp_path: Path) -> None:
    """If the real manifest exists (i.e. prepare_stream.py has been run),
    confirm it loads with our schema. Skipped otherwise."""
    real = Path("/home/dubu/manh/lab/ultralytics/data/stream_split.json")
    if not real.is_file():
        pytest.skip("Real manifest not present; run scripts/prepare_stream.py first")

    stream = load_stream(real)
    assert len(stream) == 6, "Expected 6 sessions (S0 + S1..S5)"
    assert stream.base().cumulative_count == 17
    assert stream[-1].cumulative_count == 22  # full dataset by S5
    # Cases should mix new_leaf and new_branch
    cases = [s.case for s in stream.incremental()]
    assert "new_leaf" in cases
    assert "new_branch" in cases
