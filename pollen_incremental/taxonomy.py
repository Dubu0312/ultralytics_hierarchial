"""Append-only taxonomy state for incremental learning.

CRITICAL: when a new session adds species (or new internal nodes), the IDs of
*existing* classes MUST NOT change. Otherwise the mapping head_index ↔ class_id
breaks and the model "forgets" every old class — even though the weights are intact.

The base trainer (train_hierarchical.py:78 build_species_ids) sorts class names
alphabetically, so adding a new species would re-index everything. This module
replaces that with an append-only scheme: existing IDs are loaded from the
checkpoint and new classes get max(existing) + 1, +2, ...

Convention (matches base):
    - "ordor", "familia", "genus" IDs are provided by the mapping CSV.
    - Each level uses dense IDs in [0, max_id]. Some slots may be unused (e.g.
      ordor id 0 in the Đồng Văn dataset). We size head outputs as max_id + 1
      to preserve this convention.
    - Species IDs are assigned here (max-existing + 1) and persisted in the
      checkpoint as `species2id`.
    - Parent arrays are 0-padded lists indexed by child ID, value = parent ID
      (or -1 if unknown).

This module deliberately has no torch dependency so it can be unit-tested fast.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# =========================
# Data structures
# =========================


@dataclass
class TaxonomyState:
    """Mutable container for the 4-level taxonomy across sessions.

    Attributes:
        species2id: name (str) -> species ID (int). Append-only.
        ordor_set: set of ordor IDs seen so far.
        familia_set: set of familia IDs seen so far.
        genus_set: set of genus IDs seen so far.
        familia_parent: list, index = familia ID, value = parent ordor ID (or -1).
        genus_parent: list, index = genus ID, value = parent familia ID (or -1).
        species_parent: list, index = species ID, value = parent genus ID (or -1).
    """

    species2id: dict[str, int] = field(default_factory=dict)
    ordor_set: set[int] = field(default_factory=set)
    familia_set: set[int] = field(default_factory=set)
    genus_set: set[int] = field(default_factory=set)
    familia_parent: list[int] = field(default_factory=list)
    genus_parent: list[int] = field(default_factory=list)
    species_parent: list[int] = field(default_factory=list)

    # --- Sizes (head output dimensions) ---
    @property
    def n_ordor(self) -> int:
        """Order head output size = max(ordor IDs) + 1 (matches base convention)."""
        return (max(self.ordor_set) + 1) if self.ordor_set else 0

    @property
    def n_familia(self) -> int:
        return len(self.familia_parent)

    @property
    def n_genus(self) -> int:
        return len(self.genus_parent)

    @property
    def n_species(self) -> int:
        return len(self.species2id)

    def n_classes(self) -> dict[str, int]:
        return {
            "ordor": self.n_ordor,
            "familia": self.n_familia,
            "genus": self.n_genus,
            "species": self.n_species,
        }

    # --- Serialization ---
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for torch.save / json.dump."""
        return {
            "species2id": dict(self.species2id),
            "ordor_set": sorted(self.ordor_set),
            "familia_set": sorted(self.familia_set),
            "genus_set": sorted(self.genus_set),
            "familia_parent": list(self.familia_parent),
            "genus_parent": list(self.genus_parent),
            "species_parent": list(self.species_parent),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaxonomyState":
        return cls(
            species2id=dict(d["species2id"]),
            ordor_set=set(d["ordor_set"]),
            familia_set=set(d["familia_set"]),
            genus_set=set(d["genus_set"]),
            familia_parent=list(d["familia_parent"]),
            genus_parent=list(d["genus_parent"]),
            species_parent=list(d["species_parent"]),
        )


@dataclass
class TaxonomyDelta:
    """How many *new* classes were added at each level in one session."""

    order: int = 0
    family: int = 0
    genus: int = 0
    species: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"order": self.order, "family": self.family, "genus": self.genus, "species": self.species}

    def is_empty(self) -> bool:
        return self.order == 0 and self.family == 0 and self.genus == 0 and self.species == 0


# =========================
# Builders
# =========================


def build_base_state(mapping_df: pd.DataFrame, species2id: dict[str, int]) -> TaxonomyState:
    """Build a TaxonomyState from the base mapping CSV + existing species2id.

    Use this exactly once when packaging the base checkpoint into the new
    incremental format. After that, use `extend_state()` for every session.

    Args:
        mapping_df: DataFrame with columns class_name, ordor, familia, genus.
        species2id: alphabetical mapping from base training (preserve those IDs).

    Returns:
        TaxonomyState with all base IDs populated.
    """
    mapping_df = _normalize_columns(mapping_df)
    _validate_mapping(mapping_df)

    st = TaxonomyState(species2id=dict(species2id))
    class_to_row = {r["class_name"]: r for _, r in mapping_df.iterrows()}

    # Add nodes and parent links for every base species.
    for name, sid in species2id.items():
        if name not in class_to_row:
            raise ValueError(f"Species '{name}' present in species2id but not in mapping CSV")
        r = class_to_row[name]
        o, f, g = int(r["ordor"]), int(r["familia"]), int(r["genus"])
        st.ordor_set.add(o)
        st.familia_set.add(f)
        st.genus_set.add(g)
        _ensure_len(st.familia_parent, f); st.familia_parent[f] = o
        _ensure_len(st.genus_parent, g); st.genus_parent[g] = f
        _ensure_len(st.species_parent, sid); st.species_parent[sid] = g

    return st


def extend_state(
    state: TaxonomyState,
    mapping_df_new: pd.DataFrame,
) -> tuple[TaxonomyState, TaxonomyDelta]:
    """Extend the taxonomy with classes from a new session's mapping CSV.

    APPEND-ONLY guarantees:
        - Existing species IDs are untouched.
        - Existing order/family/genus IDs are untouched.
        - Existing parent links are not rewritten (raises if conflict detected).
        - New species names get IDs = max_existing + 1, +2, ...
        - New order/family/genus IDs are read from the CSV directly; we only
          extend parent arrays to make room and write the parent link.

    Args:
        state: previous-session TaxonomyState (will not be mutated).
        mapping_df_new: DataFrame with the same schema as the base CSV.
                        May contain BOTH existing classes (re-listed) and new ones.

    Returns:
        (new_state, delta) where delta counts *new* IDs added per level.

    Raises:
        ValueError: if a re-listed class has a different parent than before
                    (taxonomy must be consistent across sessions).
    """
    mapping_df_new = _normalize_columns(mapping_df_new)
    _validate_mapping(mapping_df_new)

    new = deepcopy(state)
    delta = TaxonomyDelta()

    for _, r in mapping_df_new.iterrows():
        name = r["class_name"]
        o, f, g = int(r["ordor"]), int(r["familia"]), int(r["genus"])

        # Order
        if o not in new.ordor_set:
            new.ordor_set.add(o)
            delta.order += 1

        # Family: same ID across sessions, but parent may be assigned for the
        # first time (and must not conflict if already set).
        if f not in new.familia_set:
            new.familia_set.add(f)
            delta.family += 1
        _ensure_len(new.familia_parent, f)
        prev = new.familia_parent[f]
        if prev == -1:
            new.familia_parent[f] = o
        elif prev != o:
            raise ValueError(f"Family {f} re-assigned to a different ordor "
                             f"(was {prev}, new CSV says {o})")

        # Genus
        if g not in new.genus_set:
            new.genus_set.add(g)
            delta.genus += 1
        _ensure_len(new.genus_parent, g)
        prev = new.genus_parent[g]
        if prev == -1:
            new.genus_parent[g] = f
        elif prev != f:
            raise ValueError(f"Genus {g} re-assigned to a different familia "
                             f"(was {prev}, new CSV says {f})")

        # Species: new name → new ID = max + 1
        if name not in new.species2id:
            sid = (max(new.species2id.values()) + 1) if new.species2id else 0
            new.species2id[name] = sid
            delta.species += 1
            _ensure_len(new.species_parent, sid)
            new.species_parent[sid] = g
        else:
            # Existing species — its parent must match what we already have.
            sid = new.species2id[name]
            _ensure_len(new.species_parent, sid)
            prev = new.species_parent[sid]
            if prev != -1 and prev != g:
                raise ValueError(f"Species '{name}' (id {sid}) re-assigned to a different genus "
                                 f"(was {prev}, new CSV says {g})")
            if prev == -1:
                new.species_parent[sid] = g

    return new, delta


# =========================
# Helpers
# =========================


def _ensure_len(lst: list[int], idx: int, fill: int = -1) -> None:
    """Pad a list with `fill` until index `idx` is valid."""
    while len(lst) <= idx:
        lst.append(fill)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept both 'order' and the load-bearing typo 'ordor'."""
    df = df.copy()
    if "order" in df.columns and "ordor" not in df.columns:
        df = df.rename(columns={"order": "ordor"})
    return df


def _validate_mapping(df: pd.DataFrame) -> None:
    required = {"class_name", "ordor", "familia", "genus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing columns: {sorted(missing)}")
