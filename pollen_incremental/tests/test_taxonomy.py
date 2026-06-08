"""Unit tests for pollen_incremental.taxonomy.

Critical invariants tested:
    1. Append-only: existing species IDs never change after extend_state().
    2. Existing order/family/genus IDs are preserved.
    3. Parent links are consistent (re-listing a species cannot rewire its parent).
    4. Round-trip to_dict/from_dict preserves state.
    5. Sizes (n_species, n_familia, ...) match base convention (max_id + 1).
"""

from __future__ import annotations

import pandas as pd
import pytest

from pollen_incremental.taxonomy import (
    TaxonomyDelta,
    TaxonomyState,
    build_base_state,
    extend_state,
)


# =========================
# Fixtures: tiny base + 3 incremental scenarios
# =========================


@pytest.fixture
def base_mapping() -> pd.DataFrame:
    """Mini base: 3 species across 2 orders/2 families/2 genera.

    Tree:
        ord 1 ── fam 1 ── gen 1 ── species 'abel' (id 0)
                               └── species 'bide' (id 1)
        ord 2 ── fam 2 ── gen 2 ── species 'cler' (id 2)
    """
    return pd.DataFrame([
        {"class_name": "abel", "ordor": 1, "familia": 1, "genus": 1},
        {"class_name": "bide", "ordor": 1, "familia": 1, "genus": 1},
        {"class_name": "cler", "ordor": 2, "familia": 2, "genus": 2},
    ])


@pytest.fixture
def base_species2id() -> dict[str, int]:
    """Alphabetical order, like the base trainer's build_species_ids."""
    return {"abel": 0, "bide": 1, "cler": 2}


@pytest.fixture
def base_state(base_mapping: pd.DataFrame, base_species2id: dict[str, int]) -> TaxonomyState:
    return build_base_state(base_mapping, base_species2id)


# =========================
# build_base_state
# =========================


def test_build_base_state_sizes(base_state: TaxonomyState) -> None:
    """Sizes match base convention (max_id + 1)."""
    sizes = base_state.n_classes()
    assert sizes["ordor"] == 3, "ordor head = max(1,2) + 1 = 3"
    assert sizes["familia"] == 3, "familia head = max(1,2) + 1 = 3"
    assert sizes["genus"] == 3, "genus head = max(1,2) + 1 = 3"
    assert sizes["species"] == 3, "3 species → species head = 3"


def test_build_base_state_parent_links(base_state: TaxonomyState) -> None:
    # species 'abel' (id 0) → genus 1
    assert base_state.species_parent[0] == 1
    assert base_state.species_parent[1] == 1
    assert base_state.species_parent[2] == 2
    # genus 1 → familia 1, genus 2 → familia 2
    assert base_state.genus_parent[1] == 1
    assert base_state.genus_parent[2] == 2
    # familia 1 → ordor 1, familia 2 → ordor 2
    assert base_state.familia_parent[1] == 1
    assert base_state.familia_parent[2] == 2


def test_build_base_state_unused_slot(base_state: TaxonomyState) -> None:
    """Slot 0 of familia/genus is not used → parent should be -1 (sentinel)."""
    assert base_state.familia_parent[0] == -1
    assert base_state.genus_parent[0] == -1


def test_build_base_state_handles_order_typo() -> None:
    """Accept both 'order' and 'ordor' column names."""
    df = pd.DataFrame([{"class_name": "abel", "order": 1, "familia": 1, "genus": 1}])
    st = build_base_state(df, {"abel": 0})
    assert 1 in st.ordor_set


# =========================
# extend_state — append-only behavior
# =========================


def test_extend_new_leaf_under_existing_genus(base_state: TaxonomyState) -> None:
    """Add 1 new species under an existing genus. Only species count grows."""
    new_csv = pd.DataFrame([
        {"class_name": "abel", "ordor": 1, "familia": 1, "genus": 1},
        {"class_name": "bide", "ordor": 1, "familia": 1, "genus": 1},
        {"class_name": "cler", "ordor": 2, "familia": 2, "genus": 2},
        # NEW: 'duff' under the existing genus 1
        {"class_name": "duff", "ordor": 1, "familia": 1, "genus": 1},
    ])
    new_state, delta = extend_state(base_state, new_csv)

    # Delta: only species +1
    assert delta.species == 1
    assert delta.order == 0
    assert delta.family == 0
    assert delta.genus == 0

    # Existing species IDs untouched
    assert new_state.species2id["abel"] == 0
    assert new_state.species2id["bide"] == 1
    assert new_state.species2id["cler"] == 2
    # New species gets max + 1
    assert new_state.species2id["duff"] == 3
    assert new_state.species_parent[3] == 1

    # Head sizes
    assert new_state.n_species == 4
    assert new_state.n_ordor == 3  # unchanged


def test_extend_new_branch_with_new_internal_nodes(base_state: TaxonomyState) -> None:
    """Add 1 new species that pulls in a NEW order/family/genus."""
    new_csv = pd.DataFrame([
        # NEW: pulls in order 3, family 3, genus 3
        {"class_name": "evan", "ordor": 3, "familia": 3, "genus": 3},
    ])
    new_state, delta = extend_state(base_state, new_csv)

    assert delta.order == 1
    assert delta.family == 1
    assert delta.genus == 1
    assert delta.species == 1

    # New parent links
    assert new_state.species_parent[3] == 3  # evan → genus 3
    assert new_state.genus_parent[3] == 3
    assert new_state.familia_parent[3] == 3

    # Old species IDs preserved
    assert new_state.species2id["abel"] == 0
    assert new_state.species2id["cler"] == 2

    # Sizes grew correctly
    assert new_state.n_ordor == 4   # max(1,2,3) + 1
    assert new_state.n_familia == 4
    assert new_state.n_genus == 4
    assert new_state.n_species == 4


def test_extend_data_incremental_no_new_classes(base_state: TaxonomyState) -> None:
    """Re-listing the same classes (data-incremental session) → delta is empty."""
    # Same exact CSV as base
    same_csv = pd.DataFrame([
        {"class_name": "abel", "ordor": 1, "familia": 1, "genus": 1},
        {"class_name": "cler", "ordor": 2, "familia": 2, "genus": 2},
    ])
    new_state, delta = extend_state(base_state, same_csv)

    assert delta.is_empty()
    assert new_state.species2id == base_state.species2id
    assert new_state.species_parent == base_state.species_parent


def test_extend_does_not_mutate_input(base_state: TaxonomyState) -> None:
    """extend_state() must not modify the input state in place."""
    snapshot = base_state.to_dict()
    new_csv = pd.DataFrame([{"class_name": "duff", "ordor": 1, "familia": 1, "genus": 1}])
    _ = extend_state(base_state, new_csv)
    # base_state untouched
    assert base_state.to_dict() == snapshot


def test_extend_multiple_sessions_chain(base_state: TaxonomyState) -> None:
    """Three sequential sessions: IDs accumulate, never reshuffle."""
    s1_csv = pd.DataFrame([{"class_name": "duff", "ordor": 1, "familia": 1, "genus": 1}])
    s2_csv = pd.DataFrame([{"class_name": "evan", "ordor": 3, "familia": 3, "genus": 3}])
    s3_csv = pd.DataFrame([{"class_name": "frog", "ordor": 1, "familia": 1, "genus": 1}])

    st = base_state
    for csv in (s1_csv, s2_csv, s3_csv):
        st, _ = extend_state(st, csv)

    # Order of assignment matters: duff = 3, evan = 4, frog = 5
    assert st.species2id["abel"] == 0
    assert st.species2id["bide"] == 1
    assert st.species2id["cler"] == 2
    assert st.species2id["duff"] == 3
    assert st.species2id["evan"] == 4
    assert st.species2id["frog"] == 5

    # All parent links preserved
    assert st.species_parent[0] == 1  # abel
    assert st.species_parent[3] == 1  # duff
    assert st.species_parent[4] == 3  # evan
    assert st.species_parent[5] == 1  # frog


# =========================
# Consistency checks
# =========================


def test_extend_rejects_species_re_parented(base_state: TaxonomyState) -> None:
    """Re-listing 'abel' under a DIFFERENT genus (with consistent upper levels) must raise.

    'abel' was originally under genus 1. Here we keep familia 1 → ordor 1 (so upper
    levels don't trip first), but change abel's genus to 2 (which already exists
    under familia 2 → ordor 2). This forces the species-level conflict to fire.

    To avoid an upstream familia/genus conflict, we use a brand-new familia/genus
    that we re-parent 'abel' onto.
    """
    bad_csv = pd.DataFrame([
        # Set up a fresh genus 99 under fresh familia 99 under ordor 1 (consistent),
        # then claim abel belongs to genus 99 — but abel already belongs to genus 1.
        {"class_name": "abel", "ordor": 1, "familia": 99, "genus": 99},
    ])
    with pytest.raises(ValueError, match=r"Species 'abel'.*re-assigned"):
        extend_state(base_state, bad_csv)


def test_extend_rejects_family_re_parented(base_state: TaxonomyState) -> None:
    """Re-listing familia 1 under a different ordor must raise.

    Use a brand-new species under familia 1 to avoid hitting the species-conflict
    branch first.
    """
    bad_csv = pd.DataFrame([
        {"class_name": "newsp", "ordor": 2, "familia": 1, "genus": 1},  # familia 1 was under ordor 1
    ])
    with pytest.raises(ValueError, match=r"Family 1.*re-assigned"):
        extend_state(base_state, bad_csv)


def test_extend_rejects_genus_re_parented(base_state: TaxonomyState) -> None:
    """Re-listing genus 1 under a different familia must raise.

    Use a brand-new species to avoid species-conflict; keep the new familia
    consistent (under ordor 1, same as familia 1's existing parent) so we
    isolate the genus-conflict branch.
    """
    bad_csv = pd.DataFrame([
        # familia 7 is new under ordor 1 (no conflict). genus 1 was under familia 1,
        # here we claim genus 1 belongs to familia 7 → genus conflict.
        {"class_name": "newsp", "ordor": 1, "familia": 7, "genus": 1},
    ])
    with pytest.raises(ValueError, match=r"Genus 1.*re-assigned"):
        extend_state(base_state, bad_csv)


# =========================
# Serialization
# =========================


def test_round_trip_dict(base_state: TaxonomyState) -> None:
    """to_dict / from_dict round-trip preserves state."""
    d = base_state.to_dict()
    restored = TaxonomyState.from_dict(d)
    assert restored.species2id == base_state.species2id
    assert restored.familia_parent == base_state.familia_parent
    assert restored.genus_parent == base_state.genus_parent
    assert restored.species_parent == base_state.species_parent
    assert restored.ordor_set == base_state.ordor_set
    assert restored.n_classes() == base_state.n_classes()


# =========================
# Schema validation
# =========================


def test_build_rejects_missing_columns() -> None:
    df = pd.DataFrame([{"class_name": "abel", "ordor": 1, "familia": 1}])  # missing 'genus'
    with pytest.raises(ValueError, match="missing columns"):
        build_base_state(df, {"abel": 0})


def test_build_rejects_unknown_species() -> None:
    """species2id has 'abel' but CSV doesn't → should raise."""
    df = pd.DataFrame([{"class_name": "bide", "ordor": 1, "familia": 1, "genus": 1}])
    with pytest.raises(ValueError, match=r"'abel'.*not in mapping CSV"):
        build_base_state(df, {"abel": 0})
