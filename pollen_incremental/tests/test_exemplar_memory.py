"""Unit tests for pollen_incremental.exemplar_memory.

Covers:
    1. Herding correctness — running mean of chosen exemplars approximates class mean.
    2. Budget modes (per_class vs total).
    3. Rebalance behavior.
    4. Add/store API.
    5. Serialization round-trip.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pollen_incremental.exemplar_memory import ExemplarMemory


# =========================
# Herding
# =========================


def test_herding_returns_correct_number() -> None:
    torch.manual_seed(0)
    feats = torch.randn(50, 16)
    paths = [f"img_{i}.jpg" for i in range(50)]
    chosen = ExemplarMemory.herding(feats, paths, m=10)
    assert len(chosen) == 10
    # All from input paths
    assert set(chosen).issubset(set(paths))
    # No duplicates
    assert len(set(chosen)) == 10


def test_herding_clips_m_to_n() -> None:
    """If m > N, we just get all N candidates."""
    feats = torch.randn(5, 8)
    paths = [f"p{i}" for i in range(5)]
    chosen = ExemplarMemory.herding(feats, paths, m=100)
    assert len(chosen) == 5
    assert set(chosen) == set(paths)


def test_herding_empty_input() -> None:
    chosen = ExemplarMemory.herding(torch.empty(0, 8), [], m=5)
    assert chosen == []


def test_herding_approximates_class_mean() -> None:
    """The mean of the FIRST few chosen exemplars should be close to the class mean.

    This is the property herding is designed for — much better than random.
    """
    torch.manual_seed(42)
    # Build a class with 100 points around a non-zero mean
    true_mean = torch.tensor([1.0, 2.0, -1.0, 0.5])
    feats = true_mean.unsqueeze(0) + 0.5 * torch.randn(100, 4)
    paths = [f"p{i}" for i in range(100)]

    # Pick just 10 with herding
    chosen = ExemplarMemory.herding(feats, paths, m=10)

    # Reconstruct features of chosen
    idx = [paths.index(p) for p in chosen]
    chosen_feats = F.normalize(feats[idx], dim=1)
    target = F.normalize(feats.mean(dim=0), dim=0)
    chosen_mean = F.normalize(chosen_feats.mean(dim=0), dim=0)

    # Distance should be small; we'll use 0.1 as a generous bound
    dist = (target - chosen_mean).norm().item()
    assert dist < 0.1, f"Herding mean {dist:.3f} from class mean — too far"


def test_herding_validates_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        ExemplarMemory.herding(torch.randn(5, 4), ["a", "b"], m=2)


# =========================
# Budget modes
# =========================


def test_per_class_budget_constant() -> None:
    mem = ExemplarMemory(budget_per_class=20, mode="per_class")
    assert mem._current_budget(1) == 20
    assert mem._current_budget(100) == 20  # unchanged as classes grow


def test_total_budget_divides() -> None:
    mem = ExemplarMemory(mode="total", total_budget=100)
    assert mem._current_budget(1) == 100
    assert mem._current_budget(10) == 10
    assert mem._current_budget(50) == 2
    assert mem._current_budget(1000) == 1   # floored to 1, not 0


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        ExemplarMemory(mode="random")


def test_invalid_budget_raises() -> None:
    with pytest.raises(ValueError, match="budget_per_class must be"):
        ExemplarMemory(budget_per_class=0)


# =========================
# add_class + storage
# =========================


def test_add_class_stores_correct_count_per_class_mode() -> None:
    mem = ExemplarMemory(budget_per_class=3, mode="per_class")
    feats = torch.randn(10, 8)
    paths = [f"p{i}.jpg" for i in range(10)]
    mem.add_class(species_id=5, features=feats, paths=paths, n_classes_now=1)
    assert 5 in mem.store
    assert len(mem.store[5]) == 3


def test_add_class_stores_correct_count_total_mode() -> None:
    mem = ExemplarMemory(mode="total", total_budget=12)
    # 4 classes → 3 per class
    for sid in range(4):
        feats = torch.randn(10, 8)
        paths = [f"p{sid}_{i}.jpg" for i in range(10)]
        mem.add_class(sid, feats, paths, n_classes_now=sid + 1)
    # Sizes will be inconsistent because n_classes_now changes; the LAST add
    # had n=4 → 3 per class. Earlier adds got more.
    # Apply rebalance to force consistency.
    mem.rebalance(n_classes_now=4)
    for sid in range(4):
        assert len(mem.store[sid]) == 3


def test_rebalance_no_op_in_per_class_mode() -> None:
    mem = ExemplarMemory(budget_per_class=5, mode="per_class")
    mem.add_class(0, torch.randn(10, 4), [f"p{i}" for i in range(10)], 1)
    before = dict(mem.store)
    mem.rebalance(n_classes_now=100)
    assert mem.store == before


def test_all_items_flattens() -> None:
    mem = ExemplarMemory(budget_per_class=2, mode="per_class")
    mem.add_class(0, torch.randn(5, 4), [f"a{i}" for i in range(5)], 1)
    mem.add_class(1, torch.randn(5, 4), [f"b{i}" for i in range(5)], 2)
    items = mem.all_items()
    assert len(items) == 4
    sids = {sid for _, sid in items}
    assert sids == {0, 1}


# =========================
# Serialization
# =========================


def test_round_trip_dict() -> None:
    mem = ExemplarMemory(budget_per_class=3, mode="per_class")
    mem.add_class(0, torch.randn(5, 4), [f"a{i}" for i in range(5)], 1)
    mem.add_class(7, torch.randn(5, 4), [f"b{i}" for i in range(5)], 2)

    d = mem.to_dict()
    restored = ExemplarMemory.from_dict(d)
    assert restored.mode == "per_class"
    assert restored.budget_per_class == 3
    assert restored.store.keys() == mem.store.keys()
    for sid in mem.store:
        assert restored.store[sid] == mem.store[sid]


# =========================
# Misc
# =========================


def test_len_and_contains() -> None:
    mem = ExemplarMemory(budget_per_class=2, mode="per_class")
    mem.add_class(3, torch.randn(5, 4), [f"p{i}" for i in range(5)], 1)
    assert len(mem) == 2
    assert 3 in mem
    assert 999 not in mem
