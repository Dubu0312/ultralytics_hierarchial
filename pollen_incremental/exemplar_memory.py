"""Exemplar memory + herding for incremental rehearsal.

Implements iCaRL-style replay: after each session, we keep `m` representative
images per old species. During the next session, those exemplars are mixed
into the training batch alongside new-class data — this is what prevents
catastrophic forgetting beyond what frozen-backbone + KD alone can achieve.

Two budget modes (matching iCaRL):
    - "per_class": fixed `m` exemplars per class (memory grows with #classes).
    - "total": fixed total budget `K`, each class gets K // n_classes (memory
      stays constant, per-class budget shrinks as new classes are added).

Herding (Welling 2009 / Rebuffi 2017): greedily pick exemplars so that their
running mean matches the class mean as closely as possible. Better than random
sampling because the resulting replay set is representative even when small.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


class ExemplarMemory:
    """Persistent store of {species_id → list of image paths} across sessions.

    The herding algorithm requires features for the candidate images — those
    are computed on the fly by the caller (e.g. inside the session loop)
    using the current frozen backbone, and passed to `add_class()`.
    """

    def __init__(
        self,
        budget_per_class: int = 20,
        mode: str = "per_class",
        total_budget: int = 2000,
    ) -> None:
        if mode not in ("per_class", "total"):
            raise ValueError(f"mode must be 'per_class' or 'total', got {mode!r}")
        if budget_per_class < 1:
            raise ValueError(f"budget_per_class must be >= 1, got {budget_per_class}")

        self.mode = mode
        self.budget_per_class = budget_per_class
        self.total_budget = total_budget
        self.store: dict[int, list[str]] = {}

    # =========================
    # Serialization
    # =========================

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "budget_per_class": self.budget_per_class,
            "total_budget": self.total_budget,
            "store": {int(k): list(v) for k, v in self.store.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExemplarMemory":
        m = cls(
            budget_per_class=int(d.get("budget_per_class", 20)),
            mode=str(d.get("mode", "per_class")),
            total_budget=int(d.get("total_budget", 2000)),
        )
        m.store = {int(k): list(v) for k, v in d.get("store", {}).items()}
        return m

    # =========================
    # Budget logic
    # =========================

    def _current_budget(self, n_classes_now: int) -> int:
        """Per-class budget right now, depending on mode."""
        if self.mode == "per_class":
            return self.budget_per_class
        # 'total' mode: divide evenly
        return max(1, self.total_budget // max(1, n_classes_now))

    def all_items(self) -> list[tuple[str, int]]:
        """Flatten store to [(path, species_id), ...] for building replay loader."""
        return [(p, sid) for sid, paths in self.store.items() for p in paths]

    def n_exemplars(self) -> int:
        return sum(len(v) for v in self.store.values())

    def n_classes(self) -> int:
        return len(self.store)

    # =========================
    # Herding (the core algorithm)
    # =========================

    @staticmethod
    @torch.no_grad()
    def herding(features: torch.Tensor, paths: list[str], m: int) -> list[str]:
        """Greedy herding selection of `m` exemplars from N candidates.

        At each step k, pick the candidate that brings the running mean of
        chosen exemplars closest to the class mean. Repeats are forbidden.

        Args:
            features: (N, D) tensor — features of all candidate images for ONE class.
            paths: length-N list of image paths (parallel to features).
            m: number of exemplars to select. Clipped to N if larger.

        Returns:
            List of `min(m, N)` selected paths, in the order they were picked
            (the order matters — early picks are most representative).
        """
        if features.size(0) == 0:
            return []
        if features.size(0) != len(paths):
            raise ValueError(f"features ({features.size(0)}) and paths ({len(paths)}) length mismatch")

        feats = F.normalize(features, dim=1)
        target_mean = F.normalize(feats.mean(dim=0), dim=0)

        chosen: list[int] = []
        running_sum = torch.zeros_like(target_mean)
        m = min(m, feats.size(0))
        for k in range(m):
            # If we picked j next, the running mean would be (running_sum + feats) / (k+1)
            candidate_means = (running_sum.unsqueeze(0) + feats) / (k + 1)
            dist = (target_mean.unsqueeze(0) - candidate_means).norm(dim=1)
            for j in chosen:
                dist[j] = float("inf")
            j = int(dist.argmin())
            chosen.append(j)
            running_sum = running_sum + feats[j]

        return [paths[j] for j in chosen]

    # =========================
    # Public API
    # =========================

    def add_class(
        self,
        species_id: int,
        features: torch.Tensor,
        paths: list[str],
        n_classes_now: int,
    ) -> None:
        """Store herding-selected exemplars for one species.

        Call this for each *new* species after a session finishes (and
        optionally for old species if you want to refresh their exemplars
        because the backbone was unfrozen / image distribution drifted).

        Args:
            species_id: class ID assigned by TaxonomyState.
            features: (N, D) features for all available images of this class.
            paths: parallel list of image paths.
            n_classes_now: total number of classes in the memory AFTER this
                add (used for fixed-total budget computation).
        """
        m = self._current_budget(n_classes_now)
        self.store[species_id] = self.herding(features, paths, m)

    def rebalance(self, n_classes_now: int) -> None:
        """In fixed-total mode, shrink each class to the new per-class budget.

        No-op in per_class mode.
        """
        if self.mode != "total":
            return
        m = self._current_budget(n_classes_now)
        for sid in list(self.store.keys()):
            self.store[sid] = self.store[sid][:m]

    def species_ids(self) -> list[int]:
        return sorted(self.store.keys())

    def __len__(self) -> int:
        return self.n_exemplars()

    def __contains__(self, species_id: int) -> bool:
        return species_id in self.store

    def __repr__(self) -> str:
        return (f"ExemplarMemory(mode={self.mode!r}, budget_per_class={self.budget_per_class}, "
                f"n_classes={self.n_classes()}, n_exemplars={self.n_exemplars()})")
