"""Save / load checkpoints for incremental sessions.

Extends the base checkpoint format (HIERARCHICAL.md Phụ lục A) with:
    - tax_state: TaxonomyState as a dict (append-only ID maps + parent arrays)
    - session: int, 0 = base, 1, 2, ... = incremental
    - exemplar_memory: ExemplarMemory as a dict
    - incremental_cfg: hyperparameters used to produce this session's model

Why a separate format from the base checkpoint? Two reasons:
    1. We need the TaxonomyState explicitly stored so the next session can
       call extend_state() without re-deriving IDs from alphabetic order
       (which would break append-only).
    2. We need the exemplar memory so the next session can replay.

The base checkpoint produced by train_hierarchical_masked.py is NOT in this
format yet — convert it once via scripts/build_base_session.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pollen_incremental.exemplar_memory import ExemplarMemory
from pollen_incremental.taxonomy import TaxonomyState


@dataclass
class IncrementalCheckpoint:
    """In-memory representation of one session's saved state."""

    session: int
    base_model: str                              # e.g. "yolo11x-cls.pt"
    model_state: dict[str, torch.Tensor]         # state_dict (head sizes match tax_state)
    tax_state: TaxonomyState
    exemplar_memory: ExemplarMemory
    lambdas: dict[str, float]
    incremental_cfg: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session": int(self.session),
            "base_model": str(self.base_model),
            "model_state": self.model_state,
            "tax_state": self.tax_state.to_dict(),
            "exemplar_memory": self.exemplar_memory.to_dict(),
            "lambdas": dict(self.lambdas),
            "incremental_cfg": dict(self.incremental_cfg),
            "n_classes": self.tax_state.n_classes(),  # convenience: also at top level
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path | str, map_location: str | torch.device = "cpu") -> "IncrementalCheckpoint":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=map_location, weights_only=False)
        # Validate required fields
        for key in ("session", "base_model", "model_state", "tax_state",
                    "exemplar_memory", "lambdas"):
            if key not in payload:
                raise ValueError(f"Checkpoint at {path} missing required key: {key!r}")
        return cls(
            session=int(payload["session"]),
            base_model=str(payload["base_model"]),
            model_state=payload["model_state"],
            tax_state=TaxonomyState.from_dict(payload["tax_state"]),
            exemplar_memory=ExemplarMemory.from_dict(payload["exemplar_memory"]),
            lambdas=dict(payload["lambdas"]),
            incremental_cfg=dict(payload.get("incremental_cfg", {})),
        )

    def n_classes(self) -> dict[str, int]:
        return self.tax_state.n_classes()
