"""Evaluate an incremental model on a session's test split.

Reports per-level accuracy, HierAcc, and consistency rate — same metrics as
the base test_yolo.py so numbers compare apples-to-apples across sessions.

Why a separate module? The base test_yolo.py is tied to its own paths and
checkpoint format; we need a reusable function we can call from inside
train_incremental.py (for per-epoch logging) and from a standalone CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from pollen_incremental.masked_ops import masked_decode


@dataclass
class EvalResult:
    """Per-level + HierAcc metrics for one model on one test split."""

    n_samples: int
    acc_ordor: float
    acc_familia: float
    acc_genus: float
    acc_species: float
    hier_acc: float
    f1_macro_ordor: float
    f1_macro_familia: float
    f1_macro_genus: float
    f1_macro_species: float
    decode_mode: str  # "masked" | "indep"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n_samples": self.n_samples,
            "acc_ordor": self.acc_ordor,
            "acc_familia": self.acc_familia,
            "acc_genus": self.acc_genus,
            "acc_species": self.acc_species,
            "hier_acc": self.hier_acc,
            "f1_macro_ordor": self.f1_macro_ordor,
            "f1_macro_familia": self.f1_macro_familia,
            "f1_macro_genus": self.f1_macro_genus,
            "f1_macro_species": self.f1_macro_species,
            "decode_mode": self.decode_mode,
        }


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device | str,
    familia_parent: torch.Tensor,
    genus_parent: torch.Tensor,
    species_parent: torch.Tensor,
    decode: str = "masked",
) -> EvalResult:
    """Run model on loader, compute metrics.

    Args:
        model: a YOLOHierarchicalClassifier (or any module with .forward_dict).
        loader: DataLoader yielding (image, label_dict) batches.
        device: where to run inference.
        familia_parent / genus_parent / species_parent: parent tensors on `device`.
        decode: "masked" (top-down masked) or "indep" (independent argmax).
            "masked" is the default and matches the base pipeline's
            consistency-guaranteed decoding.

    Returns:
        EvalResult with per-level + hierarchical metrics.
    """
    if decode not in ("masked", "indep"):
        raise ValueError(f"decode must be 'masked' or 'indep', got {decode!r}")

    model.eval()
    fp = familia_parent.to(device)
    gp = genus_parent.to(device)
    sp = species_parent.to(device)

    o_p: list[int] = []; o_t: list[int] = []
    f_p: list[int] = []; f_t: list[int] = []
    g_p: list[int] = []; g_t: list[int] = []
    s_p: list[int] = []; s_t: list[int] = []

    for x, y in loader:
        x = x.to(device)
        lo, lf, lg, ls = model(x)
        if decode == "masked":
            o, f, g, s = masked_decode(lo, lf, lg, ls, fp, gp, sp)
        else:
            o = lo.argmax(1); f = lf.argmax(1); g = lg.argmax(1); s = ls.argmax(1)
        o_p.extend(o.cpu().tolist()); o_t.extend(y["ordor"].cpu().tolist())
        f_p.extend(f.cpu().tolist()); f_t.extend(y["familia"].cpu().tolist())
        g_p.extend(g.cpu().tolist()); g_t.extend(y["genus"].cpu().tolist())
        s_p.extend(s.cpu().tolist()); s_t.extend(y["species"].cpu().tolist())

    n = len(o_t)
    if n == 0:
        raise ValueError("Evaluation loader yielded no samples")

    hier = sum(
        int(a == b and c == d and e == f and g == h)
        for a, b, c, d, e, f, g, h in zip(o_p, o_t, f_p, f_t, g_p, g_t, s_p, s_t)
    ) / n

    return EvalResult(
        n_samples=n,
        acc_ordor=accuracy_score(o_t, o_p),
        acc_familia=accuracy_score(f_t, f_p),
        acc_genus=accuracy_score(g_t, g_p),
        acc_species=accuracy_score(s_t, s_p),
        hier_acc=hier,
        f1_macro_ordor=f1_score(o_t, o_p, average="macro", zero_division=0),
        f1_macro_familia=f1_score(f_t, f_p, average="macro", zero_division=0),
        f1_macro_genus=f1_score(g_t, g_p, average="macro", zero_division=0),
        f1_macro_species=f1_score(s_t, s_p, average="macro", zero_division=0),
        decode_mode=decode,
    )


def format_eval_result(name: str, result: EvalResult) -> str:
    """One-line summary for logging."""
    return (
        f"[{name}] HierAcc={result.hier_acc:.3f}  "
        f"O={result.acc_ordor:.3f} F={result.acc_familia:.3f} "
        f"G={result.acc_genus:.3f} S={result.acc_species:.3f}  "
        f"(n={result.n_samples}, decode={result.decode_mode})"
    )
