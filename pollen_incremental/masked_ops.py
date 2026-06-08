"""Masked cross-entropy loss + masked top-down decoding.

Re-implemented from train_hierarchical_masked.py:214-269 + test_yolo.py:176-203
so the incremental package is self-contained. Logic is bit-for-bit identical
to the base implementation — verified by integration test against base
checkpoints.

Two operations:

    multitask_ce_masked(logits, targets, lambdas, parents)
        Training-time loss. At each non-root level l, mask the logits to only
        the classes whose GT parent matches y_{l-1}, then compute CE. Order
        is unmasked (it has no parent). Total loss = weighted sum.

    masked_decode(lo, lf, lg, ls, parents)
        Inference-time decoder. Top-down argmax with masking by the *predicted*
        parent at each level — guarantees every output is a valid path in the
        taxonomy tree.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Default loss weights — load-bearing default for the base pipeline.
LAMBDA_DEFAULT: dict[str, float] = {
    "ordor": 1.0,
    "familia": 0.9,
    "genus": 0.8,
    "species": 0.7,
}


def multitask_ce_masked(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambdas: dict[str, float],
    familia_parent: torch.Tensor,
    genus_parent: torch.Tensor,
    species_parent: torch.Tensor,
    mask_value: float = -1e9,
) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
    """Masked CE loss across the 4 levels.

    Args:
        logits: tuple of (z_o, z_f, z_g, z_s) — each (B, n_class_at_level).
        targets: dict with keys 'ordor', 'familia', 'genus', 'species';
            values are (B,) long tensors of GT class IDs.
        lambdas: per-level weight (e.g. {"ordor": 1.0, "familia": 0.9, ...}).
        familia_parent: (n_familia,) long tensor, value = parent ordor ID.
        genus_parent: (n_genus,) long tensor, value = parent familia ID.
        species_parent: (n_species,) long tensor, value = parent genus ID.
        mask_value: large negative number used in place of masked logits
            (softmax(mask_value) ≈ 0). Default −1e9 matches base.

    Returns:
        total: scalar tensor (the loss).
        parts: tuple of per-level loss values as floats (for logging).
    """
    lo, lf, lg, ls = logits

    Lo = F.cross_entropy(lo, targets["ordor"])

    mask_f = (familia_parent.unsqueeze(0) == targets["ordor"].unsqueeze(1))
    lf_masked = lf.masked_fill(~mask_f, mask_value)
    Lf = F.cross_entropy(lf_masked, targets["familia"])

    mask_g = (genus_parent.unsqueeze(0) == targets["familia"].unsqueeze(1))
    lg_masked = lg.masked_fill(~mask_g, mask_value)
    Lg = F.cross_entropy(lg_masked, targets["genus"])

    mask_s = (species_parent.unsqueeze(0) == targets["genus"].unsqueeze(1))
    ls_masked = ls.masked_fill(~mask_s, mask_value)
    Ls = F.cross_entropy(ls_masked, targets["species"])

    total = (
        lambdas["ordor"] * Lo
        + lambdas["familia"] * Lf
        + lambdas["genus"] * Lg
        + lambdas["species"] * Ls
    )
    return total, (Lo.item(), Lf.item(), Lg.item(), Ls.item())


@torch.no_grad()
def masked_decode(
    lo: torch.Tensor,
    lf: torch.Tensor,
    lg: torch.Tensor,
    ls: torch.Tensor,
    familia_parent: torch.Tensor,
    genus_parent: torch.Tensor,
    species_parent: torch.Tensor,
    mask_value: float = -1e9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-down masked decoding — guarantees a valid taxonomy path.

    Predicts order first, then masks each child level to only those whose
    *predicted* parent matches, then argmax.

    Args:
        lo, lf, lg, ls: logits at each level, shape (B, n_class).
        familia_parent / genus_parent / species_parent: parent lookup tensors.
        mask_value: large negative value used to suppress invalid children.

    Returns:
        (o_pred, f_pred, g_pred, s_pred), each (B,) long tensor.
    """
    o = lo.argmax(1)

    mask_f = (familia_parent.unsqueeze(0) == o.unsqueeze(1))
    f = lf.masked_fill(~mask_f, mask_value).argmax(1)

    mask_g = (genus_parent.unsqueeze(0) == f.unsqueeze(1))
    g = lg.masked_fill(~mask_g, mask_value).argmax(1)

    mask_s = (species_parent.unsqueeze(0) == g.unsqueeze(1))
    s = ls.masked_fill(~mask_s, mask_value).argmax(1)

    return o, f, g, s


def to_parent_tensors(
    state,  # TaxonomyState — duck-typed to avoid circular import
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a TaxonomyState's parent arrays (lists) to long tensors on device.

    Convenience helper — masked_ce / masked_decode require torch.Tensors.
    """
    fp = torch.tensor(state.familia_parent, dtype=torch.long, device=device)
    gp = torch.tensor(state.genus_parent, dtype=torch.long, device=device)
    sp = torch.tensor(state.species_parent, dtype=torch.long, device=device)
    return fp, gp, sp
