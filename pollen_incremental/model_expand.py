"""Head expansion + weight imprinting for incremental sessions.

When a new session adds Δ classes at level L, we need to:
    1. Grow `Linear(in_features, n_old)` into `Linear(in_features, n_old + Δ)`
       while preserving the old weights exactly (frozen backbone ⇒ old logits
       must be identical pre- and post-expansion).
    2. Initialize the new rows by *imprinting*: set them to the L2-normalized
       mean feature of the new-class images (rescaled to match the magnitude
       of old rows). This makes few-shot classes immediately discriminable
       without any gradient updates — the SimpleCIL idea (Zhou et al., IJCV 2024)
       applied per-level.

This module is the second piece of the V1 foundation (after taxonomy.py).
It only handles head surgery; the actual training loop lives in
train_incremental.py (Phase 3).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def expand_linear(
    head: nn.Linear,
    n_new: int,
    proto: torch.Tensor | None = None,
) -> nn.Linear:
    """Return a new Linear with `n_new` extra output units, old weights preserved.

    Args:
        head: existing nn.Linear(in_features, n_old).
        n_new: number of new output units to add. If 0, returns `head` unchanged.
        proto: optional (n_new, in_features) tensor of L2-normalized prototypes
            for imprinting the new rows. If None, new rows are zero-initialized.

    Returns:
        New nn.Linear(in_features, n_old + n_new) with:
            - rows [0:n_old]: copied verbatim from `head`
            - rows [n_old:]: imprinted from `proto` (scaled to match old norm)
              or zero if `proto is None`
            - bias rows for new classes: 0 (if head has bias)

    The new module lives on the same device/dtype as `head`. The caller is
    responsible for assigning it back (e.g. `model.head_species = expand_linear(...)`).
    """
    if n_new < 0:
        raise ValueError(f"n_new must be >= 0, got {n_new}")
    if n_new == 0:
        return head

    in_f = head.in_features
    n_old = head.out_features
    has_bias = head.bias is not None
    device = head.weight.device
    dtype = head.weight.dtype

    new_head = nn.Linear(in_f, n_old + n_new, bias=has_bias).to(device=device, dtype=dtype)

    # Copy old weights/bias exactly.
    new_head.weight.data[:n_old].copy_(head.weight.data)
    if has_bias:
        new_head.bias.data[:n_old].copy_(head.bias.data)

    # Initialize new rows.
    if proto is not None:
        if proto.shape != (n_new, in_f):
            raise ValueError(
                f"proto shape {tuple(proto.shape)} != expected ({n_new}, {in_f})"
            )
        # Match the average magnitude of old rows so the new logits live on
        # the same scale (otherwise softmax is dominated by old classes).
        scale = head.weight.data.norm(dim=1).mean().item() if n_old > 0 else 1.0
        new_head.weight.data[n_old:].copy_(proto.to(device=device, dtype=dtype) * scale)
    else:
        new_head.weight.data[n_old:].zero_()

    if has_bias:
        new_head.bias.data[n_old:].zero_()

    return new_head


@torch.no_grad()
def compute_prototype(
    feature_extractor,
    images: Iterable[torch.Tensor],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute one L2-normalized mean-feature prototype for a single class.

    Args:
        feature_extractor: a callable mapping (B, 3, H, W) → (B, D). May be an
            nn.Module (we'll call .eval() on it) OR a plain callable / bound
            method (caller is responsible for putting the underlying module in
            eval mode and freezing it).
        images: an iterable yielding image batches (B, 3, H, W). The whole
            iterable is consumed and concatenated; all batches must belong to
            ONE class.
        device: device to run the extractor on.

    Returns:
        Tensor of shape (D,), L2-normalized.

    Raises:
        ValueError: if `images` yields no batches.
    """
    if isinstance(feature_extractor, nn.Module):
        feature_extractor.eval()

    feats: list[torch.Tensor] = []
    for batch in images:
        batch = batch.to(device)
        f = feature_extractor(batch)  # (B, D)
        if f.ndim != 2:
            raise ValueError(f"feature_extractor must output (B, D), got shape {tuple(f.shape)}")
        feats.append(F.normalize(f, dim=1))

    if not feats:
        raise ValueError("compute_prototype received no images")

    all_feats = torch.cat(feats, dim=0)  # (N, D)
    mu = all_feats.mean(dim=0)            # (D,)
    return F.normalize(mu, dim=0)


def compute_prototypes_per_class(
    feature_extractor,
    images_by_class: dict[int, Iterable[torch.Tensor]],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute prototypes for several classes, stacked in *insertion order*.

    The order of the output rows matches the iteration order of
    `images_by_class.keys()` — so the caller MUST pass an ordered dict where
    keys are the new class IDs in the order they were appended by
    `extend_state()`.

    Args:
        feature_extractor: frozen feature extractor (B,3,H,W) → (B, D).
        images_by_class: ordered mapping class_id → iterable of image batches.
        device: device to run on.

    Returns:
        Tensor of shape (n_new_classes, D), L2-normalized per row.
    """
    prototypes = [
        compute_prototype(feature_extractor, batches, device=device)
        for _, batches in images_by_class.items()
    ]
    return torch.stack(prototypes, dim=0)
