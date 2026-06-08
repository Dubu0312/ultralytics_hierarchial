"""Unit tests for pollen_incremental.model_expand.

Critical invariants tested:
    1. **Preservation**: old logits do NOT change after expansion (this is the
       core anti-forgetting guarantee at the head level).
    2. **Shape**: output Linear has the expected n_old + n_new columns.
    3. **Imprinting**: new rows match the supplied prototypes (up to the
       magnitude rescaling).
    4. **Edge cases**: n_new == 0 → identity; no bias → handled; n_old == 0
       (rare cold-start) → handled.
    5. **Prototype computation**: mean-feature + L2-norm produces a unit vector.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pollen_incremental.model_expand import (
    compute_prototype,
    compute_prototypes_per_class,
    expand_linear,
)


# =========================
# expand_linear — preservation
# =========================


def test_expand_preserves_old_logits_exactly() -> None:
    """THE invariant: any old input → old logits unchanged after expansion.

    This is what guarantees zero forgetting at the head level when the
    backbone is frozen. If this test fails, the rest of V1 is meaningless.
    """
    torch.manual_seed(42)
    in_f, n_old = 16, 5
    head = nn.Linear(in_f, n_old)

    # Random input
    x = torch.randn(8, in_f)
    y_before = head(x)  # (8, n_old)

    # Expand by 3 with random prototypes
    proto = torch.randn(3, in_f)
    proto = proto / proto.norm(dim=1, keepdim=True)
    new_head = expand_linear(head, n_new=3, proto=proto)

    y_after = new_head(x)  # (8, n_old + 3)

    # Old slice unchanged
    assert torch.allclose(y_after[:, :n_old], y_before, atol=1e-6), \
        "Old logits changed after expansion — forgetting at the head level!"


def test_expand_preserves_old_logits_when_proto_is_none() -> None:
    """Even with zero-init new rows, old logits must be untouched."""
    torch.manual_seed(0)
    head = nn.Linear(8, 3)
    x = torch.randn(4, 8)
    y_before = head(x)

    new_head = expand_linear(head, n_new=2, proto=None)
    y_after = new_head(x)

    assert torch.allclose(y_after[:, :3], y_before, atol=1e-6)
    # New logits should be exactly 0 (zero weights + zero bias)
    assert torch.allclose(y_after[:, 3:], torch.zeros(4, 2), atol=1e-6)


# =========================
# expand_linear — shape
# =========================


def test_expand_output_shape() -> None:
    head = nn.Linear(64, 10)
    new_head = expand_linear(head, n_new=4, proto=torch.randn(4, 64))
    assert new_head.in_features == 64
    assert new_head.out_features == 14
    assert new_head.weight.shape == (14, 64)
    assert new_head.bias.shape == (14,)


def test_expand_handles_no_bias() -> None:
    head = nn.Linear(8, 3, bias=False)
    new_head = expand_linear(head, n_new=2, proto=torch.randn(2, 8))
    assert new_head.bias is None
    assert new_head.weight.shape == (5, 8)
    # Old slice preserved
    assert torch.allclose(new_head.weight[:3], head.weight, atol=1e-6)


def test_expand_n_new_zero_returns_same_head() -> None:
    """No new classes → return the head as-is (identity, not a copy)."""
    head = nn.Linear(8, 3)
    result = expand_linear(head, n_new=0)
    assert result is head


def test_expand_rejects_negative_n_new() -> None:
    head = nn.Linear(8, 3)
    with pytest.raises(ValueError, match="n_new must be >= 0"):
        expand_linear(head, n_new=-1)


# =========================
# expand_linear — imprinting magnitude
# =========================


def test_imprinted_rows_match_old_norm_scale() -> None:
    """New rows should have magnitude comparable to old rows (after scaling).

    expand_linear multiplies the L2-unit prototype by `scale = mean_norm(old)`.
    So `||new_row|| == scale * ||proto|| == scale * 1.0`.
    """
    torch.manual_seed(1)
    head = nn.Linear(32, 6)
    # Make old rows have a known average norm
    head.weight.data *= 3.0  # inflate to avoid tiny init norms
    old_norms = head.weight.data.norm(dim=1)
    expected_scale = old_norms.mean().item()

    proto = torch.randn(2, 32)
    proto = proto / proto.norm(dim=1, keepdim=True)  # unit
    new_head = expand_linear(head, n_new=2, proto=proto)

    new_norms = new_head.weight.data[6:].norm(dim=1)
    assert torch.allclose(new_norms, torch.full((2,), expected_scale), atol=1e-5), \
        f"new row norms {new_norms.tolist()} != expected scale {expected_scale}"


def test_imprinted_rows_match_proto_direction() -> None:
    """After dividing by the scale, new rows should equal the prototype."""
    torch.manual_seed(2)
    head = nn.Linear(16, 4)
    proto = torch.randn(3, 16)
    proto = proto / proto.norm(dim=1, keepdim=True)

    new_head = expand_linear(head, n_new=3, proto=proto)
    scale = head.weight.data.norm(dim=1).mean().item()
    recovered = new_head.weight.data[4:] / scale
    assert torch.allclose(recovered, proto, atol=1e-5)


def test_imprinted_rows_handle_cold_start() -> None:
    """n_old == 0 (rare): fall back to scale=1.0, don't divide by zero."""
    head = nn.Linear(8, 0)  # empty head
    proto = torch.randn(3, 8)
    proto = proto / proto.norm(dim=1, keepdim=True)
    new_head = expand_linear(head, n_new=3, proto=proto)
    assert torch.allclose(new_head.weight.data, proto, atol=1e-5)


# =========================
# expand_linear — validation
# =========================


def test_expand_rejects_wrong_proto_shape() -> None:
    head = nn.Linear(8, 3)
    bad_proto = torch.randn(2, 7)  # wrong in_features
    with pytest.raises(ValueError, match="proto shape"):
        expand_linear(head, n_new=2, proto=bad_proto)

    bad_proto2 = torch.randn(3, 8)  # wrong n_new
    with pytest.raises(ValueError, match="proto shape"):
        expand_linear(head, n_new=2, proto=bad_proto2)


# =========================
# compute_prototype
# =========================


class _IdentityExtractor(nn.Module):
    """Mock feature extractor: flattens (B,3,H,W) → (B, 3*H*W). Frozen by default."""

    def __init__(self) -> None:
        super().__init__()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


def test_compute_prototype_is_unit_vector() -> None:
    """Prototype must be L2-normalized."""
    extractor = _IdentityExtractor()
    # Two batches of 4 dummy images each
    batches = [torch.randn(4, 3, 2, 2) for _ in range(2)]
    proto = compute_prototype(extractor, batches, device="cpu")
    assert proto.shape == (12,)  # 3*2*2
    assert torch.allclose(proto.norm(), torch.tensor(1.0), atol=1e-6)


def test_compute_prototype_is_mean_of_normalized_features() -> None:
    """Sanity: prototype direction matches the mean of normalized inputs."""
    extractor = _IdentityExtractor()
    # Hand-craft inputs whose flattened-and-normalized mean is predictable
    x = torch.zeros(2, 3, 1, 1)
    x[0, 0, 0, 0] = 1.0  # → [1, 0, 0]
    x[1, 1, 0, 0] = 1.0  # → [0, 1, 0]
    proto = compute_prototype(extractor, [x], device="cpu")
    # Mean of normalized = [0.5, 0.5, 0], normalized = [√2/2, √2/2, 0]
    expected = torch.tensor([0.5, 0.5, 0.0])
    expected = expected / expected.norm()
    assert torch.allclose(proto, expected, atol=1e-6)


def test_compute_prototype_rejects_empty_input() -> None:
    extractor = _IdentityExtractor()
    with pytest.raises(ValueError, match="received no images"):
        compute_prototype(extractor, [], device="cpu")


def test_compute_prototype_rejects_non_2d_output() -> None:
    """If the extractor returns (B, D, H, W) (forgot to pool), we should fail loudly."""

    class BadExtractor(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x  # returns 4D, not 2D

    with pytest.raises(ValueError, match="output \\(B, D\\)"):
        compute_prototype(BadExtractor(), [torch.randn(2, 3, 4, 4)], device="cpu")


# =========================
# compute_prototypes_per_class
# =========================


def test_compute_prototypes_per_class_preserves_order() -> None:
    """Output row order must match the dict's insertion order."""
    extractor = _IdentityExtractor()
    images_by_class = {
        5: [torch.ones(2, 3, 1, 1) * 1.0],     # all-ones → flatten = [1,1,1]
        3: [torch.ones(2, 3, 1, 1) * 2.0],     # all-twos → flatten = [2,2,2]
        7: [torch.ones(2, 3, 1, 1) * 0.5],
    }
    protos = compute_prototypes_per_class(extractor, images_by_class, device="cpu")
    assert protos.shape == (3, 3)
    # All inputs within a class are identical → normalized mean = [1/√3,1/√3,1/√3]
    # Row 0 is class 5, row 1 is class 3, row 2 is class 7.
    unit = torch.full((3,), 1.0 / (3 ** 0.5))
    assert torch.allclose(protos[0], unit, atol=1e-6)
    assert torch.allclose(protos[1], unit, atol=1e-6)
    assert torch.allclose(protos[2], unit, atol=1e-6)


# =========================
# Integration: expand a multi-head model
# =========================


class _ToyMultiHead(nn.Module):
    """Minimal stand-in for YOLOHierarchicalClassifier — just 4 Linear heads."""

    def __init__(self, in_f: int, n_o: int, n_f: int, n_g: int, n_s: int) -> None:
        super().__init__()
        self.head_ordor = nn.Linear(in_f, n_o)
        self.head_familia = nn.Linear(in_f, n_f)
        self.head_genus = nn.Linear(in_f, n_g)
        self.head_species = nn.Linear(in_f, n_s)

    def forward_heads(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "ordor": self.head_ordor(feat),
            "familia": self.head_familia(feat),
            "genus": self.head_genus(feat),
            "species": self.head_species(feat),
        }


def test_full_head_expansion_preserves_all_old_logits() -> None:
    """End-to-end: expand all 4 heads, verify old logits at every level unchanged."""
    torch.manual_seed(7)
    in_f = 32
    model = _ToyMultiHead(in_f, n_o=3, n_f=5, n_g=7, n_s=9)

    feat = torch.randn(4, in_f)
    out_before = model.forward_heads(feat)

    # Mimic one session: +1 species (no new internal), +0 elsewhere
    proto_species = torch.randn(1, in_f)
    proto_species = proto_species / proto_species.norm(dim=1, keepdim=True)

    model.head_species = expand_linear(model.head_species, n_new=1, proto=proto_species)
    # Others: no change
    model.head_ordor = expand_linear(model.head_ordor, n_new=0)
    model.head_familia = expand_linear(model.head_familia, n_new=0)
    model.head_genus = expand_linear(model.head_genus, n_new=0)

    out_after = model.forward_heads(feat)

    # Old slices: identical at every level
    for level, n_old in (("ordor", 3), ("familia", 5), ("genus", 7), ("species", 9)):
        b = out_before[level]
        a = out_after[level][:, :n_old]
        assert torch.allclose(a, b, atol=1e-6), \
            f"Level '{level}' old logits changed after expansion"

    # Species head grew by exactly 1
    assert out_after["species"].shape == (4, 10)
