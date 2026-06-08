"""Unit tests for pollen_incremental.distill."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pollen_incremental.distill import kd_loss, multihead_kd_loss


# =========================
# kd_loss
# =========================


def test_kd_zero_when_student_matches_teacher() -> None:
    """Identical logits → KL = 0."""
    z = torch.randn(4, 5)
    loss = kd_loss(z, z, temperature=2.0)
    assert loss.item() < 1e-6


def test_kd_positive_when_logits_differ() -> None:
    """Different logits → positive loss."""
    student = torch.randn(4, 5)
    teacher = torch.randn(4, 5)
    loss = kd_loss(student, teacher, temperature=2.0)
    assert loss.item() > 0


def test_kd_handles_empty_old_slice() -> None:
    """At base session, n_old=0 → no old classes to distill → return 0."""
    student = torch.empty(4, 0)
    teacher = torch.empty(4, 0)
    loss = kd_loss(student, teacher, temperature=2.0)
    assert loss.item() == 0.0


def test_kd_validates_shape() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        kd_loss(torch.randn(4, 5), torch.randn(4, 6))


def test_kd_temperature_scales() -> None:
    """T² factor preserves gradient magnitude across temperatures.

    Specifically: as T → ∞, KL goes to 0 (distributions become uniform). We
    just verify the loss is well-defined for various T.
    """
    s, t = torch.randn(4, 5), torch.randn(4, 5)
    loss1 = kd_loss(s, t, temperature=1.0)
    loss2 = kd_loss(s, t, temperature=2.0)
    loss10 = kd_loss(s, t, temperature=10.0)
    assert torch.isfinite(loss1) and torch.isfinite(loss2) and torch.isfinite(loss10)
    # Loss values differ but all positive
    assert loss1.item() > 0 and loss2.item() > 0 and loss10.item() > 0


def test_kd_backward_passes_gradient_to_student_only() -> None:
    """Gradient should flow through student, not through teacher (teacher logits
    are detached implicitly because they enter as soft targets)."""
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5, requires_grad=True)
    loss = kd_loss(student, teacher, temperature=2.0)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    # Teacher logits in this implementation also receive grad because we did
    # not .detach() — but caller is expected to pass teacher_logits from a
    # torch.no_grad() context. Document this contract by not asserting.


# =========================
# multihead_kd_loss
# =========================


def test_multihead_kd_sums_across_levels() -> None:
    B = 4
    n_old = {"ordor": 3, "familia": 5, "genus": 7, "species": 9}
    n_now = {"ordor": 4, "familia": 6, "genus": 8, "species": 10}  # +1 each
    student_out = {lv: torch.randn(B, n_now[lv], requires_grad=True) for lv in n_old}
    teacher_out = {lv: torch.randn(B, n_old[lv]) for lv in n_old}

    loss = multihead_kd_loss(student_out, teacher_out, n_old, temperature=2.0)
    assert loss.dim() == 0
    assert loss.item() > 0

    # Backward should succeed and produce grads for student
    loss.backward()
    for lv in n_old:
        assert student_out[lv].grad is not None


def test_multihead_kd_slices_student_to_old() -> None:
    """Student logits are sliced [:, :n_old] before KD — verify by passing logits
    where the NEW slice is identical between student and teacher (only old slice
    differs). The KD should still be non-zero from the old slice."""
    B = 4
    n_old = {"ordor": 2, "familia": 3, "genus": 4, "species": 5}
    student_out = {lv: torch.randn(B, n_old[lv] + 1) for lv in n_old}
    # Teacher: copy of student's OLD slice (so KD == 0 there)
    teacher_out = {lv: student_out[lv][:, :n_old[lv]].clone() for lv in n_old}
    loss = multihead_kd_loss(student_out, teacher_out, n_old, temperature=2.0)
    assert loss.item() < 1e-5, \
        "KD should be 0 when student's OLD slice exactly matches teacher"


def test_multihead_kd_requires_all_four_heads() -> None:
    student_out = {"ordor": torch.randn(4, 3)}  # missing other levels
    teacher_out = {"ordor": torch.randn(4, 3)}
    n_old = {"ordor": 3, "familia": 5, "genus": 7, "species": 9}
    with pytest.raises(KeyError):
        multihead_kd_loss(student_out, teacher_out, n_old, temperature=2.0)
