"""Knowledge distillation loss for incremental sessions.

Implements LwF-style distillation (Li & Hoiem 2017): the previous-session
model acts as the *teacher*, and we penalize the student for diverging on the
old-class logit slice. This anchors the model's behavior on old classes even
as we fine-tune for new ones.

Two functions:
    kd_loss            — KL(student || teacher) at one level, temperature T.
    multihead_kd_loss  — sums KD across all 4 heads, sliced to old classes only.

Why slice to old classes? The teacher has no opinion about classes added in
the current session (those head rows didn't exist for it). We only distill
behavior on what the teacher actually knows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Soft KL distillation between student and teacher at ONE level.

    Args:
        student_logits: (B, n_old) — current model's logits on the OLD-class slice.
        teacher_logits: (B, n_old) — frozen teacher's logits on the same slice.
        temperature: KD softness. Higher → softer distribution, weaker gradient.
            Default 2.0 (typical for LwF).

    Returns:
        Scalar tensor: T² · KL(softmax(student/T) || softmax(teacher/T)),
        averaged over the batch.

    Notes:
        - The T² factor preserves gradient scale across temperatures (Hinton 2015).
        - We pass `target=teacher_dist` and `input=student_logprob` to
          F.kl_div, which is the convention (KL(P_student || P_teacher)
          in cross-entropy form).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if student_logits.size(1) == 0:
        # No old classes at this level (e.g. base session) → zero loss.
        return student_logits.new_zeros(())

    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    logp_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(logp_student, p_teacher, reduction="batchmean") * (temperature ** 2)


def multihead_kd_loss(
    student_out: dict[str, torch.Tensor],
    teacher_out: dict[str, torch.Tensor],
    n_old: dict[str, int],
    temperature: float = 2.0,
) -> torch.Tensor:
    """Sum KD across all 4 hierarchical heads on the OLD-class slice.

    Args:
        student_out: dict level → (B, n_now) logits from current model.
        teacher_out: dict level → (B, n_old) logits from teacher snapshot.
        n_old: dict level → number of classes at that level when teacher
            was snapshotted. Slices student logits to [:, :n_old[level]] so
            shape matches teacher.
        temperature: shared KD temperature.

    Returns:
        Scalar tensor: sum of per-level KD losses (unweighted — caller can
        multiply by β for the overall L = L_CE + β · L_KD recipe).

    Raises:
        KeyError: if any of the 4 levels is missing from either dict.
    """
    total = student_out["ordor"].new_zeros(())  # scalar on correct device/dtype
    for level in ("ordor", "familia", "genus", "species"):
        s = student_out[level][:, :n_old[level]]
        t = teacher_out[level]
        # Teacher slice may differ if caller did not slice already; ensure match.
        if t.size(1) != n_old[level]:
            t = t[:, :n_old[level]]
        total = total + kd_loss(s, t, temperature=temperature)
    return total
