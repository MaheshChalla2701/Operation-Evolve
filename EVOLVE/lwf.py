"""
lwf.py – Learning without Forgetting (LwF) for Hybrid Continual Learning.

LwF prevents catastrophic forgetting by encouraging the new model's softened
output distribution to match the teacher (old) model's distribution via KL
divergence.

IMPORTANT constraint
---------------------
LwF requires the teacher and student to share the same output dimension
(num_classes).  If the architecture changes in a way that alters the output
shape, LwF CANNOT be applied.  Architecture changes that affect hidden_dim or
num_heads are handled by main.py, which only passes stable_mode=True when
architectures match.

Public API
----------
clone_model(model)                    → deep-copy teacher
distillation_loss(logits_s, logits_t, temperature)  → KL loss scalar
compute_lwf_loss(model_new, model_old, features, labels, criterion,
                 alpha, temperature) → combined CE + KL loss scalar
"""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("evolve.lwf")


# ---------------------------------------------------------------------------
# Model cloning
# ---------------------------------------------------------------------------

def clone_model(model: nn.Module) -> nn.Module:
    """
    Create a frozen deep-copy of the model to serve as a teacher.

    The clone is placed on the same device as the original and its parameters
    are frozen (no gradients).  The clone is set to eval mode.

    Parameters
    ----------
    model : nn.Module
        The current student model BEFORE training on the new task.

    Returns
    -------
    nn.Module
        A frozen, deep-copied teacher model.
    """
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def distillation_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Compute the KL-divergence distillation loss between student and teacher logits.

    Both distributions are softened by dividing by ``temperature`` before
    computing the KL divergence, which emphasises the relative ordering of
    class probabilities (dark knowledge) rather than the argmax.

    Parameters
    ----------
    logits_student : torch.Tensor, shape (B, C)
    logits_teacher : torch.Tensor, shape (B, C)
    temperature    : float > 1  (lower → harder targets; higher → softer)

    Returns
    -------
    torch.Tensor
        Scalar KL divergence.  Multiplied by T² to keep the gradient scale
        consistent across temperatures (standard practice in KD literature).
    """
    T = temperature
    p_student = F.log_softmax(logits_student / T, dim=-1)
    p_teacher = F.softmax(logits_teacher / T, dim=-1)

    # kl_div expects log-probs for input, probs for target; reduction='batchmean'
    kl = F.kl_div(p_student, p_teacher, reduction="batchmean")
    return kl * (T ** 2)


# ---------------------------------------------------------------------------
# Combined forward pass
# ---------------------------------------------------------------------------

def compute_lwf_loss(
    model_new: nn.Module,
    model_old: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Compute the combined Learning-without-Forgetting loss.

    total_loss = (1 - alpha) * CE(logits_new, labels)
               +       alpha * T² * KL(soft_new || soft_old)

    Parameters
    ----------
    model_new  : nn.Module  (student, in train mode)
    model_old  : nn.Module  (teacher, frozen clone from before training)
    features   : torch.Tensor, shape (B, D)  on correct device
    labels     : torch.Tensor, shape (B,)    on correct device
    criterion  : nn.Module  (CrossEntropyLoss or similar)
    alpha      : float  weight for distillation term (0 = CE only, 1 = KL only)
    temperature: float  softmax temperature for distillation

    Returns
    -------
    torch.Tensor
        Combined scalar loss ready for .backward().

    Notes
    -----
    model_old is evaluated inside torch.no_grad() to avoid building a graph
    for the teacher's parameters.
    """
    # Student forward pass (in train mode, gradients enabled)
    logits_new = model_new(features)
    ce_loss = criterion(logits_new, labels)

    # Teacher forward pass (no gradients)
    with torch.no_grad():
        logits_old = model_old(features)

    kl_loss = distillation_loss(logits_new, logits_old, temperature)

    total = (1.0 - alpha) * ce_loss + alpha * kl_loss

    logger.debug(
        f"[LwF] CE={ce_loss.item():.4f} | KL={kl_loss.item():.4f} | "
        f"total={total.item():.4f} | alpha={alpha} | T={temperature}"
    )

    return total
