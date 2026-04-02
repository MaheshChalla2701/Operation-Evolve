"""
weight_transfer.py – Warm-start weight transfer for evolved architectures.

When the Groq agent proposes a new architecture, the new model's layers that are
structurally identical to the previous model should reuse the trained weights so
that the student model starts from a warm state instead of random noise.

Without this, any proposed architecture change will always be rejected by the
rollback gate because random weights always produce worse loss than a trained model.

Public API
----------
transfer_compatible_weights(old_model, new_model) -> int
    Copies all matching-shape tensors from old_model to new_model in-place.
    Returns the number of successfully transferred parameter tensors.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("evolve.weight_transfer")


def transfer_compatible_weights(
    old_model: nn.Module,
    new_model: nn.Module,
) -> Tuple[int, int]:
    """
    Copy all weight tensors from `old_model` to `new_model` where the
    parameter name **and** tensor shape both match exactly.

    Layers that are new (don't exist in old_model) or have been resized
    are left at their random initialisation — which is the standard
    Net2Net-style warm-start approach.

    Parameters
    ----------
    old_model : nn.Module
        The previously trained model (source of weights).
    new_model : nn.Module
        The newly built model with potentially different architecture
        (destination for compatible weights).

    Returns
    -------
    (transferred, total) : Tuple[int, int]
        transferred – number of parameter tensors successfully copied.
        total       – total parameter tensors in the new model.

    Notes
    -----
    - The transfer is performed in-place on `new_model`.
    - Neither model's training mode is changed.
    - No gradients are needed or tracked for this operation.
    """
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    transferred = 0
    skipped_shape = []
    skipped_missing = []

    with torch.no_grad():
        for name, new_tensor in new_state.items():
            if name not in old_state:
                skipped_missing.append(name)
                continue

            old_tensor = old_state[name]

            if old_tensor.shape == new_tensor.shape:
                new_state[name] = old_tensor.clone()
                transferred += 1
            else:
                # Evolved Architecture Support: Handle Resized Tensors
                # If d_model or heads change, the shapes will differ. We want to pad/truncate.
                # Find the overlapping bounding box for all dimensions:
                overlap_slices = tuple(slice(0, min(o, n)) for o, n in zip(old_tensor.shape, new_tensor.shape))
                
                # Copy the overlapping subset into the newly initialized tensor
                new_state[name][overlap_slices] = old_tensor[overlap_slices].clone()
                transferred += 1
                
                skipped_shape.append(
                    f"{name}: old={tuple(old_tensor.shape)} → new={tuple(new_tensor.shape)} (Overlapped slice copied)"
                )

        new_model.load_state_dict(new_state)

    total = len(new_state)

    logger.info(
        f"[WeightTransfer] Transferred {transferred}/{total} tensors from old → new model."
    )
    if skipped_shape:
        logger.info(
            f"[WeightTransfer] {len(skipped_shape)} tensors skipped (shape mismatch — "
            f"new random init):\n  " + "\n  ".join(skipped_shape)
        )
    if skipped_missing:
        logger.debug(
            f"[WeightTransfer] {len(skipped_missing)} tensors are brand-new "
            f"(not in old model): {skipped_missing}"
        )

    return transferred, total
