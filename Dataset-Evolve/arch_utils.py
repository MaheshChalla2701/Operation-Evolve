"""
arch_utils.py – Architecture comparison utilities for Hybrid Continual Learning.

Used by main.py to decide whether to switch training between:
  - "stable" mode  → LwF distillation is safe (architectures match)
  - "evolve" mode  → architecture changed, skip strict LwF

Public API
----------
same_architecture(config1, config2) -> bool
    Returns True only if ALL architecture-defining fields match.

snapshot_arch(config) -> dict
    Returns a dict snapshot of the current architecture fields.

apply_snapshot(config, snapshot) -> None
    Writes a snapshot's values into the prev_* fields of a config.
"""

from dataclasses import dataclass
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Fields that define the model architecture
# ---------------------------------------------------------------------------

_ARCH_FIELDS = ("model_type", "hidden_dim", "num_heads", "num_layers", "experts")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def same_architecture(config1, config2) -> bool:
    """
    Compare two EvolveConfig instances on all architecture-defining fields.

    Parameters
    ----------
    config1, config2 : EvolveConfig
        Any two config objects exposing the fields in _ARCH_FIELDS.

    Returns
    -------
    bool
        True if ALL of {model_type, hidden_dim, num_heads, num_layers, experts}
        are identical between the two configs.  False if any differ.

    Notes
    -----
    - ``hidden_dim`` acts as ``d_model`` for Transformer models.
    - ``experts`` is 0 when MoE is not used; any non-zero value triggers a
      mismatch if it changes.
    """
    for field in _ARCH_FIELDS:
        v1 = getattr(config1, field, None)
        v2 = getattr(config2, field, None)
        if v1 != v2:
            return False
    return True


def same_architecture_from_prev(config) -> bool:
    """
    Compare the *current* architecture fields against the *prev_** snapshot
    stored in the same config object.

    This is the primary function called by main.py after config may have been
    updated (e.g. hidden_dim changed by LLM agent).

    Parameters
    ----------
    config : EvolveConfig
        Config with both current and prev_* fields.

    Returns
    -------
    bool
        True if architecture has NOT changed since the last snapshot.
    """
    mapping = {
        "model_type": "prev_model_type",
        "hidden_dim": "prev_hidden_dim",
        "num_heads":  "prev_num_heads",
        "num_layers": "prev_num_layers",
        "experts":    "prev_experts",
    }

    # If prev_* are all at their defaults (0 / ""), this is the first loop –
    # treat as "same" (no old model to distill from anyway).
    if getattr(config, "prev_hidden_dim", 0) == 0 and getattr(config, "prev_model_type", "") == "":
        return True

    for curr_field, prev_field in mapping.items():
        curr_val = getattr(config, curr_field, None)
        prev_val = getattr(config, prev_field, None)
        if curr_val != prev_val:
            return False
    return True


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def snapshot_arch(config) -> Dict[str, Any]:
    """
    Capture the current architecture fields as a plain dict.

    Parameters
    ----------
    config : EvolveConfig

    Returns
    -------
    dict with keys {model_type, hidden_dim, num_heads, num_layers, experts}
    """
    return {field: getattr(config, field, None) for field in _ARCH_FIELDS}


def apply_snapshot(config, snapshot: Dict[str, Any]) -> None:
    """
    Write a snapshot dict's values into the prev_* fields of config,
    recording the architecture that was committed in the last loop.

    Parameters
    ----------
    config   : EvolveConfig  (mutated in-place)
    snapshot : dict from snapshot_arch()
    """
    mapping = {
        "model_type": "prev_model_type",
        "hidden_dim": "prev_hidden_dim",
        "num_heads":  "prev_num_heads",
        "num_layers": "prev_num_layers",
        "experts":    "prev_experts",
    }
    for curr_field, prev_field in mapping.items():
        if curr_field in snapshot:
            setattr(config, prev_field, snapshot[curr_field])
