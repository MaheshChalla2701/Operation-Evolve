"""
utils.py – Shared utilities for the Operation Evolve system.

Provides:
  - setup_logging()            : coloured console logger with timestamps
  - save_checkpoint()          : save model weights to disk
  - load_checkpoint()          : restore model weights from disk
  - print_loop_summary()       : pretty-print per-evolution-loop stats
  - AccuracyTracker            : tracks accuracy trend across loops
"""

import os
import logging
import copy
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _ColourFormatter(logging.Formatter):
    """Minimal ANSI-colour formatter for readability in terminals & Colab."""

    COLOURS = {
        logging.DEBUG:    "\033[37m",    # White
        logging.INFO:     "\033[36m",    # Cyan
        logging.WARNING:  "\033[33m",    # Yellow
        logging.ERROR:    "\033[31m",    # Red
        logging.CRITICAL: "\033[35m",    # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelno, self.RESET)
        formatted = super().format(record)
        return f"{colour}{formatted}{self.RESET}"


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with a coloured, timestamped console handler.

    Call once at the start of main.py.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    handler.setFormatter(
        _ColourFormatter(
            fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger("evolve")
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model state_dict (+ optional metadata) to path."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logging.getLogger("evolve.utils").info(f"[Checkpoint] Saved → {path}")


def load_checkpoint(model: nn.Module, path: str) -> Dict[str, Any]:
    """Load model weights from a checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    logging.getLogger("evolve.utils").info(f"[Checkpoint] Loaded ← {path}")
    return payload


# ---------------------------------------------------------------------------
# Per-loop summary printer
# ---------------------------------------------------------------------------

def print_loop_summary(
    loop_idx: int,
    eval_results: Dict[str, Any],
    dataset_sizes: Dict[str, int],
    accepted: int,
    rejected: int,
    train_history: List[Dict[str, Any]],
    rolled_back: bool = False,
) -> None:
    """
    Print a structured, human-readable summary for one evolution loop.

    Args:
        loop_idx       : current loop number (0-indexed)
        eval_results   : dict from evaluate.evaluate()
        dataset_sizes  : {"A": n_a, "B": n_b, "C": n_c}
        accepted       : samples accepted from Dataset_C
        rejected       : samples rejected from Dataset_C
        train_history  : list of {"epoch": N, "loss": float}
        rolled_back    : whether rollback was triggered this loop
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  EVOLUTION LOOP {loop_idx + 1}  {'[ROLLED BACK]' if rolled_back else ''}")
    print(sep)

    # Training history
    if train_history:
        losses = [h["loss"] for h in train_history]
        print(f"  Training   | epochs={len(losses)} | "
              f"first_loss={losses[0]:.4f} | last_loss={losses[-1]:.4f}")

    # Evaluation
    acc = eval_results.get("accuracy", 0.0)
    loss = eval_results.get("loss", 0.0)
    conf_mean = eval_results.get("conf_mean", 0.0)
    conf_min = eval_results.get("conf_min", 0.0)
    conf_max = eval_results.get("conf_max", 0.0)
    print(f"  Evaluation | loss={loss:.4f} | accuracy={acc:.2f}%")
    print(f"  Confidence | mean={conf_mean:.3f} | min={conf_min:.3f} | max={conf_max:.3f}")

    # Per-class breakdown
    per_class = eval_results.get("per_class_acc", {})
    if per_class:
        breakdown = "  | ".join(f"class-{k}: {v:.1f}%" for k, v in sorted(per_class.items()))
        print(f"  Per-Class  | {breakdown}")

    # Dataset sizes
    sz_a = dataset_sizes.get("A", "?")
    sz_b = dataset_sizes.get("B", "?")
    sz_c = dataset_sizes.get("C", "?")
    print(f"  Datasets   | A={sz_a} (read-only) | B={sz_b} (train) | C={sz_c} (generated)")

    # Agent decisions
    print(f"  Agent      | accepted={accepted} | rejected={rejected}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Accuracy trend tracker
# ---------------------------------------------------------------------------

class AccuracyTracker:
    """Tracks and renders an accuracy trend across evolution loops."""

    def __init__(self):
        self.history: List[float] = []

    def record(self, accuracy: float) -> None:
        self.history.append(accuracy)

    def trend_str(self) -> str:
        if not self.history:
            return "(no data)"
        bars = []
        for acc in self.history:
            if acc >= 90:
                bars.append("▓▓▓")
            elif acc >= 75:
                bars.append("▓▓░")
            elif acc >= 50:
                bars.append("▓░░")
            else:
                bars.append("░░░")
        return "  ".join(
            f"L{i+1}:{bar} {a:.1f}%"
            for i, (bar, a) in enumerate(zip(bars, self.history))
        )

    def best(self) -> float:
        return max(self.history) if self.history else 0.0

    def is_improving(self) -> bool:
        """Return True if the last entry is >= all previous entries."""
        if len(self.history) < 2:
            return True
        return self.history[-1] >= max(self.history[:-1])
