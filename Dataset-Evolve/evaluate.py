"""
evaluate.py – Evaluation utilities for Operation Evolve.

Key design rule:
  - Training   → Dataset_B
  - Validation → Dataset_A (read-only ground truth)

This separation guarantees true performance measurement and prevents
data leakage between train/val sets.
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import EvolveConfig
from data import SyntheticDataset

logger = logging.getLogger("evolve.evaluate")


# ---------------------------------------------------------------------------
# Main evaluation function  (always on Dataset_A)
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataset_a: SyntheticDataset,
    config: EvolveConfig,
) -> Dict[str, Any]:
    """
    Evaluate the model on Dataset_A (core / ground-truth validation set).

    IMPORTANT: This function must ONLY ever be called with Dataset_A to
    provide a true, unbiased performance estimate.

    Returns:
        {
            "loss"          : float,
            "accuracy"      : float (0–100),
            "per_class_acc" : dict { class_id → accuracy },
            "conf_mean"     : float,
            "conf_min"      : float,
            "conf_max"      : float,
        }
    """
    device = config.get_device()
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset_a, batch_size=config.batch_size, shuffle=False)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Per-class tracking
    class_correct = {}
    class_total = {}

    all_confs = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            total_loss += loss.item() * features.shape[0]
            total_correct += (preds == labels).sum().item()
            total_samples += features.shape[0]

            all_confs.append(confs.cpu())

            # Per-class
            for cls in labels.unique().tolist():
                mask = labels == cls
                class_correct[cls] = class_correct.get(cls, 0) + (preds[mask] == labels[mask]).sum().item()
                class_total[cls] = class_total.get(cls, 0) + mask.sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)

    per_class_acc = {
        cls: 100.0 * class_correct[cls] / max(class_total[cls], 1)
        for cls in class_correct
    }

    all_confs_tensor = torch.cat(all_confs) if all_confs else torch.tensor([0.0])

    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class_acc": per_class_acc,
        "conf_mean": all_confs_tensor.mean().item(),
        "conf_min": all_confs_tensor.min().item(),
        "conf_max": all_confs_tensor.max().item(),
        "num_samples_evaluated": total_samples,
    }

    logger.info(
        f"[Eval on Dataset_A] loss={avg_loss:.4f} | "
        f"acc={accuracy:.2f}% | "
        f"conf_mean={results['conf_mean']:.3f}"
    )
    return results


# ---------------------------------------------------------------------------
# Per-sample confidence scores  (used for replay buffer seeding)
# ---------------------------------------------------------------------------

def compute_confidence(
    model: nn.Module,
    dataset: SyntheticDataset,
    config: EvolveConfig,
) -> torch.Tensor:
    """
    Compute per-sample softmax confidence (max probability) for an entire dataset.

    Returns:
        confidences : (N,) tensor of max softmax probs
    """
    device = config.get_device()
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    all_confs = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            confs, _ = probs.max(dim=-1)
            all_confs.append(confs.cpu())

    return torch.cat(all_confs)
