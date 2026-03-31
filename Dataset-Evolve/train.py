"""
train.py – Training loop for Operation Evolve.

Features:
  - Per-epoch loss logging
  - Early stopping (patience-based) watching validation loss
  - Best model state saved per loop
  - Replay buffer integration: mixes historical samples into training data
"""

import copy
import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from config import EvolveConfig
from data import SyntheticDataset, ReplayBuffer

logger = logging.getLogger("evolve.train")


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one full pass over the DataLoader.

    Returns:
        avg_loss (float): mean cross-entropy loss over all batches.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.shape[0]
        total_samples += features.shape[0]

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Full training loop (with early stopping + replay)
# ---------------------------------------------------------------------------

def train_loop(
    model: nn.Module,
    dataset_b: SyntheticDataset,
    config: EvolveConfig,
    replay_buffer: Optional[ReplayBuffer] = None,
    loop_idx: int = 0,
) -> Dict[str, Any]:
    """
    Train the model for up to config.epochs_per_loop epochs.

    Integrates replay buffer: mixes a fraction of historical samples into
    each training epoch to prevent catastrophic forgetting.

    Early stopping:
        If training loss does not improve for config.early_stopping_patience
        consecutive epochs, training stops early and the best model is restored.

    Returns a dict:
        {
            "best_state"  : best model state_dict,
            "history"     : list of per-epoch {"epoch", "loss"} dicts,
            "stopped_at"  : epoch at which training stopped,
        }
    """
    device = config.get_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # Cosine Annealing + Warmup
    warmup_epochs = max(1, config.epochs_per_loop // 5)
    sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs_per_loop - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched1, sched2], milestones=[warmup_epochs]
    )

    # --- Build training dataset (Dataset_B + replay samples) ---
    if replay_buffer is not None and len(replay_buffer) > 0:
        n_replay = int(len(dataset_b) * config.replay_mix_ratio)
        replay_sample = replay_buffer.sample(n_replay)
        if replay_sample is not None:
            train_data = ConcatDataset([dataset_b, replay_sample])
            logger.info(
                f"[Train] Replay buffer contributing {len(replay_sample)} samples "
                f"to training (total train size: {len(train_data)})"
            )
        else:
            train_data = dataset_b
    else:
        train_data = dataset_b

    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # --- Early stopping state ---
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: List[Dict[str, Any]] = []
    stopped_at = config.epochs_per_loop

    logger.info(
        f"[Loop {loop_idx}] Starting training | "
        f"dataset_B={len(dataset_b)} | epochs={config.epochs_per_loop} | "
        f"device={device}"
    )

    for epoch in range(1, config.epochs_per_loop + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "loss": loss})

        if epoch % config.print_every_n_epochs == 0:
            logger.info(
                f"  [Loop {loop_idx}] Epoch {epoch}/{config.epochs_per_loop} "
                f"| Loss: {loss:.4f}"
            )

        # Early stopping check
        if loss < best_loss - 1e-4:
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(
                    f"  [Loop {loop_idx}] Early stopping at epoch {epoch} "
                    f"(patience={config.early_stopping_patience})"
                )
                stopped_at = epoch
                break

    # Restore best weights found during training
    model.load_state_dict(best_state)
    logger.info(
        f"[Loop {loop_idx}] Training complete | best_loss={best_loss:.4f} | "
        f"stopped_at_epoch={stopped_at}"
    )

    return {
        "best_state": best_state,
        "history": history,
        "stopped_at": stopped_at,
        "best_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Hybrid Continual Training loop (LwF + Replay, architecture-aware)
# ---------------------------------------------------------------------------

def continual_train_loop(
    model: nn.Module,
    dataset_b: SyntheticDataset,
    config: EvolveConfig,
    replay_buffer=None,           # ReplayBufferV2 | None
    model_old: Optional[nn.Module] = None,
    stable_mode: bool = True,
    loop_idx: int = 0,
) -> Dict[str, Any]:
    """
    Hybrid continual training loop supporting both stable and evolve modes.

    Stable mode (same architecture as last loop):
        loss = (1 - alpha) * CE(logits, labels) + alpha * KL(logits_new || logits_old)
             = LwF combined loss   (requires model_old to be set)

    Evolve mode (architecture changed between loops):
        loss = CE(logits, labels)   only
        LwF is SKIPPED because architectures differ and old logits are invalid.

    Replay buffer integration:
        Samples int(batch_size * config.replay_sample_ratio) items from the
        reservoir replay buffer and prepends them to every mini-batch.

    Parameters
    ----------
    model        : nn.Module  (student, mutable)
    dataset_b    : SyntheticDataset  (current task training data)
    config       : EvolveConfig
    replay_buffer: ReplayBufferV2 | None
    model_old    : nn.Module | None  (frozen teacher clone; needed in stable mode)
    stable_mode  : bool  (True = LwF enabled if model_old present)
    loop_idx     : int   (logged for readability)

    Returns
    -------
    dict with keys: best_state, history, stopped_at, best_loss, mode
    """
    device = config.get_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_epochs = max(1, config.epochs_per_loop // 5)
    sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs_per_loop - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched1, sched2], milestones=[warmup_epochs]
    )

    use_lwf = stable_mode and (model_old is not None)
    mode_str = "stable (LwF)" if use_lwf else ("stable" if stable_mode else "evolve")
    logger.info(
        f"[Loop {loop_idx}] [Mode] {mode_str} | "
        f"dataset_B={len(dataset_b)} | epochs={config.epochs_per_loop} | "
        f"device={device}"
    )
    if use_lwf:
        logger.info(
            f"[Loop {loop_idx}] LwF alpha={config.lwf_alpha} T={config.lwf_temperature}"
        )
        # Move teacher to same device
        model_old = model_old.to(device)

    # ── Build training dataset (current + replay) ────────────────────────
    n_replay = int(config.batch_size * config.replay_sample_ratio)
    train_data = dataset_b
    if replay_buffer is not None and len(replay_buffer) > 0 and n_replay > 0:
        replay_sample = replay_buffer.sample(n_replay * 10)   # pre-sample a pool
        if replay_sample is not None:
            train_data = ConcatDataset([dataset_b, replay_sample])
            logger.info(
                f"[Loop {loop_idx}] Replay contributing {len(replay_sample)} samples "
                f"(total train size: {len(train_data)})"
            )

    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ── Training state ───────────────────────────────────────────────────
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: List[Dict[str, Any]] = []
    stopped_at = config.epochs_per_loop

    for epoch in range(1, config.epochs_per_loop + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if use_lwf:
                # Import lazily to avoid hard circular dep at module load
                from lwf import compute_lwf_loss  # noqa: PLC0415
                loss = compute_lwf_loss(
                    model_new=model,
                    model_old=model_old,
                    features=features,
                    labels=labels,
                    criterion=criterion,
                    alpha=config.lwf_alpha,
                    temperature=config.lwf_temperature,
                )
            else:
                logits = model(features)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.shape[0]
            total_samples += features.shape[0]

        epoch_loss = total_loss / max(total_samples, 1)
        scheduler.step()
        history.append({"epoch": epoch, "loss": epoch_loss})

        if epoch % config.print_every_n_epochs == 0:
            logger.info(
                f"  [Loop {loop_idx}] Epoch {epoch}/{config.epochs_per_loop} "
                f"| Loss: {epoch_loss:.4f}"
            )

        # Early stopping
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(
                    f"  [Loop {loop_idx}] Early stopping at epoch {epoch} "
                    f"(patience={config.early_stopping_patience})"
                )
                stopped_at = epoch
                break

    model.load_state_dict(best_state)
    logger.info(
        f"[Loop {loop_idx}] continual_train_loop done | "
        f"mode={mode_str} | best_loss={best_loss:.4f} | stopped_at={stopped_at}"
    )

    return {
        "best_state": best_state,
        "history": history,
        "stopped_at": stopped_at,
        "best_loss": best_loss,
        "mode": mode_str,
    }
