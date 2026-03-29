"""
main.py – Hybrid Continual Learning Orchestrator for Operation Evolve.

Implements the 11-step per-iteration loop:

  1.  Snapshot current config into prev_* fields
  2.  [Optional] Evolve config (change hidden_dim / num_heads / model_type)
  3.  Detect architecture mode: same_architecture_from_prev(config) → bool
  4.  If architecture changed: rebuild model, transfer embeddings where possible
  5.  Clone model → model_old (teacher for LwF)
  6.  continual_train_loop(model, dataset_b, config, replay_buffer,
                           model_old=model_old, stable_mode=same_mode)
  7.  current_acc = evaluate(model, dataset_a, config)
  8.  replay_acc  = evaluate_replay(model, replay_buffer, config)
  9.  Rollback if:
        - current_acc < best_acc - tolerance   OR
        - replay_acc drops > 10 pp below best replay accuracy seen
 10.  replay_buffer.update(filtered_c)   (reservoir sampling absorbs current task)
 11.  Update prev_* fields in config via apply_snapshot()

Run from the Dataset-Evolve directory:
    python main.py

To test evolve mode, set EVOLVE_ARCH_TEST=1 in your environment – the
orchestrator will toggle hidden_dim every other loop to simulate arch drift.
"""

import copy
import logging
import os
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from config import EvolveConfig
from data import (
    SyntheticDataset,
    generate_seed_data,
    generate_dataset_c,
    filter_by_confidence,
)
from model import build_model, BaseModel
from train import continual_train_loop
from evaluate import evaluate, evaluate_replay
from lwf import clone_model
from replay_buffer import ReplayBufferV2
from arch_utils import same_architecture_from_prev, snapshot_arch, apply_snapshot
from utils import setup_logging, AccuracyTracker

logger = logging.getLogger("evolve.main")


# ---------------------------------------------------------------------------
# Weight transfer helper
# ---------------------------------------------------------------------------

def _transfer_compatible_weights(
    model_new: BaseModel,
    model_old: BaseModel,
) -> int:
    """
    Copy parameter tensors whose names AND shapes match from old → new model.

    This preserves as much learned knowledge as possible when the architecture
    evolves (e.g. only num_layers changes, so most layers are compatible).

    Returns
    -------
    int : number of parameters transferred
    """
    old_state = model_old.state_dict()
    new_state = model_new.state_dict()
    transferred = 0
    for name, param in new_state.items():
        if name in old_state and old_state[name].shape == param.shape:
            new_state[name] = old_state[name].clone()
            transferred += param.numel()
    model_new.load_state_dict(new_state)
    return transferred


# ---------------------------------------------------------------------------
# Optional arch evolution (for testing / LLM-agent-driven changes)
# ---------------------------------------------------------------------------

def _maybe_evolve_arch(config: EvolveConfig, loop_idx: int) -> bool:
    """
    Optionally mutate config.hidden_dim to simulate architecture evolution.

    Only active when the environment variable EVOLVE_ARCH_TEST=1 is set.
    Toggles hidden_dim between 64 and 128 every other loop so you can see
    both stable and evolve modes within a single run.

    Returns True if the architecture was changed.
    """
    if os.environ.get("EVOLVE_ARCH_TEST", "0") != "1":
        return False
    if loop_idx % 2 == 1:
        new_dim = 128 if config.hidden_dim == 64 else 64
        logger.info(
            f"[ArchTest] Toggling hidden_dim: {config.hidden_dim} → {new_dim}"
        )
        config.hidden_dim = new_dim
        return True
    return False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints"


def _checkpoint_path(loop_idx: int, data_dir: str) -> str:
    """Return the path for a per-loop checkpoint file."""
    d = os.path.join(data_dir, CHECKPOINT_DIR)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"checkpoint_loop_{loop_idx}.pt")


def _best_model_path(data_dir: str) -> str:
    return os.path.join(data_dir, CHECKPOINT_DIR, "best_model.pt")


def _replay_buffer_path(data_dir: str) -> str:
    return os.path.join(data_dir, CHECKPOINT_DIR, "replay_buffer.pt")


def _save_checkpoint(
    model,
    loop_idx: int,
    acc: float,
    config,
    data_dir: str,
) -> str:
    """
    Save model weights + metadata to a per-loop checkpoint file.

    Saved dict keys:
        model_state  : model.state_dict()
        loop_idx     : int
        accuracy     : float
        model_type   : str
        hidden_dim   : int
        num_heads    : int
        num_layers   : int
        experts      : int
    """
    path = _checkpoint_path(loop_idx, data_dir)
    payload = {
        "model_state": model.state_dict(),
        "loop_idx": loop_idx,
        "accuracy": acc,
        "model_type": config.model_type,
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "experts": config.experts,
    }
    torch.save(payload, path)
    logger.info(f"[Checkpoint] Saved loop {loop_idx} → {path}  (acc={acc:.2f}%)")
    return path


def _save_best_model(model, acc: float, config, data_dir: str) -> None:
    """Overwrite best_model.pt whenever Dataset_A accuracy improves."""
    path = _best_model_path(data_dir)
    payload = {
        "model_state": model.state_dict(),
        "accuracy": acc,
        "model_type": config.model_type,
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "experts": config.experts,
    }
    torch.save(payload, path)
    logger.info(f"[BestModel] Saved → {path}  (acc={acc:.2f}%)")


def _find_latest_checkpoint(data_dir: str) -> Optional[str]:
    """Return the path of the highest-numbered checkpoint, or None."""
    d = os.path.join(data_dir, CHECKPOINT_DIR)
    if not os.path.isdir(d):
        return None
    files = [
        f for f in os.listdir(d)
        if f.startswith("checkpoint_loop_") and f.endswith(".pt")
    ]
    if not files:
        return None
    # Sort by loop number
    files.sort(key=lambda f: int(f.replace("checkpoint_loop_", "").replace(".pt", "")))
    return os.path.join(d, files[-1])


# ---------------------------------------------------------------------------
# Main hybrid continual learning loop
# ---------------------------------------------------------------------------

def run_hybrid_continual_learning(config: Optional[EvolveConfig] = None) -> None:
    """
    Main orchestration function for Hybrid Continual Learning.

    Designed to handle 100+ sequential datasets without catastrophic forgetting
    by combining:
      - Reservoir replay buffer (ReplayBufferV2)
      - Learning without Forgetting (LwF) when architecture is stable
      - Graceful architecture evolution with weight transfer + CE-only training

    Parameters
    ----------
    config : EvolveConfig | None
        Uses DEFAULT_CONFIG if None.
    """
    if config is None:
        config = EvolveConfig()

    setup_logging(config.log_level)
    config.ensure_data_dir()
    # Ensure checkpoints subdirectory exists before any path resolution
    os.makedirs(os.path.join(config.data_dir, CHECKPOINT_DIR), exist_ok=True)
    device = config.get_device()

    logger.info("=" * 70)
    logger.info("  Operation Evolve — Hybrid Continual Learning System")
    logger.info("=" * 70)
    logger.info(f"  Loops: {config.num_evolution_loops} | Device: {device}")
    logger.info(f"  Buffer capacity: {config.buffer_size} | LwF alpha: {config.lwf_alpha}")
    logger.info("=" * 70)

    # ── Immutable validation dataset (Dataset A) ─────────────────────────
    dataset_a = generate_seed_data(
        n=config.dataset_a_size,
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        seed=100,
    )
    logger.info(f"[Init] Dataset_A: {dataset_a}")

    # ── Initial training dataset (Dataset B) ─────────────────────────────
    dataset_b = generate_seed_data(
        n=config.dataset_b_initial_size,
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        seed=42,
    )
    logger.info(f"[Init] Dataset_B (initial): {dataset_b}")

    # ── Build initial model ───────────────────────────────────────────────
    model = build_model(config)
    logger.info(
        f"[Init] Model: {config.model_type} | params={model.count_parameters()}"
    )

    # ── Reservoir replay buffer ───────────────────────────────────────────
    replay_buffer_path = _replay_buffer_path(config.data_dir)
    if os.path.exists(replay_buffer_path):
        replay_buffer = ReplayBufferV2.load(replay_buffer_path)
        logger.info(f"[Init] Resumed replay buffer from {replay_buffer_path}")
    else:
        replay_buffer = ReplayBufferV2(max_size=config.buffer_size)

    # ── Auto-resume from latest checkpoint ───────────────────────────────
    start_loop = 1
    latest_ckpt = _find_latest_checkpoint(config.data_dir)
    if latest_ckpt is not None:
        ckpt = torch.load(latest_ckpt, weights_only=True)
        # Only resume if the architecture still matches config
        arch_match = (
            ckpt.get("model_type") == config.model_type
            and ckpt.get("hidden_dim") == config.hidden_dim
            and ckpt.get("num_heads") == config.num_heads
            and ckpt.get("num_layers") == config.num_layers
        )
        if arch_match:
            model.load_state_dict(ckpt["model_state"])
            start_loop = ckpt["loop_idx"] + 1
            logger.info(
                f"[Init] Resumed from checkpoint: {latest_ckpt} "
                f"(loop={ckpt['loop_idx']}, acc={ckpt['accuracy']:.2f}%) "
                f"→ continuing from loop {start_loop}"
            )
        else:
            logger.warning(
                f"[Init] Checkpoint architecture mismatch — starting fresh "
                f"(ckpt={ckpt.get('model_type')}/{ckpt.get('hidden_dim')} "
                f"vs config={config.model_type}/{config.hidden_dim})"
            )

    # ── Accuracy trackers ─────────────────────────────────────────────────
    tracker = AccuracyTracker()
    best_acc = 0.0
    best_replay_acc = -1.0       # -1 means "not yet measured"
    ROLLBACK_TOL = 2.0           # pp tolerance before triggering rollback
    REPLAY_DROP_THRESHOLD = 10.0 # pp drop in replay acc triggers rollback

    # ── Snapshot initial arch into prev_* so loop 0 compares correctly ───
    apply_snapshot(config, snapshot_arch(config))
    # Force prev fields to be set (since 0 / "" defaults would skip comparison)
    config.prev_model_type = config.model_type
    config.prev_hidden_dim = config.hidden_dim
    config.prev_num_heads = config.num_heads
    config.prev_num_layers = config.num_layers
    config.prev_experts = config.experts

    # ═════════════════════════════════════════════════════════════════════
    #  MAIN CONTINUAL LEARNING LOOP
    # ═════════════════════════════════════════════════════════════════════
    for loop_idx in range(start_loop, config.num_evolution_loops + 1):
        logger.info(f"\n{'─' * 70}")
        logger.info(f"  LOOP {loop_idx} / {config.num_evolution_loops}")
        logger.info(f"{'─' * 70}")

        # ── STEP 1: Snapshot old arch into prev_* ────────────────────────
        arch_snapshot = snapshot_arch(config)

        # ── STEP 2: (Optional) Evolve architecture ───────────────────────
        arch_changed = _maybe_evolve_arch(config, loop_idx)

        # ── STEP 3: Detect mode ──────────────────────────────────────────
        same_mode = not arch_changed and same_architecture_from_prev(config)
        mode_label = "stable" if same_mode else "evolve"
        logger.info(f"[Loop {loop_idx}] Architecture mode: [{mode_label.upper()}]")

        # ── STEP 4: Rebuild model if architecture changed ─────────────────
        if not same_mode:
            logger.info(
                f"[Loop {loop_idx}] Architecture evolved — rebuilding model "
                f"({arch_snapshot} → {snapshot_arch(config)})"
            )
            old_model_ref = model
            model = build_model(config)
            transferred_params = _transfer_compatible_weights(model, old_model_ref)
            logger.info(
                f"[Loop {loop_idx}] Weight transfer: {transferred_params:,} params copied"
            )

        # ── STEP 5: Clone model as teacher ────────────────────────────────
        model_old = clone_model(model)

        # ── STEP 6: Continual training ────────────────────────────────────
        train_result = continual_train_loop(
            model=model,
            dataset_b=dataset_b,
            config=config,
            replay_buffer=replay_buffer,
            model_old=model_old if same_mode else None,
            stable_mode=same_mode,
            loop_idx=loop_idx,
        )

        # ── STEP 7: Evaluate on Dataset_A ────────────────────────────────
        eval_result = evaluate(model, dataset_a, config)
        current_acc = eval_result["accuracy"]
        logger.info(
            f"[Loop {loop_idx}] Dataset_A acc={current_acc:.2f}% | "
            f"loss={eval_result['loss']:.4f}"
        )

        # ── STEP 8: Evaluate replay buffer retention ──────────────────────
        replay_result = evaluate_replay(model, replay_buffer, config)
        current_replay_acc = replay_result["replay_accuracy"]
        logger.info(
            f"[Loop {loop_idx}] Replay acc={current_replay_acc:.2f}% "
            f"(buffer={len(replay_buffer)} samples)"
        )

        # ── STEP 9: Rollback decision ─────────────────────────────────────
        do_rollback = False

        acc_drop = best_acc - current_acc
        if best_acc > 0 and acc_drop > ROLLBACK_TOL:
            logger.warning(
                f"[Loop {loop_idx}] ⚠ Rollback trigger: Dataset_A acc dropped "
                f"{acc_drop:.2f} pp (best={best_acc:.2f}%, current={current_acc:.2f}%)"
            )
            do_rollback = True

        if (
            best_replay_acc >= 0
            and current_replay_acc >= 0
            and (best_replay_acc - current_replay_acc) > REPLAY_DROP_THRESHOLD
        ):
            logger.warning(
                f"[Loop {loop_idx}] ⚠ Rollback trigger: replay acc dropped "
                f"{best_replay_acc - current_replay_acc:.2f} pp "
                f"(best={best_replay_acc:.2f}%, current={current_replay_acc:.2f}%)"
            )
            do_rollback = True

        if do_rollback:
            logger.info(f"[Loop {loop_idx}] Rolling back to best model state.")
            model.load_state_dict(train_result["best_state"])
            # Re-evaluate after rollback
            eval_result = evaluate(model, dataset_a, config)
            current_acc = eval_result["accuracy"]
            logger.info(f"[Loop {loop_idx}] Post-rollback acc={current_acc:.2f}%")
        else:
            # Update best trackers only on successful loops
            if current_acc > best_acc:
                best_acc = current_acc
                _save_best_model(model, current_acc, config, config.data_dir)
            if current_replay_acc > best_replay_acc:
                best_replay_acc = current_replay_acc

        # ── Save per-loop checkpoint ──────────────────────────────────────
        _save_checkpoint(model, loop_idx, current_acc, config, config.data_dir)

        tracker.record(current_acc)

        # ── STEP 10: Absorb current dataset into replay buffer ────────────
        #   Use generate_dataset_c + filter_by_confidence to select only
        #   high-confidence samples before buffering them.
        try:
            c_raw = generate_dataset_c(model, dataset_b, config)
            filtered_c, accepted, rejected = filter_by_confidence(
                c_raw, config, existing_features=dataset_b.features
            )
            if accepted > 0:
                replay_buffer.update(filtered_c)
                logger.info(
                    f"[Loop {loop_idx}] Absorbed {accepted} samples into replay "
                    f"(rejected={rejected}) | buffer={len(replay_buffer)}"
                )
            else:
                # Fallback: add raw dataset samples if all were filtered out
                replay_buffer.update(dataset_b)
                logger.info(
                    f"[Loop {loop_idx}] No confident samples; fell back to raw "
                    f"Dataset_B | buffer={len(replay_buffer)}"
                )
        except AttributeError:
            # Config may not have confidence_threshold / diversity_threshold
            # (those fields live in a different branch). Fallback gracefully.
            replay_buffer.update(dataset_b)
            logger.info(
                f"[Loop {loop_idx}] Absorbed Dataset_B into replay (direct) "
                f"| buffer={len(replay_buffer)}"
            )

        # ── STEP 11: Update prev_* from the snapshot we took in step 1 ───
        apply_snapshot(config, snapshot_arch(config))

        # Generate fresh Dataset_B for next loop (synthetic rotation)
        # In a real 100+ dataset scenario, replace this with disk / API loading.
        dataset_b = generate_seed_data(
            n=config.dataset_b_initial_size,
            num_classes=config.num_classes,
            input_dim=config.input_dim,
            seed=42 + loop_idx,   # different seed each loop = different distribution
        )
        logger.info(f"[Loop {loop_idx}] Next Dataset_B: {dataset_b}")

    # ═════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    # ── Persist replay buffer for next run ───────────────────────────────
    replay_buffer.save(replay_buffer_path)

    logger.info("\n" + "=" * 70)
    logger.info("  HYBRID CONTINUAL LEARNING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Accuracy trend : {tracker.trend_str()}")
    logger.info(f"  Best acc       : {best_acc:.2f}%")
    logger.info(f"  Best replay acc: {best_replay_acc:.2f}%")
    logger.info(f"  Buffer size    : {len(replay_buffer)} / {config.buffer_size}")
    logger.info(f"  Checkpoints    : {os.path.join(config.data_dir, CHECKPOINT_DIR)}/")
    logger.info(f"  Best model     : {_best_model_path(config.data_dir)}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_hybrid_continual_learning()
