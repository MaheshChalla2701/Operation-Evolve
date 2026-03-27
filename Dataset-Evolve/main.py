"""
main.py – Orchestrator for the Operation Evolve self-evolving AI pipeline.

Full evolution loop per iteration:
  1.  Train model on Dataset_B  (+ replay buffer)
  2.  Evaluate on Dataset_A      (true validation – never seen during training)
  3.  Generate Dataset_C         (model predictions on Dataset_B inputs)
  4.  Filter Dataset_C           (confidence + diversity)
  5.  Agent proposes updates     (add high-conf, remove noisy samples)
  6.  Mix: new_B = 0.7*old_B + 0.3*filtered_C
  7.  Apply agent proposals      (controlled surgical updates)
  8.  Validate improvement       (rollback if acc drops beyond tolerance)
  9.  Save dataset + checkpoint  (versioned)
  10. Update replay buffer
  11. Print loop summary

Usage:
    python main.py

Compatible with CPU, GPU, and Google Colab.
"""

import os
import copy
import logging
import sys

import torch

# ── Local modules ────────────────────────────────────────────────────────────
from config import EvolveConfig
from data import (
    SyntheticDataset,
    ReplayBuffer,
    generate_seed_data,
    generate_dataset_c,
    filter_by_confidence,
    mix_datasets,
    save_dataset,
    save_dataset_version,
    load_dataset_version,
)
from model import build_model
from train import train_loop
from evaluate import evaluate, compute_confidence
from agent import EvolveAgent, LLMAgent, apply_updates
from utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    print_loop_summary,
    AccuracyTracker,
)

logger = logging.getLogger("evolve.main")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: checkpoint path for a given version
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(config: EvolveConfig, version: int) -> str:
    return os.path.join(config.data_dir, f"checkpoint_v{version}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evolution(config: EvolveConfig | None = None) -> None:
    """
    Run the full self-evolving AI training pipeline.

    Args:
        config : EvolveConfig instance. Uses defaults if None.
    """
    if config is None:
        config = EvolveConfig()

    # ── Setup ─────────────────────────────────────────────────────────────────
    setup_logging(config.log_level)
    config.ensure_data_dir()
    device = config.get_device()

    logger.info(f"{'='*60}")
    logger.info(f"  Operation Evolve – Self-Evolving AI Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"  Device        : {device}")
    logger.info(f"  Model type    : {config.model_type}")
    logger.info(f"  Evolution loops: {config.num_evolution_loops}")
    logger.info(f"  Epochs/loop   : {config.epochs_per_loop}")
    logger.info(f"  Conf threshold: {config.confidence_threshold}")
    logger.info(f"  Mix ratio (B) : {config.dataset_keep_ratio:.0%} old_B + "
                f"{1-config.dataset_keep_ratio:.0%} filtered_C")
    logger.info(f"{'='*60}")

    # ── Generate seed datasets ────────────────────────────────────────────────
    logger.info("[Init] Generating seed datasets …")

    dataset_a = generate_seed_data(
        n=config.dataset_a_size,
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        seed=42,
    )
    dataset_b = generate_seed_data(
        n=config.dataset_b_initial_size,
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        seed=7,
    )

    # Dataset_A is read-only – save once and never modify
    save_dataset(dataset_a, "dataset_A", config.data_dir)
    logger.info(f"[Init] Dataset_A: {dataset_a}  ← READ-ONLY (validation only)")
    logger.info(f"[Init] Dataset_B: {dataset_b}  ← training set (will evolve)")

    # Save initial Dataset_B as version 0
    save_dataset_version(dataset_b, "dataset_B", version=0, data_dir=config.data_dir)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(config)
    logger.info(
        f"[Init] Model: {config.model_type} | "
        f"Parameters: {model.count_parameters():,}"
    )

    # Save initial checkpoint as version 0
    save_checkpoint(model, _ckpt_path(config, 0))

    # ── Agent + Replay buffer ─────────────────────────────────────────────────
    if getattr(config, "use_llm_agent", False):
        logger.info("[Init] Using LLMAgent backed by Groq.")
        from agent import LLMAgent
        agent = LLMAgent(config)
    else:
        logger.info("[Init] Using rule-based EvolveAgent.")
        agent = EvolveAgent(removal_margin=0.15)
        
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    # Seed replay buffer with a small slice of Dataset_A
    replay_buffer.populate_from(dataset_a, n=config.replay_buffer_size // 4)
    logger.info(f"[Init] Replay buffer seeded with {len(replay_buffer)} samples from Dataset_A")

    # ── Tracking state ────────────────────────────────────────────────────────
    tracker = AccuracyTracker()
    best_accuracy = 0.0
    best_version = 0
    current_version = 0

    # ── Evolution loop ────────────────────────────────────────────────────────
    for loop_idx in range(config.num_evolution_loops):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  STARTING EVOLUTION LOOP {loop_idx + 1} / {config.num_evolution_loops}")
        logger.info(f"{'─'*60}")

        # ── STEP 1: Train on Dataset_B ─────────────────────────────────────
        train_result = train_loop(
            model=model,
            dataset_b=dataset_b,
            config=config,
            replay_buffer=replay_buffer,
            loop_idx=loop_idx + 1,
        )

        # ── STEP 2: Evaluate on Dataset_A (true validation) ───────────────
        eval_results = evaluate(model, dataset_a, config)
        current_accuracy = eval_results["accuracy"]
        tracker.record(current_accuracy)

        # ── STEP 3: Generate Dataset_C ────────────────────────────────────
        dataset_c_raw = generate_dataset_c(model, dataset_b, config)
        c_size = dataset_c_raw["features"].shape[0]

        # ── STEP 4: Filter Dataset_C (confidence + diversity) ─────────────
        filtered_c, accepted, rejected = filter_by_confidence(
            dataset_c_raw,
            config,
            existing_features=dataset_b.features,
        )

        # ── STEP 5: Agent proposes updates ────────────────────────────────
        proposal = agent.propose_updates(
            dataset_b=dataset_b,
            dataset_c=dataset_c_raw,
            eval_results=eval_results,
            config=config,
        )

        # ── STEP 6: Dataset mixing (anti-drift) ───────────────────────────
        # new_B = keep_ratio * old_B  +  (1 - keep_ratio) * filtered_C
        mixed_b = mix_datasets(dataset_b, filtered_c, config)

        # ── STEP 7: Apply agent proposals on top of mixed dataset ─────────
        proposed_b = apply_updates(mixed_b, proposal)

        # ── STEP 8: Validate improvement → rollback if degraded ───────────
        rolled_back = False
        tolerance = config.rollback_tolerance  # percentage points

        if current_accuracy >= (best_accuracy - tolerance):
            # Accept the update
            logger.info(
                f"[Loop {loop_idx+1}] ✓ Accepted | "
                f"acc={current_accuracy:.2f}% >= best={best_accuracy:.2f}% - {tolerance}%"
            )
            # Commit new Dataset_B
            dataset_b = proposed_b
            best_accuracy = max(best_accuracy, current_accuracy)
            current_version += 1
            best_version = current_version

        else:
            # Rollback: restore previous best Dataset_B and checkpoint
            rolled_back = True
            logger.warning(
                f"[Loop {loop_idx+1}] ✗ ROLLBACK | "
                f"acc={current_accuracy:.2f}% < best={best_accuracy:.2f}% - {tolerance}% | "
                f"restoring v{best_version}"
            )
            dataset_b = load_dataset_version(
                "dataset_B", best_version, config.data_dir
            )
            load_checkpoint(model, _ckpt_path(config, best_version))

        # ── STEP 9: Save versioned Dataset_B + checkpoint ─────────────────
        if not rolled_back:
            save_dataset_version(
                dataset_b, "dataset_B", version=current_version, data_dir=config.data_dir
            )
            save_checkpoint(
                model,
                _ckpt_path(config, current_version),
                extra={"loop": loop_idx + 1, "accuracy": current_accuracy},
            )

        # ── STEP 10: Update replay buffer with high-conf samples ──────────
        if len(filtered_c) > 0 and not rolled_back:
            replay_buffer.add(filtered_c.features, filtered_c.labels)
            logger.info(
                f"[Replay] Buffer updated: {len(replay_buffer)} / {config.replay_buffer_size} samples"
            )

        # ── STEP 11: Print loop summary ────────────────────────────────────
        print_loop_summary(
            loop_idx=loop_idx,
            eval_results=eval_results,
            dataset_sizes={
                "A": len(dataset_a),
                "B": len(dataset_b),
                "C": c_size,
            },
            accepted=accepted,
            rejected=rejected,
            train_history=train_result["history"],
            rolled_back=rolled_back,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EVOLUTION COMPLETE")
    print("=" * 65)
    print(f"  Best accuracy  : {tracker.best():.2f}%")
    print(f"  Final Dataset_B: {len(dataset_b)} samples")
    print(f"  Best checkpoint: v{best_version}")
    print(f"\n  Accuracy trend across loops:")
    print(f"  {tracker.trend_str()}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Easy customisation: edit this config and run `python main.py` ---
    cfg = EvolveConfig(
        num_evolution_loops=5,
        epochs_per_loop=5,
        model_type="SimpleNN",          # or "SimpleTransformer"
        dataset_b_initial_size=500,
        dataset_a_size=300,
        num_classes=4,
        input_dim=16,
        confidence_threshold=0.85,
        dataset_keep_ratio=0.7,
        replay_buffer_size=150,
        early_stopping_patience=3,
        rollback_tolerance=0.5,
        log_level="INFO",
    )
    run_evolution(cfg)
