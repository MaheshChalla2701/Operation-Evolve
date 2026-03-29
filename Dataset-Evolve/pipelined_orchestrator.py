import os
import threading
import logging
from typing import List

from config import EvolveConfig
from data import SyntheticDataset, ReplayBuffer, generate_seed_data, save_dataset
from model import build_model
from train import train_loop
from evaluate import evaluate
from utils import setup_logging, AccuracyTracker

from groq_fetcher import fetch_internet_dataset

logger = logging.getLogger("evolve.orchestrator")


def _run_fetch_job(config: EvolveConfig, dataset_buffer: List[SyntheticDataset], target_idx: int):
    """Background worker that calls Groq to fetch real-world data and overwrites the target dataset slot."""
    logger.info(f"[Fetch Thread] Initiating Groq fetch for Dataset Slot {target_idx + 1}...")
    try:
        new_ds = fetch_internet_dataset(config)
        dataset_buffer[target_idx] = new_ds
        
        # Physically save to disk so the user can "see" the files
        file_name = f"dataset_{target_idx + 1}"
        save_dataset(new_ds, file_name, config.data_dir)
        
        logger.info(f"[Fetch Thread] ✓ Dataset Slot {target_idx + 1} refreshed and saved to {config.data_dir}/{file_name}.pt")
    except Exception as e:
        logger.error(f"[Fetch Thread] ! Failed to fetch data: {e}. Retaining old dataset in Slot {target_idx + 1}.")


def run_pipelined_evolution(config: EvolveConfig | None = None) -> None:
    if config is None:
        config = EvolveConfig()

    setup_logging(config.log_level)
    config.ensure_data_dir()
    device = config.get_device()

    logger.info("="*60)
    logger.info("  Operation Evolve - Pipelined Continual Learning (1-2-3 Cycle)")
    logger.info("="*60)
    
    # ── Initialize 3-Dataset Buffer ──
    # Create the initial dummy datasets before internet data kicks in
    logger.info("[Init] Generating initial starting states for Datasets 1, 2, and 3.")
    dataset_buffer: List[SyntheticDataset] = []
    for i in range(config.num_datasets_in_buffer):
        ds = generate_seed_data(
            n=config.dataset_b_initial_size,
            num_classes=config.num_classes,
            input_dim=config.input_dim,
            seed=42 + i
        )
        dataset_buffer.append(ds)

    # Fixed Evaluation / Validation Dataset (Dataset A concept)
    eval_dataset = generate_seed_data(
        n=config.dataset_a_size,
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        seed=100
    )

    # ── Initialize Model & Replay Buffer ──
    model = build_model(config)
    logger.info(f"[Init] Model: {config.model_type} on {device}")
    
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    tracker = AccuracyTracker()

    loop_idx = 0
    
    # Run the continuous looping pipeline
    while loop_idx < config.num_evolution_loops:
        current_ds_idx = loop_idx % config.num_datasets_in_buffer
        next_ds_idx = (loop_idx + 1) % config.num_datasets_in_buffer
        
        logger.info(f"\n{'-'*60}")
        logger.info(f"  CYCLE {loop_idx + 1}  |  Training: DS {current_ds_idx + 1}  |  Fetching: DS {next_ds_idx + 1}")
        logger.info(f"{'-'*60}")

        # Start Fetching thread for the NEXT dataset
        fetch_thread = threading.Thread(
            target=_run_fetch_job, 
            args=(config, dataset_buffer, next_ds_idx)
        )
        fetch_thread.start()

        # Simultaneously TRAIN on the CURRENT dataset (Main thread)
        current_dataset = dataset_buffer[current_ds_idx]
        
        logger.info(f"[Train Thread] Training on Dataset Slot {current_ds_idx + 1} with {len(current_dataset)} samples.")
        train_loop(
            model=model,
            dataset_b=current_dataset,
            config=config,
            replay_buffer=replay_buffer,
            loop_idx=loop_idx + 1
        )
        
        # Add high-quality bits of current dataset to replay buffer so it isn't forgotten when we move on
        replay_buffer.populate_from(current_dataset, n=int(config.groq_dataset_size * 0.2))

        # Wait for the fetch thread to finish before advancing to the next cycle
        fetch_thread.join()
        
        # Evaluate performance on the fixed holdout set
        eval_results = evaluate(model, eval_dataset, config)
        tracker.record(eval_results["accuracy"])
        
        logger.info(f"[Cycle {loop_idx + 1} Complete] Validation Accuracy: {eval_results['accuracy']:.2f}%")
        
        loop_idx += 1

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Final Accuracy trend: {tracker.trend_str()}")


if __name__ == "__main__":
    run_pipelined_evolution()
