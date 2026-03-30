import os
import json
import time
import copy
import threading
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from config import EvolveConfig
from data import SyntheticDataset, generate_seed_data, save_dataset
from replay_buffer import ReplayBufferV2 as ReplayBuffer  # Using the V2 from earlier if available, else standard
from model import build_model
from train import continual_train_loop
from evaluate import evaluate
from utils import setup_logging, AccuracyTracker
from groq_fetcher import fetch_internet_dataset

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("evolve.combine")

CONFIG_FILE = "config.json"
HISTORY_FILE = "evolution_log.json"
LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Background Fetcher
# ---------------------------------------------------------------------------
def _run_fetch_job(config: EvolveConfig, dataset_buffer: List[SyntheticDataset], target_idx: int):
    """Background thread to fetch new dataset."""
    logger.info(f"[Async Fetch] Start fetching Dataset {target_idx + 1}...")
    try:
        new_ds = fetch_internet_dataset(config)
        dataset_buffer[target_idx] = new_ds
        save_dataset(new_ds, f"dataset_{target_idx + 1}", config.data_dir)
        logger.info(f"[Async Fetch] ✓ Dataset {target_idx + 1} refreshed and ready.")
    except Exception as e:
        logger.error(f"[Async Fetch] ! Failed to fetch data: {e}. Keeping old dataset in Slot {target_idx + 1}.")


# ---------------------------------------------------------------------------
# Groq Agent (Model Evolve logic)
# ---------------------------------------------------------------------------
def call_agent_for_new_config(current_config: Dict[str, Any], current_metrics: Dict[str, Any], history_str: str) -> Tuple[Dict[str, Any], str]:
    """Hits the Groq API to propose a new architecture without updating weights."""
    logger.info("[Agent] Requesting architecture evolution from Groq...")
    
    prompt = f"""You are 'Operation Evolve'. 
Your goal is to optimize a Hierarchical Mixture-of-Experts (MoE) model.
CURRENT CONFIG: {json.dumps(current_config, indent=2)}
METRICS: {json.dumps(current_metrics, indent=2)}
HISTORY:
{history_str} 
    
Propose a NEW configuration JSON. You can modify:
- n_embd (e.g. 64, 128)
- expert_hidden_dim
- num_groups
- experts_per_group
- num_heads
- num_layers
- dropout
- router_temperature

Must output ONLY a JSON object with keys "rationale" and "config". No markdown blocks.
"""
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(LLM_API_URL, headers=headers, json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        })
        response.raise_for_status()
        response_data = response.json()
        
        response_text = response_data['choices'][0]['message']['content']
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        new_config_str = response_text[json_start:json_end]
        
        parsed = json.loads(new_config_str)
        proposed_config = parsed.get("config", current_config)
        rationale = parsed.get("rationale", "No rationale")
        
        merged = current_config.copy()
        merged.update(proposed_config)
        return merged, rationale
    except Exception as e:
        logger.warning(f"[Agent] Failed to get valid config from AI: {e}. Proceeding with stable architecture.")
        return current_config, "Agent failed or timed out."


# ---------------------------------------------------------------------------
# The Combined Orchestrator
# ---------------------------------------------------------------------------
def run_combined_orchestrator():
    # 1. Setup Base Configuration
    config = EvolveConfig()
    setup_logging(config.log_level)
    config.ensure_data_dir()
    device = config.get_device()
    
    logger.info("==========================================================")
    logger.info("  Operation-Combine: Infinite Pipelined LwF Evolution     ")
    logger.info("==========================================================")
    
    # Write init config
    base_config_dict = {
        "n_embd": config.d_model,
        "expert_hidden_dim": config.expert_hidden_dim,
        "num_groups": config.num_groups,
        "experts_per_group": config.experts_per_group,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "dropout": config.dropout
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(base_config_dict, f, indent=2)
        
    history_log = []

    # 2. Prepare 3 Datasets
    logger.info("[Init] Spawning Datasets 1, 2, and 3...")
    dataset_buffer: List[SyntheticDataset] = []
    for i in range(3):
        ds = generate_seed_data(n=200, num_classes=config.num_classes, input_dim=config.input_dim, seed=42+i)
        dataset_buffer.append(ds)
        
    eval_dataset = generate_seed_data(n=300, num_classes=config.num_classes, input_dim=config.input_dim, seed=999)

    # 3. Model & Memory
    model = build_model(config)
    model.to(device)
    
    # Try using ReplayBufferV2 if exists, else fallback to V1 signature
    try:
        replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size, input_dim=config.input_dim, num_classes=config.num_classes, device=device)
    except TypeError:
        from data import ReplayBuffer as RB1
        replay_buffer = RB1(max_size=config.replay_buffer_size)
    
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    
    loop_idx = 0
    # Following the Flow: Infinite cycle 1-2-3
    while True:
        # Resolve which dataset is currently active, and which is being updated
        current_ds_idx = loop_idx % 3
        
        logger.info(f"\n{'-'*60}")
        logger.info(f">>> ENTERING CYCLE {loop_idx + 1} | Training on Dataset {current_ds_idx + 1}")
        logger.info(f"{'-'*60}")
        
        # --- A. MODEL EVOLVE PHASE (Structural Only) ---
        with open(CONFIG_FILE, "r") as f:
            current_config = json.load(f)
            
        history_str = json.dumps(history_log[-5:], indent=2)
        metrics = {"best_val_loss": best_loss if best_loss != float('inf') else 9.99}
        
        proposed_config, rationale = call_agent_for_new_config(current_config, metrics, history_str)
        
        # Compare to see if architecture is actually changing
        stable_mode = (proposed_config == current_config)
        
        if stable_mode:
            logger.info("[Evolve] Architecture remains STABLE. Preparing for LwF Distillation.")
            student_model = model  # Keep current weights
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
        else:
            logger.info(f"[Evolve] Agent PROPOSED NEW ARCHITECTURE: {rationale}")
            logger.info("[Evolve] Building new Pytorch Structure (No Weights Updated)")
            # Build new shell
            config.d_model = proposed_config.get("n_embd", config.d_model)
            config.num_layers = proposed_config.get("num_layers", config.num_layers)
            config.num_heads = proposed_config.get("num_heads", config.num_heads)
            config.expert_hidden_dim = proposed_config.get("expert_hidden_dim", config.expert_hidden_dim)
            
            student_model = build_model(config)
            student_model.to(device)
            teacher_model = None  # No LwF teacher because architectures don't match
            
        # --- B. PARALLEL TASKS (Update D_Next & Train D_Current) ---
        # 1. Async Fetch Thread
        next_ds_idx = (loop_idx + 1) % 3
        fetch_thread = threading.Thread(target=_run_fetch_job, args=(config, dataset_buffer, next_ds_idx))
        fetch_thread.start()
        
        # 2. Main Train Thread
        logger.info(f"[Train] Starting optimization on Dataset {current_ds_idx + 1} (5 Epochs)")
        
        # We cap training at 5 epochs strictly as per user instructions
        config.epochs_per_loop = 5
        
        try:
            train_results = continual_train_loop(
                model=student_model,
                dataset_b=dataset_buffer[current_ds_idx],
                config=config,
                replay_buffer=replay_buffer,
                model_old=teacher_model,
                stable_mode=stable_mode,
                loop_idx=loop_idx + 1
            )
        except Exception as e:
            logger.error(f"[Train] Training loop crashed: {e}")
            train_results = {"best_loss": float('inf'), "best_state": None}

        # Wait for the next dataset to finish fetching
        fetch_thread.join()
        
        # --- C. UPDATE REPLAY BUFFER ---
        logger.info("[Replay] Updating reservoir buffer with knowledge from Dataset...")
        try:
            replay_buffer.populate_from(dataset_buffer[current_ds_idx], n=50) # small subset
        except AttributeError:
            pass # Depending on V1 or V2 implementation
            
        # --- D. EVALUATION & ROLLBACK DECISION ---
        if train_results.get("best_state") is not None:
            student_model.load_state_dict(train_results["best_state"])
            
        eval_metrics = evaluate(student_model, eval_dataset, config)
        test_loss = eval_metrics.get("loss", float('inf'))
        
        logger.info(f"[Evaluate] Validation Loss: {test_loss:.4f} (Previous Best: {best_loss:.4f})")
        
        if test_loss < best_loss or loop_idx == 0:
            logger.info("✅ ACCEPTED! The architecture and weights will be preserved.")
            best_loss = test_loss
            model = student_model  # Commit model
            best_weights = copy.deepcopy(model.state_dict())
            
            with open(CONFIG_FILE, "w") as f:
                json.dump(proposed_config, f, indent=2)
                
            status = "ACCEPTED"
        else:
            logger.warning("❌ REJECTED! Performance degraded. Rolling back to previous architecture & weights.")
            # We explicitly discard `student_model` and `proposed_config`.
            # `model` stays the same as before.
            model.load_state_dict(best_weights)
            status = "REJECTED (Rollback)"
            
        # Log History
        history_log.append({
            "cycle": loop_idx + 1,
            "dataset_slot": current_ds_idx + 1,
            "rationale": rationale,
            "test_loss": test_loss,
            "status": status
        })
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_log, f, indent=2)
            
        # Increment loop
        loop_idx += 1
        time.sleep(2)
        
if __name__ == "__main__":
    run_combined_orchestrator()
