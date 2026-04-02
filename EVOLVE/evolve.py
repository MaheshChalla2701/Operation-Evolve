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
from data import SyntheticDataset, TextDataset, generate_seed_data, generate_text_seed_data, save_dataset
from replay_buffer import ReplayBufferV2 as ReplayBuffer  # Using the V2 from earlier if available, else standard
from model import build_model
from train import continual_train_loop
from evaluate import evaluate
from utils import setup_logging, AccuracyTracker
from groq_fetcher import fetch_internet_dataset
from weight_transfer import transfer_compatible_weights

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("evolve.combine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
HISTORY_FILE = os.path.join(BASE_DIR, "evolution_log.json")
LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Background Fetcher
# ---------------------------------------------------------------------------
def _run_fetch_job(config: EvolveConfig, dataset_buffer: List[SyntheticDataset], target_idx: int, focus_prompt: str = ""):
    """Background thread to fetch new dataset."""
    logger.info(f"[Async Fetch] Start fetching Dataset {target_idx + 1}...")
    try:
        new_ds = fetch_internet_dataset(config, focus_prompt=focus_prompt)
        dataset_buffer[target_idx] = new_ds
        save_dataset(new_ds, f"dataset_{target_idx + 1}", config.data_dir)
        logger.info(f"[Async Fetch] ✓ Dataset {target_idx + 1} refreshed and ready.")
    except Exception as e:
        logger.error(f"[Async Fetch] ! Failed to fetch data: {e}. Keeping old dataset in Slot {target_idx + 1}.")


# ---------------------------------------------------------------------------
# Groq Agent (Model Evolve logic)
# ---------------------------------------------------------------------------
def call_agent_for_new_config(current_config: Dict[str, Any], current_metrics: Dict[str, Any], history_str: str, focus_prompt: str = "") -> Tuple[Dict[str, Any], str]:
    """Hits the Groq API to propose a new architecture without updating weights."""
    logger.info("[Agent] Requesting architecture evolution from Groq...")

    focus_section = ""
    if focus_prompt:
        focus_section = f"""
DOMAIN FOCUS: {focus_prompt}
Bias your proposed architecture choices (e.g. more layers, larger expert dims, more heads)
to perform best on tasks related to this domain.
"""
    
    prompt = f"""You are 'Operation Evolve'. 
Your goal is to optimize a Hierarchical Mixture-of-Experts (MoE) model.
CURRENT CONFIG: {json.dumps(current_config, indent=2)}
METRICS: {json.dumps(current_metrics, indent=2)}
HISTORY:
{history_str} 
{focus_section}
Propose a NEW configuration JSON. You can modify:
- n_embd (e.g. 64, 128)
- expert_hidden_dim
- learning_rate (e.g. 0.001, 0.0005)
- num_groups
- experts_per_group
- num_heads
- num_layers
- dropout
- router_temperature
- top_k
- temperature

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
        }, timeout=30)
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
def _read_prompt_config() -> dict:
    """Read prompt.json and return a dict with 'prompt' and 'run_time_seconds'."""
    prompt_file = os.path.join(BASE_DIR, "prompt.json")
    result = {"prompt": "", "run_time_seconds": None}
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r") as f:
                data = json.load(f)
            result["prompt"] = data.get("prompt", "").strip()
            run_time_str = data.get("run_time", "").strip()
            if run_time_str:
                parts = run_time_str.split(":")
                if len(parts) == 3:
                    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                    result["run_time_seconds"] = h * 3600 + m * 60 + s
                else:
                    logger.warning(f"[Prompt] Invalid run_time format '{run_time_str}'. Expected HH:MM:SS. Running indefinitely.")
        except Exception as e:
            logger.warning(f"[Prompt] Failed to read prompt.json: {e}")
    return result


def _read_focus_prompt() -> str:
    """Convenience wrapper — returns just the focus prompt string."""
    return _read_prompt_config()["prompt"]


def run_combined_orchestrator():
    # 1. Setup Base Configuration
    config = EvolveConfig()
    setup_logging(config.log_level)
    config.ensure_data_dir()
    device = config.get_device()
    
    logger.info("==========================================================")
    logger.info("  EVOLVE: Infinite Pipelined LwF Evolution                ")
    logger.info("==========================================================")

    # Read user focus and run-time limit from prompt.json
    cfg = _read_prompt_config()
    focus_prompt = cfg["prompt"]
    run_time_seconds = cfg["run_time_seconds"]

    if focus_prompt:
        logger.info(f"[EVOLVE] 🎯 Focus mode: '{focus_prompt}'")
    else:
        logger.info("[EVOLVE] No focus set — running generic evolution mode.")

    if run_time_seconds is not None:
        h, rem = divmod(run_time_seconds, 3600)
        m, s = divmod(rem, 60)
        logger.info(f"[EVOLVE] ⏱  Run duration: {h:02d}:{m:02d}:{s:02d} (will stop after {run_time_seconds}s)")
    else:
        logger.info("[EVOLVE] ⏱  No time limit — running infinite loop.")

    start_time = time.time()
    
    # Write init config
    base_config_dict = {
        "n_embd": config.d_model,
        "expert_hidden_dim": config.expert_hidden_dim,
        "num_groups": config.num_groups,
        "experts_per_group": config.experts_per_group,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "router_temperature": config.router_temperature
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(base_config_dict, f, indent=2)
        
    history_log = []

    # 2. Prepare 3 Datasets (LM mode: text sequences; classify mode: numeric clusters)
    logger.info("[Init] Spawning Datasets 1, 2, and 3...")
    lm_mode = config.loss_mode == "lm"

    if lm_mode:
        dataset_buffer = [
            generate_text_seed_data(
                n=config.groq_dataset_size,
                max_seq_len=config.max_seq_len,
                vocab_size=config.vocab_size,
                seed=42 + i,
            )
            for i in range(3)
        ]
        eval_dataset = generate_text_seed_data(
            n=min(200, config.groq_dataset_size),
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=999,
        )
    else:
        dataset_buffer = [
            generate_seed_data(n=200, num_classes=config.num_classes, input_dim=config.input_dim, seed=42 + i)
            for i in range(3)
        ]
        eval_dataset = generate_seed_data(n=300, num_classes=config.num_classes, input_dim=config.input_dim, seed=999)

    # 3. Model & Memory
    model = build_model(config)
    model.to(device)
    
    # Try using ReplayBufferV2 if exists, else fallback to V1 signature
    try:
        replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    except TypeError:
        from data import ReplayBuffer as RB1
        replay_buffer = RB1(max_size=config.replay_buffer_size)
    
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    
    loop_idx = 0
    # Main evolution loop — runs indefinitely or until run_time_seconds is reached
    while True:
        # Check time limit before starting a new cycle
        if run_time_seconds is not None:
            elapsed = time.time() - start_time
            remaining = run_time_seconds - elapsed
            if remaining <= 0:
                logger.info(f"[EVOLVE] ⏱  Time limit reached ({run_time_seconds}s). Stopping cleanly after {loop_idx} cycle(s).")
                break
            logger.info(f"[EVOLVE] ⏱  Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s")

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
        
        proposed_config, rationale = call_agent_for_new_config(current_config, metrics, history_str, focus_prompt=focus_prompt)
        
        # Compare ONLY structural keys to determine if the PyTorch architecture changed
        arch_keys = ["n_embd", "expert_hidden_dim", "num_groups", "experts_per_group", "num_heads", "num_layers"]
        stable_mode = all(proposed_config.get(k, current_config.get(k)) == current_config.get(k) for k in arch_keys)

        # Always apply non-structural hyperparameters
        config.learning_rate = proposed_config.get("learning_rate", config.learning_rate)
        config.dropout = proposed_config.get("dropout", config.dropout)
        config.router_temperature = proposed_config.get("router_temperature", config.router_temperature)
        config.top_k = proposed_config.get("top_k", config.top_k)
        config.temperature = proposed_config.get("temperature", config.temperature)
        
        if stable_mode:
            logger.info("[Evolve] Architecture structure is STABLE (only hyperparams changed). Preparing for LwF Distillation.")
            student_model = model  # Keep current weights
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
        else:
            logger.info(f"[Evolve] Agent PROPOSED NEW ARCHITECTURE: {rationale}")
            logger.info("[Evolve] Building new Pytorch Structure for new dimensions.")
            # Snap d_model to a multiple of num_heads BEFORE building to avoid
            # shape mismatches that cause 0-tensor weight transfer every cycle.
            proposed_heads = proposed_config.get("num_heads", config.num_heads)
            proposed_d     = proposed_config.get("n_embd", config.d_model)
            aligned_d = max(proposed_heads,
                            (proposed_d + proposed_heads - 1) // proposed_heads * proposed_heads)
            if aligned_d != proposed_d:
                logger.info(f"[Evolve] Snapping n_embd {proposed_d} -> {aligned_d} "
                            f"(must be divisible by num_heads={proposed_heads})")
            config.num_heads         = proposed_heads
            config.d_model           = aligned_d
            config.num_layers        = proposed_config.get("num_layers",        config.num_layers)
            config.expert_hidden_dim = proposed_config.get("expert_hidden_dim", config.expert_hidden_dim)
            config.num_groups        = proposed_config.get("num_groups",        config.num_groups)
            config.experts_per_group = proposed_config.get("experts_per_group", config.experts_per_group)

            student_model = build_model(config)
            student_model.to(device)

            # FIX #1 – Warm-start the new architecture by copying all
            # identically-shaped tensors from the previously trained model.
            # Without this, the student always starts from random noise and
            # the validation gate always rejects the proposed architecture.
            transferred, total = transfer_compatible_weights(model, student_model)
            logger.info(
                f"[Evolve] Weight transfer: {transferred}/{total} tensors warm-started "
                f"({total - transferred} tensors newly initialised)."
            )

            teacher_model = None  # No LwF teacher because architectures don't match
            
        # --- B. PARALLEL TASKS (Update D_Next & Train D_Current) ---
        # 1. Async Fetch Thread
        # We fetch the sequence two steps ahead so the pipeline perfectly matches 
        # the diagram (e.g. Training on D2 asynchronously updates D1)
        # Re-read prompt.json each cycle so user can update focus mid-run
        focus_prompt = _read_focus_prompt()
        if focus_prompt:
            logger.info(f"[EVOLVE] 🎯 Cycle focus: '{focus_prompt}'")

        next_ds_idx = (loop_idx + 2) % 3
        fetch_thread = threading.Thread(target=_run_fetch_job, args=(config, dataset_buffer, next_ds_idx, focus_prompt))
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
            current_ds = dataset_buffer[current_ds_idx]
            replay_buffer.update(current_ds)
        except Exception as rb_err:
            logger.warning(f"[Replay] Buffer update failed: {rb_err}")
            
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
                
            # Serialize weights to hard drive
            best_model_path = os.path.join(BASE_DIR, "best_model.pt")
            torch.save(best_weights, best_model_path)
            logger.info(f"💾 Saved best model weights to '{best_model_path}'")
                
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
