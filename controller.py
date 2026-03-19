"""
controller.py — Operation Evolve Main Loop

Orchestrates the full self-evolving training cycle:

  CYCLE N:
  ┌─────────────────────────────────────────────────────┐
  │ 1. TRAIN    → Train base model for TRAIN_ITERS      │
  │ 2. EVALUATE → Compute loss, accuracy, expert util   │
  │ 3. MUTATE   → EvolutionAgent proposes new config    │
  │ 4. TEST     → Train candidate for TEST_ITERS        │
  │ 5. SELECT   → Accept if improved, else reject       │
  │ 6. LOG      → Record version, mutations, metrics    │
  └─────────────────────────────────────────────────────┘

Logs → evolution_log.json (model_version, changes_applied, performance_history)
Safety → best_model_backup.pt is preserved before any test
"""

import json
import os
import shutil
import sys
import time
import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

from model import SparseMoETransformer
from trainer import load_data, get_batch
from agent import EvolutionAgent

# ── Evolution Hyper-params ────────────────────────────────────────────────────
EVOLUTION_CYCLES  = 3      # Number of evolution rounds
TRAIN_ITERS       = 150    # Iterations for full training round
TEST_ITERS        = 50     # Quick test iterations for candidate model
EVAL_ITERS        = 20     # Evaluation iterations per checkpoint
ACCEPT_THRESHOLD  = 0.005  # Accept if val_loss improves by ≥ 0.5%

# ── File Paths ────────────────────────────────────────────────────────────────
CONFIG_FILE      = "config.json"
BEST_WEIGHTS     = "best_model.pt"
BACKUP_WEIGHTS   = "best_model_backup.pt"
EVOLUTION_LOG    = "evolution_log.json"


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def timestamp():
    return time.strftime("%H:%M:%S")

def header(text, width=62):
    print(f"\n{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}")


# ── Core Training Function ────────────────────────────────────────────────────

def train_model(config, train_data, val_data, max_iters, device,
                init_weights_path=None, label=""):
    """
    Instantiate, optionally warm-start, and train a SparseMoETransformer.

    Returns:
        model         : trained model
        best_val_loss : float
        metrics       : dict with loss, accuracy, expert_utilization, history
    """
    model = SparseMoETransformer(config).to(device)

    # Attempt to load compatible weights (strict=False handles architecture changes)
    if init_weights_path and os.path.exists(init_weights_path):
        try:
            state = torch.load(init_weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"    [{label}] Warm-started from {init_weights_path}")
        except Exception as e:
            print(f"    [{label}] Warm-start failed ({e}), training from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    best_val_loss = float("inf")
    eval_interval = max(1, max_iters // 5)
    history       = []
    last_loss     = None

    for step in range(max_iters):
        model.train()
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss, expert_util, accuracy = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()

        # ── Evaluation checkpoint ──────────────────────────────────────
        if step % eval_interval == 0 or step == max_iters - 1:
            model.eval()
            val_loss_sum   = 0.0
            val_acc_sum    = 0.0
            val_util_accum = None

            with torch.no_grad():
                for _ in range(EVAL_ITERS):
                    x_v, y_v = get_batch(val_data, block_size, batch_size)
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    _, v_loss, v_util, v_acc = model(x_v, y_v)
                    val_loss_sum += v_loss.item()
                    val_acc_sum  += (v_acc.item() if v_acc is not None else 0.0)
                    if val_util_accum is None:
                        val_util_accum = v_util.clone()
                    else:
                        val_util_accum += v_util.clone()

            val_loss = val_loss_sum / EVAL_ITERS
            val_acc  = float(val_acc_sum) / EVAL_ITERS

            # Normalize utilization across layers
            load_dist = []
            if val_util_accum is not None:
                total = val_util_accum.sum().item()
                if total > 0:
                    load_dist = (val_util_accum.sum(dim=0) / total).tolist()

            history.append({
                "step":              step,
                "train_loss":        round(last_loss, 5),
                "val_loss":          round(val_loss, 5),
                "accuracy":          round(val_acc, 5),
                "load_distribution": load_dist,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss

    final_load = history[-1]["load_distribution"] if history else []
    metrics = {
        "best_val_loss":      best_val_loss,
        "final_train_loss":   last_loss or 0.0,
        "accuracy":           history[-1]["accuracy"] if history else 0.0,
        "expert_utilization": final_load,
        "load_distribution":  final_load,
        "history":            history,
    }
    return model, best_val_loss, metrics


# ── Main Evolution Loop ───────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("   🧬  OPERATION EVOLVE — Self-Improving AI System")
    print("=" * 62)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device : {device}")
    print(f"   Cycles : {EVOLUTION_CYCLES}")
    print(f"   Train  : {TRAIN_ITERS} iters/round | Test: {TEST_ITERS} iters")

    if not os.path.exists(CONFIG_FILE):
        print(f"[Error] {CONFIG_FILE} not found.")
        sys.exit(1)

    config = load_json(CONFIG_FILE)

    # Load data once — shared across all cycles
    train_data, val_data, vocab_size, stoi, itos = load_data("input.txt")
    config["vocab_size"] = vocab_size
    save_json(config, CONFIG_FILE)

    decode        = lambda l: "".join([itos[i] for i in l])
    agent         = EvolutionAgent()
    evolution_log = []          # Tracks all cycles
    perf_history  = []          # Performance summary across versions
    model_version = 0           # Increments on accepted mutations

    # ── State for Conditional Verification ────────────────────────────────────
    pending_verification = False
    conditional_baseline = None
    backup_config        = None

    for cycle in range(1, EVOLUTION_CYCLES + 1):
        version_id = f"v{model_version}"
        header(f"CYCLE {cycle}/{EVOLUTION_CYCLES}  [{timestamp()}]  (current: {version_id})")
        print(f"  Config: experts={config['num_experts']}, top_k={config.get('top_k',1)}, "
              f"hidden={config['expert_hidden_dim']}, lr={config['learning_rate']:.6f}, "
              f"temp={config['router_temperature']}")

        # ── STEP 1: TRAIN base model ────────────────────────────────────
        print(f"\n  [TRAIN {timestamp()}] {TRAIN_ITERS} iterations...")
        t0 = time.time()

        model, baseline_loss, metrics = train_model(
            config, train_data, val_data, TRAIN_ITERS, device,
            init_weights_path=BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) else None,
            label="TRAIN"
        )
        save_json(metrics, "metrics.json")

        t1 = time.time()
        print(f"  [TRAIN] Done in {t1-t0:.1f}s | Val Loss: {baseline_loss:.4f} | "
              f"Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  [TRAIN] Expert utilization: {[f'{v:.2f}' for v in metrics['expert_utilization']]}")

        perf_history.append({
            "cycle":       cycle,
            "version":     version_id,
            "val_loss":    round(baseline_loss, 5),
            "accuracy":    round(metrics["accuracy"], 5),
            "num_experts": int(config["num_experts"]), # type: ignore
            "top_k":       int(config.get("top_k", 1)), # type: ignore
        })

        # ── CONDITIONAL VERIFICATION ────────────────────────────────────
        if pending_verification:
            improvement = float(conditional_baseline - baseline_loss) / float(conditional_baseline + 1e-8) # type: ignore
            print(f"\n  [Verify] Testing speculative change from previous cycle...")
            print(f"  [Verify] Old Loss: {conditional_baseline:.4f} | New Loss: {baseline_loss:.4f} | Improvement: {improvement*100:.2f}%")
            
            if improvement > ACCEPT_THRESHOLD:
                print(f"  ✅  CONFIRMED — Mutation kept! (Improvement: {improvement*100:.2f}%)")
                torch.save(model.state_dict(), BEST_WEIGHTS)
                shutil.copy2(BEST_WEIGHTS, BACKUP_WEIGHTS)
                model_version += 1
            else:
                print(f"  ❌  FAILED — Mutation did not improve model. ROLLING BACK.")
                if os.path.exists(BACKUP_WEIGHTS):
                    shutil.copy2(BACKUP_WEIGHTS, BEST_WEIGHTS)
                config = backup_config
                save_json(config, CONFIG_FILE)
                # reload model structure
                model = SparseMoETransformer(config).to(device)
                if os.path.exists(BEST_WEIGHTS):
                     model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device, weights_only=True), strict=False)
                print(f"  [Verify] Rollback complete. Reverted to previous stable configuration.")
            pending_verification = False

        # Safety: save current known good/validated weights
        torch.save(model.state_dict(), BEST_WEIGHTS)
        shutil.copy2(BEST_WEIGHTS, BACKUP_WEIGHTS)

        # ── STEP 2: AGENT proposes mutation ─────────────────────────────
        print(f"\n  [AGENT {timestamp()}] Analyzing and proposing mutations...")
        proposed_config, mutations = agent.analyze(config, metrics)

        if not mutations:
            print("  [AGENT] No mutations proposed — model is performing well.")
            evolution_log.append({
                "cycle":         cycle,
                "version":       version_id,
                "baseline_loss": round(float(baseline_loss), 5), # type: ignore
                "accuracy":      round(float(metrics.get("accuracy", 0.0)), 5), # type: ignore
                "changes":       [],
                "decision":      "no_mutation",
                "config_after":  copy.deepcopy(config),
            })
            save_json(evolution_log, EVOLUTION_LOG)
            continue

        print(f"\n  [AGENT] {len(mutations)} mutation(s) proposed:")
        for m in mutations:
            print(f"    ✦ {m}")

        # ── STEP 3: CONDITIONAL ACCEPTANCE FOR STRUCTURAL CHANGES ───────
        is_structural = any("num_experts" in m for m in mutations) or \
                        any("expert_hidden_dim" in m for m in mutations) or \
                        any("top_k" in m for m in mutations)

        if is_structural:
            print(f"\n  [Controller] 🔄 STRUCTURAL mutation detected. SPECULATIVE ACCEPTANCE:")
            print(f"  [Controller] Applying changes now. Will train and evaluate in Cycle {cycle+1}.")
            
            backup_config = copy.deepcopy(config)
            conditional_baseline = baseline_loss
            pending_verification = True

            config = proposed_config
            save_json(config, CONFIG_FILE)
            
            log_entry = {
                "cycle":          cycle,
                "version":        version_id,
                "new_version":    f"{version_id} (speculative)",
                "timestamp":      timestamp(),
                "baseline_loss":  round(baseline_loss, 5),
                "accuracy":       round(metrics["accuracy"], 5),
                "changes_applied": mutations,
                "decision":       "speculative_accept",
                "config_after":   copy.deepcopy(config),
            }
            evolution_log.append(log_entry)
            save_json(evolution_log, EVOLUTION_LOG)
            continue

        # ── STEP 4: NORMAL HYPERPARAMETER TEST ──────────────────────────
        print(f"\n  [TEST {timestamp()}] Testing candidate tweak for {TEST_ITERS} iters...")
        proposed_config["vocab_size"] = vocab_size
        candidate_model  = None
        candidate_loss   = float("inf")
        candidate_metrics = {}
        test_ok = False

        try:
            t0 = time.time()
            candidate_model, candidate_loss, candidate_metrics = train_model(
                proposed_config, train_data, val_data, TEST_ITERS, device,
                init_weights_path=BEST_WEIGHTS,
                label="TEST"
            )
            t1 = time.time()
            print(f"  [TEST] Done in {t1-t0:.1f}s | Val Loss: {candidate_loss:.4f} | "
                  f"Accuracy: {candidate_metrics.get('accuracy', 0)*100:.1f}%")
            test_ok = True
        except Exception as e:
            print(f"  [TEST] ⚠️  Candidate crashed: {e}")

        improvement = float(baseline_loss - candidate_loss) / float(baseline_loss + 1e-8) # type: ignore
        accepted    = test_ok and (improvement > ACCEPT_THRESHOLD)

        if accepted:
            model_version += 1
            print(f"\n  ✅  ACCEPTED")
            config = proposed_config
            save_json(config, CONFIG_FILE)
            torch.save(candidate_model.state_dict(), BEST_WEIGHTS)
            decision = "accepted"
        else:
            print(f"\n  ❌  REJECTED")
            decision = "rejected"

        log_entry = {
            "cycle":          cycle,
            "version":        version_id,
            "new_version":    f"v{model_version}" if accepted else version_id,
            "timestamp":      timestamp(),
            "baseline_loss":  round(float(baseline_loss), 5), # type: ignore
            "accuracy":       round(float(metrics.get("accuracy", 0.0)), 5), # type: ignore
            "changes_applied": mutations,
            "candidate_loss": round(float(candidate_loss), 5) if test_ok else None, # type: ignore
            "improvement_%":  round(float(improvement) * 100, 3) if test_ok else None, # type: ignore
            "decision":       decision,
            "config_after":   copy.deepcopy(config),
        }
        evolution_log.append(log_entry)
        save_json(evolution_log, EVOLUTION_LOG)
        print(f"\n  [Log] Evolution log updated → {EVOLUTION_LOG}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"   🏁  OPERATION EVOLVE COMPLETE  —  {EVOLUTION_CYCLES} cycles")
    print(f"{'=' * 62}")

    print(f"\n  Performance History:\n")
    print(f"  {'Cycle':<7} {'Version':<8} {'Val Loss':<12} {'Accuracy':<12} {'Experts':<9} {'top_k'}")
    print(f"  {'-'*55}")
    for p in perf_history:
        print(f"  {p['cycle']:<7} {p['version']:<8} {p['val_loss']:<12.4f} "
              f"{p['accuracy']*100:<12.1f}% {p['num_experts']:<9} {p['top_k']}")

    print(f"\n  Evolution Log:\n")
    print(f"  {'Cycle':<7} {'Baseline':<12} {'Candidate':<12} {'Δ%':<10} {'Decision'}")
    print(f"  {'-'*55}")
    for e in evolution_log:
        cand  = f"{e['candidate_loss']:.4f}" if e.get("candidate_loss") is not None else "N/A"
        delta = f"{e['improvement_%']:+.2f}%" if e.get("improvement_%") is not None else "N/A"
        print(f"  {e['cycle']:<7} {e['baseline_loss']:<12.4f} {cand:<12} {delta:<10} {e['decision']}")

    print(f"\n  Final config  → {CONFIG_FILE}")
    print(f"  Best weights  → {BEST_WEIGHTS}")
    print(f"  Backup        → {BACKUP_WEIGHTS}")
    print(f"  Full log      → {EVOLUTION_LOG}\n")

    # ── GENERATE SAMPLE WITH FINAL MODEL ─────────────────────────────────────
    print("  --- Sample from Final Evolved Model (200 tokens) ---")
    final_model = SparseMoETransformer(config).to(device)
    if os.path.exists(BEST_WEIGHTS):
        try:
            final_model.load_state_dict(
                torch.load(BEST_WEIGHTS, map_location=device, weights_only=True), strict=False
            )
        except Exception as e:
            print(f"  [Warning] Could not load final weights: {e}")
    final_model.eval()
    ctx     = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_idx = final_model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=40)
    print(decode(gen_idx[0].tolist()))
    print("  ---------------------------------------------------")


if __name__ == "__main__":
    main()
