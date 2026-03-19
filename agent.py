"""
agent.py — Evolution Agent

Reads model config + performance metrics and proposes a mutated config
to improve the model. Supports:

  ✦ increase / decrease number of experts
  ✦ change top_k routing
  ✦ prune underutilized experts (< PRUNE_THRESHOLD of total tokens)
  ✦ clone high-performing experts (> CLONE_THRESHOLD of total tokens)
  ✦ adjust learning rate and router temperature

Usage:
    from agent import EvolutionAgent
    agent = EvolutionAgent()
    new_config, mutations = agent.analyze(config, metrics)
"""

import json
import os
import copy
import random

# ─── Thresholds ────────────────────────────────────────────────────────────────
IMBALANCE_THRESHOLD   = 0.60   # Single expert > 60% → router imbalance
CLONE_THRESHOLD       = 0.50   # Expert > 50% load → candidate to clone
PRUNE_THRESHOLD       = 0.05   # Expert < 5% load → candidate to prune
SLOW_IMPROVEMENT      = 0.005  # < 0.5% val_loss improvement → slow
STALL_THRESHOLD       = 0.001  # < 0.1% val_loss improvement → stalling
MAX_EXPERTS           = 8      # Safety cap on expert count
MIN_EXPERTS           = 2      # Minimum number of experts to keep


class EvolutionAgent:
    """
    AI evolution agent that analyzes model metrics and proposes config mutations.

    Input:  model config dict + metrics dict
    Output: (modified_config dict, list of mutation descriptions)

    Mutation strategies (in priority order):
      1. Prune dead experts (< PRUNE_THRESHOLD utilization)
      2. Clone dominant experts (> CLONE_THRESHOLD utilization)
      3. Add expert if router is collapsing (imbalance)
      4. Increase top_k if model is stalling
      5. Decay learning rate on stall
      6. Expand expert hidden dim on slow progress
      7. Fine-tune LR on good progress
    """

    def __init__(self):
        self.prune_threshold  = PRUNE_THRESHOLD
        self.clone_threshold  = CLONE_THRESHOLD
        self.stall_threshold  = STALL_THRESHOLD
        self.slow_threshold   = SLOW_IMPROVEMENT
        self.imbalance_thresh = IMBALANCE_THRESHOLD

    # ── Internal Analysis Helpers ────────────────────────────────────────────

    def _loss_trend(self, history: list) -> str:
        """
        Classifies the improvement rate of val_loss over the last two checkpoints.
        Returns: 'good' | 'slow' | 'stalling'
        """
        if len(history) < 2:
            return "good"
        recent   = history[-1].get("val_loss", 0)
        previous = history[-2].get("val_loss", 0)
        if previous <= 0:
            return "good"
        improvement = (previous - recent) / previous
        if improvement < self.stall_threshold:
            return "stalling"
        elif improvement < self.slow_threshold:
            return "slow"
        return "good"

    def _expert_stats(self, load_dist: list) -> dict:
        """
        Returns a dict with expert utilization statistics.
        load_dist: list of floats (one per expert, sums to ~1.0)
        """
        if not load_dist or len(load_dist) < 1:
            return {"max": 0.0, "min": 0.0, "max_idx": 0, "min_idx": 0,
                    "is_imbalanced": False, "prune_candidates": [], "clone_candidates": []}

        max_load  = max(load_dist)
        min_load  = min(load_dist)
        max_idx   = load_dist.index(max_load)
        min_idx   = load_dist.index(min_load)

        prunable = [i for i, v in enumerate(load_dist) if v < self.prune_threshold]
        clonable = [i for i, v in enumerate(load_dist) if v > self.clone_threshold]

        return {
            "max": max_load,
            "min": min_load,
            "max_idx": max_idx,
            "min_idx": min_idx,
            "is_imbalanced": max_load > self.imbalance_thresh,
            "prune_candidates": prunable,
            "clone_candidates": clonable,
        }

    # ── Core Mutation Logic ──────────────────────────────────────────────────

    def analyze(self, config: dict, metrics: dict) -> tuple[dict, list]:  # type: ignore
        """
        Analyze metrics and produce a mutated configuration.

        Args:
            config  (dict): Current model configuration.
            metrics (dict): Performance metrics with keys:
                            'best_val_loss', 'history', 'load_distribution',
                            'accuracy', 'expert_utilization'

        Returns:
            (proposed_config, mutations):
                proposed_config (dict): Modified config dict.
                mutations (list[str]):  Human-readable descriptions of changes.
        """
        proposed = copy.deepcopy(config)
        history   = metrics.get("history", [])
        load_dist = metrics.get("load_distribution", [])

        mutations = []
        trend     = self._loss_trend(history)
        stats     = self._expert_stats(load_dist)
        n_experts = config.get("num_experts", 4)
        top_k     = config.get("top_k", 1)

        print(f"  [Agent] Loss trend      : {trend}")
        print(f"  [Agent] Expert load     : {[round(v, 3) for v in load_dist]}")
        print(f"  [Agent] Router imbalance: {stats['is_imbalanced']}")
        print(f"  [Agent] Prune candidates: {stats['prune_candidates']}")
        print(f"  [Agent] Clone candidates: {stats['clone_candidates']}")

        # ── Priority 1: PRUNE underutilized experts ─────────────────────────
        if stats["prune_candidates"] and (n_experts - len(stats["prune_candidates"])) >= MIN_EXPERTS:
            n_prune = min(len(stats["prune_candidates"]), n_experts - MIN_EXPERTS)
            if n_prune > 0:
                proposed["num_experts"] = n_experts - n_prune
                mutations.append(
                    f"num_experts: {n_experts} → {proposed['num_experts']} "
                    f"(pruned {n_prune} underutilized expert(s) with < {self.prune_threshold*100:.0f}% load)"
                )
                return proposed, mutations  # Prune is a structural change — apply alone

        # ── Priority 2: CLONE dominant expert ───────────────────────────────
        if stats["clone_candidates"] and n_experts < MAX_EXPERTS:
            proposed["num_experts"] = n_experts + 1
            mutations.append(
                f"num_experts: {n_experts} → {proposed['num_experts']} "
                f"(cloned expert {stats['clone_candidates'][0]} with {stats['max']*100:.1f}% load)"
            )
            # Soften router so the clone gets traffic
            proposed["router_temperature"] = min(2.0, round(config.get("router_temperature", 1.0) * 1.15, 3))
            mutations.append(
                f"router_temperature: {config.get('router_temperature', 1.0)} → {proposed['router_temperature']} "
                f"(soften routing after clone)"
            )
            return proposed, mutations  # Structural change — apply alone

        # ── Priority 3: ADD expert on router collapse ────────────────────────
        if stats["is_imbalanced"] and n_experts < MAX_EXPERTS:
            proposed["num_experts"] = n_experts + 1
            mutations.append(
                f"num_experts: {n_experts} → {proposed['num_experts']} "
                f"(router collapse: expert {stats['max_idx']} has {stats['max']*100:.1f}% load)"
            )
            proposed["router_temperature"] = min(2.0, round(config.get("router_temperature", 1.0) * 1.15, 3))
            mutations.append(
                f"router_temperature: {config.get('router_temperature', 1.0)} → {proposed['router_temperature']} "
                f"(soften routing to reduce collapse)"
            )
            return proposed, mutations

        # ── Priority 4: INCREASE top_k if stalling ──────────────────────────
        if trend == "stalling":
            if top_k < n_experts:
                proposed["top_k"] = top_k + 1
                mutations.append(
                    f"top_k: {top_k} → {proposed['top_k']} "
                    f"(increase routing breadth on stall)"
                )
            # Also decay LR
            proposed["learning_rate"] = round(config.get("learning_rate", 3e-4) * 0.65, 7)
            mutations.append(
                f"learning_rate: {config.get('learning_rate')} → {proposed['learning_rate']} "
                f"(decay LR on stall)"
            )
            # Grow expert capacity
            proposed["expert_hidden_dim"] = config.get("expert_hidden_dim", 256) + 64
            mutations.append(
                f"expert_hidden_dim: {config.get('expert_hidden_dim')} → {proposed['expert_hidden_dim']} "
                f"(expand capacity on stall)"
            )

        # ── Priority 5: EXPAND capacity on slow progress ────────────────────
        elif trend == "slow":
            proposed["expert_hidden_dim"] = config.get("expert_hidden_dim", 256) + 32
            mutations.append(
                f"expert_hidden_dim: {config.get('expert_hidden_dim')} → {proposed['expert_hidden_dim']} "
                f"(expand capacity, slow improvement)"
            )
            new_temp = float(config.get("router_temperature", 1.0)) * 0.9
            proposed["router_temperature"] = max(0.5, round(new_temp, 3))
            mutations.append(
                f"router_temperature: {config.get('router_temperature', 1.0)} → {proposed['router_temperature']} "
                f"(sharpen routing, slow improvement)"
            )

        # ── Priority 6: FINE-TUNE on good progress ───────────────────────────
        else:
            proposed["learning_rate"] = round(config.get("learning_rate", 3e-4) * 0.95, 7)
            mutations.append(
                f"learning_rate: {config.get('learning_rate')} → {proposed['learning_rate']} "
                f"(gentle LR decay, trend is good)"
            )

        return proposed, mutations


# ── Standalone Entry Point ───────────────────────────────────────────────────

METRICS_FILE        = "metrics.json"
CONFIG_FILE         = "config.json"
PROPOSED_CONFIG     = "proposed_config.json"
MUTATION_RECORD     = "mutation_record.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def mutate_config(config, metrics):
    """Compatibility wrapper for controller.py — delegates to EvolutionAgent."""
    agent = EvolutionAgent()
    return agent.analyze(config, metrics)


def main():
    print("=" * 55)
    print(" 🧬 Operation Evolve — Evolution Agent")
    print("=" * 55)

    if not os.path.exists(METRICS_FILE):
        print(f"[Agent] No metrics file at {METRICS_FILE}. Run trainer.py first.")
        return
    if not os.path.exists(CONFIG_FILE):
        print(f"[Agent] No config file at {CONFIG_FILE}.")
        return

    metrics = load_json(METRICS_FILE)
    config  = load_json(CONFIG_FILE)

    print(f"\n[Agent] Current config:\n{json.dumps(config, indent=2)}")
    print(f"\n[Agent] Best val_loss : {metrics.get('best_val_loss', 'N/A'):.4f}")
    if metrics.get("accuracy") is not None:
        print(f"[Agent] Best accuracy : {metrics.get('accuracy', 0)*100:.2f}%")

    agent = EvolutionAgent()
    proposed_config, mutations = agent.analyze(config, metrics)

    print("\n[Agent] Mutations proposed:")
    if mutations:
        for m in mutations:
            print(f"  ✦ {m}")
    else:
        print("  (none — model is performing well)")
        proposed_config = config

    save_json(proposed_config, PROPOSED_CONFIG)
    print(f"\n[Agent] Proposed config saved → {PROPOSED_CONFIG}")

    mutation_record = {
        "mutations": mutations,
        "based_on_val_loss": metrics.get("best_val_loss"),
        "based_on_accuracy": metrics.get("accuracy"),
        "proposed_config": proposed_config
    }
    save_json(mutation_record, MUTATION_RECORD)
    print(f"[Agent] Mutation record saved → {MUTATION_RECORD}")


if __name__ == "__main__":
    main()
