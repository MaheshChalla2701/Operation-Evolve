# Operation Evolve 🧬

A **Self-Improving AI System** built with PyTorch. A Sparse Mixture of Experts (MoE) Transformer autonomously evolves its own architecture based on performance metrics.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   TRAIN     │────▶│   METRICS    │────▶│   AI AGENT      │────▶│   TEST NEW   │
│  (150 iters)│     │  metrics.json│     │   agent.py      │     │  (50 iters)  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
       ▲                                          │                       │
       │                 Accept if improved ◀────┘─────── Reject ─────── ┘
       │                 config.json updated                  (revert)
       └──────────────────────────────────────────────────────────────────
```

## Model Architecture

**SparseMoETransformer** — Character-level next-token prediction:
- 2 × Self-Attention Layers (Causal)
- 1 × Sparse MoE Layer per block (Top-1 Gating — only 1 expert active per token)
- Dynamically configured → number of experts, hidden dim, temperature all evolve

## Files

| File | Purpose |
|---|---|
| `config.json` | Current model hyperparameters and architecture config |
| `model.py` | SparseMoETransformer PyTorch implementation |
| `trainer.py` | Training loop with metric/load logging |
| `agent.py` | Mutation engine — reads metrics, proposes config changes |
| `controller.py` | **Main loop** — orchestrates Train → Agent → Test → Select |

## Run

```bash
# Install dependencies
pip install torch

# Start the self-improvement loop
python controller.py
```

## Output Files (Auto-Generated)

| File | Contents |
|---|---|
| `metrics.json` | Loss + expert load distribution per cycle |
| `proposed_config.json` | Agent's proposed new config |
| `mutation_record.json` | Detailed mutation rationale |
| `best_model.pt` | Best model weights so far |
| `evolution_log.json` | Full cycle-by-cycle evolution history |

## Agent Mutation Logic

The agent uses these rules to mutate the config:

| Condition | Mutation |
|---|---|
| Expert load imbalance (>70%) | Add an expert + soften routing temperature |
| Loss stalling (<0.1% improvement) | Decay learning rate + Increase expert capacity |
| Loss slow (<0.5% improvement) | Increase hidden dim + Sharpen routing |
| Loss improving well | Minor LR fine-tune |