"""
ask.py - Interactive Q&A using local autoregressive generation from EVOLVE.

How it works:
  1. You type a prompt / question.
  2. The prompt is tokenized with GPT-2 tiktoken encoder.
  3. The HierarchicalMoELM model generates new tokens autoregressively using
     top-k sampling (temperature-scaled) -- no internet call required.
  4. The generated tokens are decoded and printed to the terminal.
  5. Expert routing stats are shown for observability.

Usage:
    python ask.py                           # interactive loop
    python ask.py "Why is the sky blue?"    # single prompt then exit
"""

import os
import sys

import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from config import EvolveConfig
from model  import build_model

# ── constants ─────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")

BANNER = """
+----------------------------------------------------------+
|        EVOLVE  -  Local MoE Language Model               |
|  Type a prompt and the model generates text locally.     |
|  Type  'exit'  or  'quit'  to stop.                      |
+----------------------------------------------------------+
"""


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(cfg: EvolveConfig) -> torch.nn.Module:
    """Load checkpoint if it exists; otherwise use a freshly initialised model."""
    model = build_model(cfg)
    if os.path.exists(CKPT_PATH):
        saved   = torch.load(CKPT_PATH, map_location="cpu")
        current = model.state_dict()
        transferred = 0
        mismatched_printed = 0
        with torch.no_grad():
            for name, tensor in saved.items():
                if name in current:
                    if current[name].shape == tensor.shape:
                        current[name].copy_(tensor)
                        transferred += 1
                    elif mismatched_printed < 5:
                        print(f"  [DEBUG] Shape mismatch {name}: config says {current[name].shape}, checkpoint has {tensor.shape}")
                        mismatched_printed += 1
                else:
                    if mismatched_printed < 5:
                        print(f"  [DEBUG] Missing in config {name}: checkpoint has {tensor.shape}")
                        mismatched_printed += 1
        model.load_state_dict(current)
        label = f"checkpoint  ({transferred}/{len(current)} layers matched)"
    else:
        label = "untrained model  (no checkpoint found -- run train.py first)"
    print(f"  [MODEL]  {label}")
    return model


# ── sampling helpers ──────────────────────────────────────────────────────────

def _top_k_sample(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    """
    Sample a single token from logits with top-k filtering.

    logits : [V]  (un-normalised)
    Returns: sampled token id (int)
    """
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = topk_vals[-1]
        logits = logits.masked_fill(logits < threshold, -float("inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ── autoregressive generation ─────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_tokens: list,
    cfg: EvolveConfig,
    device: torch.device,
):
    """
    Autoregressively generate tokens from a prompt.

    Returns:
        generated_tokens : list[int]  (new tokens only, excluding prompt)
        expert_load      : list[float] | None
    """
    model.eval()
    ctx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated = []
    expert_load = None

    for _ in range(cfg.max_new_tokens):
        ctx_cond    = ctx[:, -cfg.max_seq_len:]
        logits      = model(ctx_cond)         # [1, S, V]
        next_logits = logits[0, -1, :]        # [V]

        next_token  = _top_k_sample(next_logits, cfg.top_k, cfg.temperature)
        generated.append(next_token)

        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        ctx = torch.cat([ctx, next_tensor], dim=1)

        # Early stop on GPT-2 EOS token
        if next_token == 50256:
            break

    # Grab expert routing stats from last forward pass
    for layer in getattr(model, "layers", []):
        moe = getattr(layer, "moe", None)
        if moe is not None and hasattr(moe, "last_expert_load"):
            expert_load = moe.last_expert_load.tolist()
            break

    return generated, expert_load


# ── display helpers ───────────────────────────────────────────────────────────

def _print_expert_stats(expert_load: list) -> None:
    n       = len(expert_load)
    active  = sum(1 for e in expert_load if e > 0.05)
    top_idx = max(range(n), key=lambda i: expert_load[i])
    print(f"\n  [MoE]  Experts: {n}  |  Active: {active}"
          f"  |  Busiest: #{top_idx}  (load={expert_load[top_idx]:.3f})")


def _print_response(prompt: str, response: str, expert_load) -> None:
    print()
    print("-" * 62)
    print(f"  PROMPT >  {prompt}")
    print("-" * 62)
    print("\n  Generated response:\n")
    words = response.split()
    line, col = "      ", 6
    for w in words:
        if col + len(w) + 1 > 80:
            print(line)
            line, col = "      ", 6
        line += w + " "
        col  += len(w) + 1
    if line.strip():
        print(line)
    print()
    if expert_load:
        _print_expert_stats(expert_load)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg    = EvolveConfig()
    cfg.device = "cpu"
    device = cfg.get_device()
    enc    = tiktoken.get_encoding("gpt2")

    # Load the latest evolved structural parameters so building the model matches the checkpoint
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            evolved = json.load(f)
        cfg.d_model           = evolved.get("n_embd", cfg.d_model)
        cfg.num_layers        = evolved.get("num_layers", cfg.num_layers)
        cfg.num_heads         = evolved.get("num_heads", cfg.num_heads)
        cfg.expert_hidden_dim = evolved.get("expert_hidden_dim", cfg.expert_hidden_dim)
        cfg.num_groups        = evolved.get("num_groups", cfg.num_groups)
        cfg.experts_per_group = evolved.get("experts_per_group", cfg.experts_per_group)
        
        # Generation specific configs
        cfg.top_k             = evolved.get("top_k", cfg.top_k)
        cfg.temperature       = evolved.get("temperature", cfg.temperature)

    print(BANNER)
    model = load_model(cfg)
    model.to(device)
    print()

    # Single-shot mode
    if len(sys.argv) > 1:
        prompts     = [" ".join(sys.argv[1:])]
        interactive = False
    else:
        prompts     = []
        interactive = True

    while True:
        if interactive:
            try:
                prompt = input("  Ask > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("  Goodbye!")
                break
        else:
            if not prompts:
                break
            prompt = prompts.pop(0)

        prompt_tokens = enc.encode(prompt)
        if len(prompt_tokens) > cfg.max_seq_len - 1:
            prompt_tokens = prompt_tokens[-(cfg.max_seq_len - 1):]
            print("  [!] Prompt truncated to fit max_seq_len.")

        print(f"\n  Generating (max {cfg.max_new_tokens} tokens,"
              f" top_k={cfg.top_k}, T={cfg.temperature}) ...")

        generated_tokens, expert_load = generate(model, prompt_tokens, cfg, device)
        response = enc.decode(generated_tokens)

        _print_response(prompt, response, expert_load)

        if not interactive:
            break


if __name__ == "__main__":
    main()
