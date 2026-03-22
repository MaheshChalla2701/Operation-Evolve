"""
trainer.py — Standalone Training Script (v2)

Improvements over v1:
  - BPE tokenizer via tiktoken (gpt2, vocab_size=50257) instead of char-level
  - Muon optimizer for weight matrices, AdamW for embeddings
  - LR warmup + cosine decay schedule
  - torch.compile() for ~2x free speedup on modern GPUs
  - val_bpb (bits-per-byte) metric alongside val_loss
"""

import torch
import torch.optim as optim
import torch.nn as nn
import json
import os
import time
import math

# BPE tokenizer
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    print("[Warning] tiktoken not installed. Run: pip install tiktoken")
    print("[Warning] Falling back to character-level tokenizer.")

from model import SparseMoETransformer


# ── BPE Tokenizer ─────────────────────────────────────────────────────────────

def load_data(filepath="input.txt"):
    """
    Load text, tokenize with BPE (tiktoken/gpt2) or fall back to char-level.

    Returns:
        train_data (Tensor), val_data (Tensor), vocab_size (int),
        encode (callable), decode (callable)
    """
    if not os.path.exists(filepath):
        print(f"[Data] '{filepath}' not found. Generating dummy dataset...")
        text = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\n" * 100
                + "The quick brown fox jumps over the lazy dog. " * 300
                + "Operation Evolve is a self-improving AI system. " * 200)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    if _TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding("gpt2")  # vocab_size = 50257
        vocab_size = enc.n_vocab
        tokens = enc.encode_ordinary(text)
        encode = lambda s: enc.encode_ordinary(s)
        decode = lambda l: enc.decode(l)
        print(f"[Data] BPE tokenizer (gpt2) | Vocab: {vocab_size:,} | Tokens: {len(tokens):,}")
    else:
        # Fallback: character-level
        chars = sorted(set(text))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: "".join([itos[i] for i in l])
        tokens = encode(text)
        print(f"[Data] Char-level tokenizer | Vocab: {vocab_size} | Tokens: {len(tokens):,}")

    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, encode, decode


def get_batch(data, block_size, batch_size):
    """Sample a random batch of (input, target) sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ── val_bpb (Bits Per Byte) ───────────────────────────────────────────────────

def compute_bpb(val_loss: float) -> float:
    """
    Convert cross-entropy loss to bits-per-byte.
    bpb = loss / log(2)  (vocab-size independent, lower is better)
    """
    return val_loss / math.log(2)


# ── Muon Optimizer ────────────────────────────────────────────────────────────

def newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate matrix orthogonalization via Newton-Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    return X.to(G.dtype)


class Muon(optim.Optimizer):
    """
    Muon optimizer for 2D weight matrices.
    Uses Nesterov momentum + Newton-Schulz orthogonalization.
    Best for attention/MLP weight matrices.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)  # Nesterov
                if g.ndim >= 2:
                    g = newton_schulz_5(g, ns_steps)
                p.add_(g, alpha=-lr)


def build_optimizer(model: nn.Module, config: dict):
    """
    Use Muon for 2D weight matrices (attention + MLP),
    AdamW for embeddings, biases, and LayerNorms.
    """
    matrix_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Muon: 2D weight matrices that aren't embeddings
        if param.ndim == 2 and "wte" not in name and "lm_head" not in name:
            matrix_params.append(param)
        else:
            adamw_params.append(param)

    lr = config.get("learning_rate", 3e-4)
    optimizer = optim.AdamW(adamw_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    muon = Muon(matrix_params, lr=lr * 0.5, momentum=0.95)
    return optimizer, muon


# ── LR Schedule: Warmup + Cosine Decay ───────────────────────────────────────

def get_lr(step: int, max_iters: int, base_lr: float,
           warmup_frac: float = 0.05, min_lr_frac: float = 0.1) -> float:
    """
    Linear warmup → cosine decay.
    warmup_frac: fraction of training for warmup (default 5%)
    min_lr_frac: final LR = base_lr * min_lr_frac
    """
    warmup_steps = max(1, int(max_iters * warmup_frac))
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_iters - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


# ── Main Training Script ──────────────────────────────────────────────────────

def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, "r") as f:
        config = json.load(f)

    print("=" * 60)
    print("  🏋  Operation Evolve — Standalone Trainer (v2)")
    print("=" * 60)
    print(json.dumps(config, indent=2))

    # ── Load Data ────────────────────────────────────────────────
    train_data, val_data, vocab_size, encode, decode = load_data("input.txt")
    config["vocab_size"] = vocab_size

    # ── Model ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Trainer] Device: {device}")

    model = SparseMoETransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Trainer] Parameters: {n_params:,}")

    # torch.compile() — free ~2x speedup (skipped on CPU / older GPUs)
    if device == "cuda":
        try:
            model = torch.compile(model)
            print("[Trainer] ✅ torch.compile() enabled")
        except Exception as e:
            print(f"[Trainer] ⚠️ torch.compile() unavailable: {e}")

    # ── Optimizer (Muon + AdamW) ─────────────────────────────────
    adamw_opt, muon_opt = build_optimizer(model, config)

    # ── Hyper-params ─────────────────────────────────────────────
    max_iters     = 100
    eval_interval = 25
    eval_iters    = 10
    block_size    = config["block_size"]
    batch_size    = config["batch_size"]
    base_lr       = config["learning_rate"]

    best_val_loss = float("inf")
    history = []

    print(f"\n[Trainer] Training for {max_iters} iterations (with LR schedule + Muon)...\n")
    t_start = time.time()

    for step in range(max_iters):

        # ── LR Schedule ──────────────────────────────────────────
        lr = get_lr(step, max_iters, base_lr)
        for opt in [adamw_opt, muon_opt]:
            for pg in opt.param_groups:
                pg["lr"] = lr if opt is adamw_opt else lr * 0.5

        # ── Training step ─────────────────────────────────────────
        model.train()
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss, expert_util, accuracy = model(xb, yb)
        adamw_opt.zero_grad(set_to_none=True)
        muon_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adamw_opt.step()
        muon_opt.step()

        # ── Periodic Evaluation ───────────────────────────────────
        if step % eval_interval == 0 or step == max_iters - 1:
            model.eval()
            val_loss_sum   = 0.0
            val_acc_sum    = 0.0
            val_util_accum = None

            with torch.no_grad():
                for _ in range(eval_iters):
                    x_v, y_v = get_batch(val_data, block_size, batch_size)
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    _, v_loss, v_util, v_acc = model(x_v, y_v)
                    val_loss_sum += v_loss.item()
                    val_acc_sum  += v_acc.item() if v_acc is not None else 0.0
                    if val_util_accum is None:
                        val_util_accum = v_util.clone()
                    else:
                        val_util_accum += v_util.clone()

            val_loss = val_loss_sum / eval_iters
            val_acc  = float(val_acc_sum) / eval_iters
            val_bpb  = compute_bpb(val_loss)  # ← NEW: bits-per-byte metric

            load_dist = []
            if val_util_accum is not None:
                total = val_util_accum.sum().item()
                if total > 0:
                    load_dist = (val_util_accum.sum(dim=0) / total).tolist()

            print(
                f"  Step {step:>4d} | LR: {lr:.2e} | Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | BPB: {val_bpb:.4f} | "
                f"Acc: {val_acc*100:.1f}% | Expert Load: {[f'{v:.2f}' for v in load_dist]}"
            )

            history.append({
                "step":              step,
                "train_loss":        round(float(loss.item()), 5),
                "val_loss":          round(float(val_loss), 5),
                "val_bpb":           round(float(val_bpb), 5),
                "accuracy":          round(val_acc, 5),
                "load_distribution": load_dist,
                "lr":                round(lr, 8),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

    t_end = time.time()
    best_bpb = compute_bpb(best_val_loss)
    print(f"\n[Trainer] Finished in {t_end - t_start:.1f}s")
    print(f"[Trainer] Best Val Loss: {best_val_loss:.4f}  (BPB: {best_bpb:.4f})")

    # ── Save Metrics ─────────────────────────────────────────────
    final_load = history[-1]["load_distribution"] if history else []
    metrics = {
        "best_val_loss":      best_val_loss,
        "best_val_bpb":       best_bpb,
        "final_train_loss":   loss.item(),
        "accuracy":           history[-1]["accuracy"] if history else 0.0,
        "expert_utilization": final_load,
        "load_distribution":  final_load,
        "history":            history,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[Trainer] Metrics saved → metrics.json")

    # ── Sample Generation ─────────────────────────────────────────
    print("\n--- Model Sample (200 tokens) ---")
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_idx = model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=40)
    # Get raw model (without torch.compile wrapper) for decode
    raw_tokens = gen_idx[0].tolist()
    print(decode(raw_tokens))
    print("---------------------------------")


if __name__ == "__main__":
    main()
