"""
trainer.py — Standalone Training Script

Trains the SparseMoETransformer for a fixed number of iterations and saves:
  - best_model.pt       (model checkpoint)
  - metrics.json        (loss, accuracy, expert utilization history)

Run directly to produce a trained baseline before launching controller.py.
"""

import torch
import torch.optim as optim
import json
import os
import time

from model import SparseMoETransformer


def load_data(filepath="input.txt"):
    """
    Load text data, build character-level vocabulary, and return train/val splits.

    Returns:
        train_data (Tensor), val_data (Tensor), vocab_size (int),
        stoi (dict), itos (dict)
    """
    if not os.path.exists(filepath):
        print(f"Dataset '{filepath}' not found. Generating a dummy dataset...")
        text  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\n" * 100
        text += "The quick brown fox jumps over the lazy dog. " * 300
        text += "Operation Evolve is a self-improving AI system. " * 200
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    chars      = sorted(set(text))
    vocab_size = len(chars)
    print(f"[Data] Vocab size: {vocab_size}, Text length: {len(text):,}")

    stoi   = {ch: i for i, ch in enumerate(chars)}
    itos   = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    n    = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, stoi, itos


def get_batch(data, block_size, batch_size):
    """Sample a random batch of (input, target) sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i + block_size]     for i in ix])
    y  = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def main():
    # ── Load Config ──────────────────────────────────────────────────────
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, "r") as f:
        config = json.load(f)

    print("=" * 55)
    print(" 🏋  Operation Evolve — Standalone Trainer")
    print("=" * 55)
    print(json.dumps(config, indent=2))

    # ── Load Data ────────────────────────────────────────────────────────
    train_data, val_data, vocab_size, stoi, itos = load_data("input.txt")
    config["vocab_size"] = vocab_size

    # ── Model & Optimizer ────────────────────────────────────────────────
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Trainer] Device: {device}")

    model     = SparseMoETransformer(config).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"[Trainer] Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # ── Hyper-params ─────────────────────────────────────────────────────
    max_iters     = 100
    eval_interval = 25
    eval_iters    = 10
    block_size    = config["block_size"]
    batch_size    = config["batch_size"]

    best_val_loss = float("inf")
    history       = []

    print(f"\n[Trainer] Training for {max_iters} iterations...\n")
    t_start = time.time()

    for step in range(max_iters):
        # ── Training step ──────────────────────────────────────────────
        model.train()
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss, expert_util, accuracy = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── Periodic Evaluation ────────────────────────────────────────
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
            val_acc  = val_acc_sum  / eval_iters

            # Expert utilization: fraction of tokens per expert (averaged across layers)
            load_dist = []
            if val_util_accum is not None:
                total = val_util_accum.sum().item()
                if total > 0:
                    load_dist = (val_util_accum.sum(dim=0) / total).tolist()

            print(
                f"  Step {step:>4d} | Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.1f}% | "
                f"Expert Load: {[f'{v:.2f}' for v in load_dist]}"
            )

            history.append({
                "step":              step,
                "train_loss":        round(float(loss.item()), 5),
                "val_loss":          round(float(val_loss), 5),
                "accuracy":          round(val_acc, 5),
                "load_distribution": load_dist,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

    t_end = time.time()
    print(f"\n[Trainer] Finished in {t_end - t_start:.1f}s")
    print(f"[Trainer] Best Val Loss: {best_val_loss:.4f}")

    # ── Save Metrics for Agent ───────────────────────────────────────────
    final_load: list = history[-1]["load_distribution"] if history else [] # type: ignore
    metrics = {
        "best_val_loss":      best_val_loss,
        "final_train_loss":   loss.item(),
        "accuracy":           history[-1]["accuracy"] if history else 0.0,
        "expert_utilization": final_load,
        "load_distribution":  final_load,
        "history":            history,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[Trainer] Metrics saved → metrics.json")

    # ── Sample Generation ────────────────────────────────────────────────
    decode = lambda l: "".join([itos[i] for i in l])
    print("\n--- Model Sample (200 tokens) ---")
    model.eval()
    ctx     = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_idx = model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=40)
    print(decode(gen_idx[0].tolist()))
    print("---------------------------------")


if __name__ == "__main__":
    main()
