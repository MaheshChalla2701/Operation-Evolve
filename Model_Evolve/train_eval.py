import sys
import json
import math
import torch
import torch.nn as nn
from hierarchical_moe import HierarchicalMoE
from datasets import load_dataset
import tiktoken
import logging

# Hide excessive huggingface/dataset logs so the controller output stays clean
logging.getLogger("datasets").setLevel(logging.ERROR)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, expert_hidden_dim, num_groups, experts_per_group, num_heads, dropout, router_temperature):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.moe = HierarchicalMoE(
            n_embd=n_embd,
            expert_hidden_dim=expert_hidden_dim,
            num_groups=num_groups,
            experts_per_group=experts_per_group,
            router_temperature=router_temperature
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, causal_mask):
        # Attention with residual, LayerNorm, and Dropout
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal_mask, is_causal=True)
        x = x + self.dropout(attn_out)
        
        # MoE with residual, LayerNorm, and Dropout
        moe_out = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x

class SimpleTransformerLM(nn.Module):
    def __init__(self, config, vocab_size, max_seq_len=256):
        super().__init__()
        self.n_embd = config["n_embd"]
        self.token_emb = nn.Embedding(vocab_size, self.n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, self.n_embd)
        self.dropout = nn.Dropout(config.get("dropout", 0.0))
        
        num_heads = config.get("num_heads", 4)
        if self.n_embd % num_heads != 0:
            num_heads = 1 # Fallback to 1 head if config tests an odd n_embd
            
        num_layers = config.get("num_layers", 1)
        
        # Stack multiple MoE-enabled Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                n_embd=self.n_embd,
                expert_hidden_dim=config["expert_hidden_dim"],
                num_groups=config["num_groups"],
                experts_per_group=config["experts_per_group"],
                num_heads=num_heads,
                dropout=config.get("dropout", 0.0),
                router_temperature=config.get("router_temperature", 1.0)
            ) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(self.n_embd)
        # Output Head projecting to vocabulary mapped logic
        self.head = nn.Linear(self.n_embd, vocab_size)

    def forward(self, x):
        B, S = x.size()
        positions = torch.arange(0, S, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token Embedding + Positional Embedding
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)
        
        # Causal Attention Mask (prevents looking at future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S).to(x.device)
        
        # Pass through all layers
        for layer in self.layers:
            h = layer(h, causal_mask)
            
        h = self.ln_f(h)
        # LM Head (Prediction logits)
        logits = self.head(h)
        return logits

def get_mixed_dataset(tokenizer, max_samples=50, max_seq_len=128):
    """
    Downloads & formats small real subsets of GSM8K, ARC, and MMLU 
    for fast MoE evolution scoring cycles.
    """    
    texts = []
    
    # GSM8K
    try:
        gsm8k = load_dataset("gsm8k", "main", split="train", streaming=True)
        for i, item in enumerate(gsm8k):
            if i >= max_samples: break
            texts.append(f"Q: {item['question']}\nA: {item['answer']}")
    except:
        pass
        
    # ARC
    try:
        arc = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=True)
        for i, item in enumerate(arc):
            if i >= max_samples: break
            choices = " ".join([f"({label}) {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])])
            texts.append(f"Q: {item['question']}\nOptions: {choices}\nA: {item['answerKey']}")
    except:
        pass
        
    # MMLU
    try:
        mmlu = load_dataset("cais/mmlu", "all", split="test", streaming=True)
        for i, item in enumerate(mmlu):
            if i >= max_samples: break
            choices = " ".join([f"({idx}): {c}" for idx, c in enumerate(item["choices"])])
            texts.append(f"Q: {item['question']}\nOptions: {choices}\nA: {item['answer']}")
    except:
        pass
        
    if not texts:
        # Fallback dummy check if HF downloading fails inside subprocess
        texts = ["The quick brown fox jumps over the lazy dog."] * 32

    # Tiktoken Encoding
    encoding = tokenizer.encode("\n\n---\n\n".join(texts), allowed_special="all")
    tokens = torch.tensor(encoding, dtype=torch.long)
    
    sequences = []
    for i in range(0, len(tokens) - max_seq_len, max_seq_len):
        sequences.append(tokens[i : i + max_seq_len])
        
    if not sequences:
        padded = torch.zeros(max_seq_len, dtype=torch.long)
        padded[:len(tokens)] = tokens[:max_seq_len]
        sequences.append(padded)

    batch = torch.stack(sequences)
    return batch


def train_and_eval(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        sys.exit(1)

    torch.manual_seed(42)

    # Tokenizer Setup (Using Tiktoken for GPT-2 BPE)
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = config.get("vocab_size", 50257) # Default to GPT-2 vocab size
    except Exception as e:
        print(f"RESULT_LOSS:100.000") # Extreme loss if BPE fails to load
        sys.exit(1)

    max_seq_len = 128
    
    model = SimpleTransformerLM(config, vocab_size=vocab_size, max_seq_len=max_seq_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Precompute token byte lengths for BPB evaluation
    token_bytes_list = []
    for i in range(vocab_size):
        try:
            b_len = len(tokenizer.decode_bytes([i]))
            token_bytes_list.append(b_len)
        except Exception:
            token_bytes_list.append(0) # Special tokens or invalid
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Load multi-task datasets 
    data = get_mixed_dataset(tokenizer, max_samples=25, max_seq_len=max_seq_len).to(device)
    
    split_idx = max(1, int(0.8 * len(data)))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    if len(val_data) == 0:
        val_data = train_data 

    epochs = 4 # Fewer epochs to keep the MoE evolution cycle quick
    batch_size = config.get("batch_size", 8) 
    
    warmup_epochs = max(1, epochs // 5)
    sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched1, sched2], milestones=[warmup_epochs]
    )
    
    metrics = {
        "best_val_loss": float('inf'),
        "best_val_bpb": float('inf'),
        "final_train_loss": float('inf'),
        "accuracy": 0.0,
        "expert_utilization": [],
        "load_distribution": [],
        "history": []
    }
    
    step_count = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        
        # Sliced Batch Evaluation Support
        for i in range(0, len(train_data), batch_size):
            X = train_data[i:i+batch_size, :-1]  # Context
            Y = train_data[i:i+batch_size, 1:]   # Next token target
            
            optimizer.zero_grad()
            logits = model(X)
            
            # Cross-entropy requires flattened inputs [B*S, V] vs [B*S]
            loss = criterion(logits.view(-1, vocab_size), Y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            step_count += 1
            
        avg_train_loss = total_train_loss / max(1, train_batches)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_val_nats = 0.0
        total_val_bytes = 0.0
        
        all_loads = []
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                X = val_data[i:i+batch_size, :-1]
                Y = val_data[i:i+batch_size, 1:]
                logits = model(X)
                
                loss = criterion(logits.view(-1, vocab_size), Y.reshape(-1))
                total_val_loss += loss.item()
                
                # --- BPB Calculation ---
                loss_unreduced = nn.functional.cross_entropy(logits.view(-1, vocab_size), Y.reshape(-1), reduction='none')
                nbytes = token_bytes_tensor[Y.reshape(-1)]
                mask = (nbytes > 0).float()
                total_val_nats += (loss_unreduced * mask).sum().item()
                total_val_bytes += nbytes.sum().item()
                # -----------------------
                
                # Accuracy tracking
                preds = torch.argmax(logits.view(-1, vocab_size), dim=-1)
                total_correct += (preds == Y.reshape(-1)).sum().item()
                total_tokens += Y.numel()
                
                # Fetch expert load logic
                batch_layer_loads = []
                for layer in model.layers:
                    if hasattr(layer.moe, 'last_expert_load'):
                        batch_layer_loads.append(layer.moe.last_expert_load.cpu())
                if batch_layer_loads:
                    all_loads.append(torch.stack(batch_layer_loads).mean(dim=0))

        # avg_val_loss adjusts for number of batches, not sequence size length anymore due to slicing mechanism.
        num_val_batches = max(1, (len(val_data) + batch_size - 1) // batch_size)
        avg_val_loss = total_val_loss / num_val_batches
        accuracy = total_correct / max(1, total_tokens)
        val_bpb = (total_val_nats / (math.log(2) * total_val_bytes)) if total_val_bytes > 0 else float('inf')
        
        if all_loads:
            final_load = torch.stack(all_loads).mean(dim=0).tolist()
        else:
            final_load = []
            
        metrics["history"].append({
            "step": step_count,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_bpb": val_bpb,
            "accuracy": accuracy,
            "load_distribution": final_load,
            "lr": config.get("learning_rate", 0.0)
        })
        
        if avg_val_loss < metrics["best_val_loss"]:
            metrics["best_val_loss"] = avg_val_loss
            metrics["best_val_bpb"] = val_bpb
            metrics["accuracy"] = accuracy
            metrics["expert_utilization"] = final_load
            metrics["load_distribution"] = final_load
            
            # Save transient weights for the controller to pick up!
            torch.save(model.state_dict(), "last_run_weights.pt")
            
        scheduler.step()
            
    metrics["final_train_loss"] = avg_train_loss

    # Export report card for the controller AI
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"RESULT_LOSS:{metrics['best_val_loss']:.6f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_eval.py <config_path.json>")
        sys.exit(1)
    train_and_eval(sys.argv[1])
