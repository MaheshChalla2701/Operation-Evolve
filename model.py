"""
model.py — Sparse MoE Transformer with RoPE

Improvements in v2:
  - Rotary Position Embeddings (RoPE) replace learned wpe
    → generalizes to longer sequences, used in LLaMA/Mistral/GPT-4
  - All existing SparseMoE logic preserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Rotary Position Embeddings (RoPE) ────────────────────────────────────────

def precompute_rope(head_dim: int, max_seq_len: int, base: int = 10000, device=None):
    """
    Precompute cosine/sine tables for RoPE.
    Returns cos, sin of shape (1, max_seq_len, 1, head_dim//2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, theta)  # (seq_len, head_dim//2)
    return freqs.cos().unsqueeze(0).unsqueeze(2), freqs.sin().unsqueeze(0).unsqueeze(2)
    # shape: (1, seq_len, 1, head_dim//2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to q or k tensor of shape (B, T, n_head, head_dim).
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos_t = cos[:, :x.shape[1], :, :]  # (1, T, 1, d)
    sin_t = sin[:, :x.shape[1], :, :]
    x_rot = torch.cat([x1 * cos_t - x2 * sin_t,
                        x1 * sin_t + x2 * cos_t], dim=-1)
    return x_rot


# ── Causal Self-Attention with RoPE ──────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_dim = config['n_embd'] // config['n_head']

        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=False)
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=False)

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config['block_size'], config['block_size']))
                  .view(1, 1, config['block_size'], config['block_size'])
        )

        # Precompute RoPE tables
        cos, sin = precompute_rope(self.head_dim, config['block_size'])
        self.register_buffer("rope_cos", cos)  # (1, block_size, 1, head_dim//2)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to (B, T, n_head, head_dim) for RoPE
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Apply RoPE to queries and keys
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Transpose to (B, n_head, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


# ── Mixture of Experts ────────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, n_embd, expert_hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expert_hidden_dim),
            nn.GELU(),
            nn.Linear(expert_hidden_dim, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer with configurable top_k routing.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config['num_experts']
        self.n_embd = config['n_embd']
        self.top_k = config.get('top_k', 1)
        self.temperature = config.get('router_temperature', 1.0)

        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(self.n_embd, config['expert_hidden_dim'], config.get('expert_dropout', 0.1))
            for _ in range(self.num_experts)
        ])

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        N = x_flat.size(0)

        logits = self.router(x_flat) / self.temperature
        prob = F.softmax(logits, dim=-1)

        k = min(self.top_k, self.num_experts)
        top_prob, top_indices = torch.topk(prob, k, dim=-1)
        top_prob = top_prob / (top_prob.sum(dim=-1, keepdim=True) + 1e-9)

        out_flat = torch.zeros_like(x_flat)
        expert_token_counts = torch.zeros(self.num_experts, device=x.device)

        for exp_idx in range(self.num_experts):
            is_selected = (top_indices == exp_idx)
            token_mask = is_selected.any(dim=-1)
            expert_token_counts[exp_idx] = token_mask.sum().float()

            if bool(token_mask.any()):
                tokens = x_flat[token_mask]
                weights = (is_selected[token_mask] * top_prob[token_mask]).sum(dim=-1, keepdim=True)
                expert_out = self.experts[exp_idx](tokens)
                out_flat[token_mask] += expert_out * weights

        total_tokens = expert_token_counts.sum() + 1e-9
        expert_utilization = expert_token_counts / total_tokens

        out = out_flat.view(B, T, C)
        return out, expert_utilization


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.moe = SparseMoELayer(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x_moe, expert_utilization = self.moe(self.ln_2(x))
        x = x + x_moe
        return x, expert_utilization


# ── Full Model ────────────────────────────────────────────────────────────────

class SparseMoETransformer(nn.Module):
    """
    Decoder-only Sparse MoE Transformer with RoPE positional encoding.

    Config keys:
        vocab_size, block_size, n_embd, n_head, n_layer, dropout,
        num_experts, expert_hidden_dim, top_k, router_temperature,
        learning_rate, batch_size
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config['vocab_size'], config['n_embd']),
            # wpe REMOVED — RoPE handles positions inside CausalSelfAttention
            drop=nn.Dropout(config['dropout']),
            h=nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f=nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config['block_size'], \
            f"Sequence length {T} exceeds block_size {self.config['block_size']}"

        # Token embeddings only — RoPE handles positions
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        x = self.transformer.drop(tok_emb)

        all_expert_utilization = []
        for block in self.transformer.h:
            x, expert_util = block(x)
            all_expert_utilization.append(expert_util)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid_mask = targets != -1
                correct = (preds == targets) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float() \
                    if valid_mask.sum() > 0 else torch.tensor(0.0)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            accuracy = None

        expert_utilization = torch.stack(all_expert_utilization) if all_expert_utilization else torch.zeros(0)
        return logits, loss, expert_utilization, accuracy

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
