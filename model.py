import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        # output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        # regularization
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


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

    Attributes:
        num_experts (int): Total number of experts.
        top_k (int): Number of experts to activate per token (sparse routing).
        temperature (float): Router temperature to control routing sharpness.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config['num_experts']
        self.n_embd = config['n_embd']
        self.top_k = config.get('top_k', 1)            # configurable top_k (default=1)
        self.temperature = config.get('router_temperature', 1.0)

        # Router/Gating network: maps token embedding → expert logits
        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)

        # Expert networks (each is an independent MLP)
        self.experts = nn.ModuleList([
            Expert(self.n_embd, config['expert_hidden_dim'], config.get('expert_dropout', 0.1))
            for _ in range(self.num_experts)
        ])

    def forward(self, x):
        B, T, C = x.size()
        # Flatten sequence dimension so each token is processed independently
        # x_flat: (B*T, C)
        x_flat = x.view(-1, C)
        N = x_flat.size(0)  # total tokens = B * T

        # Compute routing logits and apply temperature
        # logits: (N, num_experts)
        logits = self.router(x_flat) / self.temperature

        # Softmax to get routing probabilities
        prob = F.softmax(logits, dim=-1)

        # Select top_k experts per token
        # top_prob: (N, top_k), top_indices: (N, top_k)
        k = min(self.top_k, self.num_experts)
        top_prob, top_indices = torch.topk(prob, k, dim=-1)

        # Normalize top_k weights so they sum to 1 per token
        top_prob = top_prob / (top_prob.sum(dim=-1, keepdim=True) + 1e-9)

        # Initialize output
        out_flat = torch.zeros_like(x_flat)

        # Track how many tokens each expert processes (for utilization metrics)
        expert_token_counts = torch.zeros(self.num_experts, device=x.device)

        # Dispatch tokens to each expert (sparse routing)
        for exp_idx in range(self.num_experts):
            # Find all (token, slot) pairs where this expert is selected
            # is_selected: (N, top_k) boolean
            is_selected = (top_indices == exp_idx)           # (N, top_k)
            token_mask = is_selected.any(dim=-1)             # (N,) — which tokens use this expert

            expert_token_counts[exp_idx] = token_mask.sum().float()

            if bool(token_mask.any()):
                # Extract the tokens routed to this expert
                tokens = x_flat[token_mask]                  # (n_selected, C)

                # Get the weight this expert has for each selected token
                # (weighted average over the top_k slots that selected this expert)
                weights = (is_selected[token_mask] * top_prob[token_mask]).sum(dim=-1, keepdim=True)

                # Forward through expert and weight output
                expert_out = self.experts[exp_idx](tokens)   # (n_selected, C)
                out_flat[token_mask] += expert_out * weights

        # Compute expert utilization as fraction of total tokens
        total_tokens = expert_token_counts.sum() + 1e-9
        expert_utilization = expert_token_counts / total_tokens  # (num_experts,)

        # Restore original shape
        out = out_flat.view(B, T, C)
        return out, expert_utilization


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.moe = SparseMoELayer(config)

    def forward(self, x):
        # Attention sub-layer + residual
        x = x + self.attn(self.ln_1(x))
        # MoE sub-layer + residual
        x_moe, expert_utilization = self.moe(self.ln_2(x))
        x = x + x_moe
        return x, expert_utilization


class SparseMoETransformer(nn.Module):
    """
    Decoder-only Sparse Mixture-of-Experts Transformer.

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
            wpe=nn.Embedding(config['block_size'], config['n_embd']),
            drop=nn.Dropout(config['dropout']),
            h=nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f=nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # Weight tying (token embedding shares weights with output projection)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
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
            f"Cannot forward sequence of length {T}, block size is {self.config['block_size']}"

        # Token + positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (1, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks, accumulate expert utilization
        all_expert_utilization = []
        for block in self.transformer.h:
            x, expert_util = block(x)
            all_expert_utilization.append(expert_util)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training/eval mode: compute loss and token accuracy
            logits = self.lm_head(x)       # (B, T, vocab_size)

            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

            # Token-level accuracy (top-1 prediction matches target)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)           # (B, T)
                valid_mask = targets != -1              # ignore padding
                correct = (preds == targets) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float() if valid_mask.sum() > 0 else torch.tensor(0.0)  # type: ignore
        else:
            # Inference mode: logits for last token only (fast generation)
            logits = self.lm_head(x[:, [-1], :])       # (B, 1, vocab_size)
            loss = None
            accuracy = None

        # Stack utilization across layers → (n_layer, num_experts)
        expert_utilization = torch.stack(all_expert_utilization) if all_expert_utilization else torch.zeros(0)

        return logits, loss, expert_utilization, accuracy

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation. Samples tokens until max_new_tokens reached.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            # Forward pass (inference mode)
            logits, _, _, _ = self(idx_cond)
            # Focus on the last time step and apply temperature
            logits = logits[:, -1, :] / temperature
            # Optional top-k filtering for diversity control
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
