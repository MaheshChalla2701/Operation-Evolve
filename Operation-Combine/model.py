"""
model.py – Neural network architectures for Operation Evolve.

Supported models (configurable via config.model_type):
  - "SimpleNN"          → 3-layer MLP
  - "SimpleTransformer" → Transformer encoder + linear head

Both share the same interface: forward(x) → logits of shape (B, num_classes).
"""

import torch
import torch.nn as nn
from config import EvolveConfig


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModel(nn.Module):
    """Common interface for all evolve models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# SimpleNN  – 3-layer MLP
# ---------------------------------------------------------------------------

class SimpleNN(BaseModel):
    """
    Lightweight Multi-Layer Perceptron.

    Architecture:
        input_dim → hidden_dim → hidden_dim → num_classes
    with BatchNorm + ReLU + Dropout between layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SimpleTransformer  – Transformer encoder + head
# ---------------------------------------------------------------------------

class SimpleTransformer(BaseModel):
    """
    Single-sequence Transformer encoder.

    The 1-D input vector is treated as a sequence of length 1.
    This makes it trivially compatible with both CPU and GPU.

    Architecture:
        Linear projection → TransformerEncoder (N layers) → Global mean pool → Linear head
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) → treat as sequence of length 1: (B, 1, input_dim)
        x = x.unsqueeze(1)                     # (B, 1, input_dim)
        x = self.input_proj(x)                 # (B, 1, d_model)
        x = self.encoder(x)                    # (B, 1, d_model)
        x = x.mean(dim=1)                      # (B, d_model)  global mean pool
        x = self.dropout(x)
        return self.head(x)                    # (B, num_classes)


# ---------------------------------------------------------------------------
# TransformerLM  – Transformer encoder + tokenizer & embeddings
# ---------------------------------------------------------------------------

class TransformerLM(BaseModel):
    """
    Transformer Language Model for discrete token inputs.
    Incorporates TikToken tokenizer, token embeddings, and positional embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except ImportError:
            self.tokenizer = None
            
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected to be shape [B, S] containing token IDs
        B, S = x.size()
        positions = torch.arange(0, S, dtype=torch.long, device=x.device).unsqueeze(0)
        
        x = self.token_emb(x) + self.pos_emb(positions)  # (B, S, d_model)
        x = self.dropout(x)
        x = self.encoder(x)                              # (B, S, d_model)
        
        # Sequence-level global mean pooling
        x = x.mean(dim=1)                                # (B, d_model)
        return self.head(x)                              # (B, num_classes)


# ---------------------------------------------------------------------------
# HierarchicalMoELM (Wrapped MoE layer for Language Modeling)
# ---------------------------------------------------------------------------

class HierarchicalTransformerBlock(nn.Module):
    def __init__(self, n_embd, expert_hidden_dim, num_groups, experts_per_group, num_heads, dropout, router_temperature):
        super().__init__()
        # Ensure n_embd is divisible by num_heads
        if n_embd % num_heads != 0:
            num_heads = 1
            
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        
        from hierarchical_moe import HierarchicalMoE
        self.moe = HierarchicalMoE(
            n_embd=n_embd,
            expert_hidden_dim=expert_hidden_dim,
            num_groups=num_groups,
            experts_per_group=experts_per_group,
            router_temperature=router_temperature
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, causal_mask=None):
        if causal_mask is not None:
            attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal_mask, is_causal=True)
        else:
            attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
            
        x = x + self.dropout(attn_out)
        moe_out = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x


class HierarchicalMoELM(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_embd: int,
        expert_hidden_dim: int,
        num_groups: int,
        experts_per_group: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        router_temperature: float = 1.0
    ):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, self.n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, self.n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            HierarchicalTransformerBlock(
                n_embd=self.n_embd,
                expert_hidden_dim=expert_hidden_dim,
                num_groups=num_groups,
                experts_per_group=experts_per_group,
                num_heads=num_heads,
                dropout=dropout,
                router_temperature=router_temperature
            ) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        if x.dtype in [torch.float16, torch.float32, torch.float64]:
            if not hasattr(self, 'float_proj'):
                self.float_proj = nn.Linear(x.size(-1), self.n_embd).to(x.device)
            x = x.unsqueeze(1)
            h = self.float_proj(x)
            S = 1
        else:
            B, S = x.size()
            positions = torch.arange(0, S, dtype=torch.long, device=x.device).unsqueeze(0)
            h = self.token_emb(x) + self.pos_emb(positions)
            
        h = self.dropout(h)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S).to(h.device) if S > 1 else None
        
        for layer in self.layers:
            h = layer(h, causal_mask)
            
        h = self.ln_f(h)
        h = h.mean(dim=1)  # global mean pool
        return self.head(h)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(config: EvolveConfig) -> BaseModel:
    """
    Instantiate the model specified in config.model_type and move it to the
    correct device.
    """
    if config.model_type == "SimpleNN":
        model = SimpleNN(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
        )
    elif config.model_type == "SimpleTransformer":
        d_model = config.hidden_dim
        while d_model % config.num_heads != 0:
            d_model += 1
        model = SimpleTransformer(
            input_dim=config.input_dim,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
        )
    elif config.model_type == "HierarchicalMoE":
        n_embd = config.d_model
        while n_embd % config.num_heads != 0:
            n_embd += 1
            
        model = HierarchicalMoELM(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            n_embd=n_embd,
            expert_hidden_dim=config.expert_hidden_dim,
            num_groups=config.num_groups,
            experts_per_group=config.experts_per_group,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
            router_temperature=config.router_temperature
        )
    else:
        raise ValueError(
            f"Unknown model_type '{config.model_type}'."
        )

    device = config.get_device()
    model = model.to(device)
    return model
