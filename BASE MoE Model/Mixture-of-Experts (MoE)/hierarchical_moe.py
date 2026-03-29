import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import tiktoken
except ImportError:
    pass

class HierarchicalRouter(nn.Module):
    """
    Dynamic 2-Level Hierarchical Router.
    Level 1: Select groups where P(group) >= 1 / num_groups.
    Level 2: Select experts where P(expert|group) >= 1 / experts_per_group.
    """
    def __init__(self, d_model: int, num_groups: int, experts_per_group: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.top_k = top_k
        
        # Level 1 gating: W_group
        self.group_gate = nn.Linear(d_model, num_groups, bias=False)
        # Level 2 gating: W_all_experts
        self.expert_gate = nn.Linear(d_model, num_groups * experts_per_group, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [N, D]
        Returns:
            valid_mask: [N, G * E_g] boolean mask indicating selected experts
            normalized_weights: [N, G * E_g] float tensor with normalized weights (0 where masked)
        """
        N, D = x.shape
        G = self.num_groups
        E_g = self.experts_per_group

        # --- Level 1: Groups ---
        group_logits = self.group_gate(x) # [N, G]
        group_probs = F.softmax(group_logits, dim=-1) # [N, G]
        
        # Threshold: P(group) >= 1 / G
        group_mask = group_probs >= (1.0 / G) # [N, G]

        # --- Level 2: Experts within groups ---
        all_expert_logits = self.expert_gate(x) # [N, G * E_g]
        all_expert_logits = all_expert_logits.view(N, G, E_g)

        # Softmax over experts WITHIN each group
        expert_probs = F.softmax(all_expert_logits, dim=-1) # [N, G, E_g]

        # Threshold: P(expert|group) >= 1 / E_g
        expert_mask = expert_probs >= (1.0 / E_g) # [N, G, E_g]

        # --- Combine Levels ---
        # Expert is only active if its group is also active
        # group_mask is [N, G], unsqueeze to [N, G, 1] for broadcasting
        valid_mask_3d = group_mask.unsqueeze(-1) & expert_mask # [N, G, E_g]
        
        # Final weight before normalization = P(group) * P(expert | group)
        final_weights_3d = group_probs.unsqueeze(-1) * expert_probs # [N, G, E_g]
        
        # Flatten to [N, G * E_g] for the MoE layer
        valid_mask = valid_mask_3d.view(N, -1)
        final_probs_flat = final_weights_3d.view(N, -1)
        
        # --- Apply Minimum Top-K Logic ---
        # Count how many experts passed the hierarchical threshold for each token
        num_selected = valid_mask.sum(dim=-1) # [N]
        
        # Get the top-k experts globally for each token
        _, topk_indices = torch.topk(final_probs_flat, self.top_k, dim=-1) # [N, top_k]
        
        # Create a boolean mask for the top-k experts
        topk_mask = torch.zeros_like(valid_mask).scatter_(1, topk_indices, 1).bool() # [N, G * E_g]
        
        # Condition: if threshold selected less than top_k experts, fallback to topk_mask
        use_topk = num_selected < self.top_k # [N]
        
        # Final mask combination
        final_valid_mask = torch.where(use_topk.unsqueeze(-1), topk_mask, valid_mask)
        
        # Zero out unselected weights
        selected_weights = final_probs_flat * final_valid_mask.float() # [N, G * E_g]
        
        # Renormalize weights so they sum to 1 per token
        weight_sum = selected_weights.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        weight_sum = torch.clamp(weight_sum, min=1e-9)
        normalized_weights = selected_weights / weight_sum # [N, G * E_g]

        return final_valid_mask, normalized_weights


class Expert(nn.Module):
    """Standard FeedForward expert."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        return self.w2(self.activation(self.w1(x)))


class HierarchicalMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_groups: int, experts_per_group: int, top_k: int = 2):
        super().__init__()
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.top_k = top_k
        total_experts = num_groups * experts_per_group
        
        self.router = HierarchicalRouter(d_model, num_groups, experts_per_group, top_k)
        
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(total_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: [N, D] or [B, S, D]
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            B, S, D = original_shape
            x = x.view(-1, D)
        
        N, D = x.shape
        
        # 1. Routing (Dynamic mask and normalized weights)
        valid_mask, weights = self.router(x) # Both are [N, total_experts]
        
        output = torch.zeros_like(x)
        
        # 2. Sparse Execution (Loop over experts to preserve compute savings)
        total_experts = self.num_groups * self.experts_per_group
        for e_id in range(total_experts):
            # mask tells us which tokens go to this expert
            mask = valid_mask[:, e_id] # [N]
            if not mask.any():
                continue
                
            # token_indices: which tokens in the batch went here
            token_idx = mask.nonzero(as_tuple=False).squeeze(-1) # [num_active]
            
            # Gather corresponding token vectors
            token_features = x[token_idx] # [num_active, D]
            
            # Run the expert FeedForward
            expert_out = self.experts[e_id](token_features) # [num_active, D]
            
            # Apply routing weights
            w = weights[token_idx, e_id].unsqueeze(-1) # [num_active, 1]
            expert_out_weighted = expert_out * w
            
            # Scatter back to output
            output.index_add_(0, token_idx, expert_out_weighted)

        # Reshape back if input was 3D
        if len(original_shape) == 3:
            output = output.view(*original_shape)
            
        return output

class TransformerBlock(nn.Module):
    """A standard Transformer Block with the Hierarchical MoE injected."""
    def __init__(self, d_model: int, d_ff: int, num_groups: int, experts_per_group: int, num_heads: int, dropout: float = 0.1, router_top_k: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.moe = HierarchicalMoE(
            d_model=d_model,
            d_ff=d_ff,
            num_groups=num_groups,
            experts_per_group=experts_per_group,
            top_k=router_top_k
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, causal_mask=None):
        # Attention with residual, LayerNorm, and Dropout
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal_mask, is_causal=causal_mask is not None)
        x = x + self.dropout(attn_out)
        
        # MoE with residual, LayerNorm, and Dropout
        moe_out = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x

class SimpleTransformerLM(nn.Module):
    """
    A full wrapper Language Model showcasing how to integrate
    tiktoken, input embeddings, positional embeddings, and the MoE.
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 256, d_ff: int = 1024, num_groups: int = 4, experts_per_group: int = 4, num_heads: int = 4, num_layers: int = 2, max_seq_len: int = 512, dropout: float = 0.1, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # Tokenizer definition
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
        except ImportError:
            self.tokenizer = None
            
        # Input Embedding & Positional Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Stack multiple MoE-enabled Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_groups=num_groups,
                experts_per_group=experts_per_group,
                num_heads=num_heads,
                dropout=dropout,
                router_top_k=top_k
            ) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

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
            h = layer(h, causal_mask=causal_mask)
            
        h = self.ln_f(h)
        logits = self.head(h)
        return logits
