import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalRouter(nn.Module):
    """
    Dynamic 2-Level Hierarchical Router.
    Level 1: Select groups where P(group) >= 1 / num_groups.
    Level 2: Select experts where P(expert|group) >= 1 / experts_per_group.
    """
    def __init__(self, n_embd: int, num_groups: int, experts_per_group: int, router_temperature: float = 1.0):
        super().__init__()
        self.n_embd = n_embd
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.router_temperature = router_temperature
        
        # Level 1 gating: W_group
        self.group_gate = nn.Linear(n_embd, num_groups, bias=False)
        # Level 2 gating: W_all_experts
        self.expert_gate = nn.Linear(n_embd, num_groups * experts_per_group, bias=False)

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
        group_probs = F.softmax(group_logits / self.router_temperature, dim=-1) # [N, G]
        
        # Threshold: P(group) >= 1 / G
        group_mask = group_probs >= (1.0 / G) # [N, G]

        # --- Level 2: Experts within groups ---
        all_expert_logits = self.expert_gate(x) # [N, G * E_g]
        all_expert_logits = all_expert_logits.view(N, G, E_g)

        # Softmax over experts WITHIN each group
        expert_probs = F.softmax(all_expert_logits / self.router_temperature, dim=-1) # [N, G, E_g]

        # Threshold: P(expert|group) >= 1 / E_g
        expert_mask = expert_probs >= (1.0 / E_g) # [N, G, E_g]

        # --- Combine Levels ---
        # Expert is only active if its group is also active
        # group_mask is [N, G], unsqueeze to [N, G, 1] for broadcasting
        valid_mask_3d = group_mask.unsqueeze(-1) & expert_mask # [N, G, E_g]
        
        # Final weight before normalization = P(group) * P(expert | group)
        final_weights_3d = group_probs.unsqueeze(-1) * expert_probs # [N, G, E_g]
        
        # Zero out unselected weights
        selected_weights_3d = final_weights_3d * valid_mask_3d.float() # [N, G, E_g]
        
        # Flatten to [N, G * E_g] for the MoE layer
        valid_mask = valid_mask_3d.view(N, -1)
        selected_weights = selected_weights_3d.view(N, -1)
        
        # Renormalize weights so they sum to 1 per token
        # Some tokens might sum to very small numbers if thresholds are weird, but theoretically sum >= avg
        weight_sum = selected_weights.sum(dim=-1, keepdim=True)
        # Avoid division by zero just in case (e.g. padding tokens or numerical issues)
        weight_sum = torch.clamp(weight_sum, min=1e-9)
        normalized_weights = selected_weights / weight_sum # [N, G * E_g]

        return valid_mask, normalized_weights


class Expert(nn.Module):
    """Standard FeedForward expert."""
    def __init__(self, n_embd: int, expert_hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(n_embd, expert_hidden_dim)
        self.w2 = nn.Linear(expert_hidden_dim, n_embd)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        return self.w2(self.activation(self.w1(x)))


class HierarchicalMoE(nn.Module):
    def __init__(self, n_embd: int, expert_hidden_dim: int, num_groups: int, experts_per_group: int, router_temperature: float = 1.0):
        super().__init__()
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        total_experts = num_groups * experts_per_group
        
        self.router = HierarchicalRouter(n_embd, num_groups, experts_per_group, router_temperature)
        
        self.experts = nn.ModuleList([
            Expert(n_embd, expert_hidden_dim) for _ in range(total_experts)
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
        
        self.last_expert_load = valid_mask.float().mean(dim=0)
        
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
