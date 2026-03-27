import torch
import torch.nn as nn
from hierarchical_moe import HierarchicalMoE

class TransformerBlockWithMoE(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, num_groups: int, experts_per_group: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.moe = HierarchicalMoE(d_model, d_ff, num_groups, experts_per_group)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        moe_out = self.moe(x)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)
        return x

def demo():
    print("--- Dynamic Hierarchical MoE Transformer Block Demo ---")
    d_model = 256
    nhead = 4
    d_ff = 512
    num_groups = 4 # 4 clusters
    experts_per_group = 8 # 8 experts per cluster (Total 32 experts)
    
    model = TransformerBlockWithMoE(d_model, nhead, d_ff, num_groups, experts_per_group)
    
    B, S = 2, 16
    X = torch.randn(B, S, d_model)
    
    with torch.no_grad():
        out = model(X)
        valid_mask, weights = model.moe.router(X.view(-1, d_model))
        avg_active = valid_mask.float().sum(dim=-1).mean().item()
        
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Average active experts per token: {avg_active:.2f} (variable dynamic routing)")
    print("\n✓ Dynamic Hierarchical MoE integrated & executed smoothly!")

if __name__ == "__main__":
    demo()
