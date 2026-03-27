import torch
from hierarchical_moe import HierarchicalMoE

def test_hierarchical_moe():
    print("Initializing Dynamic HierarchicalMoE...")
    d_model = 128
    d_ff = 256
    num_groups = 4
    experts_per_group = 4
    
    moe = HierarchicalMoE(d_model, d_ff, num_groups, experts_per_group)
    
    # 1. Test 3D Input [B, S, D]
    B, S = 2, 8
    X = torch.randn(B, S, d_model)
    print(f"\nInput shape: {X.shape}")
    
    out = moe(X)
    print(f"Output shape: {out.shape}")
    assert out.shape == X.shape, f"Shape mismatch: {out.shape} != {X.shape}"
    print("✓ 3D Input Shape Test Passed")
    
    # 2. Test 2D Input [N, D]
    N = 16
    X2 = torch.randn(N, d_model, requires_grad=True)
    
    out2 = moe(X2)
    assert out2.shape == X2.shape, f"Shape mismatch: {out2.shape} != {X2.shape}"
    print("✓ 2D Input Shape Test Passed")
    
    # 3. Test Backprop
    loss = out2.sum()
    loss.backward()
    print("✓ Backprop Test Passed (No Crash)")
    
    # 4. Print dynamic routing stats
    with torch.no_grad():
        valid_mask, weights = moe.router(X2)
        
        # Check normalization
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums)), "Weights did not normalize to 1"
        print("✓ Weight Normalization Passed")
        
        experts_per_token = valid_mask.sum(dim=-1).float()
        avg_experts = experts_per_token.mean().item()
        
        print(f"\nDynamic Routing Stats for {N} tokens:")
        print(f" Average experts per token: {avg_experts:.2f} (out of {num_groups * experts_per_group})")
        print(f" Min experts per token: {experts_per_token.min().item()}")
        print(f" Max experts per token: {experts_per_token.max().item()}")

if __name__ == "__main__":
    test_hierarchical_moe()
    print("\nAll tests passed successfully!")
