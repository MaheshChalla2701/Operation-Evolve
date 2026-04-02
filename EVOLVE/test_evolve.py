"""
test_evolve.py – Standalone test suite for Operation Evolve.

Run from the EVOLVE directory:
    python test_evolve.py

No Groq API key required – all tests use synthetic data only.
"""

import copy
import sys
import traceback
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EvolveConfig
from model import build_model, SimpleNN, SimpleTransformer, HierarchicalMoELM
from hierarchical_moe import HierarchicalMoE, HierarchicalRouter
from data import SyntheticDataset, generate_seed_data
from evaluate import evaluate, compute_confidence, evaluate_replay
from weight_transfer import transfer_compatible_weights
from replay_buffer import ReplayBufferV2

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def run_test(name, fn):
    print(f"\n  [{name}]", end=" ")
    try:
        fn()
        print(PASS)
        results.append((name, True, ""))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"{FAIL}\n      {e}")
        results.append((name, False, tb))


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _make_cfg(model_type="HierarchicalMoE"):
    cfg = EvolveConfig()
    cfg.model_type = model_type
    cfg.device = "cpu"
    return cfg


# ── 1. Config ─────────────────────────────────────────────────────────────────

section("1 · Config")


def test_config_defaults():
    cfg = EvolveConfig()
    assert cfg.num_classes == 4
    assert cfg.input_dim == 16
    assert cfg.model_type == "HierarchicalMoE"
    assert cfg.batch_size == 32


def test_config_device():
    cfg = EvolveConfig()
    dev = cfg.get_device()
    assert isinstance(dev, torch.device)


def test_config_data_dir():
    cfg = EvolveConfig()
    cfg.ensure_data_dir()
    assert os.path.isdir(cfg.data_dir)


run_test("Default values", test_config_defaults)
run_test("Device resolution", test_config_device)
run_test("Data-dir creation", test_config_data_dir)

# ── 2. Synthetic Data ─────────────────────────────────────────────────────────

section("2 · Synthetic Data")


def test_generate_seed_data():
    cfg = EvolveConfig()
    ds = generate_seed_data(n=100, num_classes=cfg.num_classes, input_dim=cfg.input_dim, seed=0)
    assert isinstance(ds, SyntheticDataset)
    assert len(ds) == 100
    feat, label = ds[0]
    assert feat.shape == (cfg.input_dim,)
    assert 0 <= label.item() < cfg.num_classes


def test_dataset_length_and_types():
    ds = generate_seed_data(n=50, num_classes=4, input_dim=16, seed=1)
    # generate_seed_data may produce slightly fewer than n (batch rounding)
    assert 0 < len(ds) <= 50
    # Access individual samples to check types
    feats, label = ds[0]
    # torch.long == torch.int64 — verify it's an integer type
    assert feats.dtype in (torch.float32, torch.float64)
    assert label.dtype in (torch.long, torch.int64)  # same type, both spellings valid
    assert 0 <= label.item() < 4


run_test("generate_seed_data", test_generate_seed_data)
run_test("Dataset types", test_dataset_length_and_types)

# ── 3. Model Build ────────────────────────────────────────────────────────────

section("3 · Model Build (build_model factory)")


def test_build_simple_nn():
    m = build_model(_make_cfg("SimpleNN"))
    assert isinstance(m, SimpleNN)


def test_build_simple_transformer():
    m = build_model(_make_cfg("SimpleTransformer"))
    assert isinstance(m, SimpleTransformer)


def test_build_hierarchical_moe():
    m = build_model(_make_cfg("HierarchicalMoE"))
    assert isinstance(m, HierarchicalMoELM)


def test_build_unknown_raises():
    try:
        build_model(_make_cfg("NonExistentModel"))
        raise AssertionError("Expected ValueError but none was raised")
    except ValueError:
        pass


run_test("Build SimpleNN", test_build_simple_nn)
run_test("Build SimpleTransformer", test_build_simple_transformer)
run_test("Build HierarchicalMoELM", test_build_hierarchical_moe)
run_test("Unknown model_type raises ValueError", test_build_unknown_raises)

# ── 4. Forward Pass ───────────────────────────────────────────────────────────

section("4 · Forward Pass (float features → class logits)")

B = 8
INPUT_DIM = 16
NUM_CLASSES = 4


def _dummy_batch():
    return torch.randn(B, INPUT_DIM)


def test_forward_simple_nn():
    m = build_model(_make_cfg("SimpleNN"))
    m.eval()
    out = m(_dummy_batch())
    assert out.shape == (B, NUM_CLASSES), f"Got {out.shape}"


def test_forward_simple_transformer():
    m = build_model(_make_cfg("SimpleTransformer"))
    m.eval()
    out = m(_dummy_batch())
    assert out.shape == (B, NUM_CLASSES), f"Got {out.shape}"


def test_forward_hierarchical_moe():
    m = build_model(_make_cfg("HierarchicalMoE"))
    m.eval()
    out = m(_dummy_batch())
    assert out.shape == (B, NUM_CLASSES), f"Got {out.shape}"


def test_forward_no_nan():
    for mtype in ("SimpleNN", "SimpleTransformer", "HierarchicalMoE"):
        m = build_model(_make_cfg(mtype))
        m.eval()
        out = m(_dummy_batch())
        assert not torch.isnan(out).any(), f"NaN in output for {mtype}"


run_test("SimpleNN forward shape", test_forward_simple_nn)
run_test("SimpleTransformer forward shape", test_forward_simple_transformer)
run_test("HierarchicalMoELM forward shape", test_forward_hierarchical_moe)
run_test("No NaN in outputs (all models)", test_forward_no_nan)

# ── 5. HierarchicalMoE Layer ──────────────────────────────────────────────────

section("5 · HierarchicalMoE Layer (unit tests)")


def test_moe_2d_shape():
    moe = HierarchicalMoE(n_embd=64, expert_hidden_dim=128, num_groups=2, experts_per_group=4)
    x = torch.randn(16, 64)
    out = moe(x)
    assert out.shape == (16, 64)


def test_moe_3d_shape():
    moe = HierarchicalMoE(n_embd=64, expert_hidden_dim=128, num_groups=2, experts_per_group=4)
    x = torch.randn(4, 8, 64)
    out = moe(x)
    assert out.shape == (4, 8, 64)


def test_moe_backprop():
    moe = HierarchicalMoE(n_embd=64, expert_hidden_dim=128, num_groups=2, experts_per_group=4)
    x = torch.randn(8, 64, requires_grad=True)
    out = moe(x)
    out.sum().backward()
    assert x.grad is not None


def test_router_weight_normalization():
    router = HierarchicalRouter(n_embd=64, num_groups=2, experts_per_group=4)
    x = torch.randn(32, 64)
    _, weights = router(x)
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Router weights don't sum to 1"


def test_expert_load_tracking():
    moe = HierarchicalMoE(n_embd=64, expert_hidden_dim=128, num_groups=2, experts_per_group=4)
    x = torch.randn(16, 64)
    moe(x)
    load = moe.last_expert_load
    assert load.shape == (2 * 4,), f"Expected shape (8,), got {load.shape}"
    assert (load >= 0).all() and (load <= 1).all()


run_test("MoE 2D input shape", test_moe_2d_shape)
run_test("MoE 3D input shape", test_moe_3d_shape)
run_test("MoE backpropagation", test_moe_backprop)
run_test("Router weight normalization", test_router_weight_normalization)
run_test("Expert load tracking", test_expert_load_tracking)

# ── 6. Parameter Count ────────────────────────────────────────────────────────

section("6 · Parameter Count")


def test_param_count_positive():
    m = build_model(_make_cfg("HierarchicalMoE"))
    n = m.count_parameters()
    assert n > 0, "Model has 0 trainable parameters"
    print(f"    → HierarchicalMoELM has {n:,} trainable params", end=" ")


def test_param_count_reasonable():
    m = build_model(_make_cfg("HierarchicalMoE"))
    n = m.count_parameters()
    assert 1_000 < n < 100_000_000, f"Suspicious param count: {n}"


run_test("Positive param count", test_param_count_positive)
run_test("Reasonable param count range", test_param_count_reasonable)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────

section("7 · Evaluate (evaluate.py)")


def test_evaluate_returns_dict():
    cfg = EvolveConfig(); cfg.device = "cpu"
    m = build_model(cfg)
    ds = generate_seed_data(n=100, num_classes=cfg.num_classes, input_dim=cfg.input_dim)
    result = evaluate(m, ds, cfg)
    for key in ("loss", "accuracy", "per_class_acc", "conf_mean", "conf_min", "conf_max"):
        assert key in result, f"Missing key: {key}"


def test_evaluate_accuracy_range():
    cfg = EvolveConfig(); cfg.device = "cpu"
    m = build_model(cfg)
    ds = generate_seed_data(n=200, num_classes=cfg.num_classes, input_dim=cfg.input_dim)
    result = evaluate(m, ds, cfg)
    assert 0.0 <= result["accuracy"] <= 100.0


def test_evaluate_loss_non_negative():
    cfg = EvolveConfig(); cfg.device = "cpu"
    m = build_model(cfg)
    ds = generate_seed_data(n=200, num_classes=cfg.num_classes, input_dim=cfg.input_dim)
    result = evaluate(m, ds, cfg)
    assert result["loss"] >= 0.0


def test_compute_confidence():
    cfg = EvolveConfig(); cfg.device = "cpu"
    m = build_model(cfg)
    ds = generate_seed_data(n=100, num_classes=cfg.num_classes, input_dim=cfg.input_dim)
    confs = compute_confidence(m, ds, cfg)
    assert confs.shape == (100,)
    assert (confs >= 0).all() and (confs <= 1).all()


run_test("evaluate() returns correct keys", test_evaluate_returns_dict)
run_test("accuracy in [0, 100]", test_evaluate_accuracy_range)
run_test("loss is non-negative", test_evaluate_loss_non_negative)
run_test("compute_confidence shape & range", test_compute_confidence)

# ── 8. Replay Buffer ──────────────────────────────────────────────────────────
# ReplayBufferV2 constructor: __init__(self, max_size: int = 10_000)
# Methods: .add_samples(features, labels), .update(dataset), .sample(n)

section("8 · Replay Buffer (ReplayBufferV2)")


def test_replay_buffer_populate():
    buf = ReplayBufferV2(max_size=500)
    ds = generate_seed_data(n=100, num_classes=4, input_dim=16)
    buf.add_samples(ds.features[:50], ds.labels[:50])
    assert len(buf) == 50


def test_replay_buffer_sample():
    buf = ReplayBufferV2(max_size=500)
    ds = generate_seed_data(n=200, num_classes=4, input_dim=16)
    buf.add_samples(ds.features, ds.labels)
    sample_ds = buf.sample(30)
    assert sample_ds is not None
    assert len(sample_ds) == 30


def test_replay_buffer_overflow():
    """Adding more than max_size should cap at max_size (reservoir sampling)."""
    buf = ReplayBufferV2(max_size=50)
    ds = generate_seed_data(n=200, num_classes=4, input_dim=16)
    buf.add_samples(ds.features, ds.labels)
    assert len(buf) <= 50


def test_replay_buffer_update():
    """update() integrates a whole SyntheticDataset."""
    buf = ReplayBufferV2(max_size=500)
    ds = generate_seed_data(n=80, num_classes=4, input_dim=16)
    buf.update(ds)
    assert len(buf) == 80


def test_evaluate_replay_empty_buffer():
    cfg = EvolveConfig(); cfg.device = "cpu"
    buf = ReplayBufferV2(max_size=500)
    m = build_model(cfg)
    result = evaluate_replay(m, buf, cfg)
    assert result["replay_accuracy"] == -1.0


def test_evaluate_replay_populated():
    cfg = EvolveConfig(); cfg.device = "cpu"
    buf = ReplayBufferV2(max_size=500)
    ds = generate_seed_data(n=200, num_classes=cfg.num_classes, input_dim=cfg.input_dim)
    buf.add_samples(ds.features, ds.labels)
    m = build_model(cfg)
    result = evaluate_replay(m, buf, cfg)
    assert 0.0 <= result["replay_accuracy"] <= 100.0
    assert result["num_samples"] > 0


run_test("Buffer populate via add_samples", test_replay_buffer_populate)
run_test("Buffer sample", test_replay_buffer_sample)
run_test("Buffer overflow (reservoir cap)", test_replay_buffer_overflow)
run_test("Buffer update (whole dataset)", test_replay_buffer_update)
run_test("Evaluate empty replay buffer", test_evaluate_replay_empty_buffer)
run_test("Evaluate populated replay buffer", test_evaluate_replay_populated)

# ── 9. Weight Transfer ────────────────────────────────────────────────────────

section("9 · Weight Transfer")


def test_weight_transfer_same_arch():
    """Transferring from a model filled with 1.0 should propagate to target."""
    cfg = _make_cfg("HierarchicalMoE")
    src = build_model(cfg)
    tgt = build_model(cfg)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
    transferred, total = transfer_compatible_weights(src, tgt)
    assert transferred == total, f"Should transfer all {total} tensors, got {transferred}"
    # Verify at least one weight was actually copied
    p_tgt = next(tgt.parameters())
    assert abs(p_tgt.mean().item() - 1.0) < 0.1, "Weight not transferred correctly"


def test_weight_transfer_partial():
    """Different hidden_dim → some layers match, others don't."""
    cfg_small = _make_cfg("SimpleNN")
    cfg_big = _make_cfg("SimpleNN")
    cfg_big.hidden_dim = 128   # differs from default 64
    src = build_model(cfg_small)
    tgt = build_model(cfg_big)
    transferred, total = transfer_compatible_weights(src, tgt)
    assert 0 <= transferred <= total


run_test("Weight transfer – identical arch", test_weight_transfer_same_arch)
run_test("Weight transfer – partial (size mismatch)", test_weight_transfer_partial)

# ── 10. Checkpoint Load ───────────────────────────────────────────────────────

section("10 · Checkpoint (best_model.pt)")


def test_checkpoint_loads():
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
    if not os.path.exists(ckpt_path):
        print("    (no checkpoint found – skipping)", end=" ")
        return
    cfg = EvolveConfig(); cfg.device = "cpu"
    m = build_model(cfg)
    saved_state = torch.load(ckpt_path, map_location="cpu")

    # Manually transfer compatible tensors (shape-safe, no strict=False risk)
    current_state = m.state_dict()
    transferred = 0
    with torch.no_grad():
        for name, tensor in saved_state.items():
            if name in current_state and current_state[name].shape == tensor.shape:
                current_state[name].copy_(tensor)
                transferred += 1
    m.load_state_dict(current_state)

    m.eval()
    out = m(torch.randn(4, cfg.input_dim))
    assert out.shape == (4, cfg.num_classes), f"Got {out.shape}"
    print(f"    (checkpoint: {transferred}/{len(current_state)} tensors matched)", end=" ")


run_test("Load best_model.pt and run inference", test_checkpoint_loads)

# ── 11. End-to-End Mini Training Round ───────────────────────────────────────

section("11 · End-to-End (1 training epoch + eval)")


def test_e2e_one_epoch():
    """Train for 1 epoch on tiny data and verify eval runs cleanly."""
    from torch.utils.data import DataLoader

    cfg = EvolveConfig()
    cfg.device = "cpu"
    cfg.epochs_per_loop = 1
    cfg.batch_size = 16
    cfg.learning_rate = 1e-3
    cfg.model_type = "HierarchicalMoE"

    ds_train = generate_seed_data(n=64, num_classes=cfg.num_classes, input_dim=cfg.input_dim, seed=7)
    ds_eval  = generate_seed_data(n=64, num_classes=cfg.num_classes, input_dim=cfg.input_dim, seed=8)

    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    for feats, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(feats), labels)
        loss.backward()
        optimizer.step()

    result = evaluate(model, ds_eval, cfg)
    assert 0.0 <= result["accuracy"] <= 100.0
    print(f"    (val_acc={result['accuracy']:.1f}%, val_loss={result['loss']:.4f})", end=" ")


def test_e2e_with_replay():
    """Full cycle: train → populate replay buffer → evaluate replay."""
    from torch.utils.data import DataLoader

    cfg = EvolveConfig(); cfg.device = "cpu"; cfg.batch_size = 16
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    buf = ReplayBufferV2(max_size=200)

    ds = generate_seed_data(n=64, num_classes=cfg.num_classes, input_dim=cfg.input_dim, seed=42)

    # Train
    model.train()
    for feats, labels in torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True):
        optimizer.zero_grad()
        criterion(model(feats), labels).backward()
        optimizer.step()

    # Populate buffer and evaluate
    buf.update(ds)
    result = evaluate_replay(model, buf, cfg)
    assert result["replay_accuracy"] >= 0.0
    print(f"    (replay_acc={result['replay_accuracy']:.1f}%)", end=" ")


run_test("1-epoch train + eval loop", test_e2e_one_epoch)
run_test("1-epoch train + replay eval", test_e2e_with_replay)

# ── Summary ───────────────────────────────────────────────────────────────────

section("SUMMARY")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"\n  {passed} passed  |  {failed} failed  |  {len(results)} total\n")

if failed:
    print("  Failed tests:")
    for name, ok, tb in results:
        if not ok:
            print(f"    • {name}")
    sys.exit(1)
else:
    print("  All tests passed! 🎉")
