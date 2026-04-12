"""
multimodal_wrappers.py
======================

This file contains five modality-specific wrapper classes that all share the
same core engine:  TransformerBlock + HierarchicalMoE from hierarchical_moe.py

The MoE block only processes numeric tensors of shape (Batch, Sequence, D_model).
Each wrapper below is responsible for converting its raw domain input into that
shape (the "embedding" step) and for converting the MoE output back into
something useful for that domain (the "head" step).

Wrappers included:
  1. TextGenerationMoE        – autoregressive next-token language model
  2. VisionTransformerMoE     – image patch classification / feature extraction
  3. AudioMoE                 – raw waveform or spectrogram understanding
  4. ClassificationMoE        – sequence / document / image classification
  5. RLActorCriticMoE         – Actor-Critic agent for Reinforcement Learning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hierarchical_moe import TransformerBlock


# ──────────────────────────────────────────────────────────────────────────────
#  Shared helper: build a stack of TransformerBlocks
# ──────────────────────────────────────────────────────────────────────────────

def _build_transformer_stack(
    num_layers: int,
    d_model: int,
    d_ff: int,
    num_groups: int,
    experts_per_group: int,
    num_heads: int,
    dropout: float,
    top_k: int,
) -> nn.ModuleList:
    """Returns a ModuleList of `num_layers` TransformerBlocks."""
    return nn.ModuleList([
        TransformerBlock(
            d_model=d_model,
            d_ff=d_ff,
            num_groups=num_groups,
            experts_per_group=experts_per_group,
            num_heads=num_heads,
            dropout=dropout,
            router_top_k=top_k,
        )
        for _ in range(num_layers)
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  1.  TEXT GENERATION
#      Input : (B, S)  integer token ids
#      Output: (B, S, vocab_size)  next-token logits
# ══════════════════════════════════════════════════════════════════════════════

class TextGenerationMoE(nn.Module):
    """
    Autoregressive Language Model using the Hierarchical MoE Transformer.

    Architecture:
        Token Embedding + Learned Positional Embedding
        → N × TransformerBlock (causal attention + HierarchicalMoE)
        → LayerNorm
        → Linear head  (d_model → vocab_size)

    Usage:
        model = TextGenerationMoE()
        input_ids = torch.randint(0, 50257, (2, 128))   # (B, S)
        logits = model(input_ids)                        # (B, S, 50257)

        # For training, pair with nn.CrossEntropyLoss:
        #   loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        # For generation (greedy):
        #   next_token = logits[:, -1, :].argmax(-1)
    """

    def __init__(
        self,
        vocab_size: int = 50257,    # GPT-2 vocabulary size by default
        d_model: int = 256,
        d_ff: int = 1024,
        num_groups: int = 4,
        experts_per_group: int = 4,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        top_k: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ── Embedding layers ──────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop       = nn.Dropout(dropout)

        # ── Transformer stack ─────────────────────────────────────────────────
        self.layers = _build_transformer_stack(
            num_layers, d_model, d_ff, num_groups, experts_per_group,
            num_heads, dropout, top_k
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between token embedding and output projection (like GPT-2)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) long tensor of token ids
        Returns:
            logits: (B, S, vocab_size)
        """
        B, S = input_ids.shape
        assert S <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)  # (1, S)
        h = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        # Causal mask so each position can only attend to previous positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S).to(input_ids.device)

        for layer in self.layers:
            h = layer(h, causal_mask=causal_mask)

        return self.head(self.ln_f(h))          # (B, S, vocab_size)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """
        Greedy / temperature-sampled autoregressive generation.

        Args:
            prompt_ids     : (1, S) starting token ids
            max_new_tokens : how many tokens to generate
            temperature    : > 1 → more random,  < 1 → more greedy
        Returns:
            (1, S + max_new_tokens) token ids
        """
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            context = ids[:, -self.max_seq_len:]
            logits  = self(context)                    # (1, S, vocab)
            logits  = logits[:, -1, :] / temperature   # (1, vocab)
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids     = torch.cat([ids, next_id], dim=1)
        return ids


# ══════════════════════════════════════════════════════════════════════════════
#  2.  COMPUTER VISION  (Vision Transformer – ViT style)
#      Input : (B, C, H, W)  image tensors  (e.g. C=3, H=W=224)
#      Output: (B, num_classes)  class logits
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and projects each patch
    linearly into a d_model-dimensional vector (exactly as in ViT).
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, d_model: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        # A single Conv2d with kernel = stride = patch_size acts as a patch projector
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, d_model)
        """
        x = self.proj(x)            # (B, d_model, H/P, W/P)
        x = x.flatten(2)            # (B, d_model, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, d_model)
        return x


class VisionTransformerMoE(nn.Module):
    """
    Vision Transformer (ViT) that uses HierarchicalMoE instead of standard FFN.

    Architecture:
        Image → Patch Embedding
        → Prepend [CLS] token
        → Add Learned Positional Embedding
        → N × TransformerBlock (bidirectional attention + HierarchicalMoE)
        → Take [CLS] output → LayerNorm → Linear classifier head

    Usage:
        model = VisionTransformerMoE(num_classes=1000)
        images = torch.randn(4, 3, 224, 224)    # (B, C, H, W)
        logits = model(images)                   # (B, 1000)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        d_model: int = 256,
        d_ff: int = 1024,
        num_groups: int = 4,
        experts_per_group: int = 4,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        top_k: int = 2,
    ):
        super().__init__()

        # ── Patch embedding ────────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches

        # ── [CLS] token & positional embedding ───────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.drop       = nn.Dropout(dropout)

        # ── Transformer stack (NO causal mask – bidirectional) ────────────────
        self.layers = _build_transformer_stack(
            num_layers, d_model, d_ff, num_groups, experts_per_group,
            num_heads, dropout, top_k
        )

        # ── Classification head ───────────────────────────────────────────────
        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, num_classes)

        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,   std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor,  pixel values in [0, 1]
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]

        # Patch embedding:  (B, num_patches, d_model)
        x = self.patch_embed(x)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)          # (B, num_patches+1, d_model)

        # Add positional embedding
        x = self.drop(x + self.pos_emb)

        # Bidirectional transformer (no causal mask)
        for layer in self.layers:
            x = layer(x, causal_mask=None)

        # Use [CLS] output for classification
        cls_out = self.ln_f(x[:, 0, :])           # (B, d_model)
        return self.head(cls_out)                   # (B, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  AUDIO PROCESSING
#      Input : (B, 1, samples)  raw waveform  OR  (B, n_mels, T)  spectrogram
#      Output: (B, num_classes)  class logits  (e.g. speech command recognition)
# ══════════════════════════════════════════════════════════════════════════════

class AudioMoE(nn.Module):
    """
    Audio model using the HierarchicalMoE Transformer.

    Works with either:
      (a) Raw waveform:  (B, 1, T_samples) — set use_spectrogram=False
      (b) Pre-computed log-mel spectrogram frames:  (B, n_mels, T_frames)

    A 1-D convolutional stem "tokenizes" the audio into frame-level embeddings,
    then the MoE Transformer models temporal dependencies.

    Usage (raw waveform):
        model = AudioMoE(in_channels=1, num_classes=35, use_spectrogram=False)
        waveform = torch.randn(4, 1, 16000)     # 1 s at 16 kHz
        logits   = model(waveform)              # (4, 35)

    Usage (log-mel spectrogram):
        model = AudioMoE(in_channels=80, num_classes=35, use_spectrogram=True)
        spec  = torch.randn(4, 80, 128)         # (B, n_mels, time)
        logits = model(spec)                    # (4, 35)
    """

    def __init__(
        self,
        in_channels: int = 1,       # 1 for raw waveform; n_mels for spectrogram
        num_classes: int = 35,
        d_model: int = 256,
        d_ff: int = 1024,
        num_groups: int = 4,
        experts_per_group: int = 4,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        top_k: int = 2,
        use_spectrogram: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # ── Audio frontend (1-D CNN stem) ─────────────────────────────────────
        if use_spectrogram:
            # Treat frequency axis as channels; kernel slides over time
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
            )
        else:
            # Downsample raw waveform aggressively before the Transformer
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, d_model // 4, kernel_size=10, stride=5, padding=4),
                nn.GELU(),
                nn.Conv1d(d_model // 4, d_model // 2, kernel_size=8, stride=4, padding=3),
                nn.GELU(),
                nn.Conv1d(d_model // 2, d_model, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
            )

        # ── Positional embedding (after CNN output) ───────────────────────────
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop     = nn.Dropout(dropout)

        # ── Transformer stack (bidirectional – no future masking needed) ──────
        self.layers = _build_transformer_stack(
            num_layers, d_model, d_ff, num_groups, experts_per_group,
            num_heads, dropout, top_k
        )

        # ── Classification head (mean-pool over time) ─────────────────────────
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, channels, time)
        Returns:
            logits: (B, num_classes)
        """
        # CNN stem: (B, channels, T) → (B, d_model, T')
        x = self.stem(x)

        # Rearrange to sequence: (B, T', d_model)
        x = x.transpose(1, 2)

        # Truncate if needed
        S = min(x.size(1), self.max_seq_len)
        x = x[:, :S, :]

        pos = torch.arange(S, device=x.device).unsqueeze(0)
        x   = self.drop(x + self.pos_emb(pos))

        for layer in self.layers:
            x = layer(x, causal_mask=None)

        # Mean-pool over time for a fixed-size representation
        x = self.ln_f(x.mean(dim=1))               # (B, d_model)
        return self.head(x)                          # (B, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  CLASSIFICATION (general-purpose: text, tabular, or any sequence)
#      Input : (B, S, feature_dim)  pre-projected feature vectors
#      Output: (B, num_classes)  class logits
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationMoE(nn.Module):
    """
    General-purpose sequence classifier using HierarchicalMoE.

    Use this wrapper when your data is already a sequence of feature vectors
    (e.g., sentence embeddings, tabular rows, time-series steps).

    If your data is raw text, tokenise it first and pass the token ids through
    an embedding layer before this model.

    Architecture:
        Input projection  (feature_dim → d_model)
        → Learned Positional Embedding
        → N × TransformerBlock (bidirectional + HierarchicalMoE)
        → [CLS] pooling  OR  mean-pool
        → LayerNorm → Dropout → Linear classifier

    Usage:
        model = ClassificationMoE(feature_dim=768, num_classes=10)
        features = torch.randn(8, 64, 768)    # (B, S, feature_dim)
        logits   = model(features)            # (8, 10)
    """

    def __init__(
        self,
        feature_dim: int = 768,     # dimension of each input vector
        num_classes: int = 10,
        d_model: int = 256,
        d_ff: int = 1024,
        num_groups: int = 4,
        experts_per_group: int = 4,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        top_k: int = 2,
        pool: str = "cls",          # "cls" or "mean"
    ):
        super().__init__()
        assert pool in ("cls", "mean"), "pool must be 'cls' or 'mean'"
        self.pool        = pool
        self.max_seq_len = max_seq_len

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Linear(feature_dim, d_model)

        # ── [CLS] token & positional embedding ───────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Embedding(max_seq_len + 1, d_model)
        self.drop       = nn.Dropout(dropout)

        # ── Transformer stack (bidirectional) ─────────────────────────────────
        self.layers = _build_transformer_stack(
            num_layers, d_model, d_ff, num_groups, experts_per_group,
            num_heads, dropout, top_k
        )

        # ── Classifier head ───────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)
        self.head  = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, feature_dim) – arbitrary feature vectors
        Returns:
            logits: (B, num_classes)
        """
        B, S, _ = x.shape
        S = min(S, self.max_seq_len)
        x = x[:, :S, :]

        # Project to d_model
        x = self.input_proj(x)                        # (B, S, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)              # (B, S+1, d_model)

        pos = torch.arange(S + 1, device=x.device).unsqueeze(0)
        x   = self.drop(x + self.pos_emb(pos))

        for layer in self.layers:
            x = layer(x, causal_mask=None)

        if self.pool == "cls":
            rep = x[:, 0, :]                          # [CLS] token  (B, d_model)
        else:
            rep = x[:, 1:, :].mean(dim=1)             # mean over sequence (B, d_model)

        rep = self.drop2(self.ln_f(rep))
        return self.head(rep)                          # (B, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
#  5.  REINFORCEMENT LEARNING  –  Actor-Critic
#      Input : (B, S, state_dim)  observation / state history
#      Output: policy logits (B, action_dim)  &  value estimate (B, 1)
# ══════════════════════════════════════════════════════════════════════════════

class RLActorCriticMoE(nn.Module):
    """
    Transformer-based Actor-Critic agent using HierarchicalMoE.

    The agent receives a history of state observations and outputs:
      - action_logits  : raw scores for each discrete action  (for policy)
      - value          : estimated expected return  (for critic / baseline)

    Works with both single-step and multi-step (recurrent-style) environments.

    Architecture:
        State projection  (state_dim → d_model)
        → Learned Positional Embedding
        → N × TransformerBlock (causal – agent should NOT see future states)
        → LayerNorm
        ┌─→ Actor head   Linear(d_model → action_dim)  → action logits
        └─→ Critic head  Linear(d_model → 1)           → state value

    Usage (single-step gym environment):
        model  = RLActorCriticMoE(state_dim=8, action_dim=4)
        obs    = torch.randn(1, 1, 8)              # (B=1, S=1, state_dim)
        policy_logits, value = model(obs)
        action = torch.distributions.Categorical(logits=policy_logits).sample()

    Usage (multi-step trajectory):
        model  = RLActorCriticMoE(state_dim=8, action_dim=4)
        traj   = torch.randn(4, 16, 8)             # (B=4, S=16, state_dim)
        policy_logits, values = model(traj)        # (4,16,4) and (4,16,1)
        # Use last step for action, all steps for value loss
    """

    def __init__(
        self,
        state_dim: int = 8,         # dimensionality of a single observation
        action_dim: int = 4,        # number of discrete actions
        d_model: int = 128,
        d_ff: int = 512,
        num_groups: int = 2,
        experts_per_group: int = 4,
        num_heads: int = 4,
        num_layers: int = 3,
        max_history: int = 128,     # maximum number of past steps to remember
        dropout: float = 0.0,       # usually 0 for RL to avoid stochasticity
        top_k: int = 2,
    ):
        super().__init__()
        self.max_history = max_history

        # ── State projection ──────────────────────────────────────────────────
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
        )

        # ── Positional embedding ──────────────────────────────────────────────
        self.pos_emb = nn.Embedding(max_history, d_model)
        self.drop     = nn.Dropout(dropout)

        # ── Transformer stack (CAUSAL – agent sees only current + past states) ─
        self.layers = _build_transformer_stack(
            num_layers, d_model, d_ff, num_groups, experts_per_group,
            num_heads, dropout, top_k
        )

        self.ln_f = nn.LayerNorm(d_model)

        # ── Actor head (policy) ───────────────────────────────────────────────
        self.actor_head  = nn.Linear(d_model, action_dim)

        # ── Critic head (value function) ──────────────────────────────────────
        self.critic_head = nn.Linear(d_model, 1)

    def forward(self, states: torch.Tensor):
        """
        Args:
            states: (B, S, state_dim)  –  history of observations
        Returns:
            action_logits: (B, S, action_dim)  –  policy output for each step
            values:        (B, S, 1)           –  critic estimate for each step
        """
        B, S, _ = states.shape
        S = min(S, self.max_history)
        states = states[:, :S, :]

        # Project states to d_model
        h = self.state_proj(states)                    # (B, S, d_model)

        pos = torch.arange(S, device=states.device).unsqueeze(0)
        h   = self.drop(h + self.pos_emb(pos))

        # Causal mask – agent cannot look into the future
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S).to(states.device)

        for layer in self.layers:
            h = layer(h, causal_mask=causal_mask)

        h = self.ln_f(h)                               # (B, S, d_model)

        action_logits = self.actor_head(h)             # (B, S, action_dim)
        values        = self.critic_head(h)            # (B, S, 1)

        return action_logits, values

    def select_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Convenience method for single-step inference during rollout.

        Args:
            obs           : (1, S, state_dim)  current observation history
            deterministic : if True, pick argmax; otherwise sample
        Returns:
            action (int), log_prob (scalar tensor), value (scalar tensor)
        """
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(obs)        # (1, S, action_dim), (1, S, 1)
            logits = logits[:, -1, :]                # take last step  (1, action_dim)
            value  = value[:, -1, 0]                 # take last step  (1,)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
            return action.item(), dist.log_prob(action), value


# ────────────────────────────────────────────────────────────────────────────
#  Quick sanity test  (run this file directly: python multimodal_wrappers.py)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Multimodal MoE Wrapper – Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # 1 — Text Generation
    print("[1] TextGenerationMoE")
    m = TextGenerationMoE(vocab_size=50257, d_model=128, num_layers=2).to(device)
    ids = torch.randint(0, 50257, (2, 32)).to(device)
    out = m(ids)
    print(f"    Input : {tuple(ids.shape)}  →  Output logits: {tuple(out.shape)}")
    gen = m.generate(ids[:1, :5], max_new_tokens=10)
    print(f"    Generated ids shape: {tuple(gen.shape)}\n")

    # 2 — Vision
    print("[2] VisionTransformerMoE")
    m = VisionTransformerMoE(img_size=64, patch_size=8, num_classes=10, d_model=128, num_layers=2).to(device)
    imgs = torch.randn(4, 3, 64, 64).to(device)
    out  = m(imgs)
    print(f"    Input : {tuple(imgs.shape)}  →  Output logits: {tuple(out.shape)}\n")

    # 3 — Audio
    print("[3] AudioMoE (raw waveform)")
    m = AudioMoE(in_channels=1, num_classes=35, d_model=128, num_layers=2).to(device)
    wav = torch.randn(4, 1, 16000).to(device)
    out = m(wav)
    print(f"    Input : {tuple(wav.shape)}  →  Output logits: {tuple(out.shape)}\n")

    # 4 — Classification
    print("[4] ClassificationMoE")
    m = ClassificationMoE(feature_dim=64, num_classes=5, d_model=128, num_layers=2).to(device)
    feats = torch.randn(8, 20, 64).to(device)
    out   = m(feats)
    print(f"    Input : {tuple(feats.shape)}  →  Output logits: {tuple(out.shape)}\n")

    # 5 — RL Actor-Critic
    print("[5] RLActorCriticMoE")
    m = RLActorCriticMoE(state_dim=8, action_dim=4, d_model=64, num_layers=2).to(device)
    states = torch.randn(2, 16, 8).to(device)
    logits, values = m(states)
    print(f"    States: {tuple(states.shape)}  →  Policy: {tuple(logits.shape)},  Values: {tuple(values.shape)}")
    action, lp, v = m.select_action(states[:1])
    print(f"    Single action: {action},  log_prob: {lp.item():.4f},  value: {v.item():.4f}\n")

    print("=" * 60)
    print("  All wrappers passed sanity check!")
    print("=" * 60)
