"""
config.py – Central configuration for the Operation Evolve system.

All hyperparameters, paths, and safety thresholds live here.
Change values here to tune the self-evolving pipeline without
touching any other module.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvolveConfig:
    # ------------------------------------------------------------------ #
    # Dataset paths                                                        #
    # ------------------------------------------------------------------ #
    data_dir: str = "data"           # All versioned .pt files are stored here

    # ------------------------------------------------------------------ #
    # Dataset generation (seed data)                                       #
    # ------------------------------------------------------------------ #
    num_classes: int = 4             # Number of output classes
    input_dim: int = 16              # Feature vector dimension
    dataset_a_size: int = 300        # Samples in Dataset_A (core / read-only)
    dataset_b_initial_size: int = 500  # Initial samples in Dataset_B

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model_type: str = "SimpleNN"     # "SimpleNN" or "SimpleTransformer"
    hidden_dim: int = 64             # Hidden layer size for SimpleNN
    num_heads: int = 4               # Attention heads for SimpleTransformer
    num_layers: int = 2              # Transformer encoder layers

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    epochs_per_loop: int = 5         # Epochs trained per evolution loop
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Early stopping
    early_stopping_patience: int = 3  # Stop if val-loss doesn't improve for N epochs

    # ------------------------------------------------------------------ #
    # Evolution loop                                                        #
    # ------------------------------------------------------------------ #
    num_evolution_loops: int = 5     # How many self-evolution cycles to run

    # ------------------------------------------------------------------ #
    # Dataset_C generation                                                  #
    # ------------------------------------------------------------------ #
    generate_c_size: int = 200       # Samples to generate per loop

    # ------------------------------------------------------------------ #
    # Confidence filtering                                                  #
    # ------------------------------------------------------------------ #
    confidence_threshold: float = 0.85   # Min softmax confidence to accept
    diversity_threshold: float = 0.02    # L2 distance below which samples are near-duplicates

    # ------------------------------------------------------------------ #
    # Dataset mixing (anti-drift)                                           #
    # "new_B = keep_ratio * old_B + (1-keep_ratio) * filtered_C"           #
    # ------------------------------------------------------------------ #
    dataset_keep_ratio: float = 0.7  # Fraction of old Dataset_B to retain
    # (1 - dataset_keep_ratio) fraction comes from filtered Dataset_C

    # ------------------------------------------------------------------ #
    # Replay buffer                                                         #
    # ------------------------------------------------------------------ #
    replay_buffer_size: int = 150    # Max samples stored in replay buffer
    replay_mix_ratio: float = 0.15   # Fraction of each training batch from replay buffer

    # ------------------------------------------------------------------ #
    # Rollback safety                                                       #
    # ------------------------------------------------------------------ #
    rollback_tolerance: float = 0.5  # Accept new model if acc >= best - tolerance (%)

    # ------------------------------------------------------------------ #
    # Logging                                                               #
    # ------------------------------------------------------------------ #
    log_level: str = "INFO"          # "DEBUG" | "INFO" | "WARNING"
    print_every_n_epochs: int = 1    # How often to print epoch-level logs

    # ------------------------------------------------------------------ #
    # LLM Agent                                                             #
    # ------------------------------------------------------------------ #
    use_llm_agent: bool = True
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model_name: str = "llama-3.3-70b-versatile"

    # ------------------------------------------------------------------ #
    # Device                                                                #
    # ------------------------------------------------------------------ #
    device: str = "auto"             # "auto" | "cpu" | "cuda"

    # ------------------------------------------------------------------ #
    # Derived helpers (not to be overridden)                                #
    # ------------------------------------------------------------------ #
    def get_device(self):
        """Return the correct torch device."""
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def ensure_data_dir(self):
        """Create data directory if it doesn't exist."""
        os.makedirs(self.data_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Default singleton – import and use directly
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = EvolveConfig()
