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
    model_type: str = "SimpleNN"     # "SimpleNN", "SimpleTransformer", "TransformerLM"
    hidden_dim: int = 64             # Hidden layer size for SimpleNN / d_model for Transformers
    num_heads: int = 4               # Attention heads for SimpleTransformer/TransformerLM
    num_layers: int = 2              # Transformer encoder layers
    vocab_size: int = 50257          # Vocabulary size for tokenizer (TransformerLM)
    max_seq_len: int = 128           # Maximum sequence length (TransformerLM)

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
    # Replay buffer                                                         #
    # ------------------------------------------------------------------ #
    replay_buffer_size: int = 150    # Max samples stored in replay buffer
    replay_mix_ratio: float = 0.15   # Fraction of each training batch from replay buffer



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
    # Pipelined Loop                                                        #
    # ------------------------------------------------------------------ #
    num_datasets_in_buffer: int = 3
    parallel_fetch: bool = True
    groq_dataset_size: int = 60      # Smaller batch size to save tokens / time

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
