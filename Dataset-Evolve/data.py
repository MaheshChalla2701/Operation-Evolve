"""
data.py – Dataset management for the Operation Evolve system.

Responsibilities:
  - Define SyntheticDataset (wraps feature/label tensors)
  - Generate seed data for Dataset_A and Dataset_B
  - generate_dataset_c() – run model forward pass to create candidates
  - filter_by_confidence() – apply threshold + diversity check
  - merge_datasets() – mix old Dataset_B with filtered Dataset_C
  - save/load versioned .pt files
  - Manage the replay buffer
"""

import os
import math
import logging
from typing import Optional, Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader

from config import EvolveConfig

logger = logging.getLogger("evolve.data")


# ---------------------------------------------------------------------------
# Core dataset wrapper
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Wraps (features, labels) tensors as a PyTorch Dataset."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0], "features/labels size mismatch"
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def __repr__(self) -> str:
        return f"SyntheticDataset(n={len(self)}, dim={self.features.shape[1]}, classes={self.labels.unique().tolist()})"


# ---------------------------------------------------------------------------
# Seed data generation
# ---------------------------------------------------------------------------

def generate_seed_data(
    n: int,
    num_classes: int,
    input_dim: int,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """
    Create a synthetic classification dataset.

    Each class is a Gaussian cluster centred at a distinct point in R^input_dim.
    """
    if seed is not None:
        torch.manual_seed(seed)

    features_list, labels_list = [], []
    samples_per_class = n // num_classes

    for cls in range(num_classes):
        # Unique cluster centre per class
        centre = torch.zeros(input_dim)
        centre[cls % input_dim] = float(cls + 1) * 2.0
        noise = torch.randn(samples_per_class, input_dim) * noise_std
        feats = centre.unsqueeze(0).expand(samples_per_class, -1) + noise
        labels = torch.full((samples_per_class,), cls, dtype=torch.long)
        features_list.append(feats)
        labels_list.append(labels)

    features = torch.cat(features_list)
    labels = torch.cat(labels_list)

    # Shuffle
    perm = torch.randperm(features.shape[0])
    return SyntheticDataset(features[perm], labels[perm])


# ---------------------------------------------------------------------------
# Dataset_C generation  (MANDATORY per requirements)
# ---------------------------------------------------------------------------

def generate_dataset_c(
    model: torch.nn.Module,
    dataset_b: SyntheticDataset,
    config: EvolveConfig,
) -> Dict[str, torch.Tensor]:
    """
    Run Dataset_B inputs through the model to generate candidate samples.

    Returns a dict with:
        "features"     : (N, D) tensor
        "pred_labels"  : (N,)   predicted class indices
        "confidences"  : (N,)   max softmax probability per sample
    """
    device = config.get_device()
    model.eval()
    loader = DataLoader(dataset_b, batch_size=config.batch_size, shuffle=False)

    all_features: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_confs: List[torch.Tensor] = []

    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_features.append(features.cpu())
            all_preds.append(preds.cpu())
            all_confs.append(confs.cpu())

    dataset_c = {
        "features": torch.cat(all_features),
        "pred_labels": torch.cat(all_preds),
        "confidences": torch.cat(all_confs),
    }

    logger.info(
        f"[Dataset_C] Generated {dataset_c['features'].shape[0]} candidate samples "
        f"| mean_conf={dataset_c['confidences'].mean():.3f} "
        f"| min={dataset_c['confidences'].min():.3f} "
        f"| max={dataset_c['confidences'].max():.3f}"
    )
    return dataset_c


# ---------------------------------------------------------------------------
# Confidence + diversity filtering
# ---------------------------------------------------------------------------

def filter_by_confidence(
    dataset_c: Dict[str, torch.Tensor],
    config: EvolveConfig,
    existing_features: Optional[torch.Tensor] = None,
) -> Tuple[SyntheticDataset, int, int]:
    """
    Filter Dataset_C candidate samples.

    Criteria:
      1. confidence >= config.confidence_threshold
      2. (optional) L2 distance from existing samples > config.diversity_threshold
         to avoid near-duplicate injection.

    Returns:
        filtered_dataset  : SyntheticDataset of accepted samples
        accepted_count    : int
        rejected_count    : int
    """
    features = dataset_c["features"]
    pred_labels = dataset_c["pred_labels"]
    confidences = dataset_c["confidences"]

    total = features.shape[0]

    # --- Step 1: confidence threshold ---
    conf_mask = confidences >= config.confidence_threshold
    accepted_features = features[conf_mask]
    accepted_labels = pred_labels[conf_mask]
    accepted_confs = confidences[conf_mask]

    # --- Step 2: diversity check ---
    if existing_features is not None and accepted_features.shape[0] > 0:
        keep_indices = _diversity_filter(
            accepted_features, existing_features, config.diversity_threshold
        )
        accepted_features = accepted_features[keep_indices]
        accepted_labels = accepted_labels[keep_indices]

    accepted = accepted_features.shape[0]
    rejected = total - accepted

    logger.info(
        f"[Filter] threshold={config.confidence_threshold:.2f} | "
        f"accepted={accepted} | rejected={rejected}"
    )

    if accepted == 0:
        # Return an empty dataset (won't break anything downstream)
        empty_feats = torch.zeros(0, features.shape[1])
        empty_labels = torch.zeros(0, dtype=torch.long)
        return SyntheticDataset(empty_feats, empty_labels), 0, rejected

    return SyntheticDataset(accepted_features, accepted_labels), accepted, rejected


def _diversity_filter(
    candidates: torch.Tensor,
    existing: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Return indices of candidates that are NOT near-duplicates of any existing sample.

    Uses pairwise L2 distance; O(N*M) – acceptable for small datasets.
    """
    keep = []
    for i in range(candidates.shape[0]):
        diffs = existing - candidates[i].unsqueeze(0)          # (M, D)
        dists = diffs.norm(dim=-1)                              # (M,)
        if dists.min().item() > threshold:
            keep.append(i)
    if len(keep) == 0:
        return torch.tensor([], dtype=torch.long)
    return torch.tensor(keep, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset mixing  (anti-drift)
# ---------------------------------------------------------------------------

def mix_datasets(
    dataset_b: SyntheticDataset,
    filtered_c: SyntheticDataset,
    config: EvolveConfig,
) -> SyntheticDataset:
    """
    new_B = keep_ratio * old_B  +  (1 - keep_ratio) * filtered_C

    Rows of old_B are randomly sub-sampled; filtered_C samples are appended.
    """
    keep_n = math.ceil(len(dataset_b) * config.dataset_keep_ratio)
    perm = torch.randperm(len(dataset_b))[:keep_n]

    kept_features = dataset_b.features[perm]
    kept_labels = dataset_b.labels[perm]

    if len(filtered_c) > 0:
        # How many from C we want
        c_quota = math.ceil(
            len(dataset_b) * (1.0 - config.dataset_keep_ratio)
        )
        c_perm = torch.randperm(len(filtered_c))[:c_quota]
        new_features = torch.cat([kept_features, filtered_c.features[c_perm]])
        new_labels = torch.cat([kept_labels, filtered_c.labels[c_perm]])
    else:
        new_features = kept_features
        new_labels = kept_labels

    logger.info(
        f"[Mix] old_B contributed {keep_n} | C contributed {len(new_features) - keep_n} "
        f"| new_B size = {len(new_features)}"
    )
    return SyntheticDataset(new_features, new_labels)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Stores a rolling window of high-quality historical samples.

    Prevents catastrophic forgetting by mixing past data into training.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.features: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

    def add(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Add samples; evict oldest if buffer overflows."""
        self.features.append(features)
        self.labels.append(labels)
        # Merge and trim
        all_f = torch.cat(self.features)
        all_l = torch.cat(self.labels)
        if all_f.shape[0] > self.max_size:
            all_f = all_f[-self.max_size:]
            all_l = all_l[-self.max_size:]
        self.features = [all_f]
        self.labels = [all_l]

    def sample(self, n: int) -> Optional[SyntheticDataset]:
        """Random sample of n items (or all if buffer < n)."""
        if not self.features:
            return None
        all_f = torch.cat(self.features)
        all_l = torch.cat(self.labels)
        n = min(n, all_f.shape[0])
        idx = torch.randperm(all_f.shape[0])[:n]
        return SyntheticDataset(all_f[idx], all_l[idx])

    def __len__(self) -> int:
        if not self.features:
            return 0
        return torch.cat(self.features).shape[0]

    def populate_from(self, dataset: SyntheticDataset, n: int) -> None:
        """Seed the buffer with high-confidence samples from an existing dataset."""
        n = min(n, len(dataset))
        idx = torch.randperm(len(dataset))[:n]
        self.add(dataset.features[idx], dataset.labels[idx])


# ---------------------------------------------------------------------------
# Save / load versioned datasets
# ---------------------------------------------------------------------------

def save_dataset_version(
    dataset: SyntheticDataset,
    name: str,
    version: int,
    data_dir: str,
) -> str:
    """Save dataset to  <data_dir>/<name>_v<version>.pt"""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{name}_v{version}.pt")
    torch.save({"features": dataset.features, "labels": dataset.labels}, path)
    logger.info(f"[Version] Saved {path}  ({len(dataset)} samples)")
    return path


def load_dataset_version(
    name: str,
    version: int,
    data_dir: str,
) -> SyntheticDataset:
    """Load a versioned dataset from disk."""
    path = os.path.join(data_dir, f"{name}_v{version}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset version not found: {path}")
    data = torch.load(path, weights_only=True)
    return SyntheticDataset(data["features"], data["labels"])


def save_dataset(dataset: SyntheticDataset, name: str, data_dir: str) -> str:
    """Save a non-versioned dataset (e.g. dataset_A)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{name}.pt")
    torch.save({"features": dataset.features, "labels": dataset.labels}, path)
    logger.info(f"[Data] Saved {path}  ({len(dataset)} samples)")
    return path


def load_dataset(name: str, data_dir: str) -> SyntheticDataset:
    """Load a non-versioned dataset from disk."""
    path = os.path.join(data_dir, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = torch.load(path, weights_only=True)
    return SyntheticDataset(data["features"], data["labels"])
