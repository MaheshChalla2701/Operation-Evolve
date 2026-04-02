"""
replay_buffer.py – Production-grade Replay Buffer for Hybrid Continual Learning.

Uses reservoir sampling so that ALL historical samples have an equal probability
of being retained regardless of insertion order, preventing early-task domination.

Public API
----------
ReplayBufferV2
    .add_samples(features, labels)   – Add a raw tensor batch (reservoir enforced)
    .update(dataset)                  – Convenience: add entire SyntheticDataset
    .sample(batch_size)               – Random sample → SyntheticDataset | None
    .save(path)                       – Persist buffer to disk
    .load(path)                       – Restore buffer from disk (classmethod)
    .__len__()                        – Current number of stored samples
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("evolve.replay_buffer")


# ---------------------------------------------------------------------------
# Lazy import to avoid a circular import with data.py
# ---------------------------------------------------------------------------

def _make_dataset(features: torch.Tensor, labels: torch.Tensor):
    """Return TextDataset for long tensors, SyntheticDataset for floats."""
    from data import SyntheticDataset, TextDataset  # noqa: PLC0415
    if features.dtype in (torch.int32, torch.int64):
        return TextDataset(features, labels)
    return SyntheticDataset(features, labels)


# ---------------------------------------------------------------------------
# ReplayBufferV2  – reservoir sampling implementation
# ---------------------------------------------------------------------------

class ReplayBufferV2:
    """
    Memory-bounded replay buffer using reservoir sampling (Algorithm R).

    Reservoir sampling guarantees that every past sample has an equal
    probability of being retained, regardless of insertion order or total
    number of samples seen.  This is crucial for 100+ sequential datasets
    where naive FIFO would erase early-task knowledge.

    Attributes
    ----------
    max_size : int
        Maximum number of samples stored at any time.
    _features : torch.Tensor | None
        Stored feature tensor, shape (N, D).
    _labels : torch.Tensor | None
        Stored label tensor, shape (N,).
    _total_seen : int
        Total samples seen since creation (for reservoir probability).
    """

    def __init__(self, max_size: int = 10_000):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self._features: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        self._total_seen: int = 0

    # ------------------------------------------------------------------
    # Core insertion
    # ------------------------------------------------------------------

    def add_samples(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Add a batch of (features, labels) tensors using reservoir sampling.

        For each incoming sample:
        - If the buffer has space, add directly.
        - Otherwise, randomly replace an existing slot with probability
          max_size / total_seen_so_far  (standard Algorithm R).

        Parameters
        ----------
        features : torch.Tensor, shape (N, D)
        labels   : torch.Tensor, shape (N,)
        """
        features = features.cpu()
        labels = labels.cpu()
        # Preserve dtype: long for token sequences, float for features
        if features.is_floating_point():
            features = features.float()
        else:
            features = features.long()
        labels = labels.long()
        assert features.shape[0] == labels.shape[0], (
            f"features/labels size mismatch: {features.shape[0]} vs {labels.shape[0]}"
        )

        n_new = features.shape[0]

        for i in range(n_new):
            self._total_seen += 1

            if self._features is None:
                # First ever sample — initialise tensors
                self._features = features[i:i+1].clone()
                self._labels = labels[i:i+1].clone()

            elif self._features.shape[0] < self.max_size:
                # Buffer not yet full — just append
                self._features = torch.cat([self._features, features[i:i+1]])
                self._labels = torch.cat([self._labels, labels[i:i+1]])

            else:
                # Reservoir: replace a random slot with probability max_size / total_seen
                j = int(torch.randint(0, self._total_seen, (1,)).item())
                if j < self.max_size:
                    self._features[j] = features[i]
                    self._labels[j] = labels[i]

        logger.debug(
            f"[ReplayBuffer] add_samples: added {n_new} | "
            f"buffer_size={len(self)} | total_seen={self._total_seen}"
        )

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def update(self, dataset) -> None:
        """
        Add an entire SyntheticDataset or TextDataset (anything with .features/.labels)
        to the buffer using reservoir sampling.

        Call this AFTER each training loop finishes to absorb the current task.

        Parameters
        ----------
        dataset : SyntheticDataset | TextDataset
            Any dataset exposing .features (Tensor) and .labels (Tensor).
        """
        self.add_samples(dataset.features, dataset.labels)
        logger.info(
            f"[ReplayBuffer] update: absorbed dataset of {len(dataset)} samples | "
            f"buffer now has {len(self)} / {self.max_size} samples"
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int):
        """
        Draw a random sample from the buffer.

        Returns None if the buffer is empty; otherwise returns a SyntheticDataset
        with min(batch_size, len(self)) samples.

        Parameters
        ----------
        batch_size : int
            Desired number of samples.

        Returns
        -------
        SyntheticDataset | None
        """
        if self._features is None or self._features.shape[0] == 0:
            return None

        n = min(batch_size, self._features.shape[0])
        idx = torch.randperm(self._features.shape[0])[:n]
        return _make_dataset(self._features[idx], self._labels[idx])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the buffer to disk as a .pt file.

        Parameters
        ----------
        path : str
            File path (directories are created automatically).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "features": self._features,
            "labels": self._labels,
            "total_seen": self._total_seen,
            "max_size": self.max_size,
        }
        torch.save(payload, path)
        logger.info(f"[ReplayBuffer] saved {len(self)} samples → {path}")

    @classmethod
    def load(cls, path: str) -> "ReplayBufferV2":
        """
        Restore a buffer from disk.

        Parameters
        ----------
        path : str
            Path to a file previously created by .save().

        Returns
        -------
        ReplayBufferV2
        """
        payload = torch.load(path, weights_only=True)
        buf = cls(max_size=payload["max_size"])
        buf._features = payload.get("features")
        buf._labels = payload.get("labels")
        buf._total_seen = payload.get("total_seen", 0)
        logger.info(f"[ReplayBuffer] loaded {len(buf)} samples ← {path}")
        return buf

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._features is None:
            return 0
        return self._features.shape[0]

    def __repr__(self) -> str:
        return (
            f"ReplayBufferV2(size={len(self)}/{self.max_size}, "
            f"total_seen={self._total_seen})"
        )
