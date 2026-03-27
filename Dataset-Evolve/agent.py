"""
agent.py – Decision logic for the Operation Evolve self-evolving pipeline.

Design principles:
  1. Agent NEVER directly overwrites datasets.
  2. Agent PROPOSES a structured update plan.
  3. Updates are APPLIED only by apply_updates() after external validation.
  4. The EvolveAgent class can be subclassed (e.g. LLMAgent) for future upgrades.

Rule-based logic:
  - Remove samples with model confidence < threshold (likely noisy)
  - Add high-confidence generated samples from Dataset_C
  - Detect repeated per-class errors and log warnings
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import torch
from groq import Groq

from config import EvolveConfig
from data import SyntheticDataset, filter_by_confidence

logger = logging.getLogger("evolve.agent")


# ---------------------------------------------------------------------------
# Proposal dataclass
# ---------------------------------------------------------------------------

class UpdateProposal:
    """
    Structured proposal produced by the agent.

    Fields:
        add_features    : tensor of features to add to Dataset_B
        add_labels      : tensor of labels  to add to Dataset_B
        remove_indices  : list of Dataset_B indices to remove
        notes           : human-readable log of reasoning
        accepted_count  : number of new samples accepted
        rejected_count  : number of Dataset_C samples rejected
    """

    def __init__(self):
        self.add_features: Optional[torch.Tensor] = None
        self.add_labels: Optional[torch.Tensor] = None
        self.remove_indices: List[int] = []
        self.notes: List[str] = []
        self.accepted_count: int = 0
        self.rejected_count: int = 0

    def summary(self) -> str:
        return (
            f"accepted={self.accepted_count} | "
            f"rejected={self.rejected_count} | "
            f"removals={len(self.remove_indices)}"
        )


# ---------------------------------------------------------------------------
# Abstract base (swap EvolveAgent for LLMAgent later)
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract interface all agents must implement."""

    @abstractmethod
    def propose_updates(
        self,
        dataset_b: SyntheticDataset,
        dataset_c: Dict[str, torch.Tensor],
        eval_results: Dict[str, Any],
        config: EvolveConfig,
    ) -> UpdateProposal:
        ...

    def analyze(
        self,
        dataset_c: Dict[str, torch.Tensor],
        eval_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Pre-analysis hook – override to add richer analysis."""
        return {
            "mean_conf": dataset_c["confidences"].mean().item(),
            "min_conf": dataset_c["confidences"].min().item(),
            "max_conf": dataset_c["confidences"].max().item(),
            "current_accuracy": eval_results.get("accuracy", 0.0),
        }


# ---------------------------------------------------------------------------
# Rule-based agent
# ---------------------------------------------------------------------------

class EvolveAgent(BaseAgent):
    """
    Rule-based decision agent.

    Rules (in priority order):
      1. Accept Dataset_C samples with confidence >= threshold (additions).
      2. Remove Dataset_B samples whose re-scored confidence <
         (threshold - removal_margin) – marks as likely noisy.
      3. Flag classes with repeated errors for logging.
    """

    def __init__(self, removal_margin: float = 0.15):
        """
        Args:
            removal_margin: Buffer below threshold to trigger Dataset_B removals.
                            E.g. if threshold=0.85, removals happen at conf < 0.70.
        """
        self.removal_margin = removal_margin

    def propose_updates(
        self,
        dataset_b: SyntheticDataset,
        dataset_c: Dict[str, torch.Tensor],
        eval_results: Dict[str, Any],
        config: EvolveConfig,
    ) -> UpdateProposal:
        """
        Produce a structured UpdateProposal without touching any data directly.
        """
        proposal = UpdateProposal()

        # ---- Pre-analysis ----
        analysis = self.analyze(dataset_c, eval_results)
        logger.info(
            f"[Agent] Analysis | "
            f"Dataset_C mean_conf={analysis['mean_conf']:.3f} | "
            f"model_accuracy={analysis['current_accuracy']:.2f}%"
        )

        # ---- Rule 1: Accept high-confidence Dataset_C samples ----
        confs = dataset_c["confidences"]
        features = dataset_c["features"]
        preds = dataset_c["pred_labels"]

        conf_mask = confs >= config.confidence_threshold
        accepted_features = features[conf_mask]
        accepted_labels = preds[conf_mask]
        proposal.accepted_count = int(conf_mask.sum().item())
        proposal.rejected_count = int((~conf_mask).sum().item())

        if proposal.accepted_count > 0:
            proposal.add_features = accepted_features
            proposal.add_labels = accepted_labels
            proposal.notes.append(
                f"Rule-1 ADD: {proposal.accepted_count} high-confidence samples "
                f"(conf >= {config.confidence_threshold:.2f})"
            )
        else:
            proposal.notes.append(
                "Rule-1 ADD: No samples passed confidence threshold – skipping addition."
            )

        # ---- Rule 2: Remove noisy Dataset_B samples ----
        removal_threshold = config.confidence_threshold - self.removal_margin
        # We use per_class_acc as a proxy for difficulty – flag low-performing classes
        per_class_acc = eval_results.get("per_class_acc", {})
        struggling_classes = [
            cls for cls, acc in per_class_acc.items() if acc < 50.0
        ]

        if struggling_classes:
            proposal.notes.append(
                f"Rule-3 WARN: Classes with accuracy < 50%: {struggling_classes}"
            )
            logger.warning(
                f"[Agent] Repeated errors detected in classes: {struggling_classes}"
            )

        # Identify Dataset_B samples to remove heuristically.
        # Strategy: for each struggling class, flag samples of that class
        # if Dataset_B is large enough to survive trimming.
        if struggling_classes and len(dataset_b) > 100:
            remove_candidates: List[int] = []
            for idx in range(len(dataset_b)):
                lbl = dataset_b.labels[idx].item()
                if lbl in struggling_classes:
                    remove_candidates.append(idx)

            # Limit removals to at most 10% of Dataset_B to stay conservative
            max_removals = max(1, len(dataset_b) // 10)
            proposal.remove_indices = remove_candidates[:max_removals]
            if proposal.remove_indices:
                proposal.notes.append(
                    f"Rule-2 REMOVE: {len(proposal.remove_indices)} noisy samples "
                    f"from struggling classes {struggling_classes}"
                )

        for note in proposal.notes:
            logger.info(f"[Agent] {note}")

        logger.info(f"[Agent] Proposal → {proposal.summary()}")
        return proposal


# ---------------------------------------------------------------------------
# LLM Agent stub  (drop-in replacement for future use)
# ---------------------------------------------------------------------------

class LLMAgent(BaseAgent):
    """
    LLM-based agent using Groq API.

    It analyzes evaluation results and dataset statistics,
    then queries the LLM to propose an update strategy.
    """

    def __init__(self, config: EvolveConfig):
        if not config.groq_api_key:
            raise ValueError("groq_api_key is not set in config.")
        
        self.client = Groq(api_key=config.groq_api_key)
        self.model_name = config.llm_model_name

    def propose_updates(
        self,
        dataset_b: SyntheticDataset,
        dataset_c: Dict[str, torch.Tensor],
        eval_results: Dict[str, Any],
        config: EvolveConfig,
    ) -> UpdateProposal:
        proposal = UpdateProposal()
        
        # We need to construct a summary of the current iteration to feed to the LLM
        analysis = self.analyze(dataset_c, eval_results)
        c_size = dataset_c["features"].shape[0]
        b_size = len(dataset_b)
        
        per_class_acc = eval_results.get("per_class_acc", {})
        
        prompt = f"""
You are an expert AI agent orchestrating a self-evolving dataset.
Your job is to decide which rules to apply to update the training dataset (Dataset_B).

CURRENT STATE:
- Dataset_B (Training) size: {b_size} samples
- Dataset_C (Candidate/Generated) size: {c_size} samples
- Model Overall Accuracy: {analysis['current_accuracy']:.2f}%
- Candidate Dataset_C Confidence (Mean: {analysis['mean_conf']:.3f}, Min: {analysis['min_conf']:.3f}, Max: {analysis['max_conf']:.3f})
- Per-class accuracy: {per_class_acc}

You have the following hard-coded system parameters:
- Confidence Threshold: {config.confidence_threshold}

Based on this information, dynamically set the strategies:
1. `accept_threshold`: What minimum confidence score should we accept generated samples from Dataset_C? Must be >= {config.confidence_threshold}.
2. `removal_margin`: How far below the confidence threshold should we tolerate existing training samples before flagging them as noisy? (e.g. 0.15)
3. `classes_to_purge`: A list of integer class IDs that are struggling heavily (accuracy < 50%) from which we should explicitly remove noisy samples.

Reply ONLY with a valid JSON object matching this schema:
{{
    "accept_threshold": float,
    "removal_margin": float,
    "classes_to_purge": [int],
    "reasoning": "A short summary of why these choices were made"
}}
"""
        logger.info(f"[LLMAgent] Querying Groq ({self.model_name})...")
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            decision = json.loads(content)
            
            accept_threshold = max(float(decision.get("accept_threshold", config.confidence_threshold)), config.confidence_threshold)
            removal_margin = float(decision.get("removal_margin", 0.15))
            classes_to_purge = decision.get("classes_to_purge", [])
            reasoning = decision.get("reasoning", "No reasoning provided.")
            
            logger.info(f"[LLMAgent] LLM Reasoning: {reasoning}")
            proposal.notes.append(f"LLM Reasoning: {reasoning}")
            
        except Exception as e:
            logger.error(f"[LLMAgent] Failed to query LLM or parse response: {e}")
            logger.warning("[LLMAgent] Falling back to default rule-based parameters.")
            accept_threshold = config.confidence_threshold
            removal_margin = 0.15
            classes_to_purge = [cls for cls, acc in per_class_acc.items() if acc < 50.0]

        # ---- Apply the LLM's dynamic parameters ----
        
        # Additions
        confs = dataset_c["confidences"]
        features = dataset_c["features"]
        preds = dataset_c["pred_labels"]

        conf_mask = confs >= accept_threshold
        proposal.accepted_count = int(conf_mask.sum().item())
        proposal.rejected_count = int((~conf_mask).sum().item())

        if proposal.accepted_count > 0:
            proposal.add_features = features[conf_mask]
            proposal.add_labels = preds[conf_mask]
            proposal.notes.append(f"LLM Rule ADD: {proposal.accepted_count} samples (threshold {accept_threshold:.2f})")
        else:
            proposal.notes.append(f"LLM Rule ADD: 0 samples passed threshold {accept_threshold:.2f}")

        # Removals
        removal_threshold = config.confidence_threshold - removal_margin
        
        if classes_to_purge and len(dataset_b) > 100:
            remove_candidates: List[int] = []
            for idx in range(len(dataset_b)):
                lbl = dataset_b.labels[idx].item()
                if lbl in classes_to_purge:
                    remove_candidates.append(idx)
            
            max_removals = max(1, len(dataset_b) // 10)
            proposal.remove_indices = remove_candidates[:max_removals]
            if proposal.remove_indices:
                proposal.notes.append(f"LLM Rule REMOVE: {len(proposal.remove_indices)} noisy samples from classes {classes_to_purge}")

        for note in proposal.notes:
            logger.info(f"[LLMAgent] {note}")

        logger.info(f"[LLMAgent] Proposal → {proposal.summary()}")
        return proposal


# ---------------------------------------------------------------------------
# Apply updates  (called AFTER validation)
# ---------------------------------------------------------------------------

def apply_updates(
    dataset_b: SyntheticDataset,
    proposal: UpdateProposal,
) -> SyntheticDataset:
    """
    Apply a validated UpdateProposal to Dataset_B.

    Steps:
      1. Remove flagged indices from Dataset_B.
      2. Append accepted new samples.

    The dataset is NEVER replaced wholesale – only surgical changes are applied.
    """
    features = dataset_b.features.clone()
    labels = dataset_b.labels.clone()

    # Step 1: removals (build keep mask)
    n = features.shape[0]
    keep_mask = torch.ones(n, dtype=torch.bool)
    for idx in proposal.remove_indices:
        if 0 <= idx < n:
            keep_mask[idx] = False

    features = features[keep_mask]
    labels = labels[keep_mask]

    # Step 2: additions
    if proposal.add_features is not None and proposal.add_features.shape[0] > 0:
        features = torch.cat([features, proposal.add_features])
        labels = torch.cat([labels, proposal.add_labels])

    logger.info(
        f"[Agent Apply] Dataset_B updated: {n} → {features.shape[0]} samples "
        f"(removed={n - keep_mask.sum().item()} | added={proposal.accepted_count})"
    )
    return SyntheticDataset(features, labels)
