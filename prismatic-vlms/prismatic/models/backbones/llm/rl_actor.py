"""
rl_actor.py

Lightweight actor networks for LLM token pruning policies.
Each actor consumes the hidden states produced by a transformer layer
and outputs logits over {prune, keep}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LLMActorConfig:
    """Configuration for installing pruning actors inside LLM transformer layers."""

    target_layers: Tuple[int, ...]
    hidden_dim: int


class TokenPruningActor(nn.Module):
    """Simple per-token classifier that outputs prune/keep logits."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape [batch, seq_len, embed_dim]

        Returns:
            Tensor of shape [batch, seq_len, 2] representing logits for {prune, keep}.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"TokenPruningActor expects hidden states in shape [batch, seq_len, embed_dim], "
                f"but received tensor with shape {tuple(hidden_states.shape)}."
            )
        return self.net(hidden_states)


class PruningActorRegistry(nn.Module):
    """Module container that keeps actor networks indexed by layer id."""

    def __init__(self) -> None:
        super().__init__()
        self.actors = nn.ModuleDict()

    def add_actor(self, layer_id: int, actor: TokenPruningActor) -> None:
        self.actors[str(layer_id)] = actor

    def get_actor(self, layer_id: int) -> TokenPruningActor:
        if str(layer_id) not in self.actors:
            raise KeyError(f"No pruning actor registered for layer {layer_id}.")
        return self.actors[str(layer_id)]

    def __contains__(self, layer_id: int) -> bool:
        return str(layer_id) in self.actors

    def forward(self, layer_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.get_actor(layer_id)(hidden_states)


