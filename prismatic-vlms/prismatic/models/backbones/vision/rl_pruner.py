"""
rl_pruner.py

Utilities for integrating a MAPPO-based token pruning controller into TIMM Vision Transformers.
The controller consumes intermediate patch embeddings and produces binary masks that zero-out
attention contributions from low-utility visual tokens while leaving the ViT architecture intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence
import warnings

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MAPPOPruningConfig:
    """Configuration for MAPPO token pruning."""

    target_layers: Sequence[int]
    threshold: float = 0.5
    min_keep_ratio: float = 0.1
    zero_out_queries: bool = True
    policy_path: str | None = None


class MAPPOActor(nn.Module):
    """A lightweight MAPPO actor network that scores tokens for pruning."""

    def __init__(self, embed_dim: int, hidden_dim: int, layer_vocab_size: int) -> None:
        super().__init__()
        self.layer_embedding = nn.Embedding(layer_vocab_size, embed_dim)
        self.policy = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, layer_index: int, layer_idx_lookup: Dict[int, int]) -> torch.Tensor:
        """
        Args:
            tokens: Tensor[B, N, C] of token embeddings.
            layer_index: 1-based ViT block index.
            layer_idx_lookup: mapping from absolute block id to embedding index.

        Returns:
            Tensor[B, N] of logits prior to sigmoid.
        """
        if tokens.dim() != 3:
            raise ValueError(f"Expected `tokens` to be rank-3 (B, N, C). Got shape {tuple(tokens.shape)}.")

        device = tokens.device
        batch_size, num_tokens, _ = tokens.shape
        layer_token_idx = layer_idx_lookup[layer_index]
        layer_ids = torch.full((batch_size, num_tokens), layer_token_idx, dtype=torch.long, device=device)

        contextual_tokens = tokens + self.layer_embedding(layer_ids)
        global_context = contextual_tokens.mean(dim=1, keepdim=True)
        contextual_tokens = contextual_tokens + global_context

        logits = self.policy(contextual_tokens).squeeze(-1)
        return logits


class MAPPOPruningController(nn.Module):
    """Controller that uses a MAPPO actor to produce token masks and apply them to attention outputs."""

    def __init__(
        self,
        embed_dim: int,
        config: MAPPOPruningConfig,
    ) -> None:
        super().__init__()
        if len(config.target_layers) == 0:
            raise ValueError("`target_layers` must be non-empty for MAPPO pruning.")

        self.config = config
        self._fallback_keep_all = config.policy_path is None
        if self._fallback_keep_all:
            warnings.warn(
                "MAPPO policy path not provided; vision token pruning will keep all tokens.",
                RuntimeWarning,
            )
        self.register_buffer(
            "_layer_lookup",
            torch.tensor(config.target_layers, dtype=torch.long),
            persistent=False,
        )
        self._layer_index_mapping = {
            int(layer_id.item()): idx for idx, layer_id in enumerate(self._layer_lookup)
        }

        hidden_dim = max(embed_dim // 2, 128)
        self.actor = MAPPOActor(embed_dim, hidden_dim, len(config.target_layers))

        if config.policy_path is not None:
            self._load_policy_weights(config.policy_path)

        # Keep most recent masks for inspection/debugging.
        self.latest_masks: Dict[int, torch.Tensor] = {}

    @property
    def target_layers(self) -> Sequence[int]:
        return tuple(int(idx.item()) for idx in self._layer_lookup)

    def _load_policy_weights(self, policy_path: str) -> None:
        path = Path(policy_path)
        if not path.exists():
            raise FileNotFoundError(f"MAPPO policy checkpoint not found at `{path}`.")

        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.actor.load_state_dict(state_dict, strict=False)

    def compute_mask(self, tokens: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Return a boolean mask indicating which tokens to keep."""
        if self._fallback_keep_all:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
            self.latest_masks[layer_index] = mask.detach().cpu()
            return mask

        logits = self.actor(tokens, layer_index, self._layer_index_mapping)
        probs = torch.sigmoid(logits)

        if self.config.threshold is None:
            binary_mask = torch.ones_like(probs, dtype=torch.bool)
        else:
            binary_mask = probs >= self.config.threshold

        # Ensure a minimum keep ratio per sample.
        batch_size, num_tokens = binary_mask.shape
        min_keep = max(int(num_tokens * self.config.min_keep_ratio), 1)
        min_keep = min(min_keep, num_tokens)

        topk_indices = probs.topk(k=min_keep, dim=1).indices
        for batch_idx in range(batch_size):
            if torch.count_nonzero(binary_mask[batch_idx]) < min_keep:
                binary_mask[batch_idx].zero_()
                binary_mask[batch_idx, topk_indices[batch_idx]] = True

        # Always keep the first token (CLS) and any distillation tokens if present.
        binary_mask[:, 0] = True
        self.latest_masks[layer_index] = binary_mask.detach().cpu()
        return binary_mask

    def build_attention_forward(self, attn_module: nn.Module, layer_index: int):
        """Return a forward function that applies pruning to the given attention module."""

        def forward(module_self: nn.Module, x: torch.Tensor) -> torch.Tensor:
            mask = self.compute_mask(x, layer_index)
            return self._apply_attention_with_mask(module_self, x, mask)

        return forward

    def _apply_attention_with_mask(
        self,
        attn_module: nn.Module,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replicate TIMM attention forward pass with MAPPO mask application."""
        B, N, C = tokens.shape
        qkv = (
            attn_module.qkv(tokens)
            .reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if hasattr(attn_module, "q_norm"):
            q = attn_module.q_norm(q)
        if hasattr(attn_module, "k_norm"):
            k = attn_module.k_norm(k)

        q = q * attn_module.scale

        key_mask = mask[:, None, None, :].to(dtype=torch.bool, device=tokens.device)
        scores = torch.matmul(q, k.transpose(-2, -1))
        mask_fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~key_mask, mask_fill_value)

        attn = scores.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)

        value_mask = mask[:, None, :, None].to(dtype=v.dtype, device=tokens.device)
        v = v * value_mask

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = attn_module.proj(out)
        out = attn_module.proj_drop(out)

        if self.config.zero_out_queries:
            out = out * mask.to(out.dtype).unsqueeze(-1)

        return out


