"""
Stage 2: Reinforcement learning (GRPO-style) fine-tuning of pruning actors.
Requires a reference checkpoint produced by Stage 1.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from prismatic.conf.models import ModelConfig, ModelRegistry
from prismatic.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from prismatic.models.vlms.prismatic import IGNORE_INDEX, PrismaticVLM


def load_config(model_id: str) -> ModelConfig:
    for variant in ModelRegistry:
        if variant.model_id == model_id:
            return variant.value()
    raise ValueError(f"Unknown model id: {model_id}")


class TorchFileDataset(Dataset):
    REQUIRED_KEYS = {"input_ids", "attention_mask", "labels", "pixel_values", "image_token_mask"}

    def __init__(self, data_path: Path) -> None:
        raw = torch.load(data_path)
        if isinstance(raw, dict):
            missing = self.REQUIRED_KEYS - raw.keys()
            if missing:
                raise ValueError(f"Dataset at {data_path} missing keys: {missing}")
            length = len(next(iter(raw.values())))
            self.items = [
                {key: raw[key][i] for key in self.REQUIRED_KEYS}
                for i in range(length)
            ]
        elif isinstance(raw, Iterable):
            self.items = list(raw)
            for item in self.items:
                missing = self.REQUIRED_KEYS - item.keys()
                if missing:
                    raise ValueError(f"Dataset item missing keys: {missing}")
        else:
            raise ValueError(f"Unsupported dataset format stored in {data_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def load_model(cfg: ModelConfig, device: torch.device) -> PrismaticVLM:
    vision_backbone, _ = get_vision_backbone_and_transform(
        cfg.vision_backbone_id,
        cfg.image_resize_strategy,
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.llm_backbone_id,
        llm_max_length=cfg.llm_max_length,
        inference_mode=False,
        llm_pruning_actor_layers=getattr(cfg, "llm_pruning_actor_layers", None),
        llm_pruning_actor_hidden_dim=getattr(cfg, "llm_pruning_actor_hidden_dim", None),
        llm_pruning_actor_num_samples=getattr(cfg, "llm_pruning_actor_num_samples", None),
        llm_pruning_reward_alpha=getattr(cfg, "llm_pruning_reward_alpha", 1.0),
        llm_pruning_reward_beta=getattr(cfg, "llm_pruning_reward_beta", 1.0),
    )
    vlm = get_vlm(cfg.model_id, cfg.arch_specifier, vision_backbone, llm_backbone)
    vlm.to(device)
    vlm.train()
    return vlm


def collect_actor_parameters(vlm: PrismaticVLM) -> Iterable[torch.nn.Parameter]:
    for actor in vlm.llm_backbone.pruning_actors.actors.values():
        yield from actor.parameters()


def load_reference_state(checkpoint_path: Path, vlm: PrismaticVLM) -> Dict[int, Dict[str, torch.Tensor]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        vlm.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "actor_state_dict" in ckpt:
        return {int(k): v for k, v in ckpt["actor_state_dict"].items()}
    # Compatibility: if full state_dict only, extract actors
    actor_state = {}
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", {}))
    for layer_id, actor in vlm.llm_backbone.pruning_actors.actors.items():
        sub = {name.split(".", 5)[-1]: tensor for name, tensor in state_dict.items() if f"pruning_actors.actors.{layer_id}" in name}
        if sub:
            actor_state[layer_id] = sub
    return actor_state


def compute_ce_per_sample(outputs, labels) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_logits.size(0), -1)
    per_sample = loss.mean(dim=-1)
    return per_sample.mean(), per_sample


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.mean()
    std = tensor.std().clamp_min(1e-6)
    return (tensor - mean) / std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2: RL fine-tuning of pruning actors (GRPO-style).")
    parser.add_argument("--model-id", required=True, help="ModelConfig identifier")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to training data (.pt)")
    parser.add_argument("--reference-checkpoint", type=Path, required=True, help="Stage1 checkpoint path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples per layer (G)")
    parser.add_argument("--gamma-ce", type=float, default=1.0)
    parser.add_argument("--gamma-prune", type=float, default=1.0)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--kl-coef", type=float, default=1e-3)
    return parser.parse_args()


def train_stage2(
    vlm: PrismaticVLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    num_samples: int,
    gamma_ce: float,
    gamma_prune: float,
    clip_eps: float,
    kl_coef: float,
) -> None:
    backbone = vlm.llm_backbone

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            outputs = vlm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                image_token_mask=batch["image_token_mask"],
            )

            ce_loss, ce_per_sample = compute_ce_per_sample(outputs, batch["labels"])
            stats = backbone.get_rl_statistics()
            if not stats["log_prob"]:
                raise RuntimeError("RL statistics are empty; ensure pruning actors are enabled and rl_training_mode is active.")

            # Total pruning ratio per sample (mean across layers)
            prune_ratios = torch.stack(list(stats["prune_ratio"].values()), dim=0)  # [layers, batch]
            total_prune_ratio = prune_ratios.mean(dim=0)

            global_reward = gamma_ce * ce_per_sample.detach() + gamma_prune * total_prune_ratio.detach()
            global_advantage = normalize(global_reward)

            total_loss = torch.tensor(0.0, device=device)
            for layer_id, log_prob in stats["log_prob"].items():
                ref_log_prob = stats["ref_log_prob"][layer_id]
                sampling_adv = stats["sampling_advantage"][layer_id]
                advantage = sampling_adv + global_advantage

                ratio = torch.exp(log_prob - ref_log_prob)
                clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

                policy_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))

                keep_prob = stats["keep_prob"][layer_id].clamp(1e-6, 1 - 1e-6)
                ref_keep_prob = stats["ref_keep_prob"][layer_id].clamp(1e-6, 1 - 1e-6)
                image_mask = stats["image_mask"][layer_id].to(dtype=keep_prob.dtype)

                kl = keep_prob * (torch.log(keep_prob) - torch.log(ref_keep_prob)) + \
                     (1 - keep_prob) * (torch.log(1 - keep_prob) - torch.log(1 - ref_keep_prob))
                kl = (kl * image_mask).sum(dim=-1).mean()

                layer_loss = policy_loss + kl_coef * kl
                total_loss = total_loss + layer_loss

            total_loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"[Stage2][Epoch {epoch+1}] Step {step} "
                    f"CE={ce_loss.item():.4f}  "
                    f"PruneRatio={total_prune_ratio.mean().item():.4f}  "
                    f"Loss={total_loss.item():.4f}"
                )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_id)
    if getattr(cfg, "llm_pruning_actor_layers", None) in (None, ()):  # 如果未指定，使用默认层
        cfg.llm_pruning_actor_layers = (3, 6, 9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm = load_model(cfg, device)

    reference_state = load_reference_state(args.reference_checkpoint, vlm)
    vlm.llm_backbone.enable_pruning_rl(reference_state, num_samples=args.num_samples)

    dataset = TorchFileDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    actor_parameters = list(collect_actor_parameters(vlm))
    if not actor_parameters:
        raise RuntimeError("No pruning actors available for RL fine-tuning.")
    optimizer = torch.optim.AdamW(actor_parameters, lr=args.actor_lr)

    train_stage2(
        vlm,
        dataloader,
        optimizer,
        device,
        args.epochs,
        args.num_samples,
        args.gamma_ce,
        args.gamma_prune,
        args.clip_eps,
        args.kl_coef,
    )

    checkpoint = {
        "model_state_dict": {name: param.cpu() for name, param in vlm.state_dict().items()},
        "actor_state_dict": vlm.llm_backbone.get_actor_state_dict(),
        "config_id": args.model_id,
    }
    output_path = Path(f"stage2_pruning_actor_{args.model_id.replace('+', '_')}.pt")
    torch.save(checkpoint, output_path)
    print(f"Stage 2 training finished. Checkpoint saved at {output_path}")


if __name__ == "__main__":
    main()

