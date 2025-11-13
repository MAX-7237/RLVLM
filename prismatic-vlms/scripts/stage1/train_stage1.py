"""
Stage 1: Supervised pretraining of pruning actors.

Optimizes actor MLPs with the objective:
    loss = cross_entropy_loss - pruning_reward
where pruning_reward is computed in base_llm (log-scale reward).
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

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


def compute_cross_entropy(outputs, labels) -> torch.Tensor:
    if outputs.loss is not None:
        return outputs.loss
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def train(
    vlm: PrismaticVLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> None:
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

            ce_loss = compute_cross_entropy(outputs, batch["labels"])
            pruning_reward = vlm.llm_backbone.get_total_pruning_reward()
            loss = ce_loss - pruning_reward

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"[Stage1][Epoch {epoch+1}] Step {step}  "
                    f"CE={ce_loss.item():.4f}  "
                    f"PruningReward={pruning_reward.item():.4f}  "
                    f"Objective={loss.item():.4f}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1: Supervised actor pretraining.")
    parser.add_argument("--model-id", required=True, help="ModelConfig identifier, e.g., reproduction-llava-v15+7b")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to torch serialized dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_id)
    if getattr(cfg, "llm_pruning_actor_layers", None) in (None, ()):  # 默认层
        cfg.llm_pruning_actor_layers = (3, 6, 9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm = load_model(cfg, device)

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
        raise RuntimeError("No pruning actors were installed. Set `llm_pruning_actor_layers` in the config.")
    optimizer = torch.optim.AdamW(actor_parameters, lr=args.actor_lr)

    train(vlm, dataloader, optimizer, device, args.epochs)

    checkpoint = {
        "model_state_dict": {name: param.cpu() for name, param in vlm.state_dict().items()},
        "actor_state_dict": {layer_id: actor.state_dict() for layer_id, actor in vlm.llm_backbone.pruning_actors.actors.items()},
        "config_id": args.model_id,
    }
    output_path = Path(f"stage1_pruning_actor_{args.model_id.replace('+', '_')}.pt")
    torch.save(checkpoint, output_path)
    print(f"Stage 1 training finished. Checkpoint saved at {output_path}")


if __name__ == "__main__":
    main()

