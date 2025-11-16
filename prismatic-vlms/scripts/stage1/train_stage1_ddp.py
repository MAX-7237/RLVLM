"""
Stage 1: Supervised pretraining of pruning actors (DDP version)
Safe and stable version for 8×3090, NO FSDP.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------
# Prismatic imports
# --------------------------------------------------
from prismatic.conf.models import ModelConfig, ModelRegistry
from prismatic.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from prismatic.models.vlms.prismatic import IGNORE_INDEX, PrismaticVLM


#####################################################################
#   Dataset
#####################################################################

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


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device, dtype: Optional[torch.dtype] = None):
    integer_keys = {"input_ids", "attention_mask", "labels"}
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if dtype is not None and v.dtype.is_floating_point and k not in integer_keys:
                v = v.to(dtype)
        result[k] = v
    return result


#####################################################################
#   Model loading
#####################################################################

def load_config(model_id: str) -> ModelConfig:
    for variant in ModelRegistry:
        if variant.model_id == model_id:
            return variant.value()
    raise ValueError(f"Unknown model id: {model_id}")


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


#####################################################################
#   DDP setup
#####################################################################

def setup_distributed(args):
    """Setup torch.distributed for DDP, preventing double initialization."""

    if dist.is_available() and dist.is_initialized():
        # Already initialized (e.g., Prismatic or HF did it)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(f"[DDP] Process group already initialized → rank={args.rank}")
        return True

    # torchrun will set these automatically
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # single GPU fallback
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return False

    torch.cuda.set_device(args.local_rank)

    # Initialize only once
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return True



#####################################################################
#   Training loop
#####################################################################

def train(vlm: PrismaticVLM, dataloader: DataLoader, optimizer, device, epochs, rank):

    if rank == 0:
        print(f"\n=== Training start (DDP) ===")
        print(f"Device = {device}, Epochs = {epochs}, Batches = {len(dataloader)}\n")

    for epoch in range(epochs):
        if rank == 0:
            print(f"--- Epoch {epoch+1} ---")

        for step, batch in enumerate(dataloader, start=1):

            if device.type == "cuda":
                model_dtype = next(vlm.parameters()).dtype
                batch = move_to_device(batch, device, dtype=model_dtype)
            else:
                batch = move_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            # FP16 autocast to reduce activation memory
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )

            ce_loss = compute_cross_entropy(outputs, batch["labels"])
            pruning_reward = vlm.llm_backbone.get_total_pruning_reward()
            loss = ce_loss - pruning_reward

            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"[NaN] Step {step}: CE={ce_loss}, Reward={pruning_reward}")
                continue

            loss.backward()
            optimizer.step()

            if rank == 0 and step % 5 == 0:
                print(
                    f"[Epoch {epoch+1}] Step {step}  "
                    f"CE={ce_loss.item():.4f}  Reward={pruning_reward.item():.4f}  "
                    f"Loss={loss.item():.4f}"
                )


#####################################################################
#   Argument parsing
#####################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="phi-2+3b")
    p.add_argument("--data-path", default="/home/ubuntu/vlm_prune/RLVLM/ai2d/training_data_1024.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


#####################################################################
#   Main
#####################################################################

def main():
    args = parse_args()

    # DDP init
    is_distributed = setup_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    if args.rank == 0:
        print(f"Using device: {device}")
        print(f"Distributed: {is_distributed}")

    cfg = load_config(args.model_id)
    # ----------------------
    # 强制开启 LLM 剪枝 actor（Stage1 必须训练这个）
    # ----------------------
    cfg.llm_pruning_actor_layers = (3, 6, 9)
    cfg.llm_pruning_actor_hidden_dim = 128
    cfg.llm_pruning_actor_num_samples = 4
    cfg.llm_pruning_reward_alpha = 1.0
    cfg.llm_pruning_reward_beta = 1.0

    if args.rank == 0:
        print(">> Enabled pruning actors at layers:", cfg.llm_pruning_actor_layers)


    # Load model components
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
    vlm = vlm.to(device)

    # ================================
    # 🔒 Freeze ALL non-actor weights
    # ================================
    for p in vlm.vision_backbone.parameters():
        p.requires_grad = False

    for p in vlm.llm_backbone.llm.parameters():
        p.requires_grad = False

    for p in vlm.projector.parameters():
        p.requires_grad = False

    # 如果 Prismatic 还有 image-to-text adapter（某些版本有）
    if hasattr(vlm, "image_text_adapter"):
        for p in vlm.image_text_adapter.parameters():
            p.requires_grad = False
            
    # Wrap DDP
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        vlm = DDP(
            vlm,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        if args.rank == 0:
            print("Model wrapped with DDP.")

    # Dataset
    dataset = TorchFileDataset(args.data_path)
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )

    # Optimizer: only train pruning actors
    actor_params = list(collect_actor_parameters(vlm.module if is_distributed else vlm))
    optimizer = torch.optim.AdamW(actor_params, lr=args.actor_lr)

    train(vlm, dataloader, optimizer, device, args.epochs, args.rank)

    if args.rank == 0:
        save_path = f"stage1_ddp_{args.model_id.replace('+','_')}.pt"
        torch.save(
            {
                "actor_state_dict": {
                    k: v.cpu()
                    for k, v in (vlm.module if is_distributed else vlm).llm_backbone.pruning_actors.state_dict().items()
                }
            },
            save_path,
        )
        print(f"\nTraining finished. Saved checkpoint → {save_path}\n")


if __name__ == "__main__":
    main()
