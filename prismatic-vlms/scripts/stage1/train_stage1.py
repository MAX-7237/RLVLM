"""
Stage 1: Supervised pretraining of pruning actors with FSDP support.

Optimizes actor MLPs with the objective:
    loss = cross_entropy_loss - pruning_reward
where pruning_reward is computed in base_llm (log-scale reward).

Usage:
    # Single GPU
    python scripts/stage1/train_stage1.py

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=8 scripts/stage1/train_stage1.py --use_fsdp
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from prismatic.conf.models import ModelConfig, ModelRegistry
from prismatic.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from prismatic.models.vlms.prismatic import IGNORE_INDEX, PrismaticVLM
from prismatic.training.strategies.fsdp import FSDPStrategy


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


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    # Integer tensors that should not be converted to model dtype
    integer_keys = {"input_ids", "attention_mask", "labels"}
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            # Only convert floating point tensors to model dtype, keep integers as is
            if dtype is not None and k not in integer_keys and v.dtype.is_floating_point:
                v = v.to(dtype)
            result[k] = v
        else:
            result[k] = v
    return result


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
    rank: int,
) -> None:
    if rank == 0:
        print(f"Starting training with {epochs} epochs, device: {device}")
        print(f"Dataloader has {len(dataloader)} batches")
    for epoch in range(epochs):
        if rank == 0:
            print(f"Starting epoch {epoch+1}")
        for step, batch in enumerate(dataloader, start=1):
            if rank == 0:
                print(f"Processing step {step}, batch keys: {batch.keys()}")
                print(f"Batch shapes - input_ids: {batch['input_ids'].shape}, labels: {batch['labels'].shape}")
            # Get model dtype for data conversion (only for GPU training)
            if device.type == "cuda":
                model_dtype = next(vlm.parameters()).dtype
                batch = move_to_device(batch, device, dtype=model_dtype)
            else:
                batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            # 前向启用 autocast(fp16) 降低激活显存；参数保持 FP32 由 FSDP 管理
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                    outputs = vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
            else:
                outputs = vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )

            ce_loss = compute_cross_entropy(outputs, batch["labels"])
            pruning_reward = vlm.llm_backbone.get_total_pruning_reward()
            loss = ce_loss - pruning_reward

            # 数值保护：若出现 NaN/Inf，跳过该步，防止梯度污染
            if not torch.isfinite(loss):
                if rank == 0:
                    print("[Stage1] Detected non-finite loss. Skipping step to maintain stability.")
                    print(f"  CE={ce_loss.detach().cpu().item() if torch.isfinite(ce_loss) else 'nan/inf'}  "
                          f"PruningReward={pruning_reward.detach().cpu().item() if torch.isfinite(pruning_reward) else 'nan/inf'}")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            optimizer.step()

            if step % 1 == 0 and rank == 0:  # Print every step for debugging
                print(
                    f"[Stage1][Epoch {epoch+1}] Step {step}  "
                    f"CE={ce_loss.item():.4f}  "
                    f"PruningReward={pruning_reward.item():.4f}  "
                    f"Objective={loss.item():.4f}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1: Supervised actor pretraining with FSDP support.")
    parser.add_argument("--model-id", default="phi-2+3b")
    parser.add_argument("--data-path", default="/home/ubuntu/vlm_prune/RLVLM/ai2d/training_data_1024.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)  # Small batch size for GPU memory efficiency
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-fsdp", action="store_true", help="Use FSDP for multi-GPU training")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load LLM in 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load LLM in 8-bit quantization")
    return parser.parse_args()


def setup_distributed(args):
    """Setup distributed training if using FSDP"""
    if args.use_fsdp:
        # Check if distributed is already initialized (by torchrun)
        if dist.is_initialized():
            args.rank = dist.get_rank()
            args.world_size = dist.get_world_size()
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            print(f"Distributed already initialized: rank {args.rank}/{args.world_size}")
        else:
            # Manual initialization for single process testing
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12345'
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
                os.environ['LOCAL_RANK'] = str(args.local_rank if args.local_rank != -1 else 0)

                args.rank = 0
                args.world_size = torch.cuda.device_count()
                args.local_rank = args.local_rank if args.local_rank != -1 else 0

                torch.cuda.set_device(args.local_rank)
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=args.world_size,
                    rank=args.rank
                )
                print(f"Manually initialized distributed: rank {args.rank}/{args.world_size}")
            else:
                # Single GPU
                args.rank = 0
                args.world_size = 1
                args.local_rank = 0
                return False

        torch.cuda.set_device(args.local_rank)
        return True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return False


def main() -> None:
    print("=== Starting main function ===")
    args = parse_args()
    print(f"Args parsed: model_id={args.model_id}, batch_size={args.batch_size}, use_fsdp={args.use_fsdp}")

    # Setup distributed training
    is_distributed = setup_distributed(args)

    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    print("CUDA cache cleared")

    print(f"Loading config for {args.model_id}")
    cfg = load_config(args.model_id)
    print("Config loaded")

    if args.use_fsdp:
        # For FSDP, we need pruning actors
        if getattr(cfg, "llm_pruning_actor_layers", None) in (None, ()):
            cfg.llm_pruning_actor_layers = (3, 6, 9)
        print(f"Pruning actor layers set to: {cfg.llm_pruning_actor_layers}")
    else:
        # For single GPU/CPU, disable pruning actors for debugging
        cfg.llm_pruning_actor_layers = None
        print("Pruning actors disabled for single GPU/CPU training")

    device = torch.device(f"cuda:{args.local_rank}") if args.use_fsdp and torch.cuda.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model components...")
    vision_backbone, _ = get_vision_backbone_and_transform(
        cfg.vision_backbone_id,
        cfg.image_resize_strategy,
    )
    print("Vision backbone loaded")

    print("Loading LLM backbone...")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.llm_backbone_id,
        llm_max_length=cfg.llm_max_length,
        inference_mode=False,
        llm_pruning_actor_layers=cfg.llm_pruning_actor_layers,
        llm_pruning_actor_hidden_dim=getattr(cfg, "llm_pruning_actor_hidden_dim", None),
        llm_pruning_actor_num_samples=getattr(cfg, "llm_pruning_actor_num_samples", None),
        llm_pruning_reward_alpha=getattr(cfg, "llm_pruning_reward_alpha", 1.0),
        llm_pruning_reward_beta=getattr(cfg, "llm_pruning_reward_beta", 1.0),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    print("LLM backbone loaded")

    print("Assembling VLM...")
    vlm = get_vlm(cfg.model_id, cfg.arch_specifier, vision_backbone, llm_backbone)
    print("VLM assembled")

    # 放置到设备：
    # - FSDP 模式：保持在 CPU，交由 FSDP 在各 rank 上按需搬运/重构分片，避免先把整模型放到单卡导致 OOM
    # - 非 FSDP：直接移动到目标设备
    if args.use_fsdp:
        print("Keep VLM on CPU before FSDP wrapping to avoid pre-wrap GPU OOM")
    else:
        vlm.to(device)
        print(f"VLM moved to {device}")

    # 视觉主干与投影头放到本 rank GPU，并使用 FP16 权重；输入同样在 autocast(fp16) 下计算
    if torch.cuda.is_available():
        target_device = torch.device(f"cuda:{args.local_rank}") if args.use_fsdp else device
        vlm.vision_backbone = vlm.vision_backbone.to(device=target_device, dtype=torch.float16)
        if hasattr(vlm, "projector") and vlm.projector is not None:
            vlm.projector = vlm.projector.to(device=target_device, dtype=torch.float16)

    if args.use_fsdp:
        # 使用原生 PyTorch FSDP 进行最小改动包裹，保持现有训练循环
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        # 不使用 FSDP 混合精度，统一保持 FP32，由训练时 autocast 控制计算精度
        fsdp_precision = None
        # 仅对 LLM Backbone 做 FSDP 包裹，并按 Transformer Layer 自动 wrap，降低初始化峰值
        # 在包裹前将所有浮点参数与缓冲区统一为 FP32，避免 FSDP flatten dtype 冲突
        for _, param in vlm.llm_backbone.named_parameters(recurse=True):
            if param.dtype.is_floating_point:
                param.data = param.data.to(torch.float32)
        for _, buffer in vlm.llm_backbone.named_buffers(recurse=True):
            if buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(torch.float32)
        llm_layer_cls = getattr(vlm.llm_backbone, "transformer_layer_cls", None)
        # 通过 partial 预先指定 transformer 层类别；阈值设置偏小以更细粒度 wrap，降低峰值
        auto_policy = None
        if llm_layer_cls is not None:
            auto_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={llm_layer_cls},
                nonwrapped_numel=0,  # 更细粒度分块
            )
        # 忽略包含整型参数的子模块，及嵌入层（Embedding）以避免权重被扁平化成 1-D
        ignored = []
        for submodule in vlm.llm_backbone.modules():
            has_int_param = False
            for p in submodule.parameters(recurse=False):
                if isinstance(p, torch.nn.Parameter) and not p.dtype.is_floating_point:
                    has_int_param = True
                    break
            if has_int_param or isinstance(submodule, torch.nn.Embedding):
                ignored.append(submodule)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,   # LLaMA 3B 支持 BF16
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        vlm.llm_backbone = FSDP(
            vlm.llm_backbone,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,         # 必开
            auto_wrap_policy=auto_policy, # Transformer layer-wise wrapping
            cpu_offload=False,            # 禁止 offload，3090 的 PCIe 太慢
            limit_all_gathers=True,
            ignored_modules=ignored or None,
        )
        print("Wrapped LLM backbone with native FSDP (auto-wrap transformer layers)")

    if (not args.use_fsdp) and device.type == "cuda" and not (args.load_in_4bit or args.load_in_8bit):
        print("Converting to bfloat16 mixed precision...")
        vlm = vlm.to(torch.bfloat16)
        torch.cuda.empty_cache()

    print("Enabling gradient checkpointing...")
    vlm.llm_backbone.enable_gradient_checkpointing()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading dataset from {args.data_path}")
    dataset = TorchFileDataset(args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")

    if args.use_fsdp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True, rank=args.rank, num_replicas=args.world_size)
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
    print(f"DataLoader created with batch_size={args.batch_size}")

    print("Checking for pruning actors...")
    actor_parameters = list(collect_actor_parameters(vlm))
    print(f"Found {len(actor_parameters)} actor parameters")

    if actor_parameters:
        # Training with pruning actors (original stage1 training)
        print("Training with pruning actors (Stage 1)")
        optimizer = torch.optim.AdamW(actor_parameters, lr=args.actor_lr)
    else:
        # Fallback: train all parameters (modified training)
        print("No pruning actors found, training all model parameters instead")
        all_parameters = list(vlm.parameters())
        print(f"Training {len(all_parameters)} total parameters")
        optimizer = torch.optim.AdamW(all_parameters, lr=args.actor_lr)

    print("Optimizer created successfully")
    print("Starting training...")
    train(vlm, dataloader, optimizer, device, args.epochs, args.rank)

    if args.rank == 0:
        checkpoint = {
            "model_state_dict": {name: param.cpu() for name, param in vlm.state_dict().items()},
            "config_id": args.model_id,
        }

        # Only save actor state dict if actors exist
        if hasattr(vlm, 'llm_backbone') and hasattr(vlm.llm_backbone, 'pruning_actors'):
            checkpoint["actor_state_dict"] = {
                layer_id: actor.state_dict()
                for layer_id, actor in vlm.llm_backbone.pruning_actors.actors.items()
            }
        output_path = Path(f"stage1_pruning_actor_{args.model_id.replace('+', '_')}.pt")
        torch.save(checkpoint, output_path)
        print(f"Stage 1 training finished. Checkpoint saved at {output_path}")


if __name__ == "__main__":
    main()

