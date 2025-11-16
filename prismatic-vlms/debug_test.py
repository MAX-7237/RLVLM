"""
Debug script to test model loading and data loading without training.
"""

import torch
from pathlib import Path
from prismatic.conf.models import ModelConfig, ModelRegistry
from prismatic.models.materialize import get_vision_backbone_and_transform, get_llm_backbone_and_tokenizer, get_vlm
from prismatic.models.vlms.prismatic import IGNORE_INDEX, PrismaticVLM
from scripts.stage1.train_stage1 import TorchFileDataset

def load_config(model_id: str) -> ModelConfig:
    for variant in ModelRegistry:
        if variant.model_id == model_id:
            return variant.value()
    raise ValueError(f"Unknown model id: {model_id}")

def main():
    print("=== Starting debug test ===")

    # Load config
    print("Loading config...")
    cfg = load_config("reproduction-llava-v15+7b")
    cfg.llm_pruning_actor_layers = (3, 6, 9)

    # Load model on CPU
    print("Loading model on CPU...")
    cpu_device = torch.device("cpu")
    vision_backbone, _ = get_vision_backbone_and_transform(
        cfg.vision_backbone_id,
        cfg.image_resize_strategy,
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.llm_backbone_id,
        llm_max_length=cfg.llm_max_length,
        inference_mode=False,
        llm_pruning_actor_layers=cfg.llm_pruning_actor_layers,
        llm_pruning_actor_hidden_dim=getattr(cfg, "llm_pruning_actor_hidden_dim", None),
        llm_pruning_actor_num_samples=getattr(cfg, "llm_pruning_actor_num_samples", None),
        llm_pruning_reward_alpha=getattr(cfg, "llm_pruning_reward_alpha", 1.0),
        llm_pruning_reward_beta=getattr(cfg, "llm_pruning_reward_beta", 1.0),
    )
    vlm = get_vlm(cfg.model_id, cfg.arch_specifier, vision_backbone, llm_backbone)
    vlm.to(cpu_device)
    print("Model loaded successfully")

    # Load dataset
    print("Loading dataset...")
    data_path = "/home/ubuntu/vlm_prune/RLVLM/ai2d/training_data_1024.pt"
    dataset = TorchFileDataset(data_path)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Test forward pass with one batch
    print("Testing forward pass...")
    # Create a batch with one sample
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Pixel values shape: {batch['pixel_values'].shape}")

    # Move to device
    batch = {k: v.to(cpu_device) for k, v in batch.items()}

    try:
        outputs = vlm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        print("Forward pass successful!")
        print(f"Outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    print("=== Debug test completed ===")

if __name__ == "__main__":
    main()
