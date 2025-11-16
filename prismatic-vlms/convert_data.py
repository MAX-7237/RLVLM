"""
Convert AI2D JSON dataset to PyTorch .pt format expected by TorchFileDataset.
"""

import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np


def convert_ai2d_to_pt(json_path: str, output_path: str, max_samples: int = None):
    """Convert AI2D JSON dataset to PyTorch tensor format."""

    # Load JSON data
    print(f"Loading data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert to list of examples
    examples = []
    count = 0
    for question_id, example in data.items():
        if max_samples and count >= max_samples:
            break

        examples.append(example)
        count += 1

    print(f"Loaded {len(examples)} examples")

    # Convert to the format expected by TorchFileDataset
    # We need: input_ids, attention_mask, labels, pixel_values, image_token_mask

    # For now, create dummy data with correct structure
    # In a real implementation, you would:
    # 1. Load and tokenize the text
    # 2. Load and process images
    # 3. Create proper labels and masks

    num_examples = len(examples)

    # Create dummy tensors with realistic shapes
    # These would be replaced with actual processed data
    input_ids = torch.randint(0, 32000, (num_examples, 512))  # Dummy token IDs
    attention_mask = torch.ones(num_examples, 512, dtype=torch.long)  # All tokens are attended to
    # Labels should be 2D for sequence labeling (batch_size, seq_len)
    labels = torch.randint(0, 32000, (num_examples, 512))  # Token-level labels for sequence
    labels[:, 1:] = -100  # Set non-first tokens to ignore index (typical for causal LM)
    pixel_values = torch.randn(num_examples, 3, 336, 336)  # Dummy image tensors (336x336 for CLIP ViT-L-336px)
    image_token_mask = torch.zeros(num_examples, 512, dtype=torch.bool)  # No image tokens for now

    # Create the data dictionary
    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_token_mask': image_token_mask
    }

    # Save to .pt file
    torch.save(data_dict, output_path)
    print(f"Saved converted data to {output_path}")
    print(f"Data shapes:")
    for key, tensor in data_dict.items():
        print(f"  {key}: {tensor.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert AI2D dataset to PyTorch format")
    parser.add_argument("--input-json", required=True, help="Path to input JSON file")
    parser.add_argument("--output-pt", required=True, help="Path to output .pt file")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")

    args = parser.parse_args()

    convert_ai2d_to_pt(args.input_json, args.output_pt, args.max_samples)


if __name__ == "__main__":
    main()
