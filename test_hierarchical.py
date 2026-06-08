#!/usr/bin/env python3
"""
Test script to verify hierarchical classification implementation
"""

import torch
from ultralytics.nn.modules.head import HierarchicalClassify

def test_hierarchical_classify():
    """Test the hierarchical classification head"""
    print("Testing HierarchicalClassify head...")

    # Create a hierarchical classification head with 4 levels
    levels = {
        "order": 5,      # 5 orders
        "family": 10,    # 10 families
        "genus": 20,     # 20 genera
        "species": 50    # 50 species
    }

    # Create head with 1280 input channels (like efficientnet_b0)
    head = HierarchicalClassify(c1=1280, levels=levels)

    # Create a dummy input tensor (batch_size=2, channels=1280, height=1, width=1)
    dummy_input = torch.randn(2, 1280, 1, 1)

    # Forward pass
    with torch.no_grad():
        outputs = head(dummy_input)

    print("Forward pass successful!")
    print(f"Output keys: {list(outputs.keys())}")

    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

    print("HierarchicalClassify test passed!")

if __name__ == "__main__":
    test_hierarchical_classify()