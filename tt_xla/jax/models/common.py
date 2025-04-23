"""Common utilities for model loading and manipulation."""

import os
import time
from typing import Dict, Any

import jax
import jax.numpy as jnp


def load_params_from_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Load model parameters from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint files
        
    Returns:
        Dictionary of model parameters
    """
    print(f"Loading parameters from {checkpoint_dir}")
    
    # Here we would typically load parameters from files
    # This is a placeholder implementation
    params = {}
    
    # In a real implementation, you would load parameters from files
    # For example:
    # from safetensors import safe_open
    # for filename in os.listdir(checkpoint_dir):
    #     if filename.endswith(".safetensors"):
    #         filepath = os.path.join(checkpoint_dir, filename)
    #         with safe_open(filepath, framework="flax") as f:
    #             for tensor_name in f.keys():
    #                 params[tensor_name] = f.get_tensor(tensor_name)
    
    print(f"Loaded {len(params)} parameters")
    return params 