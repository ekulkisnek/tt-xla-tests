# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration utilities for Qwen2.5-7B model.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, Mesh
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

def load_qwen_config(
    weights_path: str, mesh: Optional[Mesh] = None
) -> Dict[str, Any]:
    """Load Qwen2 configuration from JSON file and add mesh and config dict."""
    # Load configuration from JSON
    config_path = os.path.join(weights_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract key dimensions needed for JAX model
    config = {
        "hidden_size": config.get("hidden_size", 3584),
        "intermediate_size": config.get("intermediate_size", 14336),
        "num_hidden_layers": config.get("num_hidden_layers", 28),
        "num_attention_heads": config.get("num_attention_heads", 28),
        "num_key_value_heads": config.get("num_key_value_heads", 4),
        "vocab_size": config.get("vocab_size", 152064),
        "attention_bias": False,
        "tie_word_embeddings": False,
        "rope_theta": config.get("rope_theta", 10000.0),
        "initializer_range": config.get("initializer_range", 0.02),
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
    }
    
    # Set mesh if provided
    if mesh is not None:
        config["mesh"] = mesh
    
    # Calculate derived values needed for the model
    # This is critical for proper parameter loading
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    
    # Ensure these dimensions match the loaded parameters
    config["head_dim"] = head_dim
    config["qwen_attention_heads_match_actual_weights"] = True
    
    return config

def get_mesh_config(mesh_shape: Tuple[int, ...], axis_names: Tuple[str, ...] = ('batch', 'model')):
    """
    Create a mesh configuration dictionary for the given mesh shape.
    
    Args:
        mesh_shape: Shape of the device mesh, e.g. (2, 4) for 2x4
        axis_names: Names for the mesh axes
        
    Returns:
        Dictionary containing mesh configuration
    """
    return {
        'mesh_shape': mesh_shape,
        'axis_names': axis_names,
    }

def create_partition_specs(
    config: Dict[str, Any], 
    mesh_shape: Tuple[int, ...], 
    axis_names: Tuple[str, ...] = ('batch', 'model')
) -> Dict[str, P]:
    """
    Create partition specs for Qwen2.5 model parameters based on mesh configuration.
    
    Args:
        config: Model configuration
        mesh_shape: Shape of the device mesh
        axis_names: Names of the mesh axes
        
    Returns:
        Dictionary mapping parameter paths to partition specs
    """
    # Find the index of the 'model' axis
    model_axis_idx = axis_names.index('model')
    model_axis_size = mesh_shape[model_axis_idx]
    
    partition_specs = {}
    
    # Embedding layer - typically not sharded in this implementation
    partition_specs['model/embed_tokens/embedding'] = None
    
    # Layer norm parameters - not sharded
    for i in range(config['num_hidden_layers']):
        partition_specs[f'model/layers_{i}/input_layernorm/weight'] = None
        partition_specs[f'model/layers_{i}/post_attention_layernorm/weight'] = None
    
    partition_specs['model/norm/weight'] = None
    
    # Attention parameters
    for i in range(config['num_hidden_layers']):
        # Attention projection matrices are sharded along the head dimension
        # Query projection - sharded across model parallel dimension
        partition_specs[f'model/layers_{i}/self_attn/q_proj/kernel'] = P(None, 'model')
        partition_specs[f'model/layers_{i}/self_attn/q_proj/bias'] = P('model',)
        
        # Key projection - sharded across model parallel dimension
        partition_specs[f'model/layers_{i}/self_attn/k_proj/kernel'] = P(None, 'model')
        partition_specs[f'model/layers_{i}/self_attn/k_proj/bias'] = P('model',)
        
        # Value projection - sharded across model parallel dimension
        partition_specs[f'model/layers_{i}/self_attn/v_proj/kernel'] = P(None, 'model')
        partition_specs[f'model/layers_{i}/self_attn/v_proj/bias'] = P('model',)
        
        # Output projection - sharded across model parallel dimension in the opposite direction
        partition_specs[f'model/layers_{i}/self_attn/o_proj/kernel'] = P('model', None)
        partition_specs[f'model/layers_{i}/self_attn/o_proj/bias'] = None
        
        # MLP parameters
        # Gate and up projections are sharded across model parallel dimension
        partition_specs[f'model/layers_{i}/mlp/gate_proj/kernel'] = P(None, 'model')
        partition_specs[f'model/layers_{i}/mlp/gate_proj/bias'] = P('model',)
        
        partition_specs[f'model/layers_{i}/mlp/up_proj/kernel'] = P(None, 'model')
        partition_specs[f'model/layers_{i}/mlp/up_proj/bias'] = P('model',)
        
        # Down projection - sharded across model parallel dimension in the opposite direction
        partition_specs[f'model/layers_{i}/mlp/down_proj/kernel'] = P('model', None)
        partition_specs[f'model/layers_{i}/mlp/down_proj/bias'] = None
    
    # Language modeling head - typically not sharded
    partition_specs['lm_head/kernel'] = None
    
    return partition_specs

def create_device_mesh(
    mesh_shape: Tuple[int, ...], 
    axis_names: Tuple[str, ...] = ('batch', 'model')
) -> jax.sharding.Mesh:
    """
    Create a device mesh for tensor parallelism.
    
    Args:
        mesh_shape: Tuple specifying the shape of the mesh, e.g., (2, 4) for 2x4
        axis_names: Names for the mesh axes, typically ('batch', 'model')
        
    Returns:
        A device mesh for use in tensor parallelism
    """
    # Get available devices
    devices = jax.devices('cpu')
    
    # Calculate required number of devices
    required_devices = np.prod(mesh_shape)
    
    # Check if we have enough devices
    if len(devices) < required_devices:
        # Allow simulation by setting XLA_FLAGS
        if "XLA_FLAGS" not in os.environ:
            print(f"Warning: Not enough devices ({len(devices)}) for mesh shape {mesh_shape}.")
            print(f"To simulate more devices, run with: XLA_FLAGS='--xla_force_host_platform_device_count={required_devices}'")
            print("Continuing with available devices...")
    
    # Create device mesh using JAX utilities
    try:
        # Use JAX's mesh_utils to create the device mesh
        import jax.experimental.mesh_utils as mesh_utils
        device_mesh = mesh_utils.create_device_mesh(mesh_shape)
        return jax.sharding.Mesh(device_mesh, axis_names)
    except ValueError as e:
        # If mesh creation fails, raise a more informative error
        raise ValueError(
            f"Failed to create device mesh with shape {mesh_shape}. "
            f"Available devices: {len(devices)}. Required: {required_devices}. "
            f"Use XLA_FLAGS='--xla_force_host_platform_device_count={required_devices}' "
            f"to simulate more devices. Original error: {str(e)}"
        )

def supported_mesh_configs():
    """Return a list of supported mesh configurations for Qwen2.5-7B."""
    return [
        {'shape': (2, 4), 'axis_names': ('batch', 'model'), 'description': '2x4 mesh with 8 devices total'},
        {'shape': (1, 8), 'axis_names': ('batch', 'model'), 'description': '1x8 mesh with 8 devices total'},
        {'shape': (1, 32), 'axis_names': ('batch', 'model'), 'description': '1x32 mesh with 32 devices total'},
        {'shape': (8, 4), 'axis_names': ('batch', 'model'), 'description': '8x4 mesh with 32 devices total'},
    ]

def get_qwen2_7b_config() -> Dict[str, Any]:
    """
    Get the configuration for Qwen2.5-7B model.
    
    Returns:
        Dict: Configuration for the Qwen2.5-7B model
    """
    return {
        "vocab_size": 152064,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,  # GQA with 4 KV heads
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": False,
        "rope_theta": 1000000.0,
        "attention_dropout": 0.0,
    }

def get_qwen2_7b_gqa_config() -> Dict[str, Any]:
    """
    Get the configuration for Qwen2.5-7B model with grouped-query attention.
    
    Returns:
        Dict: Configuration for the Qwen2.5-7B model with GQA
    """
    config = get_qwen2_7b_config()
    # Already has GQA with 4 KV heads
    return config

def get_small_config(hidden_size: int = 16, num_layers: int = 2) -> Dict[str, Any]:
    """
    Get a small configuration for testing.
    
    Args:
        hidden_size: Size of the hidden layers
        num_layers: Number of transformer blocks
        
    Returns:
        Dict: Configuration for a small model
    """
    return {
        "vocab_size": 151936,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 3,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
    } 