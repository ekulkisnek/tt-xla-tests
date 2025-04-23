#!/usr/bin/env python3
"""
Qwen 2.5 Model Tensor Parallel Implementation in JAX/Flax

This file provides a complete implementation of the Qwen 2.5 model
with tensor parallelism using JAX and Flax. The implementation supports
loading weights from HuggingFace safetensors files and running on multiple devices.

Usage:
python qwen25_tp_model.py --weights_dir /path/to/qwen25-weights --mesh_shape 1,8
"""

import os
import re
import json
import math
import time
import logging
import argparse
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.experimental.pjit import with_sharding_constraint
from jax.sharding import PartitionSpec as P

import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training.common_utils import shard_prng_key

# Import safetensors for weight loading
try:
    from safetensors import safe_open
    from safetensors.flax import load_file as safe_load_file
except ImportError:
    raise ImportError("safetensors package is required. Install with: pip install safetensors")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

###################
# Configuration
###################

class Qwen25Config:
    """Configuration class for Qwen 2.5 model."""
    
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        sliding_window: int = 131072,
        max_window_layers: int = 28,
        use_sliding_window: bool = False,
        use_mrope: bool = False,
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        pad_token_id: int = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window
        self.use_mrope = use_mrope
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        # Calculate head dimensions explicitly to avoid inconsistencies
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_head_dim = self.hidden_size // self.num_attention_heads  # Same dimension for K/V heads
        
        # Set up partitioning strategy for tensor parallelism
        # These will be used for parameter and activation sharding
        self.mesh_axes = {
            'hidden': 'model',  # Partition hidden dimension
            'heads': 'model',   # Partition attention heads
            'kv_heads': 'model' # Partition KV heads for GQA
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen25Config":
        """Create a configuration from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, json_file: Union[str, Path]) -> "Qwen25Config":
        """Load configuration from a JSON file."""
        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
    
    def __repr__(self) -> str:
        return f"Qwen25Config({', '.join(f'{k}={v}' for k, v in self.to_dict().items())})"


###################
# Utility Functions
###################

def create_sinusoidal_positions(num_pos: int, dim: int, theta: float = 10000.0) -> jnp.ndarray:
    """
    Create sinusoidal position embeddings.
    
    Args:
        num_pos: Maximum number of positions
        dim: Dimension of the embeddings
        theta: Base value for frequencies
        
    Returns:
        Sinusoidal position embeddings of shape (num_pos, dim)
    """
    # Create position indices
    positions = jnp.arange(0, num_pos, dtype=jnp.float32)
    
    # Create frequency indices
    half_dim = dim // 2
    freq_indices = jnp.arange(0, half_dim, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (freq_indices / half_dim))
    
    # Create position embeddings
    sinusoid_inp = jnp.einsum("i,j->ij", positions, inv_freq)
    
    # Apply sin and cos
    sin_pos = jnp.sin(sinusoid_inp)
    cos_pos = jnp.cos(sinusoid_inp)
    
    # Interleave sin and cos
    sincos = jnp.stack([sin_pos, cos_pos], axis=-1).reshape(num_pos, -1)
    
    return sincos


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate half the hidden dims of the input.
    
    Args:
        x: Input tensor of shape (..., d)
        
    Returns:
        Rotated tensor where the second half of hidden dimensions are negated and 
        swapped with the first half.
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, 
                         sinu_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads/kv_heads, seq_len, head_dim)
        sinu_pos: Sinusoidal position embeddings (seq_len, head_dim)
        
    Returns:
        q, k: Tensors with rotary position embeddings applied
    """
    # Extract sin and cos components
    sin, cos = jnp.split(sinu_pos, 2, axis=-1)
    
    # Make sin and cos match the query and key shapes
    sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim/2)
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim/2)
    
    # Duplicate for full head dimension
    sin = jnp.concatenate([sin, sin], axis=-1)  # (1, 1, seq_len, head_dim)
    cos = jnp.concatenate([cos, cos], axis=-1)  # (1, 1, seq_len, head_dim)
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def make_causal_mask(attention_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Create a causal mask from an attention mask.
    
    Args:
        attention_mask: Boolean mask of shape (batch_size, seq_len)
        
    Returns:
        Causal attention mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_length = attention_mask.shape
    
    # Create a causal mask that blocks attending to future tokens
    causal_mask = jnp.triu(
        jnp.ones((seq_length, seq_length), dtype=jnp.bool_), k=1
    )
    
    # Combine with the attention mask (1 = attend, 0 = mask)
    # attention_mask has shape (batch_size, seq_length)
    causal_mask = jnp.logical_or(
        causal_mask[None, None, :, :],
        (1 - attention_mask[:, None, None, :]).astype(jnp.bool_)
    )
    
    # Convert to float
    causal_mask = jnp.where(causal_mask, -1e10, 0.0)
    
    return causal_mask 


###################
# Weight Loading
###################

def create_mesh_from_string(mesh_shape_str: str) -> Mesh:
    """
    Create a device mesh from a string representation.
    
    Args:
        mesh_shape_str: String representation of mesh shape (e.g. "1,8")
        
    Returns:
        JAX device mesh
    """
    # Parse mesh shape from string
    mesh_shape = tuple(int(x) for x in mesh_shape_str.split(","))
    
    # Calculate total number of devices needed
    total_devices = np.prod(mesh_shape)
    
    # Get available devices
    devices = jax.devices()
    
    if len(devices) < total_devices:
        raise ValueError(
            f"Requested mesh shape {mesh_shape} requires {total_devices} devices, "
            f"but only {len(devices)} devices are available."
        )
    
    # Create mesh
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    
    # Create axis names
    axis_names = ["batch", "model"][:len(mesh_shape)]
    
    return Mesh(device_mesh, axis_names)


def map_tensor_parallelism(params: Dict, mesh: Mesh, config: Qwen25Config) -> Dict:
    """
    Map model parameters to tensor parallel devices.
    
    Args:
        params: Model parameters
        mesh: Device mesh
        config: Model configuration
        
    Returns:
        Parameters with tensor parallel mapping
    """
    # Flatten parameters for easier processing
    flat_params = flatten_dict(params)
    
    # Create new dictionary for sharded parameters
    sharded_flat_params = {}
    
    # Process each parameter
    for path, param in flat_params.items():
        # Default to no sharding
        pspec = P()
        
        # Check for embedding parameters
        if "embed_tokens" in path:
            # Partition embedding on vocab dim (not hidden dim)
            pspec = P(None, "model")
        
        # Check for attention query projection
        elif "attention" in path and "q_proj" in path and "kernel" in path:
            # Partition attention heads
            pspec = P(None, "model")
        
        # Check for attention key/value projections
        elif "attention" in path and ("k_proj" in path or "v_proj" in path) and "kernel" in path:
            # Partition key/value heads
            pspec = P(None, "model")
        
        # Check for attention output projection
        elif "attention" in path and "o_proj" in path and "kernel" in path:
            # Partition on hidden dimension
            pspec = P("model", None)
        
        # Check for MLP gate/up projections
        elif "mlp" in path and ("gate_proj" in path or "up_proj" in path) and "kernel" in path:
            # Partition on output dimension
            pspec = P(None, "model")
        
        # Check for MLP down projection
        elif "mlp" in path and "down_proj" in path and "kernel" in path:
            # Partition on input dimension
            pspec = P("model", None)
        
        # Check for LM head
        elif "lm_head" in path and "kernel" in path:
            # Partition on input dimension
            pspec = P("model", None)
        
        # Create named sharding with appropriate partition spec
        named_sharding = NamedSharding(mesh, pspec)
        
        # Apply sharding to parameter
        sharded_param = jax.device_put(param, named_sharding)
        
        # Add to new parameters dictionary
        sharded_flat_params[path] = sharded_param
    
    # Unflatten parameters
    sharded_params = unflatten_dict(sharded_flat_params)
    
    return sharded_params


def map_safetensors_to_flax(key: str) -> Tuple:
    """
    Map a safetensors parameter key to a Flax parameter path.
    
    Args:
        key: Parameter key from safetensors
        
    Returns:
        Tuple representing the path in Flax parameters
    """
    # Define commonly used mappings
    common_mappings = {
        "self_attn.q_proj": "attention.q_proj",
        "self_attn.k_proj": "attention.k_proj",
        "self_attn.v_proj": "attention.v_proj",
        "self_attn.o_proj": "attention.o_proj",
    }
    
    # Process embeddings
    if key == "model.embed_tokens.weight":
        return ("params", "model", "embed_tokens", "embedding")
    
    # Process final layernorm
    elif key == "model.norm.weight":
        return ("params", "model", "norm", "scale")
    
    # Process LM head
    elif key == "lm_head.weight":
        return ("params", "lm_head", "kernel")
    
    # Process transformer layers
    elif key.startswith("model.layers."):
        # Extract layer index
        match = re.match(r"model\.layers\.(\d+)\.(.*)", key)
        if match:
            layer_idx, remainder = match.groups()
            
            # Map remainder to Flax structure
            for safetensors_key, flax_key in common_mappings.items():
                if remainder.startswith(safetensors_key):
                    param_type = remainder.split(".")[-1]  # weight or bias
                    flax_param = "kernel" if param_type == "weight" else param_type
                    return ("params", "model", "layers", f"{layer_idx}", flax_key, flax_param)
            
            # Handle layernorms
            if remainder.startswith("input_layernorm.weight"):
                return ("params", "model", "layers", f"{layer_idx}", "input_layernorm", "scale")
            elif remainder.startswith("post_attention_layernorm.weight"):
                return ("params", "model", "layers", f"{layer_idx}", "post_attention_layernorm", "scale")
            
            # Handle MLP
            elif remainder.startswith("mlp."):
                mlp_part = remainder.split(".")
                if len(mlp_part) >= 3:
                    mlp_type = mlp_part[1]  # gate_proj, up_proj, down_proj
                    param_type = mlp_part[2]  # weight or bias
                    flax_param = "kernel" if param_type == "weight" else param_type
                    return ("params", "model", "layers", f"{layer_idx}", "mlp", mlp_type, flax_param)
    
    # If no match found, return the key as is
    return key


def convert_safetensors_to_flax(weights: Dict) -> Dict:
    """
    Convert safetensors weights to Flax parameter structure.
    
    Args:
        weights: Dictionary of weights from safetensors
        
    Returns:
        Dictionary with weights in Flax structure
    """
    # Initialize output dictionary
    flax_params = {}
    
    # Process each parameter
    for key, tensor in weights.items():
        # Map key to Flax path
        flax_path = map_safetensors_to_flax(key)
        
        # If key was not mapped successfully, skip
        if isinstance(flax_path, str):
            logger.warning(f"Could not map parameter {key} to Flax structure")
            continue
        
        # For kernels, transpose to match Flax convention
        if flax_path[-1] == "kernel":
            # Process attention heads specially for GQA
            if "q_proj" in flax_path or "k_proj" in flax_path or "v_proj" in flax_path:
                tensor = tensor.T
            # Standard dense kernels
            elif "mlp" in flax_path or "lm_head" in flax_path or "o_proj" in flax_path:
                tensor = tensor.T
        
        # Add to Flax parameters
        # Need to build the nested structure
        current = flax_params
        for i, path_part in enumerate(flax_path):
            if i == len(flax_path) - 1:
                current[path_part] = tensor
            else:
                if path_part not in current:
                    current[path_part] = {}
                current = current[path_part]
    
    return flax_params


def validate_and_reshape_tensors(
    params: Dict, model_params: Dict, config: Qwen25Config
) -> Dict:
    """
    Validate and reshape tensors to match model parameter shapes.
    
    Args:
        params: Loaded parameters
        model_params: Model parameters
        config: Model configuration
        
    Returns:
        Validated and reshaped parameters
    """
    # Flatten dictionaries for easier processing
    flat_params = flatten_dict(params)
    flat_model_params = flatten_dict(model_params)
    
    # Create new dictionary for validated parameters
    validated_params = {}
    
    # Check for missing and extra parameters
    missing_params = [k for k in flat_model_params if k not in flat_params]
    extra_params = [k for k in flat_params if k not in flat_model_params]
    
    if missing_params:
        logger.warning(f"Missing {len(missing_params)} parameters: {missing_params[:5]} ...")
    
    if extra_params:
        logger.warning(f"Extra {len(extra_params)} parameters: {extra_params[:5]} ...")
    
    # Process each parameter
    for path, param in flat_params.items():
        # Check if parameter exists in model
        if path not in flat_model_params:
            continue
        
        # Get expected shape
        expected_shape = flat_model_params[path].shape
        
        # Check for shape mismatch
        if param.shape != expected_shape:
            logger.warning(
                f"Shape mismatch for {path}: expected {expected_shape}, got {param.shape}"
            )
            
            # If it's a simple transposition issue, fix it
            if len(param.shape) == 2 and param.shape[::-1] == expected_shape:
                param = param.T
                logger.info(f"Transposed parameter {path}")
            
            # If shapes still don't match, try more complex reshaping
            if param.shape != expected_shape:
                try:
                    # For GQA, may need special handling for query/key/value projections
                    if ("attention" in str(path)) and (
                        "q_proj" in str(path) or "k_proj" in str(path) or "v_proj" in str(path)
                    ):
                        # Attention projections may need reshaping for GQA
                        param = handle_attention_projection_reshaping(
                            param, expected_shape, path, config
                        )
                    else:
                        # Try general reshaping if sizes match
                        if np.prod(param.shape) == np.prod(expected_shape):
                            param = param.reshape(expected_shape)
                            logger.info(f"Reshaped parameter {path}")
                        else:
                            # If sizes don't match, create a new tensor and warn
                            logger.warning(
                                f"Cannot reshape {path}: {param.shape} to {expected_shape}. "
                                f"Using random initialization."
                            )
                            param = jnp.zeros(expected_shape)
                except Exception as e:
                    logger.error(f"Error reshaping {path}: {e}")
                    # Use a zero tensor as fallback
                    param = jnp.zeros(expected_shape)
        
        # Add validated parameter
        validated_params[path] = param
    
    # Check for completely missing parameters and use zeros
    for path, param in flat_model_params.items():
        if path not in validated_params:
            logger.warning(f"Missing parameter {path}, using zeros")
            validated_params[path] = jnp.zeros_like(param)
    
    # Unflatten validated parameters
    return unflatten_dict(validated_params)


def handle_attention_projection_reshaping(
    param: jnp.ndarray, expected_shape: Tuple, path: Tuple, config: Qwen25Config
) -> jnp.ndarray:
    """
    Special handling for attention projection reshaping with GQA.
    
    Args:
        param: Parameter tensor
        expected_shape: Expected shape
        path: Parameter path
        config: Model configuration
        
    Returns:
        Reshaped parameter
    """
    # Extract query/key/value part from path
    if "q_proj" in str(path):
        is_query = True
        num_heads = config.num_attention_heads
    elif "k_proj" in str(path) or "v_proj" in str(path):
        is_query = False
        num_heads = config.num_key_value_heads
    else:
        # If not a query/key/value projection, reshape normally
        return param.reshape(expected_shape)
    
    head_dim = config.head_dim
    hidden_size = config.hidden_size
    
    # Calculate shapes
    if is_query:
        # Query projection should output (hidden_size, num_heads * head_dim)
        expected_out_dim = num_heads * head_dim
    else:
        # Key/value projection should output (hidden_size, num_kv_heads * head_dim)
        expected_out_dim = num_heads * head_dim
    
    # Check if we need to reshape
    if param.shape[1] != expected_out_dim:
        # Handle GQA case
        if is_query and param.shape[1] < expected_out_dim:
            # If loaded query projection is smaller than expected, expand
            # This can happen if loading from a model with fewer query heads
            logger.warning(
                f"Expanding query projection from {param.shape} to {expected_shape}"
            )
            
            # Calculate how many times to repeat
            repeat_factor = expected_out_dim // param.shape[1]
            
            # Repeat the projection weights
            param = jnp.repeat(param, repeat_factor, axis=1)
            
        elif not is_query and param.shape[1] > expected_out_dim:
            # If loaded key/value projection is larger than expected, reduce
            # This can happen if loading from a model with more key/value heads
            logger.warning(
                f"Reducing key/value projection from {param.shape} to {expected_shape}"
            )
            
            # Take only needed part
            param = param[:, :expected_out_dim]
            
        elif not is_query and param.shape[1] < expected_out_dim:
            # If loaded key/value projection is smaller than expected, expand
            logger.warning(
                f"Expanding key/value projection from {param.shape} to {expected_shape}"
            )
            
            # Calculate how many times to repeat
            repeat_factor = expected_out_dim // param.shape[1]
            
            # Repeat the projection weights
            param = jnp.repeat(param, repeat_factor, axis=1)
    
    # Ensure final shape matches expected shape
    if param.shape != expected_shape:
        param = param.reshape(expected_shape)
    
    return param


def load_and_validate_flax_params(
    safetensors_dir: str, model_params: Dict, config: Qwen25Config
) -> Dict:
    """
    Load safetensors weights, convert to Flax and validate.
    
    Args:
        safetensors_dir: Directory containing safetensors files
        model_params: Model parameters
        config: Model configuration
        
    Returns:
        Validated Flax parameters
    """
    # Find all safetensors files
    file_pattern = os.path.join(safetensors_dir, "*.safetensors")
    safetensors_files = sorted(glob.glob(file_pattern))
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found at {safetensors_dir}")
    
    logger.info(f"Found {len(safetensors_files)} safetensors files")
    
    # Check for index file
    index_file = os.path.join(safetensors_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        logger.info(f"Found safetensors index file: {index_file}")
        try:
            with open(index_file, "r") as f:
                index = json.load(f)
                weight_map = index.get("weight_map", {})
                if weight_map:
                    logger.info(f"Index file contains {len(weight_map)} parameter mappings")
        except Exception as e:
            logger.warning(f"Error reading index file: {e}")
    
    # Initialize weights dictionary
    weights = {}
    
    # Load weights from all files
    for file_path in safetensors_files:
        logger.info(f"Loading weights from {file_path}")
        try:
            # Load file
            file_weights = safe_load_file(file_path)
            
            # Add weights to dictionary
            weights.update(file_weights)
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    logger.info(f"Loaded {len(weights)} parameters from safetensors files")
    
    # Convert to Flax structure
    logger.info("Converting to Flax parameter structure")
    flax_params = convert_safetensors_to_flax(weights)
    
    # Clean up raw weights to save memory
    del weights
    gc.collect()
    
    # Validate and reshape parameters
    logger.info("Validating parameters")
    validated_params = validate_and_reshape_tensors(flax_params, model_params, config)
    
    return validated_params


def load_qwen_model(
    weights_dir: str,
    mesh_shape: str = "1,8",
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: Optional[jnp.dtype] = None
) -> Tuple[TPQwenForCausalLM, Dict]:
    """
    Load Qwen model with tensor parallelism.
    
    Args:
        weights_dir: Directory containing weights
        mesh_shape: Device mesh shape (e.g. "1,8")
        dtype: Data type for model
        param_dtype: Data type for parameters (defaults to dtype)
        
    Returns:
        Tuple of model and parameters
    """
    # Use param_dtype = dtype if not specified
    if param_dtype is None:
        param_dtype = dtype
    
    # Load config
    config_path = os.path.join(weights_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")
    
    logger.info(f"Loading config from {config_path}")
    config = Qwen25Config.from_json_file(config_path)
    
    # Create device mesh
    logger.info(f"Creating device mesh with shape {mesh_shape}")
    mesh = create_mesh_from_string(mesh_shape)
    
    # Initialize a base model to get parameter structure
    logger.info("Initializing model with random weights")
    model = TPQwenForCausalLM(
        config=config,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    
    # Create initialization inputs
    batch_size = 1
    seq_length = 8
    inputs = {
        "input_ids": jnp.ones((batch_size, seq_length), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_length), dtype=jnp.float32),
    }
    
    # Initialize parameters
    logger.info("Generating parameter structure")
    with mesh:
        # Use PRNG key
        key = jax.random.PRNGKey(0)
        variables = model.init(key, **inputs)
    
    # Get model parameters
    model_params = variables["params"]
    
    # Load and validate parameters
    logger.info(f"Loading weights from {weights_dir}")
    params = load_and_validate_flax_params(weights_dir, model_params, config)
    
    # Apply tensor parallelism to parameters
    logger.info("Mapping parameters for tensor parallelism")
    with mesh:
        params = map_tensor_parallelism(params, mesh, config)
    
    return model, params


###################
# Model Components
###################

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a simpler version of LayerNorm that only normalizes
    standard deviation but not mean.
    """
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Scale parameter (no bias in RMSNorm)
        self.weight = self.param(
            "scale",
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMS normalization to input tensor."""
        # Cast input to working dtype
        x = x.astype(self.dtype)
        
        # Calculate RMS
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jnp.reciprocal(jnp.sqrt(variance + self.eps))
        
        # Apply scale parameter (weight) and convert to output dtype
        return (self.weight * x).astype(self.dtype)


class QwenMLP(nn.Module):
    """
    Multi-Layer Perceptron for Qwen2.5 with tensor parallelism.
    
    Uses SwiGLU activation function: x * silu(gate(x)).
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Extract dimensions
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        
        # Gate projection
        self.gate_proj = nn.Dense(
            features=intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
            name="gate_proj",
        )
        
        # Up projection
        self.up_proj = nn.Dense(
            features=intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
            name="up_proj",
        )
        
        # Down projection
        self.down_proj = nn.Dense(
            features=hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
            name="down_proj",
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply MLP to input tensor with SwiGLU activation."""
        # Gate and up projections
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # SwiGLU activation: x * silu(gate(x))
        intermediate = nn.silu(gate_output) * up_output
        
        # Add explicit tensor sharding constraint for tensor parallelism
        intermediate = with_sharding_constraint(
            intermediate, P("batch", "length", "model")
        )
        
        # Down projection
        output = self.down_proj(intermediate)
        
        return output


class QwenAttention(nn.Module):
    """
    Multi-headed attention for Qwen2.5 with tensor parallelism.
    
    Supports Grouped Query Attention (GQA) where the number of key/value heads
    can be fewer than the number of query heads.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    causal: bool = True
    
    def setup(self):
        # Extract dimensions
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        
        # Query projection
        self.q_proj = nn.Dense(
            features=num_heads * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
            name="q_proj",
        )
        
        # Key projection
        self.k_proj = nn.Dense(
            features=num_kv_heads * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
            name="k_proj",
        )
        
        # Value projection
        self.v_proj = nn.Dense(
            features=num_kv_heads * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
            name="v_proj",
        )
        
        # Output projection
        self.o_proj = nn.Dense(
            features=hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
            name="o_proj",
        )
        
        # Save dimensions for reference
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Calculate ratio for GQA - how many query heads per key/value head
        if num_kv_heads < num_heads:
            self.gqa_ratio = num_heads // num_kv_heads
        else:
            self.gqa_ratio = 1
    
    def _split_heads(self, x: jnp.ndarray, n_heads: int) -> jnp.ndarray:
        """Split hidden dim into heads and head_dim."""
        batch, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(batch, seq_len, n_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
    
    def _merge_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """Merge heads back into hidden dim."""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, head_dim)
        return x.reshape(batch, seq_len, -1)
    
    def apply_gqa_heads_expansion(self, kv: jnp.ndarray) -> jnp.ndarray:
        """
        Expand key/value heads to match query heads for GQA.
        
        Args:
            kv: Key or value tensor of shape (batch, kv_heads, seq_len, head_dim)
            
        Returns:
            Expanded tensor of shape (batch, heads, seq_len, head_dim)
        """
        if self.gqa_ratio == 1:
            return kv
        
        # Reshape to prepare for repeat pattern
        batch, kv_heads, seq_len, head_dim = kv.shape
        
        # Expand using proper tile operation
        kv = jnp.tile(kv, [1, self.gqa_ratio, 1, 1])
        
        # Apply explicit tensor sharding constraint
        kv = with_sharding_constraint(kv, P("batch", "model", "length", None))
        
        return kv
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Apply multi-headed attention to input.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
            position_ids: Position IDs of shape (batch, seq_len)
            past_key_value: Cached key/value states for generation
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value states for generation
            deterministic: Whether to use deterministic behavior (no dropout)
            
        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size)
            past_key_value: Optionally, key/value states for generation
            attention_weights: Optionally, attention weights
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        
        # Project hidden states to query, key, and value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Split hidden dimension into heads and head_dim
        query_states = self._split_heads(query_states, self.num_heads)
        key_states = self._split_heads(key_states, self.num_kv_heads)
        value_states = self._split_heads(value_states, self.num_kv_heads)
        
        # Apply explicit tensor sharding constraints for tensor parallelism
        query_states = with_sharding_constraint(query_states, P("batch", "model", "length", None))
        key_states = with_sharding_constraint(key_states, P("batch", "model", "length", None))
        value_states = with_sharding_constraint(value_states, P("batch", "model", "length", None))
        
        # Handle KV caching for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=2)
            value_states = jnp.concatenate([past_value, value_states], axis=2)
        
        # Cache key and value for later use if needed
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # Apply rotary position embeddings
        if position_ids is not None:
            # Create sinusoidal position embeddings
            max_position = self.config.max_position_embeddings
            sinu_pos = create_sinusoidal_positions(
                max_position, self.head_dim, self.config.rope_theta
            )
            
            # Extract the positions we need
            position_embeds = jnp.take(sinu_pos, position_ids, axis=0)
            
            # Apply rotary embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, position_embeds
            )
        
        # Expand key and value heads if using GQA
        if self.num_kv_heads < self.num_heads:
            key_states = self.apply_gqa_heads_expansion(key_states)
            value_states = self.apply_gqa_heads_expansion(value_states)
        
        # Compute attention weights and perform attention
        query_length = query_states.shape[2]
        key_length = key_states.shape[2]
        
        # Handle attention masking
        if attention_mask is None:
            # Create a simple causal mask if none provided
            attention_mask = jnp.ones((batch_size, query_length))
            
        # Convert to float32 for stability in attention math
        attn_weights = jnp.matmul(query_states, jnp.swapaxes(key_states, -1, -2)) / jnp.sqrt(self.head_dim)
        
        # Add attention mask
        attn_weights = attn_weights + attention_mask
        
        # Apply softmax and handle dropout
        attn_weights = nn.softmax(attn_weights, axis=-1)
        
        if not deterministic and self.config.attention_dropout > 0:
            attn_weights = nn.dropout(
                attn_weights, rate=self.config.attention_dropout, deterministic=False
            )
        
        # Apply attention weights to values
        attn_output = jnp.matmul(attn_weights, value_states)
        
        # Apply sharding constraint after attention
        attn_output = with_sharding_constraint(attn_output, P("batch", "model", "length", None))
        
        # Merge heads back into hidden dimension
        attn_output = self._merge_heads(attn_output)
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        # Prepare return values
        outputs = (attn_output,)
        
        if use_cache:
            outputs = outputs + (past_key_value,)
            
        if output_attentions:
            outputs = outputs + (attn_weights,)
            
        return outputs 


class QwenDecoderLayer(nn.Module):
    """
    Decoder layer for Qwen2.5 with tensor parallelism.
    
    This layer includes:
    1. Input layernorm
    2. Self-attention
    3. Post-attention layernorm
    4. MLP (SwiGLU)
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Layernorms
        self.input_layernorm = RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        self.post_attention_layernorm = RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        # Self-attention
        self.attention = QwenAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        # MLP
        self.mlp = QwenMLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Apply transformer decoder layer to input.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
            position_ids: Position IDs of shape (batch, seq_len)
            past_key_value: Cached key/value states for generation
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value states for generation
            deterministic: Whether to use deterministic behavior (no dropout)
            
        Returns:
            hidden_states: Output tensor
            past_key_value: Optionally, key/value states for generation
            attention_weights: Optionally, attention weights
        """
        # Residual connection 1
        residual = hidden_states
        
        # Apply norm before attention (pre-layernorm architecture)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply self-attention
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        # Extract attention output and optional cache/weights
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:] if len(attn_outputs) > 1 else tuple()
        
        # Apply tensor sharding to attention output
        attn_output = with_sharding_constraint(attn_output, P("batch", "length", "model"))
        
        # Add back residual connection
        hidden_states = attn_output + residual
        
        # Residual connection 2
        residual = hidden_states
        
        # Apply norm before MLP (pre-layernorm architecture)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP
        feed_forward_output = self.mlp(hidden_states)
        
        # Apply tensor sharding to MLP output
        feed_forward_output = with_sharding_constraint(feed_forward_output, P("batch", "length", "model"))
        
        # Add back residual connection
        hidden_states = feed_forward_output + residual
        
        # Ensure output has proper sharding
        hidden_states = with_sharding_constraint(hidden_states, P("batch", "length", "model"))
        
        # Add hidden states to outputs
        outputs = (hidden_states,) + outputs
        
        return outputs


class QwenModel(nn.Module):
    """
    Qwen2.5 base transformer model with tensor parallelism.
    
    Includes token embedding layer and transformer decoder layers,
    but no language modeling head.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Embeddings
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=nn.initializers.normal(self.config.initializer_range),
        )
        
        # Decoder layers
        self.layers = [
            QwenDecoderLayer(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        
        # Final layernorm
        self.norm = RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply Qwen model to input tokens.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            position_ids: Position IDs of shape (batch, seq_len)
            past_key_values: Cached key/value states for generation
            output_attentions: Whether to return all attention weights
            output_hidden_states: Whether to return all hidden states
            use_cache: Whether to return key/value states for generation
            deterministic: Whether to use deterministic behavior (no dropout)
            
        Returns:
            Dictionary with model outputs:
                - last_hidden_state: Final hidden states
                - past_key_values: Optionally, key/value states for generation
                - hidden_states: Optionally, all hidden states
                - attentions: Optionally, all attention weights
        """
        batch_size, seq_length = input_ids.shape
        
        # Default values for optional inputs
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)
            
        if position_ids is None:
            position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None, :]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_length))
        
        # Create causal mask from attention mask
        causal_mask = make_causal_mask(attention_mask)
        
        # Convert input IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply tensor sharding constraint to embeddings
        hidden_states = with_sharding_constraint(hidden_states, P("batch", "length", "model"))
        
        # Lists to store all hidden states and attention weights if needed
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        
        # Apply each layer in sequence
        for i, layer in enumerate(self.layers):
            # Add to all_hidden_states if needed
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get past_key_value for this layer if available
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Apply layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                deterministic=deterministic,
            )
            
            # Unpack layer outputs
            hidden_states = layer_outputs[0]
            
            # Add to caches if needed
            if use_cache:
                all_past_key_values = all_past_key_values + (layer_outputs[1],)
                
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)
        
        # Apply final layernorm
        hidden_states = self.norm(hidden_states)
        
        # Add to all_hidden_states if needed
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Prepare outputs dictionary
        outputs = {
            "last_hidden_state": hidden_states,
        }
        
        if use_cache:
            outputs["past_key_values"] = all_past_key_values
            
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
            
        if output_attentions:
            outputs["attentions"] = all_attentions
            
        return outputs


class QwenForCausalLM(nn.Module):
    """
    Qwen2.5 model with a language modeling head.
    
    Includes the base Qwen2.5 model with tensor parallelism and
    adds a language modeling head on top.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Base Qwen model
        self.model = QwenModel(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        # Language modeling head - weighted tied with embeddings
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
        )
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply Qwen model with language modeling head to input tokens.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            position_ids: Position IDs of shape (batch, seq_len)
            past_key_values: Cached key/value states for generation
            output_attentions: Whether to return all attention weights
            output_hidden_states: Whether to return all hidden states
            use_cache: Whether to return key/value states for generation
            deterministic: Whether to use deterministic behavior (no dropout)
            
        Returns:
            Dictionary with model outputs:
                - logits: Output logits
                - past_key_values: Optionally, key/value states for generation
                - hidden_states: Optionally, all hidden states
                - attentions: Optionally, all attention weights
        """
        # Apply base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        # Extract hidden states
        hidden_states = outputs["last_hidden_state"]
        
        # Apply LM head to predict next token logits
        logits = self.lm_head(hidden_states)
        
        # Add logits to outputs
        outputs["logits"] = logits
        
        return outputs


class TPQwenForCausalLM(nn.Module):
    """
    Tensor Parallel version of Qwen2.5 for Causal Language Modeling.
    
    This is a wrapper around QwenForCausalLM that handles sharding
    for tensor parallelism across multiple devices.
    """
    config: Qwen25Config
    mesh: Mesh
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Create the inner model
        self.model = QwenForCausalLM(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
    
    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
        params: Optional[Dict] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply tensor parallel Qwen model with language modeling head.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            position_ids: Position IDs of shape (batch, seq_len)
            past_key_values: Cached key/value states for generation
            output_attentions: Whether to return all attention weights
            output_hidden_states: Whether to return all hidden states
            use_cache: Whether to return key/value states for generation
            deterministic: Whether to use deterministic behavior (no dropout)
            params: Optional override for model parameters
            
        Returns:
            Dictionary with model outputs:
                - logits: Output logits
                - past_key_values: Optionally, key/value states for generation
                - hidden_states: Optionally, all hidden states
                - attentions: Optionally, all attention weights
        """
        # Ensure input has proper sharding constraints
        input_ids = with_sharding_constraint(input_ids, P("batch", "length"))
        
        if attention_mask is not None:
            attention_mask = with_sharding_constraint(attention_mask, P("batch", "length"))
            
        if position_ids is not None:
            position_ids = with_sharding_constraint(position_ids, P("batch", "length"))
        
        # Run model with params provided or use model's params
        if params is None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                deterministic=deterministic,
            )
        else:
            outputs = self.model.apply(
                {"params": params},
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                deterministic=deterministic,
            )
        
        # Ensure output has proper sharding
        outputs["logits"] = with_sharding_constraint(outputs["logits"], P("batch", "length", None))
        
        return outputs 


###################
# Generation Utilities
###################

def generate_step(
    model: TPQwenForCausalLM,
    params: Dict,
    input_ids: jnp.ndarray,
    max_length: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    deterministic: bool = False,
    past_key_values: Optional[Dict] = None,
    mesh: Optional[Mesh] = None,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Autoregressive generation for a single step.
    
    Args:
        model: The model to use for generation
        params: Model parameters
        input_ids: Input token IDs
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        deterministic: Whether to use deterministic sampling
        past_key_values: Past key/value cache for generation
        mesh: Device mesh for tensor parallelism
        
    Returns:
        Tuple of generated token IDs and updated past_key_values
    """
    # Track input shape
    batch_size, seq_length = input_ids.shape
    
    # Create attention mask and position IDs
    attention_mask = jnp.ones_like(input_ids)
    
    if past_key_values is None:
        # For first step, use full sequence
        position_ids = jnp.arange(seq_length)[None, :]
    else:
        # For subsequent steps, only need the last position
        position_ids = jnp.array([[past_key_values.get("cache_index", seq_length)]])
    
    # Run model forward pass
    with mesh:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            deterministic=True,
            params=params,
        )
    
    # Get logits and cache
    logits = outputs["logits"]
    current_past_key_values = outputs.get("past_key_values")
    
    # Get logits for the last token
    next_token_logits = logits[:, -1, :]
    
    # Temperature sampling
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
        
        # Apply top-p sampling
        if top_p < 1.0 and not deterministic:
            next_token_logits = top_p_logits(next_token_logits, top_p)
            
        # Sample from the distribution
        if deterministic:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        else:
            key = jax.random.PRNGKey(int(time.time() * 1e6) % 2**32)
            next_token = jax.random.categorical(key, next_token_logits, axis=-1)
    else:
        # Greedy decoding
        next_token = jnp.argmax(next_token_logits, axis=-1)
    
    # Add to input IDs
    next_input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=-1)
    
    return next_input_ids, current_past_key_values


def top_p_logits(
    logits: jnp.ndarray, top_p: float = 0.9, filter_value: float = -float("Inf")
) -> jnp.ndarray:
    """
    Filter logits using nucleus (top-p) sampling.
    
    Args:
        logits: Logits to filter, shape (batch_size, vocab_size)
        top_p: Keep top tokens with cumulative probability >= top_p
        filter_value: Value to assign to filtered logits
        
    Returns:
        Filtered logits
    """
    # Sort logits in descending order
    sorted_logits = -jnp.sort(-logits, axis=-1)
    
    # Get cumulative probabilities
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    
    # Create a mask for logits to keep
    sorted_indices_to_keep = cumulative_probs < top_p
    
    # Keep also the first one above threshold
    sorted_indices_to_keep = jnp.concatenate([
        jnp.ones_like(sorted_indices_to_keep[:, :1]),
        sorted_indices_to_keep[:, :-1]
    ], axis=-1)
    
    # Get indices of sorted elements
    _, sorted_indices = jax.lax.sort_key_val(
        -logits,  # Sort keys in descending order
        jnp.broadcast_to(jnp.arange(logits.shape[-1]), logits.shape),  # Values to sort
    )
    
    # Create a mask for original logits
    indices_to_keep = jnp.take_along_axis(
        sorted_indices_to_keep, sorted_indices, axis=-1
    )
    
    # Filter logits
    logits = jnp.where(indices_to_keep, logits, filter_value)
    
    return logits


def generate_text(
    model: TPQwenForCausalLM,
    tokenizer,
    params: Dict,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    deterministic: bool = False,
    mesh: Optional[Mesh] = None,
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        params: Model parameters
        prompt: Text prompt
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        deterministic: Whether to use deterministic sampling
        mesh: Device mesh for tensor parallelism
        
    Returns:
        Generated text as a string
    """
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = jnp.array([token_ids])
    
    # Initialize past key values
    past_key_values = None
    
    # Track generated IDs
    generated_ids = input_ids
    
    # Generate tokens
    for i in range(max_length):
        # Trim sequence if it's getting too long
        if generated_ids.shape[1] > 4096:
            generated_ids = generated_ids[:, -2048:]
            past_key_values = None  # Reset cache after trimming
        
        # Generate next token
        if past_key_values is not None:
            # If we have past key values, only need the last token
            current_ids = generated_ids[:, -1:]
        else:
            # Otherwise, use the full sequence
            current_ids = generated_ids
        
        # Generate next token
        next_ids, past_key_values = generate_step(
            model=model,
            params=params,
            input_ids=current_ids,
            temperature=temperature,
            top_p=top_p,
            deterministic=deterministic,
            past_key_values=past_key_values,
            mesh=mesh,
        )
        
        # Handle case when we don't use cache
        if past_key_values is None:
            # Append the new token
            generated_ids = next_ids
        else:
            # Only append the newly generated token
            generated_ids = jnp.concatenate([generated_ids, next_ids[:, -1:]], axis=-1)
        
        # Check for end of sequence token
        if generated_ids[0, -1] == tokenizer.eos_token_id:
            break
            
        # Print progress
        if i % 10 == 0:
            print(".", end="", flush=True)
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return generated_text


###################
# Main Function
###################

def main():
    """
    Main function for command-line usage.
    """
    import contextlib
    import glob
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Qwen 2.5 Tensor Parallel Model"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Directory containing the model weights",
    )
    parser.add_argument(
        "--mesh_shape",
        type=str,
        default="1,8",
        help="Device mesh shape for tensor parallelism (e.g. '1,8', '2,4', etc.)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of generated text",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic decoding",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a basic test without generation",
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        
    # Map dtype string to jnp.dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {args.weights_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.weights_dir)
    except ImportError:
        logger.error("Failed to import AutoTokenizer from transformers")
        logger.error("Please install with: pip install transformers")
        return
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return
    
    # Create mesh
    mesh = create_mesh_from_string(args.mesh_shape)
    
    try:
        # Load model with mesh context
        with mesh:
            model, params = load_qwen_model(
                weights_dir=args.weights_dir,
                mesh_shape=args.mesh_shape,
                dtype=dtype,
            )
        
        logger.info("Model loaded successfully!")
        
        # Run a basic test if requested
        if args.test:
            logger.info("Running a basic test...")
            
            # Create input IDs
            input_ids = jnp.ones((1, 10), dtype=jnp.int32)
            
            # Try running the model
            with mesh:
                outputs = model(
                    input_ids=input_ids,
                    params=params,
                )
            
            logger.info(f"Basic test successful! Output shape: {outputs['logits'].shape}")
            return
        
        # Generate text
        logger.info(f"Generating text with prompt: {args.prompt}")
        
        start_time = time.time()
        
        # Generate text with mesh context
        with mesh:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                params=params,
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                deterministic=args.deterministic,
                mesh=mesh,
            )
        
        generation_time = time.time() - start_time
        tokens_generated = len(tokenizer.encode(generated_text)) - len(tokenizer.encode(args.prompt))
        
        logger.info(f"\nGenerated {tokens_generated} tokens in {generation_time:.2f}s "
                   f"({tokens_generated/generation_time:.2f} tokens/s)")
        
        print("\nPrompt:")
        print(args.prompt)
        print("\nGenerated text:")
        print(generated_text)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up JAX flags
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton=true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # Run main function
    main() 