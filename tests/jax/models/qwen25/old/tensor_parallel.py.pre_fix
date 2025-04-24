#!/usr/bin/env python3
# Tensor parallel utilities for Qwen 2.5 model

import logging
import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import freeze, unfreeze
from typing import Dict, List, Tuple, Optional, Union, Any
import flax.linen as nn
from jax.sharding import Mesh, PartitionSpec as P
from jax import lax

# Try to import model components, with fallbacks
try:
    from model_implementation import (
        RMSNorm,
        precompute_freqs_cis,
        apply_rotary_emb,
        QwenAttention,
        QwenMLP,
        QwenTransformerBlock,
        Qwen2Model,
        Qwen2ForCausalLM,
    )
except ImportError:
    logging.warning("Could not import model_implementation components, some functions may not work")

logger = logging.getLogger(__name__)

# Global flag to control shape-related debug prints
DEBUG_SHAPES = False

def log_shape_debug(message):
    """Conditionally log shape debug messages based on DEBUG_SHAPES flag"""
    if DEBUG_SHAPES:
        logger.debug(message)

def create_device_mesh(mesh_shape):
    """
    Create a device mesh with the specified shape.
    
    Args:
        mesh_shape: Tuple of (rows, cols) for the mesh shape
        
    Returns:
        jax.sharding.Mesh: A JAX device mesh
    """
    devices = jax.devices()
    required_devices = mesh_shape[0] * mesh_shape[1]
    
    logger.info(f"Creating mesh with shape {mesh_shape}, requiring {required_devices} devices")
    logger.info(f"Available devices: {len(devices)}")
    
    if len(devices) < required_devices:
        raise ValueError(
            f"Not enough devices ({len(devices)}) for mesh shape {mesh_shape}. "
            f"Required: {required_devices}. Set XLA_FLAGS to simulate more devices."
        )
    
    if len(devices) > required_devices:
        logger.info(f"Warning: Using only {required_devices} of {len(devices)} available devices")
        devices = devices[:required_devices]
    
    try:
        # Create a flat array of devices with the required shape
        devices_array = np.array(devices).reshape(mesh_shape)
        mesh = Mesh(devices_array, ('batch', 'model'))
        logger.info(f"Mesh created with shape {mesh_shape}")
        return mesh
    except ValueError as e:
        logger.error(f"Error creating mesh with np.array.reshape: {e}")
        try:
            # Try using mesh_utils with the sliced devices
            from jax.experimental import mesh_utils
            device_mesh = mesh_utils.create_device_mesh(mesh_shape, devices=devices[:required_devices])
            mesh = Mesh(device_mesh, ('batch', 'model'))
            logger.info(f"Mesh created using mesh_utils")
            return mesh
        except Exception as ex:
            logger.error(f"Error creating mesh with mesh_utils: {ex}")
            raise ValueError(
                f"Failed to create device mesh with shape {mesh_shape}. "
                f"Available devices: {len(devices)}. Required: {required_devices}."
            )

def get_partition_specs(config):
    """
    Create partition specifications for the model parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dict: Partition specs for the model parameters
    """
    hidden_size = config.get('hidden_size', 4096)
    intermediate_size = config.get('intermediate_size', 14336)
    num_attention_heads = config.get('num_attention_heads', 32)
    
    # Partition specs for embeddings
    embed_p = P(None, 'model')
    
    # Partition specs for attention
    q_p = P(None, 'model')
    k_p = P(None, 'model')
    v_p = P(None, 'model')
    o_p = P('model', None)
    
    # Partition specs for MLP
    gate_p = P(None, 'model')
    up_p = P(None, 'model')
    down_p = P('model', None)
    
    # Weights partition specs
    weight_p = P(None)
    
    # Create complete partition specs
    return {
        'model': {
            'embed_tokens': {
                'embedding': embed_p,
            },
            'layers_.*': {
                'self_attn': {
                    'q_proj': {
                        'kernel': q_p,
                    },
                    'k_proj': {
                        'kernel': k_p,
                    },
                    'v_proj': {
                        'kernel': v_p,
                    },
                    'o_proj': {
                        'kernel': o_p,
                    },
                },
                'mlp': {
                    'gate_proj': {
                        'kernel': gate_p,
                    },
                    'up_proj': {
                        'kernel': up_p,
                    },
                    'down_proj': {
                        'kernel': down_p,
                    },
                },
                'input_layernorm': {
                    'weight': weight_p,
                },
                'post_attention_layernorm': {
                    'weight': weight_p,
                }
            },
            'norm': {
                'weight': weight_p,
            }
        },
        'lm_head': {
            'kernel': P('model', None),  # Transpose of embed_p
        }
    }

def shard_parameters(params: Dict, num_shards: int, config: Optional[Dict] = None) -> List[Dict]:
    """
    Shard model parameters for tensor parallelism.
    
    Args:
        params: Model parameters
        num_shards: Number of shards to create
        config: Optional model configuration
        
    Returns:
        List of parameter dictionaries, one for each shard
    """
    # If config not provided, attempt to infer dimensions
    if config is None:
        config = infer_model_config_from_params(params)
        
    # Unfreeze parameters for modification if necessary
    params = unfreeze(params) if hasattr(params, 'unfreeze') else dict(params)
    
    # Flatten parameters for easier processing
    flat_params = flatten_dict(params)
    
    # Create empty shards
    sharded_flat_params = [{} for _ in range(num_shards)]
    
    # Process each parameter
    for path, param in flat_params.items():
        path_str = '.'.join(str(p) for p in path)
        
        # Handle different parameter types
        if isinstance(param, (np.ndarray, jnp.ndarray)):
            # Embedding parameters
            if 'embed_tokens' in path_str and path_str.endswith('embedding'):
                # Shard embeddings by vocab partitioning
                sharded_params = shard_embedding_weights(param, num_shards, config)
                for i, sharded_param in enumerate(sharded_params):
                    sharded_flat_params[i][path] = sharded_param
            
            # Attention parameters
            elif 'attn' in path_str and any(path_str.endswith(f'{proj}.kernel') for proj in ['q', 'k', 'v']):
                # Shard QKV projection by output dimension
                sharded_params = shard_qkv_weights(param, num_shards, config)
                for i, sharded_param in enumerate(sharded_params):
                    sharded_flat_params[i][path] = sharded_param
            
            # Attention output projection
            elif 'attn' in path_str and path_str.endswith('o.kernel'):
                # Shard output projection by input dimension
                sharded_params = shard_attention_output_weights(param, num_shards, config)
                for i, sharded_param in enumerate(sharded_params):
                    sharded_flat_params[i][path] = sharded_param
            
            # MLP parameters
            elif 'mlp' in path_str and any(path_str.endswith(f'w{idx}.kernel') for idx in [1, 2]):
                # Shard MLP intermediate weights by output dimension
                sharded_params = shard_mlp_intermediate_weights(param, num_shards, config)
                for i, sharded_param in enumerate(sharded_params):
                    sharded_flat_params[i][path] = sharded_param
            
            # MLP output projection
            elif 'mlp' in path_str and path_str.endswith('w3.kernel'):
                # Shard MLP output weights by input dimension
                sharded_params = shard_mlp_output_weights(param, num_shards, config)
                for i, sharded_param in enumerate(sharded_params):
                    sharded_flat_params[i][path] = sharded_param
            
            # Layer normalization parameters (copy to all shards)
            elif any(ln in path_str for ln in ['ln_1', 'ln_2', 'ln_f']):
                for i in range(num_shards):
                    sharded_flat_params[i][path] = param
            
            # LM head weights
            elif 'lm_head' in path_str and path_str.endswith('weight'):
                # Shard LM head by vocab partitioning (same as embedding)
                sharded_params = shard_embedding_weights(param, num_shards, config)
                for i, sharded_param in enumerate(sharded_params):
                    sharded_flat_params[i][path] = sharded_param
            
            # Default: copy parameter to all shards
            else:
                for i in range(num_shards):
                    sharded_flat_params[i][path] = param
        else:
            # Non-tensor values: copy to all shards
            for i in range(num_shards):
                sharded_flat_params[i][path] = param
    
    # Unflatten each shard
    sharded_params = [unflatten_dict(flat_params) for flat_params in sharded_flat_params]
    
    # Return frozen parameters if input was frozen
    return [freeze(p) if hasattr(params, 'freeze') else p for p in sharded_params]

def infer_model_config_from_params(params: Dict) -> Dict:
    """
    Infer model configuration from parameter shapes.
    
    Args:
        params: Model parameters
        
    Returns:
        Dictionary with inferred configuration
    """
    # Default configuration
    config = {
        'hidden_size': 4096,
        'intermediate_size': 14336,
        'num_attention_heads': 32,
        'vocab_size': 151936,
    }
    
    # Flatten parameters
    flat_params = flatten_dict(params)
    
    # Check embedding shape for vocab size and hidden size
    for path, param in flat_params.items():
        path_str = '.'.join(str(p) for p in path)
        
        if isinstance(param, (np.ndarray, jnp.ndarray)):
            if 'embed_tokens' in path_str and path_str.endswith(('embedding', 'weight')):
                if len(param.shape) == 2:
                    config['vocab_size'] = param.shape[0]
                    config['hidden_size'] = param.shape[1]
                    logger.info(f"Inferred vocab_size={config['vocab_size']}, hidden_size={config['hidden_size']}")
            
            # Check MLP intermediate size
            elif 'mlp' in path_str and any(gate in path_str for gate in ['w1', 'gate_proj']):
                if len(param.shape) == 2:
                    config['intermediate_size'] = param.shape[1]
                    logger.info(f"Inferred intermediate_size={config['intermediate_size']}")
            
            # Check attention heads
            elif 'attn' in path_str and any(proj in path_str for proj in ['q', 'q_proj']):
                if len(param.shape) == 2:
                    # Using heuristic: if proj shape is (hidden, X), then X = num_heads * head_dim
                    # And typically head_dim = hidden_size / num_heads
                    # So X / (hidden_size / num_heads) = num_heads
                    qkv_dim = param.shape[1]
                    hidden_size = config['hidden_size']  # From embedding or default
                    
                    # Try common head counts
                    for heads in [16, 32, 40, 64, 128]:
                        head_dim = hidden_size // heads
                        if qkv_dim == heads * head_dim:
                            config['num_attention_heads'] = heads
                            logger.info(f"Inferred num_attention_heads={heads}")
                            break
    
    return config

def shard_embedding_weights(embed_weight: Union[np.ndarray, jnp.ndarray], 
                          num_shards: int, 
                          config: Dict) -> List[Union[np.ndarray, jnp.ndarray]]:
    """
    Shard embedding weights for tensor parallelism.
    Shards by partitioning the vocabulary dimension.
    
    Args:
        embed_weight: Embedding weight tensor of shape (vocab_size, hidden_size)
        num_shards: Number of shards
        config: Model configuration
        
    Returns:
        List of sharded embedding weights
    """
    vocab_size, hidden_size = embed_weight.shape
    
    # Calculate shard size and possible remainder
    shard_size = vocab_size // num_shards
    remainder = vocab_size % num_shards
    
    # Create sharded weights
    sharded_weights = []
    start_idx = 0
    
    for i in range(num_shards):
        # Handle uneven division of vocab
        current_shard_size = shard_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_shard_size
        
        # Extract shard
        shard = embed_weight[start_idx:end_idx, :]
        sharded_weights.append(shard)
        
        start_idx = end_idx
    
    return sharded_weights

def shard_qkv_weights(qkv_weight: Union[np.ndarray, jnp.ndarray], 
                    num_shards: int, 
                    config: Dict) -> List[Union[np.ndarray, jnp.ndarray]]:
    """
    Shard query, key, or value projection weights for tensor parallelism.
    Shards by partitioning the head dimension.
    
    Args:
        qkv_weight: QKV projection weight tensor of shape (hidden_size, num_heads * head_dim)
        num_shards: Number of shards
        config: Model configuration
        
    Returns:
        List of sharded QKV weights
    """
    hidden_size, qkv_dim = qkv_weight.shape
    
    # Infer num_heads from config or weight shape
    num_heads = config.get('num_attention_heads', qkv_dim // (hidden_size // 32))
    head_dim = hidden_size // num_heads
    
    # Calculate heads per shard
    heads_per_shard = num_heads // num_shards
    remainder = num_heads % num_shards
    
    # Create sharded weights
    sharded_weights = []
    start_idx = 0
    
    for i in range(num_shards):
        # Handle uneven division of heads
        current_heads = heads_per_shard + (1 if i < remainder else 0)
        head_width = current_heads * head_dim
        end_idx = start_idx + head_width
        
        # Extract shard
        shard = qkv_weight[:, start_idx:end_idx]
        sharded_weights.append(shard)
        
        start_idx = end_idx
    
    return sharded_weights

def shard_attention_output_weights(output_weight: Union[np.ndarray, jnp.ndarray], 
                                 num_shards: int, 
                                 config: Dict) -> List[Union[np.ndarray, jnp.ndarray]]:
    """
    Shard attention output projection weights for tensor parallelism.
    Shards by partitioning the input dimension (concatenated heads).
    
    Args:
        output_weight: Output projection weight tensor of shape (num_heads * head_dim, hidden_size)
        num_shards: Number of shards
        config: Model configuration
        
    Returns:
        List of sharded output weights
    """
    qkv_dim, hidden_size = output_weight.shape
    
    # Infer num_heads from config or weight shape
    num_heads = config.get('num_attention_heads', qkv_dim // (hidden_size // 32))
    head_dim = hidden_size // num_heads
    
    # Calculate heads per shard
    heads_per_shard = num_heads // num_shards
    remainder = num_heads % num_shards
    
    # Create sharded weights
    sharded_weights = []
    start_idx = 0
    
    for i in range(num_shards):
        # Handle uneven division of heads
        current_heads = heads_per_shard + (1 if i < remainder else 0)
        head_width = current_heads * head_dim
        end_idx = start_idx + head_width
        
        # Extract shard
        shard = output_weight[start_idx:end_idx, :]
        sharded_weights.append(shard)
        
        start_idx = end_idx
    
    return sharded_weights

def shard_mlp_intermediate_weights(mlp_weight: Union[np.ndarray, jnp.ndarray], 
                                 num_shards: int, 
                                 config: Dict) -> List[Union[np.ndarray, jnp.ndarray]]:
    """
    Shard MLP intermediate weights (W1/W2) for tensor parallelism.
    Shards by partitioning the output dimension.
    
    Args:
        mlp_weight: MLP weight tensor of shape (hidden_size, intermediate_size)
        num_shards: Number of shards
        config: Model configuration
        
    Returns:
        List of sharded MLP weights
    """
    hidden_size, intermediate_size = mlp_weight.shape
    
    # Calculate shard size and possible remainder
    shard_size = intermediate_size // num_shards
    remainder = intermediate_size % num_shards
    
    # Create sharded weights
    sharded_weights = []
    start_idx = 0
    
    for i in range(num_shards):
        # Handle uneven division
        current_shard_size = shard_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_shard_size
        
        # Extract shard
        shard = mlp_weight[:, start_idx:end_idx]
        sharded_weights.append(shard)
        
        start_idx = end_idx
    
    return sharded_weights

def shard_mlp_output_weights(mlp_weight: Union[np.ndarray, jnp.ndarray], 
                           num_shards: int, 
                           config: Dict) -> List[Union[np.ndarray, jnp.ndarray]]:
    """
    Shard MLP output weights (W3) for tensor parallelism.
    Shards by partitioning the input dimension.
    
    Args:
        mlp_weight: MLP weight tensor of shape (intermediate_size, hidden_size)
        num_shards: Number of shards
        config: Model configuration
        
    Returns:
        List of sharded MLP weights
    """
    intermediate_size, hidden_size = mlp_weight.shape
    
    # Calculate shard size and possible remainder
    shard_size = intermediate_size // num_shards
    remainder = intermediate_size % num_shards
    
    # Create sharded weights
    sharded_weights = []
    start_idx = 0
    
    for i in range(num_shards):
        # Handle uneven division
        current_shard_size = shard_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_shard_size
        
        # Extract shard
        shard = mlp_weight[start_idx:end_idx, :]
        sharded_weights.append(shard)
        
        start_idx = end_idx
    
    return sharded_weights

def merge_sharded_parameters(sharded_params_list: List[Dict], config: Optional[Dict] = None) -> Dict:
    """
    Merge sharded parameters back into a single parameter dictionary.
    
    Args:
        sharded_params_list: List of sharded parameter dictionaries
        config: Optional model configuration
        
    Returns:
        Merged parameter dictionary
    """
    if not sharded_params_list:
        return {}
    
    num_shards = len(sharded_params_list)
    
    # If config not provided, attempt to infer dimensions
    if config is None:
        config = infer_model_config_from_params(sharded_params_list[0])
    
    # Unfreeze parameters for modification if necessary
    sharded_params_list = [
        unfreeze(p) if hasattr(p, 'unfreeze') else dict(p) 
        for p in sharded_params_list
    ]
    
    # Flatten parameters for easier processing
    flat_sharded_params = [flatten_dict(p) for p in sharded_params_list]
    
    # Create merged parameters
    merged_flat_params = {}
    
    # Get all parameter paths from the first shard
    all_paths = set(flat_sharded_params[0].keys())
    for flat_params in flat_sharded_params[1:]:
        all_paths.update(flat_params.keys())
    
    # Process each parameter path
    for path in all_paths:
        path_str = '.'.join(str(p) for p in path)
        
        # Check if parameter exists in all shards
        if not all(path in flat_params for flat_params in flat_sharded_params):
            logger.warning(f"Parameter {path_str} does not exist in all shards, skipping")
            continue
        
        # Get parameter from first shard to check type
        param = flat_sharded_params[0][path]
        
        # Handle different parameter types
        if isinstance(param, (np.ndarray, jnp.ndarray)):
            # Embedding parameters
            if 'embed_tokens' in path_str and path_str.endswith('embedding'):
                # Merge embeddings by concatenating along vocabulary dimension
                merged_param = merge_embedding_weights([flat_params[path] for flat_params in flat_sharded_params], config)
                merged_flat_params[path] = merged_param
            
            # Attention parameters
            elif 'attn' in path_str and any(path_str.endswith(f'{proj}.kernel') for proj in ['q', 'k', 'v']):
                # Merge QKV projection by concatenating along head dimension
                merged_param = merge_qkv_weights([flat_params[path] for flat_params in flat_sharded_params], config)
                merged_flat_params[path] = merged_param
            
            # Attention output projection
            elif 'attn' in path_str and path_str.endswith('o.kernel'):
                # Merge output projection by concatenating along input dimension
                merged_param = merge_attention_output_weights([flat_params[path] for flat_params in flat_sharded_params], config)
                merged_flat_params[path] = merged_param
            
            # MLP parameters
            elif 'mlp' in path_str and any(path_str.endswith(f'w{idx}.kernel') for idx in [1, 2]):
                # Merge MLP intermediate weights by concatenating along output dimension
                merged_param = merge_mlp_intermediate_weights([flat_params[path] for flat_params in flat_sharded_params], config)
                merged_flat_params[path] = merged_param
            
            # MLP output projection
            elif 'mlp' in path_str and path_str.endswith('w3.kernel'):
                # Merge MLP output weights by concatenating along input dimension
                merged_param = merge_mlp_output_weights([flat_params[path] for flat_params in flat_sharded_params], config)
                merged_flat_params[path] = merged_param
            
            # Layer normalization parameters (use value from any shard)
            elif any(ln in path_str for ln in ['ln_1', 'ln_2', 'ln_f']):
                merged_flat_params[path] = param
            
            # LM head weights
            elif 'lm_head' in path_str and path_str.endswith('weight'):
                # Merge LM head by concatenating along vocabulary dimension
                merged_param = merge_embedding_weights([flat_params[path] for flat_params in flat_sharded_params], config)
                merged_flat_params[path] = merged_param
            
            # Default: use value from first shard
            else:
                merged_flat_params[path] = param
        else:
            # Non-tensor values: use value from first shard
            merged_flat_params[path] = param
    
    # Unflatten merged parameters
    merged_params = unflatten_dict(merged_flat_params)
    
    # Return frozen parameters if input was frozen
    return freeze(merged_params) if hasattr(sharded_params_list[0], 'freeze') else merged_params

def merge_embedding_weights(sharded_weights: List[Union[np.ndarray, jnp.ndarray]], 
                           config: Dict) -> Union[np.ndarray, jnp.ndarray]:
    """
    Merge sharded embedding weights by concatenating along vocabulary dimension.
    
    Args:
        sharded_weights: List of sharded embedding weights
        config: Model configuration
        
    Returns:
        Merged embedding weights
    """
    return np.concatenate(sharded_weights, axis=0) if isinstance(sharded_weights[0], np.ndarray) else jnp.concatenate(sharded_weights, axis=0)

def merge_qkv_weights(sharded_weights: List[Union[np.ndarray, jnp.ndarray]], 
                     config: Dict) -> Union[np.ndarray, jnp.ndarray]:
    """
    Merge sharded query, key, or value projection weights by concatenating along head dimension.
    
    Args:
        sharded_weights: List of sharded QKV weights
        config: Model configuration
        
    Returns:
        Merged QKV weights
    """
    return np.concatenate(sharded_weights, axis=1) if isinstance(sharded_weights[0], np.ndarray) else jnp.concatenate(sharded_weights, axis=1)

def merge_attention_output_weights(sharded_weights: List[Union[np.ndarray, jnp.ndarray]], 
                                 config: Dict) -> Union[np.ndarray, jnp.ndarray]:
    """
    Merge sharded attention output projection weights by concatenating along input dimension.
    
    Args:
        sharded_weights: List of sharded output weights
        config: Model configuration
        
    Returns:
        Merged output weights
    """
    return np.concatenate(sharded_weights, axis=0) if isinstance(sharded_weights[0], np.ndarray) else jnp.concatenate(sharded_weights, axis=0)

def merge_mlp_intermediate_weights(sharded_weights: List[Union[np.ndarray, jnp.ndarray]], 
                                 config: Dict) -> Union[np.ndarray, jnp.ndarray]:
    """
    Merge sharded MLP intermediate weights (W1/W2) by concatenating along output dimension.
    
    Args:
        sharded_weights: List of sharded MLP weights
        config: Model configuration
        
    Returns:
        Merged MLP weights
    """
    return np.concatenate(sharded_weights, axis=1) if isinstance(sharded_weights[0], np.ndarray) else jnp.concatenate(sharded_weights, axis=1)

def merge_mlp_output_weights(sharded_weights: List[Union[np.ndarray, jnp.ndarray]], 
                           config: Dict) -> Union[np.ndarray, jnp.ndarray]:
    """
    Merge sharded MLP output weights (W3) by concatenating along input dimension.
    
    Args:
        sharded_weights: List of sharded MLP weights
        config: Model configuration
        
    Returns:
        Merged MLP weights
    """
    return np.concatenate(sharded_weights, axis=0) if isinstance(sharded_weights[0], np.ndarray) else jnp.concatenate(sharded_weights, axis=0)

class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    precision: Optional[Union[str, lax.Precision]] = None
    mesh: Mesh = None
    shard_axes: Tuple[str, str] = ('model', None)  # (kernel_in, kernel_out)
    
    @nn.compact
    def __call__(self, inputs):
        """Apply the dense layer with tensor parallelism."""
        kernel_shape = (inputs.shape[-1], self.features)
        
        # Log input shape and kernel shape for debugging
        log_shape_debug(f"TensorParallelDense: inputs.shape={inputs.shape}, kernel_shape={kernel_shape}")
        
        # Create sharding rules based on mesh
        kernel_sharding = None
        if self.mesh is not None:
            kernel_in, kernel_out = self.shard_axes
            kernel_sharding = P(kernel_in, kernel_out)
            log_shape_debug(f"TensorParallelDense: kernel_sharding={kernel_sharding}")
        
        # Create kernel parameter with appropriate sharding
        kernel = self.param(
            'kernel',
            self.kernel_init,
            kernel_shape,
            self.param_dtype,
            kernel_sharding
        )
        
        # Cast kernel to compute precision
        kernel = kernel.astype(self.dtype)
        
        # Apply dense transformation
        y = jnp.matmul(inputs, kernel, precision=self.precision)
        
        # Handle bias if needed
        if self.use_bias:
            bias_shape = (self.features,)
            bias_sharding = None
            if self.mesh is not None and self.shard_axes[1] is not None:
                bias_sharding = P(self.shard_axes[1])
            
            bias = self.param(
                'bias',
                self.bias_init,
                bias_shape,
                self.param_dtype,
                bias_sharding
            )
            bias = bias.astype(self.dtype)
            y = y + bias
        
        return y

# Placeholder definition for required classes, which will be defined later
# (or may already be defined in the __init__.py)
TensorParallelQwenAttention = None
TensorParallelQwenMLP = None 
TensorParallelQwenTransformerBlock = None
TensorParallelQwen2Model = None

class TensorParallelQwen2ForCausalLM(nn.Module):
    """Tensor parallel implementation of Qwen2 for causal language modeling."""
    config: Dict[str, Any]
    mesh: Mesh = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None
    
    def setup(self):
        """Initialize model components."""
        self.transformer_config = {**self.config}
        self.transformer = TensorParallelQwen2Model(
            config=self.transformer_config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            mesh=self.mesh,
        )
        
        # LM head uses the embedding transposed
        hidden_size = self.config["hidden_size"]
        vocab_size = self.config["vocab_size"]
        
        # Initialize LM head
        self.lm_head = TensorParallelDense(
            features=vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            mesh=self.mesh,
            shard_axes=('model', None),  # Shard along model dim
        )
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
        mems=None,
        **kwargs
    ):
        """Run the model forward pass."""
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs
        )
        
        hidden_states = transformer_outputs['last_hidden_state'] if return_dict else transformer_outputs[0]
        
        # Apply LM head to get logits
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return (logits,) + transformer_outputs[1:]
        
        return {
            'logits': logits,
            'past_key_values': transformer_outputs.get('past_key_values'),
            'hidden_states': transformer_outputs.get('hidden_states'),
            'attentions': transformer_outputs.get('attentions'),
        }
    
    def get_partition_rules(self):
        """Get model-specific partition rules."""
        return self.transformer.get_partition_rules()
    
    def get_params_partition_spec(self):
        """Get model-specific parameter partition specs."""
        return self.transformer.get_params_partition_spec()
    
    def input_sharding_spec(self, dtype=jnp.int32):
        """Return the sharding spec for input tensors."""
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

# Minimal implementation of TensorParallelQwen2Model for compatibility
class TensorParallelQwen2Model(nn.Module):
    """Tensor parallel implementation of Qwen2 model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None
    mesh: Mesh = None
    
    def setup(self):
        """Initialize model components."""
        # This is just a placeholder implementation
        # In a real scenario, this would be a full implementation
        pass
        
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
        **kwargs
    ):
        """Placeholder forward pass."""
        # Create a minimal placeholder implementation that returns a valid structure
        # with the expected shape of the outputs
        batch_size, seq_length = input_ids.shape
        hidden_size = self.config.get("hidden_size", 4096)
        
        # Create a dummy hidden state with the correct shape
        dummy_hidden_states = jnp.zeros((batch_size, seq_length, hidden_size), dtype=self.dtype)
        
        # Return a dictionary with the expected fields
        return {
            'last_hidden_state': dummy_hidden_states,
            'past_key_values': None,
            'hidden_states': None,
            'attentions': None,
        }
        
    def get_partition_rules(self):
        """Return partition rules for tensor parallelism."""
        return [
            # Embedding partitioning
            ("embed_tokens/embedding", P(None, "model")),
            
            # Layer norm parameters
            ("layers_.*/(input|post_attention)_layernorm/weight", P(None)),
            ("norm/weight", P(None)),
            
            # Attention parameters
            ("layers_.*/self_attn/(q|k|v)_proj/kernel", P(None, "model")),
            ("layers_.*/self_attn/o_proj/kernel", P("model", None)),
            
            # MLP parameters
            ("layers_.*/mlp/(gate|up)_proj/kernel", P(None, "model")),
            ("layers_.*/mlp/down_proj/kernel", P("model", None)),
        ]
    
    def get_params_partition_spec(self):
        """Return the partition spec for the parameters."""
        return {
            # Embedding specs
            "embed_tokens": {
                "embedding": P(None, "model"),
            },
            # Layer norms
            "norm": {
                "weight": P(None),
            },
            # Recursive specs for layers
            "layers": {
                "[0-9]+": {
                    "input_layernorm": {
                        "weight": P(None),
                    },
                    "post_attention_layernorm": {
                        "weight": P(None),
                    },
                    "self_attn": {
                        "q_proj": {
                            "kernel": P(None, "model"),
                        },
                        "k_proj": {
                            "kernel": P(None, "model"),
                        },
                        "v_proj": {
                            "kernel": P(None, "model"),
                        },
                        "o_proj": {
                            "kernel": P("model", None),
                        },
                    },
                    "mlp": {
                        "gate_proj": {
                            "kernel": P(None, "model"),
                        },
                        "up_proj": {
                            "kernel": P(None, "model"),
                        },
                        "down_proj": {
                            "kernel": P("model", None),
                        },
                    },
                }
            }
        }
    
    def input_sharding_spec(self, dtype=jnp.int32):
        """Return the sharding spec for input tensors."""
        if self.mesh is None:
            return None
        return jax.sharding.NamedSharding(self.mesh, P('batch', None))

def get_tensor_parallel_model(
    model_name: str = "qwen2_5",
    mesh_shape: Tuple[int, int] = (1, 8),
    config: Optional[Dict] = None,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: Optional[jnp.dtype] = None,
):
    """Get a tensor-parallel model instance with appropriate mesh."""
    # Create device mesh
    mesh = create_device_mesh(mesh_shape)
    
    # Default param dtype to dtype if not specified
    if param_dtype is None:
        param_dtype = dtype
    
    # Return TensorParallelQwen2ForCausalLM
    return TensorParallelQwen2ForCausalLM(
        config=config,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
    )

# Additional utility function to help with imports
def load_params_from_checkpoint(model, model_path):
    """Load parameters from checkpoint files and apply parameter mapping."""
    # Just a placeholder that delegates to the model's method
    if hasattr(model, 'load_params_from_checkpoint'):
        return model.load_params_from_checkpoint(model_path)
    else:
        # Try to import from weight_loading
        from weight_loading import load_qwen_weights
        return load_qwen_weights(model_path, model)

# Parameter mapping utility
def map_parameter_paths(params):
    """Map parameter paths from one format to another."""
    # This is just a placeholder that returns the params unchanged
    # In a real implementation, this would apply transformations
    return params

if __name__ == "__main__":
    # Sample usage code
    logging.basicConfig(level=logging.INFO)
    logger.info("Tensor parallel utilities for Qwen 2.5 model")