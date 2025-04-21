#!/usr/bin/env python3
# Tensor parallel utilities for Qwen 2.5 model

import logging
import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import freeze, unfreeze
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    # Sample usage code
    logging.basicConfig(level=logging.INFO)
    logger.info("Tensor parallel utilities for Qwen 2.5 model")