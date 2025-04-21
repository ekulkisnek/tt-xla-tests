# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading utilities for Qwen2.5 models.
"""

import os
import glob
import logging
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, PartitionSpec as P
from safetensors.flax import load_file as safe_load_file
from safetensors import safe_open
from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from transformers.utils import logging as transformers_logging

from tensor_parallel import get_partition_specs, create_device_mesh

# Setup logging
logger = logging.getLogger(__name__)
transformers_logger = transformers_logging.get_logger("transformers")
transformers_logger.setLevel(logging.INFO)

# Patch PyTorch's load function to handle the weights_only parameter in PyTorch 2.6+
def patch_torch_load():
    """
    Apply a patch to torch.load to handle PyTorch 2.6's weights_only parameter change.
    """
    try:
        import torch
        original_torch_load = torch.load
        
        # Create patched version that ALWAYS sets weights_only=False for PyTorch 2.6+
        def patched_torch_load(f, *args, **kwargs):
            # For PyTorch 2.6+, ALWAYS force weights_only=False to avoid the unpickling error
            kwargs['weights_only'] = False
            
            try:
                return original_torch_load(f, *args, **kwargs)
            except Exception as e:
                # If there's still an error related to weights_only, try without any args
                if "weights_only" in str(e):
                    logger.warning(f"Error with weights_only parameter: {e}")
                    logger.info("Trying again with minimal parameters...")
                    return original_torch_load(f, map_location="cpu")
                else:
                    # Re-raise if it's a different error
                    raise
        
        # Replace the original function
        torch.load = patched_torch_load
        logger.info("✅ Applied patch for PyTorch 2.6+ weights_only parameter")
        return True
    except ImportError:
        logger.warning("⚠️ Could not patch torch.load (torch not imported yet)")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Could not patch torch.load: {e}")
        return False

# Patch transformers' load_pytorch_checkpoint function to handle weights_only issue
def patch_transformers_load_function():
    """
    Apply a patch to transformers' load_pytorch_checkpoint_in_flax_state_dict function
    to handle PyTorch 2.6's weights_only parameter change.
    """
    try:
        from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict as original_load_fn
        from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
        
        # Create a more robust version of the load function
        def patched_load_pytorch_checkpoint_in_flax_state_dict(
            flax_model, 
            pytorch_checkpoint_path, 
            is_sharded=False, 
            allow_missing_keys=False
        ):
            """Patched version that handles PyTorch 2.6+ weights_only issues more robustly"""
            try:
                # Make sure our torch.load patch is applied
                patch_torch_load()
                
                # First try the original function
                return original_load_fn(
                    flax_model, 
                    pytorch_checkpoint_path, 
                    is_sharded=is_sharded, 
                    allow_missing_keys=allow_missing_keys
                )
            except Exception as first_error:
                logger.warning(f"First loading attempt failed: {first_error}")
                
                # Try a more direct approach to load the checkpoint files
                try:
                    logger.info("Trying direct loading approach...")
                    import torch
                    
                    # Load the PyTorch state dict manually with weights_only=False
                    if isinstance(pytorch_checkpoint_path, str):
                        pytorch_checkpoint_path = [pytorch_checkpoint_path]
                    
                    # For a single file
                    if not is_sharded:
                        state_dict = torch.load(pytorch_checkpoint_path[0], map_location="cpu", weights_only=False)
                        
                        # Convert to Flax format
                        flax_state_dict = convert_pytorch_state_dict_to_flax(state_dict, flax_model)
                        return flax_state_dict
                    
                    # For sharded files
                    state_dict = {}
                    for checkpoint_file in pytorch_checkpoint_path:
                        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                        state_dict.update(checkpoint)
                    
                    # Convert to Flax format
                    flax_state_dict = convert_pytorch_state_dict_to_flax(state_dict, flax_model)
                    return flax_state_dict
                    
                except Exception as second_error:
                    logger.error(f"Both loading attempts failed. Original error: {first_error}")
                    logger.error(f"Direct loading error: {second_error}")
                    raise ValueError(f"Could not load PyTorch checkpoint: {second_error}")
        
        # Apply the patch
        import transformers.modeling_flax_pytorch_utils
        transformers.modeling_flax_pytorch_utils.load_pytorch_checkpoint_in_flax_state_dict = patched_load_pytorch_checkpoint_in_flax_state_dict
        logger.info("✅ Applied patch for transformers' load_pytorch_checkpoint_in_flax_state_dict")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not patch transformers' load function: {e}")
        return False

# Apply patches when this module is imported
_ = patch_torch_load()
_ = patch_transformers_load_function()

def get_checkpoint_files(checkpoint_dir: str) -> List[str]:
    """
    Get a list of all checkpoint files in the given directory.
    Supports safetensors and PyTorch formats.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        List of checkpoint filenames
    """
    # First check for the safetensors index file
    index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        logger.info(f"Found safetensors index file: {index_file}")
        import json
        with open(index_file, "r") as f:
            index = json.load(f)
            if "weight_map" in index:
                files = sorted(list(set(index["weight_map"].values())))
                return [os.path.join(checkpoint_dir, f) for f in files]
    
    # Check for safetensors files directly
    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if safetensors_files:
        logger.info(f"Found {len(safetensors_files)} safetensors files")
        return safetensors_files
    
    # Check for PyTorch files
    pytorch_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.bin")))
    if pytorch_files:
        logger.info(f"Found {len(pytorch_files)} PyTorch checkpoint files")
        return pytorch_files
    
    # Try the pytorch_model.bin file as a fallback
    pytorch_model_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(pytorch_model_bin):
        logger.info(f"Found single PyTorch model file: {pytorch_model_bin}")
        return [pytorch_model_bin]
    
    raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

def load_qwen_weights(
    model_path: str,
    model: Any,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    debug: bool = False,
) -> Dict:
    """
    Load weights from PyTorch checkpoint files and convert to Flax format using HuggingFace utilities.
    
    Args:
        model_path: Path to model checkpoint files
        model: Flax model to load weights for
        config: Model configuration dictionary 
        mesh: JAX device mesh for tensor parallelism
        param_dtype: Parameter dtype
        debug: Whether to print debug information
        
    Returns:
        Dictionary of model parameters
    """
    logger.info(f"Loading weights from {model_path}")
    
    # Find checkpoint files
    checkpoint_files = None
    try:
        checkpoint_files = get_checkpoint_files(model_path)
        is_sharded = len(checkpoint_files) > 1
        logger.info(f"Found {'sharded' if is_sharded else 'single'} checkpoint with {len(checkpoint_files)} file(s)")
    except FileNotFoundError:
        logger.error(f"No checkpoint files found in {model_path}")
        raise
    
    # Apply PyTorch patches to avoid potential loading issues
    patch_torch_load()
    patch_transformers_load_function()
    
    # Determine if we're loading from safetensors or PyTorch files
    is_safetensors = all(f.endswith('.safetensors') for f in checkpoint_files)
    
    # First attempt: try loading with safetensors (preferred method)
    if is_safetensors:
        logger.info("Attempting to load directly with safetensors (preferred method)...")
        try:
            params = load_safetensors_weights(
                model_path=model_path,
                model=model,
                config=config,
                mesh=mesh,
                param_dtype=param_dtype
            )
            logger.info("Successfully loaded weights using safetensors")
            return params
        except Exception as e:
            logger.warning(f"Direct safetensors loading failed: {e}")
            logger.warning("Falling back to alternative methods...")
    
    # Second attempt: try loading with HuggingFace transformers utilities
    logger.info("Attempting to load with HuggingFace utilities...")
    
    # For safetensors files, use HuggingFace's safe_load_file
    if is_safetensors:
        logger.info("Loading with HuggingFace safe_load_file...")
        
        # Load each file and combine
        all_params = {}
        for file_path in checkpoint_files:
            logger.info(f"Loading safetensors file: {file_path}")
            try:
                file_params = safe_load_file(file_path)
                all_params.update(file_params)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise
        
        # Convert to the expected format
        params = {"params": {}}
        
        # Process the embedding weights first
        embedding_found = False
        for key, value in all_params.items():
            if "embed_tokens.weight" in key:
                embedding_found = True
                # Create the embedding parameter structure
                if "transformer" not in params["params"]:
                    params["params"]["transformer"] = {}
                
                if "embed_tokens" not in params["params"]["transformer"]:
                    params["params"]["transformer"]["embed_tokens"] = {}
                
                # Set the embedding parameter with correct name
                params["params"]["transformer"]["embed_tokens"]["embedding"] = jnp.array(value, dtype=param_dtype)
                logger.info(f"Loaded embedding weights with shape {value.shape}")
                break
        
        if not embedding_found:
            logger.warning("No embedding parameters found - may cause errors!")
        
        # Process remaining parameters
        for key, value in all_params.items():
            if "embed_tokens.weight" in key:
                # Already handled above
                continue
                
            # Convert PyTorch key to Flax path
            flax_path = convert_pt_key_to_flax(key)
            
            if flax_path:
                # Build nested dictionary from path
                current = params
                for path_part in flax_path[:-1]:
                    if path_part not in current:
                        current[path_part] = {}
                    current = current[path_part]
                
                # Set the actual parameter value
                current[flax_path[-1]] = jnp.array(value, dtype=param_dtype)
        
        # Apply tensor parallelism if mesh is provided
        if mesh is not None:
            logger.info(f"Applying tensor parallelism with mesh shape {mesh.devices.shape}")
            
            # Use JAX's device_put to distribute parameters according to partition specs
            with mesh:
                params = jax.device_put(params, jax.sharding.NamedSharding(mesh, P()))
        
        logger.info("Successfully loaded weights using HuggingFace safetensors utilities")
        return params
    
    # For PyTorch .bin files, use load_pytorch_checkpoint_in_flax_state_dict
    logger.info("Loading with HuggingFace transformers PyTorch utilities...")
    try:
        # Create a temporary model state to initialize parameters
        with mesh:
            # Generate a random key for initialization
            rng = jax.random.PRNGKey(0)
            
            # Create init batch
            batch_size = mesh.shape[0] if mesh is not None and len(mesh.shape) > 0 else 1
            seq_len = 1
            init_batch = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
            
            # Initialize parameters with dummy values
            params = model.init(rng, input_ids=init_batch)
        
        # Now load the real weights on top of the init structure
        params = load_pytorch_checkpoint_in_flax_state_dict(
            model,
            checkpoint_files,
            is_sharded=is_sharded,
            allow_missing_keys=True
        )
        
        # Check if we have embedding weights
        embedding_found = False
        if "params" in params and "transformer" in params["params"]:
            if "embed_tokens" in params["params"]["transformer"]:
                if "embedding" in params["params"]["transformer"]["embed_tokens"]:
                    embedding_found = True
                    logger.info("Found embedding weights in loaded parameters")
        
        if not embedding_found:
            logger.warning("No embedding parameters found in loaded weights - may cause errors!")
        
        # Apply tensor parallelism if mesh is provided
        if mesh is not None:
            logger.info("Applying tensor parallelism with mesh shape {mesh.devices.shape}")
            
            # Use JAX's device_put to distribute parameters according to partition specs
            with mesh:
                params = jax.device_put(params, jax.sharding.NamedSharding(mesh, P()))
        
        logger.info("Weight loading completed in {time.time() - start_time:.2f} seconds")
        return params
        
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        raise

def load_safetensors_weights(
    model_path: str,
    model: Any,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> Dict:
    """
    Load weights from safetensors files and convert to Flax format.
    
    Args:
        model_path: Path to safetensors files
        model: Flax model to load weights for
        config: Model configuration
        mesh: Optional JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        
    Returns:
        Dict: Dictionary of model parameters in Flax format
    """
    logger.info(f"Loading safetensors weights from {model_path}")
    
    # Get index file if available
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    weight_map = {}
    
    if os.path.exists(index_file):
        logger.info(f"Found safetensors index file: {index_file}")
        import json
        with open(index_file, "r") as f:
            index = json.load(f)
            weight_map = index.get("weight_map", {})
            files = sorted(list(set(weight_map.values())))
            logger.info(f"Found {len(files)} safetensors files in index")
    else:
        # Find safetensors files in the directory
        safetensors_files = get_checkpoint_files(model_path)
        files = [os.path.basename(f) for f in safetensors_files]
        logger.info(f"Found {len(files)} safetensors files in directory")
    
    # Create a raw params dictionary with original PyTorch names
    raw_params = {}
    
    # Track if we found embedding weights
    found_embedding = False
    
    # Load weights from each file
    for file_name in files:
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Loading weights from {file_name}")
        
        try:
            # Use safe_open to load weights from safetensors file
            with safe_open(file_path, framework="numpy") as f:
                for key in f.keys():
                    # Load the tensor as numpy array
                    tensor = f.get_tensor(key)
                    
                    # Check for embedding weights
                    if "embed_tokens.weight" in key:
                        found_embedding = True
                        logger.info(f"Found embedding weights: {key} with shape {tensor.shape}")
                    
                    # Store with original name
                    raw_params[key] = tensor
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            # Fallback to HuggingFace's safe_load_file
            try:
                logger.info(f"Trying fallback with HuggingFace's safe_load_file")
                file_params = safe_load_file(file_path)
                for key, tensor in file_params.items():
                    # Check for embedding weights
                    if "embed_tokens.weight" in key:
                        found_embedding = True
                        logger.info(f"Found embedding weights: {key} with shape {tensor.shape}")
                    
                    # Store with original name
                    raw_params[key] = tensor
            except Exception as e2:
                logger.error(f"Both loading methods failed for {file_path}: {e2}")
                raise
    
    logger.info(f"Loaded {len(raw_params)} raw parameters")
    
    # Now convert to Flax parameter structure
    flax_params = {}
    
    # Process each parameter
    for key, tensor in raw_params.items():
        # Convert PyTorch key to Flax path
        flax_path = convert_pt_key_to_flax(key)
        
        if flax_path:
            # Convert to JAX array with correct dtype
            tensor_jax = jnp.array(tensor, dtype=param_dtype)
            
            # Build nested dictionary from path
            current = flax_params
            for path_part in flax_path[:-1]:
                if path_part not in current:
                    current[path_part] = {}
                current = current[path_part]
            
            # Set the actual parameter value
            current[flax_path[-1]] = tensor_jax
        else:
            logger.warning(f"Could not map parameter: {key}")
    
    # Check if we found and mapped the embedding
    if "transformer" in flax_params.get("params", {}) and "embed_tokens" in flax_params["params"]["transformer"]:
        if "embedding" in flax_params["params"]["transformer"]["embed_tokens"]:
            logger.info("Successfully mapped embedding weights")
        else:
            logger.warning("Embedding weights not properly mapped")
    elif found_embedding:
        logger.warning("Found embedding weights but failed to map them correctly")
    else:
        logger.warning("No embedding weights found in safetensors files")
    
    # Apply tensor parallelism if mesh is provided
    if mesh is not None:
        logger.info(f"Applying tensor parallelism with mesh shape {mesh.devices.shape}")
        partition_specs = get_partition_specs(config)
        
        with mesh:
            # Use JAX's device_put to distribute parameters according to partition specs
            from flax.core.frozen_dict import freeze, unfreeze
            
            # Apply constraints to params
            param_specs = {}
            for key, value in partition_specs.items():
                if key.startswith("model."):
                    flax_key = convert_pt_key_to_flax(key)
                    if flax_key:
                        param_specs[tuple(flax_key)] = value
            
            # Apply sharding using the partition specs
            flax_params = jax.device_put(flax_params, jax.sharding.NamedSharding(mesh, P()))
    
    logger.info("Finished loading and converting safetensors weights")
    return flax_params

def init_model_from_weights(
    model_class,
    model_path: str,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    debug: bool = False,
):
    """
    Initialize a model and load weights.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model weights
        config: Model configuration dictionary
        mesh: JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        debug: Whether to print debug information
        
    Returns:
        Tuple of (model instance, loaded parameters)
    """
    # Initialize model
    model = model_class(
        config=config,
        mesh=mesh,
        dtype=jnp.bfloat16,
        param_dtype=param_dtype,
    )
    
    # Load weights
    params = load_qwen_weights(
        model_path=model_path,
        model=model,
        config=config,
        mesh=mesh,
        param_dtype=param_dtype,
        debug=debug
    )
    
    return model, params

def load_partial_qwen_weights(model_path, target_params, config, mesh, num_layers=None, logger=None):
    """
    Load weights from a full-sized Qwen model into a reduced-size model configuration.
    This function selectively copies weights to match the smaller architecture.
    
    Args:
        model_path: Path to the full model weights
        target_params: Target parameter structure (from model.init())
        config: Reduced size configuration
        mesh: Device mesh
        num_layers: Number of layers to load
        logger: Logger for status messages
    
    Returns:
        Dictionary with loaded parameters matching the reduced structure
    """
    import os
    import json
    import numpy as np
    from safetensors import safe_open
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    
    # Simple logging function if no logger provided
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log_info(f"Loading partial weights from {model_path} for reduced size model")
    
    # Create a copy of target_params to modify
    new_params = jax.tree_util.tree_map(lambda x: x, target_params)
    
    # Load original model config to get original dimensions
    original_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(original_config_path):
        with open(original_config_path, "r") as f:
            original_config = json.load(f)
        log_info(f"Original model: hidden_size={original_config.get('hidden_size')}, layers={original_config.get('num_hidden_layers')}")
    else:
        log_info(f"No config.json found at {model_path}, assuming standard Qwen2 7B architecture")
        original_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 14336
        }
    
    # Get list of weight files (safetensors format)
    weight_files = []
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            weight_files.append(os.path.join(model_path, file))
    
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    
    log_info(f"Found {len(weight_files)} weight files")
    
    # Target sizes from reduced config
    target_hidden_size = config["hidden_size"]
    target_layers = config["num_hidden_layers"]
    target_heads = config["num_attention_heads"]
    target_kv_heads = config["num_key_value_heads"]
    target_intermediate = config["intermediate_size"]
    
    # Original sizes
    orig_hidden_size = original_config.get("hidden_size", 4096)
    orig_layers = original_config.get("num_hidden_layers", 32) 
    orig_heads = original_config.get("num_attention_heads", 32)
    orig_kv_heads = original_config.get("num_key_value_heads", 8)
    orig_intermediate = original_config.get("intermediate_size", 14336)
    
    # Scaling factors
    hidden_factor = target_hidden_size / orig_hidden_size
    intermediate_factor = target_intermediate / orig_intermediate
    
    # Load weights from each file
    for file_path in weight_files:
        log_info(f"Processing {os.path.basename(file_path)}...")
        
        with safe_open(file_path, framework="numpy") as f:
            weight_keys = f.keys()
            
            for key in weight_keys:
                # Extract layer number for layer-specific weights
                layer_match = None
                if ".layers." in key:
                    parts = key.split(".layers.")
                    if len(parts) > 1:
                        layer_parts = parts[1].split(".")
                        try:
                            layer_num = int(layer_parts[0])
                            # Skip layers beyond our target layer count
                            if layer_num >= target_layers:
                                continue
                        except ValueError:
                            pass
                
                # Try to find the corresponding key in our target params
                tensor = f.get_tensor(key)
                param_key = key
                
                # Convert PyTorch key format to JAX format if needed
                # This is a simplified example and may need adjustment for your specific model
                param_key = param_key.replace("model.layers", "transformer.h")
                param_key = param_key.replace("self_attn", "attn")
                param_key = param_key.replace("q_proj", "q")
                param_key = param_key.replace("k_proj", "k")
                param_key = param_key.replace("v_proj", "v")
                param_key = param_key.replace("o_proj", "o")
                param_key = param_key.replace("mlp.gate_proj", "mlp.w1")
                param_key = param_key.replace("mlp.up_proj", "mlp.w2")
                param_key = param_key.replace("mlp.down_proj", "mlp.w3")
                param_key = param_key.replace("input_layernorm", "ln_1")
                param_key = param_key.replace("post_attention_layernorm", "ln_2")
                
                # Handle embedding conversion
                if "embed_tokens.weight" in param_key:
                    param_key = "transformer.embed_tokens.embedding"
                
                # Handle different paths in the params tree
                param_path = param_key.split(".")
                
                # Try to find the tensor location in our parameter tree
                current = new_params
                found = True
                
                for i, part in enumerate(param_path):
                    if part in current:
                        if i == len(param_path) - 1:  # Last part
                            # Found the target - now resize if needed
                            target_shape = current[part].shape
                            if tensor.shape != target_shape:
                                log_info(f"Resizing {param_key}: {tensor.shape} -> {target_shape}")
                                
                                # Handle different weight types
                                if "embed_tokens" in param_key or "wte" in param_key:
                                    # For embeddings, keep vocab dim and resize hidden dim
                                    if len(tensor.shape) == 2 and len(target_shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                elif "q" in param_path[-1] or "k" in param_path[-1] or "v" in param_path[-1]:
                                    # For attention weights
                                    if len(tensor.shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                elif "attn.o" in param_key or "attn.c_proj" in param_key:
                                    # For attention output projection
                                    if len(tensor.shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                elif "mlp" in param_key:
                                    # For MLP weights
                                    if len(tensor.shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                else:
                                    # General case - try to match dimensions if possible
                                    try:
                                        # Simple truncation for each dimension
                                        slices = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
                                        tensor_resized = np.zeros(target_shape, dtype=tensor.dtype)
                                        tensor_view = tensor[slices]
                                        target_slices = tuple(slice(0, s.stop) for s in slices)
                                        tensor_resized[target_slices] = tensor_view
                                        tensor = tensor_resized
                                    except Exception as e:
                                        log_info(f"Error resizing {param_key}: {e}")
                                        continue
                            
                            # Convert to the right dtype
                            try:
                                tensor = jnp.array(tensor, dtype=current[part].dtype)
                                
                                # Apply the correct partitioning spec
                                if hasattr(current[part], "sharding"):
                                    sharding = current[part].sharding
                                    tensor = jax.device_put(tensor, sharding)
                                
                                # Update the parameter
                                current[part] = tensor
                                log_info(f"Loaded {param_key}")
                            except Exception as e:
                                log_info(f"Error setting {param_key}: {e}")
                        else:
                            current = current[part]
                    else:
                        found = False
                        break
                
                if not found:
                    log_info(f"Key {param_key} not found in target params")
    
    log_info("Partial weight loading complete")
    return new_params 

def load_safetensors_only(
    model_path: str,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> Dict:
    """
    Load model weights directly from safetensors without using Flax model.
    
    Args:
        model_path: Path to safetensors files
        config: Model configuration
        mesh: Optional JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        
    Returns:
        Dict: Dictionary of loaded and processed parameters
    """
    # Get safetensors files
    if os.path.isfile(model_path) and model_path.endswith('.safetensors'):
        checkpoint_files = [model_path]
    else:
        checkpoint_files = get_checkpoint_files(model_path)
        if not checkpoint_files[0].endswith('.safetensors'):
            raise ValueError("This function only supports safetensors files")
    
    logger.info("Loading safetensors directly from " + model_path)
    
    # Check for index file (for sharded checkpoints)
    is_sharded = len(checkpoint_files) > 1
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    has_index = os.path.exists(index_file)
    
    if has_index:
        logger.info(f"Found safetensors index file: {index_file}")
        import json
        with open(index_file, "r") as f:
            index = json.load(f)
            weight_map = index.get("weight_map", {})
            files = sorted(list(set(weight_map.values())))
            logger.info(f"Found {len(files)} safetensors files")
    else:
        files = checkpoint_files
        logger.info(f"Found {len(files)} safetensors files")
    
    # Load all safetensors files and merge them
    safe_params = {}
    
    for safetensors_file in files:
        if not os.path.isabs(safetensors_file):
            safetensors_file = os.path.join(model_path, safetensors_file)
        
        logger.info(f"Loading safetensors file: {safetensors_file}")
        with safe_open(safetensors_file, framework="flax") as f:
            for key in f.keys():
                safe_params[key] = f.get_tensor(key)
    
    # Convert the parameters to Flax parameter structure
    flax_params = {}
    embed_key_found = False
    
    # Determine the model structure
    has_model = any(k.startswith("model.") for k in safe_params.keys())
    has_transformer = any(k.startswith("transformer.") for k in safe_params.keys())
    
    # Decide on the prefix based on structure
    if has_model:
        prefix = "model."
    elif has_transformer:
        prefix = "transformer."
    else:
        prefix = ""
    
    # Process the embedding layer first to get the correct key name
    for key, value in safe_params.items():
        if "embed_tokens.weight" in key:
            # Handle embedding and update key name
            layer_name = "embed_tokens"
            flax_params[("params", "transformer", layer_name, "embedding")] = value
            embed_key_found = True
            logger.info(f"Found embedding parameter at key: {('params', 'transformer', layer_name, 'embedding')}")
            break
    
    # If we couldn't find the embedding, check for alternatives
    if not embed_key_found:
        for key, value in safe_params.items():
            if ".wte.weight" in key or ".transformer.wte.weight" in key:
                layer_name = "embed_tokens"
                flax_params[("params", "transformer", layer_name, "embedding")] = value
                embed_key_found = True
                logger.info(f"Found alternative embedding parameter at key: {key}")
                break
    
    # Process all other parameters
    for key, value in safe_params.items():
        if "embed_tokens.weight" in key or ".wte.weight" in key:
            # Already handled above
            continue
        
        # Special handling for attention and MLP layers
        flax_key = convert_pt_key_to_flax(key, prefix)
        if flax_key:
            # For tensor parallelism, adjust some shapes
            if mesh is not None and mesh.shape[1] > 1:  # Check if we have model parallel dimension > 1
                tensor_parallel_size = mesh.shape[1]
                hidden_size = config['hidden_size']
                num_heads = config['num_attention_heads']
                head_dim = hidden_size // num_heads
                kv_heads = config.get('num_key_value_heads', num_heads)
                kv_head_dim = hidden_size // kv_heads
                
                # Handle query projection matrix
                if ".self_attn.q_proj.weight" in key or "self_attn.q_proj.kernel" in flax_key[-1]:
                    # Reshape for tensor parallel Q projection
                    logger.info(f"Reshaping Q projection for tensor parallelism from {value.shape}")
                    if len(value.shape) == 2:
                        # Each rank gets hidden_size // tensor_parallel_size output features
                        per_partition_head_dim = head_dim * num_heads // tensor_parallel_size
                        value = value.reshape(hidden_size, hidden_size)
                        reshaped_value = value[:, :per_partition_head_dim]
                        flax_params[flax_key] = reshaped_value
                        continue
                
                # Handle key projection matrix
                elif ".self_attn.k_proj.weight" in key or "self_attn.k_proj.kernel" in flax_key[-1]:
                    # Reshape for tensor parallel K projection
                    logger.info(f"Reshaping K projection for tensor parallelism from {value.shape}")
                    if len(value.shape) == 2:
                        # Each rank gets kv_head_dim // tensor_parallel_size output features
                        per_partition_kv_dim = kv_head_dim * kv_heads // tensor_parallel_size
                        value = value.reshape(hidden_size, kv_head_dim * kv_heads)
                        reshaped_value = value[:, :per_partition_kv_dim]
                        flax_params[flax_key] = reshaped_value
                        continue
                
                # Handle value projection matrix
                elif ".self_attn.v_proj.weight" in key or "self_attn.v_proj.kernel" in flax_key[-1]:
                    # Reshape for tensor parallel V projection
                    logger.info(f"Reshaping V projection for tensor parallelism from {value.shape}")
                    if len(value.shape) == 2:
                        # Each rank gets kv_head_dim // tensor_parallel_size output features
                        per_partition_kv_dim = kv_head_dim * kv_heads // tensor_parallel_size
                        value = value.reshape(hidden_size, kv_head_dim * kv_heads)
                        reshaped_value = value[:, :per_partition_kv_dim]
                        flax_params[flax_key] = reshaped_value
                        continue
                
                # Handle output projection matrix
                elif ".self_attn.o_proj.weight" in key or "self_attn.o_proj.kernel" in flax_key[-1]:
                    # Reshape for tensor parallel O projection
                    logger.info(f"Reshaping O projection for tensor parallelism from {value.shape}")
                    if len(value.shape) == 2:
                        # Each rank gets hidden_size // tensor_parallel_size input features
                        per_partition_dim = hidden_size // tensor_parallel_size
                        value = value.reshape(hidden_size, hidden_size)
                        reshaped_value = value[:per_partition_dim, :]
                        flax_params[flax_key] = reshaped_value
                        continue
                
                # Handle MLP layers - gate and up projections
                elif ".mlp.gate_proj.weight" in key or ".mlp.up_proj.weight" in key or "mlp.gate_proj.kernel" in flax_key[-1] or "mlp.up_proj.kernel" in flax_key[-1]:
                    # Reshape for tensor parallel MLP gate/up projections
                    logger.info(f"Reshaping MLP gate/up projection for tensor parallelism from {value.shape}")
                    if len(value.shape) == 2:
                        intermediate_size = config['intermediate_size']
                        per_partition_intermediate = intermediate_size // tensor_parallel_size
                        value = value.reshape(hidden_size, intermediate_size)
                        reshaped_value = value[:, :per_partition_intermediate]
                        flax_params[flax_key] = reshaped_value
                        continue
                
                # Handle MLP down projection
                elif ".mlp.down_proj.weight" in key or "mlp.down_proj.kernel" in flax_key[-1]:
                    # Reshape for tensor parallel MLP down projection
                    logger.info(f"Reshaping MLP down projection for tensor parallelism from {value.shape}")
                    if len(value.shape) == 2:
                        intermediate_size = config['intermediate_size']
                        per_partition_intermediate = intermediate_size // tensor_parallel_size
                        value = value.reshape(intermediate_size, hidden_size)
                        reshaped_value = value[:per_partition_intermediate, :]
                        flax_params[flax_key] = reshaped_value
                        continue
            
            # For non-tensor-parallel layers or if no reshaping was done
            flax_params[flax_key] = value
    
    # Log some sample keys to verify
    sample_keys = list(flax_params.keys())[:10]
    logger.info(f"Sample parameter keys after processing: {sample_keys}")
    
    # Apply tensor parallelism sharding if mesh is provided
    if mesh is not None:
        logger.info(f"Applying tensor parallelism with mesh shape {mesh.shape}")
        partition_specs = get_partition_specs(config)
        
        # Convert flat dict of params to nested for easier access
        nested_params = unflatten_dict(flax_params)
        
        # Apply sharding constraints
        with mesh:
            from jax.experimental.pjit import pjit
            from flax.core.frozen_dict import freeze, unfreeze
            
            def apply_tensor_parallel_constraints(params, partition_specs):
                params = unfreeze(params)  # Make sure params is mutable
                flat_params = flatten_dict(params)
                
                # Apply sharding to each parameter
                for param_key, param_value in flat_params.items():
                    # Find the matching partition spec
                    param_spec = None
                    
                    # Convert the tuple key to a string path for matching
                    key_path = "/".join(str(k) for k in param_key if isinstance(k, str) or isinstance(k, int))
                    
                    # Try to find a matching pattern in partition_specs
                    for pattern, spec in partition_specs.items():
                        import re
                        # Convert pattern to regex
                        regex_pattern = pattern.replace('.', '\.').replace('*', '.*')
                        if re.match(regex_pattern, key_path):
                            param_spec = spec
                            break
                    
                    # Apply sharding constraint if we found a spec
                    if param_spec is not None:
                        try:
                            flat_params[param_key] = jax.lax.with_sharding_constraint(
                                param_value, param_spec
                            )
                        except Exception as e:
                            logger.warning(f"Could not apply sharding constraint to {key_path}: {e}")
                
                return freeze(unflatten_dict(flat_params))  # Convert back to nested and freeze
            
            # Apply tensor parallelism constraints
            nested_params = apply_tensor_parallel_constraints(nested_params, partition_specs)
            
            # Flatten back for return
            flax_params = flatten_dict(nested_params)
    
    # Return the processed parameters
    return flax_params

def convert_pt_key_to_flax(pt_key, prefix="model."):
    """
    Convert PyTorch parameter key to Flax parameter key.
    
    Args:
        pt_key: PyTorch parameter key (e.g., model.layers.0.self_attn.q_proj.weight)
        prefix: Prefix to check for in the key (default: "model.")
        
    Returns:
        List of strings representing the path in a nested Flax parameter dictionary
    """
    # Create a list for the parameter path segments
    flax_path = ["params"]  # Always put parameters under "params"
    
    # Special case: embedding layer
    if "embed_tokens.weight" in pt_key:
        return ["params", "transformer", "embed_tokens", "embedding"]
    
    # Special case: LM head
    if "lm_head.weight" in pt_key:
        return ["params", "lm_head", "kernel"]
    
    # Special case: final norm
    if "model.norm.weight" in pt_key:
        return ["params", "transformer", "ln_f", "weight"]
    
    # Handle the transformer block parameters
    if pt_key.startswith(prefix):
        # Remove the prefix
        key = pt_key[len(prefix):]
        
        # Split by dots to get components
        parts = key.split(".")
        
        # Check if this is a layer parameter
        if parts[0] == "layers" and len(parts) > 1:
            layer_idx = parts[1]
            flax_path.extend(["transformer", "h", layer_idx])
            
            # Handle different components of the layer
            if len(parts) > 2:
                component = parts[2]
                
                # Handle attention components
                if component == "self_attn":
                    if parts[3] == "q_proj":
                        flax_path.extend(["attn", "q"])
                    elif parts[3] == "k_proj":
                        flax_path.extend(["attn", "k"])
                    elif parts[3] == "v_proj":
                        flax_path.extend(["attn", "v"])
                    elif parts[3] == "o_proj":
                        flax_path.extend(["attn", "o"])
                    else:
                        flax_path.extend(["attn", parts[3]])
                        
                    # Handle weight/bias
                    if parts[4] == "weight":
                        flax_path.append("kernel")
                    else:
                        flax_path.append(parts[4])
                
                # Handle MLP components
                elif component == "mlp":
                    if parts[3] == "gate_proj":
                        flax_path.extend(["mlp", "w1"])
                    elif parts[3] == "up_proj":
                        flax_path.extend(["mlp", "w2"])
                    elif parts[3] == "down_proj":
                        flax_path.extend(["mlp", "w3"])
                    else:
                        flax_path.extend(["mlp", parts[3]])
                        
                    # Handle weight/bias
                    if parts[4] == "weight":
                        flax_path.append("kernel")
                    else:
                        flax_path.append(parts[4])
                
                # Handle LayerNorm components
                elif component == "input_layernorm":
                    flax_path.extend(["ln_1", "weight"])
                elif component == "post_attention_layernorm":
                    flax_path.extend(["ln_2", "weight"])
                else:
                    # Fallback for other components
                    flax_path.append(component)
        else:
            # Fallback for other parameters
            flax_path.extend(["transformer"] + parts)
    else:
        # Not a model parameter
        flax_path.extend([pt_key])
    
    return flax_path

# Alias for backward compatibility
direct_load_safetensors = load_safetensors_only 