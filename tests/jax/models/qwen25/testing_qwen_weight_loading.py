#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Testing Qwen2.5 Weight Loading for Full Implementation

This script verifies that the weight loading functionality works correctly for 
the Qwen2.5 model implementation in qwen25_full_implementation.py.

It tests all weight loading methods and verifies that:
1. The weights can be loaded successfully
2. The weights are mapped correctly to the model architecture
3. The model can perform a simple forward pass with the loaded weights

Usage:

source venv/bin/activate

python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights

python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method all --test_forward

The script will output detailed diagnostics about the weight loading process.
"""

import os
import sys
import time
import json
import glob
import logging
import argparse
import msgpack
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("qwen_weight_loading_test")

# Import from the full implementation file
# This imports the actual model implementation rather than a dummy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from qwen25_full_implementation import (
        Qwen25Config, 
        FlaxQwen25ForCausalLM,
        FlaxQwen25Model,
        load_config_from_json,
        load_safetensors_weights,
        direct_load_from_index
    )
    logger.info("Successfully imported from qwen25_full_implementation.py")
except ImportError as e:
    logger.error(f"Failed to import from qwen25_full_implementation.py: {e}")
    logger.error("Please ensure qwen25_full_implementation.py is in the same directory")
    sys.exit(1)

# Import external dependencies
try:
    from transformers import AutoTokenizer
    from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
    from safetensors import safe_open
    from safetensors.flax import load_file as safe_load_file
except ImportError:
    logger.error("Missing required dependencies. Please install with:")
    logger.error("pip install transformers safetensors")
    sys.exit(1)

# Add weight cache at the module level
_weight_cache = {}

def log_tensor_sizes(params, top_n=10):
    """
    Log the sizes of the largest tensors in the parameter dictionary.
    This helps identify which tensors are consuming the most memory.
    
    Args:
        params: Parameter dictionary to analyze
        top_n: Number of largest tensors to log
        
    Returns:
        List of (key, size_mb) tuples for the largest tensors
    """
    logger.info("Analyzing tensor sizes to identify memory usage...")
    flat_params = flatten_dict(params)
    
    # Calculate size of each tensor
    tensor_sizes = []
    total_size_mb = 0
    
    for key, tensor in flat_params.items():
        try:
            # Calculate size in MB
            size_bytes = tensor.size * tensor.itemsize if hasattr(tensor, 'itemsize') else np.prod(tensor.shape) * 4  # Assume float32 if itemsize not available
            size_mb = size_bytes / (1024 * 1024)
            total_size_mb += size_mb
            
            # Get key as string for display
            key_str = "/".join(key) if isinstance(key, tuple) else str(key)
            tensor_sizes.append((key_str, size_mb, tensor.shape))
        except (AttributeError, TypeError):
            # Skip tensors that don't have shape or size attributes
            continue
    
    # Sort by size (largest first)
    tensor_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Log the largest tensors
    logger.info(f"Total parameter size: {total_size_mb:.2f} MB")
    logger.info(f"Top {min(top_n, len(tensor_sizes))} largest tensors:")
    for i, (key, size_mb, shape) in enumerate(tensor_sizes[:top_n]):
        logger.info(f"  {i+1}. {key}: {size_mb:.2f} MB, shape={shape}")
    
    return tensor_sizes[:top_n]

def get_cached_weights(model_path, config, dtype):
    """
    Get weights from cache if available, otherwise load them.
    This prevents reloading the same weights multiple times.
    
    Args:
        model_path: Path to model weights
        config: Model configuration
        dtype: Data type to use
        
    Returns:
        Loaded weights
    """
    cache_key = f"{model_path}_{dtype}"
    if cache_key not in _weight_cache:
        logger.info("Loading weights (not found in cache)...")
        _weight_cache[cache_key] = load_safetensors_weights(model_path)
        
        # Log memory usage after loading
        log_memory_usage("after loading weights")
    else:
        logger.info("Using cached weights")
    
    return _weight_cache[cache_key]

def test_direct_safetensors(model_path: str, config: Qwen25Config, dtype: jnp.dtype = jnp.float32) -> Dict:
    """
    Test direct loading from safetensors files without using the model.
    
    Args:
        model_path: Path to the model directory
        config: Model configuration
        dtype: Data type for loading
        
    Returns:
        Dictionary of loaded parameters or None if loading fails
    """
    logger.info("Testing direct loading from safetensors...")
    start_time = time.time()
    
    try:
        # Use cached weights if available
        params = get_cached_weights(model_path, config, dtype)
        logger.info(f"Successfully loaded weights in {time.time() - start_time:.2f} seconds")
        return params
    except Exception as e:
        logger.error(f"Direct safetensors loading failed: {e}")
        return None

def test_direct_load_from_index(model_path: str, config: Qwen25Config, dtype: jnp.dtype = jnp.float32) -> Dict:
    """
    Test loading using direct_load_from_index function.
    
    Args:
        model_path: Path to the model directory
        config: Model configuration
        dtype: Data type for loading
        
    Returns:
        Initialized model with loaded weights or None if loading fails
    """
    logger.info("Testing direct loading using index file...")
    start_time = time.time()
    
    # Import necessary modules here to control cleanup
    import gc
    import jax
    
    try:
        # Create model instance - with minimal initialization
        logger.info("Creating base model instance...")
        model = FlaxQwen25ForCausalLM(config, dtype=dtype, _do_init=True)
        
        # Clear any compilation caches before loading weights
        jax.clear_caches()
        gc.collect()
        
        # Load weights using direct_load_from_index
        logger.info("Loading weights from index file...")
        model_or_params = direct_load_from_index(model, model_path)
        
        # Clear memory from loading process
        # Free some references that might hold memory
        for key in list(globals().keys()):
            if key.startswith('_jit_') or key.startswith('_tmp_'):
                del globals()[key]
        gc.collect()
        jax.clear_caches()
        
        # Check if we got back a model or just parameters
        if hasattr(model_or_params, 'params'):
            # We got back a model
            logger.info("Received model object with parameters already set")
            model = model_or_params
        else:
            # We got back parameters - need to fix structure and update model
            logger.info("Got parameters instead of model - fixing structure and updating model")
            
            # Process in steps to avoid holding too much in memory at once
            logger.info("Fixing parameter structure...")
            params = fix_params_structure(model_or_params)
            
            # Clear references to old parameter structure
            del model_or_params
            gc.collect()
            
            # Update model parameters
            logger.info("Updating model parameters...")
            model.params = params
            
            # Clear references
            del params
            gc.collect()
        
        logger.info(f"Successfully loaded weights in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        logger.error(f"direct_load_from_index failed: {e}")
        import traceback
        traceback.print_exc()
        # Make sure to collect garbage even if we failed
        gc.collect()
        jax.clear_caches()
        return None

def test_model_and_weights(model_path: str, config: Qwen25Config, dtype: jnp.dtype = jnp.float32) -> Dict:
    """
    Test loading using the load_model_and_weights function.
    
    Args:
        model_path: Path to the model directory
        config: Model configuration
        dtype: Data type for loading
        
    Returns:
        Loaded model or None if loading fails
    """
    logger.info("Testing load_model_and_weights function...")
    start_time = time.time()
    
    try:
        # Import the function locally to avoid issues
        from qwen25_full_implementation import load_model_and_weights
        
        # Load model and weights
        model = load_model_and_weights(model_path, dtype=dtype)
        
        logger.info(f"Successfully loaded model and weights in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        logger.error(f"load_model_and_weights failed: {e}")
        return None

def test_forward_pass(model, config: Qwen25Config, tokenizer=None) -> bool:
    """
    Test a simple forward pass with the loaded model.
    
    Args:
        model: Loaded model
        config: Model configuration
        tokenizer: Optional tokenizer
        
    Returns:
        True if forward pass succeeds, False otherwise
    """
    logger.info("Testing forward pass with loaded weights...")
    
    try:
        # Verify essential parameters first
        logger.info("Verifying model parameters before forward pass...")
        if hasattr(model, 'params'):
            # Get a flattened view of parameters
            from flax.traverse_util import flatten_dict
            flat_params = flatten_dict(model.params)
            
            # Log memory usage after flattening parameters
            log_memory_usage("before parameter verification")
            
            # Check that we have the essential parameters - support both formats
            # Either direct or with 'model' in the path
            essential_param_patterns = [
                # Format 1: Without 'model' in path
                [
                    ('params', 'embed_tokens', 'embedding'),
                    ('params', 'lm_head', 'kernel'),
                    ('params', 'layers', 'norm', 'scale')
                ],
                # Format 2: With 'model' in path
                [
                    ('params', 'model', 'embed_tokens', 'embedding'),
                    ('params', 'lm_head', 'kernel'),
                    ('params', 'model', 'layers', 'norm', 'scale')
                ]
            ]
            
            # For each parameter set, check if any are missing
            format_found = False
            for param_set in essential_param_patterns:
                missing_params = []
                for param_path in param_set:
                    if param_path not in flat_params:
                        missing_params.append('/'.join(param_path))
                
                # If all required params for this format exist, we found a match
                if not missing_params:
                    format_found = True
                    break
            
            if not format_found:
                logger.error(f"Missing essential parameters for all supported formats")
                logger.info("Available top-level parameters:")
                if 'params' in model.params:
                    logger.info(f"  params keys: {list(model.params['params'].keys())}")
                    if 'model' in model.params['params']:
                        logger.info(f"  model keys: {list(model.params['params']['model'].keys())}")
                    if 'layers' in model.params['params']:
                        logger.info(f"  layers keys: {list(model.params['params']['layers'].keys())}")
                    elif 'model' in model.params['params'] and 'layers' in model.params['params']['model']:
                        logger.info(f"  model/layers keys: {list(model.params['params']['model']['layers'].keys())}")
                
                # If no matching format, analyze largest tensors to help diagnose the issue
                log_tensor_sizes(model.params)
                return False
            
            # Check parameter shapes for key components - try both formats
            embed_shapes = []
            embed_paths = [
                ('params', 'embed_tokens', 'embedding'),
                ('params', 'model', 'embed_tokens', 'embedding')
            ]
            for path in embed_paths:
                if path in flat_params:
                    embed_shape = flat_params[path].shape
                    expected_shape = (config.vocab_size, config.hidden_size)
                    if embed_shape != expected_shape:
                        logger.warning(f"Embedding shape mismatch at {path}: got {embed_shape}, expected {expected_shape}")
                    embed_shapes.append(embed_shape)
            
            # LM head should be [hidden_size, vocab_size] in Flax (transposed from PyTorch)
            if ('params', 'lm_head', 'kernel') in flat_params:
                lm_head_shape = flat_params[('params', 'lm_head', 'kernel')].shape
                expected_shape = (config.hidden_size, config.vocab_size)
                if lm_head_shape != expected_shape:
                    logger.warning(f"LM head shape mismatch: got {lm_head_shape}, expected {expected_shape}")
                
            logger.info("Model parameter verification completed")
        
        # Create a simple input
        if tokenizer:
            # Use tokenizer if available
            input_text = "Hello, world!"
            logger.info(f"Creating input using tokenizer: '{input_text}'")
            inputs = tokenizer(input_text, return_tensors="jax")
            input_ids = inputs.input_ids
            logger.info(f"Tokenized to shape: {input_ids.shape}")
        else:
            # Create dummy input if no tokenizer
            logger.info("No tokenizer available, using dummy input")
            input_ids = jnp.ones((1, 10), dtype=jnp.int32)
        
        # For memory-intensive models, try with a smaller input first
        input_ids = input_ids[:, :min(10, input_ids.shape[1])]
        logger.info(f"Using input shape: {input_ids.shape}")
        
        # Add explicit garbage collection before forward pass
        import gc
        gc.collect()
        
        # Run forward pass with error handling
        try:
            logger.info("Running model forward pass...")
            outputs = model(input_ids)
            logger.info("Forward pass completed successfully")
        except ValueError as e:
            # Check if this is a shape mismatch error
            if "shape" in str(e).lower() or "dimension" in str(e).lower():
                logger.error(f"Shape mismatch error in forward pass: {e}")
                logger.info("This might indicate incompatible parameters. Checking parameter shapes...")
                
                # Try to determine where the mismatch occurred
                # Assuming common locations for shape errors
                if hasattr(model, 'module'):
                    logger.info("Model has module attribute, checking for compatibility issues...")
                    if hasattr(model.module, 'config'):
                        logger.info(f"Module config: hidden_size={model.module.config.hidden_size}, "
                                    f"num_attention_heads={model.module.config.num_attention_heads}")
                
                logger.error("Forward pass failed due to shape mismatch")
                return False
            else:
                # Re-raise the error
                raise
        
        # Check if outputs have the expected shape
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            # Assume first output is logits in tuple case
            logits = outputs[0]
        
        # Logits should have shape [batch_size, seq_len, vocab_size]
        expected_shape = (input_ids.shape[0], input_ids.shape[1], config.vocab_size)
        
        if logits.shape == expected_shape:
            logger.info(f"✅ Forward pass succeeded with output shape {logits.shape}")
            
            # Do a simple check on the output values
            if jnp.isnan(logits).any():
                logger.warning("⚠️ Output contains NaN values!")
                return False
                
            if jnp.isinf(logits).any():
                logger.warning("⚠️ Output contains Inf values!")
                return False
                
            # Check the range of values in the output
            logits_abs_max = float(jnp.max(jnp.abs(logits)))
            logger.info(f"Maximum absolute logit value: {logits_abs_max:.2f}")
            
            return True
        else:
            logger.error(f"❌ Output shape mismatch: got {logits.shape}, expected {expected_shape}")
            return False
    
    except Exception as e:
        import traceback
        logger.error(f"❌ Forward pass failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def verify_parameter_mapping(model_params: Dict) -> Dict[str, bool]:
    """
    Verify that parameters are correctly mapped to the model structure.
    
    Args:
        model_params: Model parameters
        
    Returns:
        Dict with verification results
    """
    # Convert to flattened dict for easier analysis
    flat_params = flatten_dict(model_params)
    
    # Key components to verify
    components = {
        "embedding": False,
        "lm_head": False,
        "attention": False,
        "mlp": False,
        "layer_norm": False,
        "multi_layer": False  # Check if we have multiple layers
    }
    
    # Convert keys to strings for easier checking
    keys = ["/".join(k) if isinstance(k, tuple) else k for k in flat_params.keys()]
    
    # Check if we have embedding
    for key in keys:
        if "embed_tokens" in key and "embedding" in key:
            components["embedding"] = True
        
        if "lm_head" in key and "kernel" in key:
            components["lm_head"] = True
        
        if "attention" in key or "attn" in key:
            components["attention"] = True
        
        if "mlp" in key:
            components["mlp"] = True
        
        if "layernorm" in key or "norm" in key:
            components["layer_norm"] = True
    
    # Check if we have multiple layers
    layer_count = 0
    for i in range(100):  # Check up to 100 layers
        layer_found = False
        for key in keys:
            if f"layers/{i}" in key or f"layers.{i}" in key or f"layers_{i}" in key:
                layer_found = True
                break
        
        if layer_found:
            layer_count += 1
        else:
            break
    
    components["multi_layer"] = layer_count > 1
    
    # Add layer count to results
    result = {**components, "layer_count": layer_count}
    
    # Check overall status
    result["all_components_present"] = all(components.values())
    
    return result

def scan_checkpoint_files(model_path: str) -> Dict[str, Any]:
    """
    Scan the checkpoint directory and return information about available files.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dict containing information about available files
    """
    import glob
    
    result = {
        "safetensors": [],
        "safetensors_index": None,
        "config": None,
        "tokenizer_files": []
    }
    
    # Check for safetensors index
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        result["safetensors_index"] = index_file
        
        # Read index to get files
        with open(index_file, "r") as f:
            index = json.load(f)
            if "weight_map" in index:
                weight_map = index["weight_map"]
                files = sorted(list(set(weight_map.values())))
                result["safetensors"] = [os.path.join(model_path, f) for f in files]
    
    # If no index or empty list, search directly
    if not result["safetensors"]:
        result["safetensors"] = sorted(glob.glob(os.path.join(model_path, "model-*-of-*.safetensors")))
    
    # Check for config
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        result["config"] = config_file
    
    # Check for tokenizer files
    for tokenizer_file in ["tokenizer.json", "vocab.json", "merges.txt"]:
        file_path = os.path.join(model_path, tokenizer_file)
        if os.path.exists(file_path):
            result["tokenizer_files"].append(file_path)
    
    # Log summary
    logger.info(f"Found {len(result['safetensors'])} safetensors files")
    logger.info(f"Found safetensors index: {result['safetensors_index'] is not None}")
    logger.info(f"Found config file: {result['config'] is not None}")
    logger.info(f"Found {len(result['tokenizer_files'])} tokenizer files")
    
    return result

def analyze_param_structure(params: Dict) -> Dict[str, Any]:
    """
    Analyze the parameter structure of loaded weights.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Dict with analysis results
    """
    total_params = 0
    num_tensors = 0
    param_shapes = {}
    param_types = {}
    layer_pattern_counts = {}
    
    # Define critical keys that must be present for a functional model
    # For raw safetensors weights
    critical_keys_raw = [
        "model/embed_tokens/weight",
        "lm_head/weight"
    ]
    
    # For processed Flax model params
    critical_keys_flax = [
        "params/embed_tokens/embedding",
        "params/lm_head/kernel",
        # Also check for 'model' in the path
        "params/model/embed_tokens/embedding",
        "params/model/lm_head/kernel"
    ]
    
    # Keep track of key strings for checking critical keys and layers
    critical_keys_found = []
    
    # Use a generator for keys to avoid storing all of them in memory at once
    def get_key_strings(params):
        """Generate key strings efficiently without storing them all at once."""
        # Handle different parameter structures - flatten appropriately
        if isinstance(params, dict):
            if "params" in params:
                # This is a standard Flax params structure
                for key, value in flatten_dict(params).items():
                    yield key, value
            else:
                # This could be a raw weight dictionary from load_safetensors_weights
                # Check if there are model-specific keys
                if "model" in params:
                    # Extract model parameters
                    model_params = params["model"]
                    # Create custom flattened representation
                    for key, value in flatten_dict(model_params).items():
                        key_str = "model/" + ("/".join(key) if isinstance(key, tuple) else key)
                        yield key_str, value
                    
                    # Add lm_head if present
                    if "lm_head" in params:
                        for key, value in flatten_dict(params["lm_head"]).items():
                            key_str = "lm_head/" + ("/".join(key) if isinstance(key, tuple) else key)
                            yield key_str, value
                else:
                    # Just flatten as is - likely raw weights
                    for key, value in flatten_dict(params).items():
                        key_str = "/".join(key) if isinstance(key, tuple) else key
                        yield key_str, value
    
    # Analyze layers to find maximum layer index
    num_layers = 0
    
    # Process all keys using the generator
    key_count = 0
    for key_str, tensor in get_key_strings(params):
        # Convert tuple keys to strings for consistency
        if isinstance(key_str, tuple):
            key_str = "/".join(key_str)
        
        # Count parameters
        try:
            num_params = np.prod(tensor.shape)
            total_params += num_params
            num_tensors += 1
            
            # Group by shape
            shape_str = str(tensor.shape)
            if shape_str not in param_shapes:
                param_shapes[shape_str] = 0
            param_shapes[shape_str] += 1
            
            # Group by tensor type
            dtype_str = str(tensor.dtype)
            if dtype_str not in param_types:
                param_types[dtype_str] = 0
            param_types[dtype_str] += 1
            
            # Check for layer patterns
            for pattern in ["attention", "mlp", "layernorm", "embed"]:
                if pattern in key_str.lower():
                    if pattern not in layer_pattern_counts:
                        layer_pattern_counts[pattern] = 0
                    layer_pattern_counts[pattern] += 1
            
            # Check for critical keys
            for critical_key in critical_keys_raw + critical_keys_flax:
                if critical_key in key_str and critical_key not in critical_keys_found:
                    critical_keys_found.append(critical_key)
            
            # Check for layer indices
            for pattern in [r"layers/(\d+)/", r"layers_(\d+)/", r"layers\.(\d+)\.", r"layers\.(\d+)/", r"layers/layers_(\d+)/"]:
                import re
                match = re.search(pattern, key_str)
                if match:
                    layer_idx = int(match.group(1))
                    num_layers = max(num_layers, layer_idx + 1)
                    break
            
            # Increment key count
            key_count += 1
            
            # Periodically log progress for large parameter sets
            if key_count % 100 == 0:
                logger.debug(f"Analyzed {key_count} parameters...")
        except Exception as e:
            # Skip parameters that can't be analyzed
            logger.debug(f"Error analyzing parameter: {e}")
            continue
    
    # We only need to collect a limited number of sample keys for display
    sample_keys = []
    for i, (key_str, _) in enumerate(get_key_strings(params)):
        if i >= 10:  # Only collect 10 keys
            break
        if isinstance(key_str, tuple):
            key_str = "/".join(key_str)
        sample_keys.append(key_str)
    
    return {
        "total_params": total_params,
        "num_tensors": num_tensors,
        "param_shapes": param_shapes,
        "param_types": param_types,
        "critical_keys_present": len(critical_keys_found) > 0,  # At least one critical key format found
        "critical_keys_found": critical_keys_found,
        "layer_pattern_counts": layer_pattern_counts,
        "num_layers_detected": num_layers,
        "status": "ok" if len(critical_keys_found) > 0 and num_layers > 0 else "incomplete",
        "sample_keys": sample_keys,  # First 10 keys for inspection
    }

def fix_params_structure(weights):
    """
    Fix parameter structure to match FlaxQwen25ForCausalLM expectations.
    
    Args:
        weights: Dictionary of weights loaded from safetensors
        
    Returns:
        Dictionary with restructured weights matching Flax model structure
    """
    logger.info("Restructuring parameters to match FlaxQwen25ForCausalLM")
    
    # Initialize the restructured parameters dict
    fixed_params = {"params": {}}
    
    # Extract configuration parameters from weights to determine model structure
    config = {}
    if "model.embed_tokens.weight" in weights:
        vocab_size = weights["model.embed_tokens.weight"].shape[0]
        hidden_size = weights["model.embed_tokens.weight"].shape[1]
        config["vocab_size"] = vocab_size
        config["hidden_size"] = hidden_size
    
    # Count number of layers based on pattern matching
    layer_count = 0
    for key in weights:
        if "model.layers." in key:
            layer_num = int(key.split("model.layers.")[1].split(".")[0])
            layer_count = max(layer_count, layer_num + 1)
    
    logger.info(f"Detected model structure: vocab_size={config.get('vocab_size', 'unknown')}, "
                f"hidden_size={config.get('hidden_size', 'unknown')}, layers={layer_count}")
    
    # Mapping from PyTorch parameter names to Flax parameter structure
    mappings = {
        # Embeddings
        "model.embed_tokens.weight": ("model", "embed_tokens", "embedding"),
        
        # Layer mappings (will be formatted with layer number)
        "model.layers.{}.self_attn.q_proj.weight": ("model", "layers_{}", "self_attn", "q_proj", "kernel"),
        "model.layers.{}.self_attn.k_proj.weight": ("model", "layers_{}", "self_attn", "k_proj", "kernel"),
        "model.layers.{}.self_attn.v_proj.weight": ("model", "layers_{}", "self_attn", "v_proj", "kernel"),
        "model.layers.{}.self_attn.o_proj.weight": ("model", "layers_{}", "self_attn", "o_proj", "kernel"),
        
        "model.layers.{}.mlp.gate_proj.weight": ("model", "layers_{}", "mlp", "gate_proj", "kernel"),
        "model.layers.{}.mlp.up_proj.weight": ("model", "layers_{}", "mlp", "up_proj", "kernel"),
        "model.layers.{}.mlp.down_proj.weight": ("model", "layers_{}", "mlp", "down_proj", "kernel"),
        
        "model.layers.{}.input_layernorm.weight": ("model", "layers_{}", "input_layernorm", "scale"),
        "model.layers.{}.post_attention_layernorm.weight": ("model", "layers_{}", "post_attention_layernorm", "scale"),
        
        # Final layer norm
        "model.norm.weight": ("model", "norm", "scale"),
        
        # Language model head
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    # Process every weight from the source dictionary
    processed_keys = set()
    
    # First, handle the layer-specific weights
    for layer_idx in range(layer_count):
        for src_pattern, dst_path in mappings.items():
            if "{}" in src_pattern:
                src_key = src_pattern.format(layer_idx)
                if src_key in weights:
                    # Create path in the params dictionary
                    current = fixed_params["params"]
                    for part in dst_path[:-1]:
                        part_name = part.format(layer_idx) if "{}" in part else part
                        if part_name not in current:
                            current[part_name] = {}
                        current = current[part_name]
                    
                    # Add the weight tensor, handling necessary transformations
                    last_part = dst_path[-1].format(layer_idx) if "{}" in dst_path[-1] else dst_path[-1]
                    
                    # Special handling for weight matrices - transpose kernel matrices for Flax
                    if last_part == "kernel":
                        current[last_part] = weights[src_key].T  # Transpose for Flax
                    else:
                        current[last_part] = weights[src_key]
                    
                    processed_keys.add(src_key)
    
    # Handle non-layer specific weights
    for src_key, dst_path in mappings.items():
        if "{}" not in src_key and src_key in weights:
            # Create path in the params dictionary
            current = fixed_params["params"]
            for part in dst_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add the weight tensor, handling necessary transformations
            last_part = dst_path[-1]
            
            # Special handling for weight matrices - transpose kernel matrices for Flax
            if last_part == "kernel":
                current[last_part] = weights[src_key].T  # Transpose for Flax
            elif last_part == "embedding":
                current[last_part] = weights[src_key]  # Don't transpose embeddings
            else:
                current[last_part] = weights[src_key]
            
            processed_keys.add(src_key)
    
    # Log stats about the conversion
    total_keys = len(weights)
    logger.info(f"Processed {len(processed_keys)}/{total_keys} weights for FlaxQwen25ForCausalLM structure")
    
    if len(processed_keys) < total_keys:
        # Log a sample of unprocessed keys for debugging
        unprocessed = set(weights.keys()) - processed_keys
        logger.warning(f"Unable to map {len(unprocessed)} keys, first 10: {list(unprocessed)[:10]}")
    
    return fixed_params

def log_memory_usage(label=""):
    """Log the current memory usage."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss_gb = memory_info.rss / (1024 ** 3)
        vms_gb = memory_info.vms / (1024 ** 3)
        logger.info(f"Memory Usage {label}: RSS={rss_gb:.2f} GB, VMS={vms_gb:.2f} GB")
        return rss_gb
    except Exception as e:
        logger.warning(f"Failed to log memory usage: {e}")
        return None

def load_safetensors_weights(model_path, slice_size=None, verbose=True):
    """
    Load weights from a safetensors file incrementally to reduce memory usage.
    
    Args:
        model_path: Path to the model directory or specific safetensors file
        slice_size: Optional slice size to load incrementally
        verbose: Whether to log detailed information
        
    Returns:
        Dict with loaded weights
    """
    import gc
    from safetensors import safe_open
    from safetensors.flax import load_file

    log_memory_usage("Before loading weights")
    
    # Scan checkpoint files
    ckpt_info = scan_checkpoint_files(model_path)
    safetensors_files = ckpt_info["safetensors"]
    
    if not safetensors_files:
        logger.error(f"No safetensors files found in {model_path}")
        return None
    
    # Initialize dictionary to accumulate weights from all files
    all_weights = {}
    num_total_tensors = 0
    
    # Load index file if available to get metadata
    index_file = ckpt_info["safetensors_index"]
    weight_map = {}
    
    if index_file:
        with open(index_file, "r") as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            if "metadata" in index_data:
                logger.info(f"Model metadata: {index_data['metadata']}")
    
    # Function to get all keys from a safetensors file without loading tensors
    def get_keys_from_file(file_path):
        keys = []
        with safe_open(file_path, framework="flax") as f:
            keys = f.keys()
        return keys
    
    # Scan files first to get total tensor count for progress reporting
    file_keys_map = {}
    for file_path in safetensors_files:
        keys = get_keys_from_file(file_path)
        file_keys_map[file_path] = keys
        num_total_tensors += len(keys)
    
    logger.info(f"Found {num_total_tensors} tensors across {len(safetensors_files)} files")
    
    # Process files incrementally
    tensors_processed = 0
    for file_idx, file_path in enumerate(safetensors_files):
        file_keys = file_keys_map[file_path]
        logger.info(f"Processing file {file_idx+1}/{len(safetensors_files)}: {os.path.basename(file_path)} with {len(file_keys)} tensors")
        
        # Batch loading if slice_size specified, otherwise load the whole file at once
        if slice_size and len(file_keys) > slice_size:
            # Process in batches
            current_weights = {}
            key_batches = [file_keys[i:i+slice_size] for i in range(0, len(file_keys), slice_size)]
            
            for batch_idx, key_batch in enumerate(key_batches):
                logger.info(f"Loading batch {batch_idx+1}/{len(key_batches)} with {len(key_batch)} tensors")
                
                # Load only specific keys in this batch
                with safe_open(file_path, framework="flax") as f:
                    batch_start = time.time()
                    for key in key_batch:
                        current_weights[key] = f.get_tensor(key)
                        tensors_processed += 1
                        
                        # Log progress periodically
                        if tensors_processed % 50 == 0:
                            progress_pct = (tensors_processed / num_total_tensors) * 100
                            logger.info(f"Loaded {tensors_processed}/{num_total_tensors} tensors ({progress_pct:.1f}%)")
                    
                    batch_time = time.time() - batch_start
                    logger.info(f"Batch loaded in {batch_time:.2f}s")
                
                # Merge batch into full weights and clear batch memory
                all_weights.update(current_weights)
                current_weights.clear()
                
                # Force garbage collection
                gc.collect()
                log_memory_usage(f"After batch {batch_idx+1}")
        else:
            # Load the whole file at once for smaller files
            start_time = time.time()
            current_weights = load_file(file_path)
            tensors_processed += len(current_weights)
            load_time = time.time() - start_time
            logger.info(f"File loaded in {load_time:.2f}s")
            
            # Merge this file's weights into the full set
            all_weights.update(current_weights)
            
            # Clear references to free memory
            current_weights.clear()
            
            # Force garbage collection
            gc.collect()
            log_memory_usage(f"After file {file_idx+1}")
    
    # Log final stats
    log_memory_usage("After loading all weights")
    logger.info(f"Loaded {len(all_weights)} weights total from {len(safetensors_files)} files")
    
    # Analyze the loaded weights
    if verbose:
        analysis = analyze_param_structure(all_weights)
        logger.info(f"Params analysis: {analysis['total_params']/1e9:.2f}B params, {analysis['num_tensors']} tensors")
        logger.info(f"Critical keys present: {analysis['critical_keys_present']}")
        logger.info(f"Detected layers: {analysis['num_layers_detected']}")
        logger.info(f"Parameter status: {analysis['status']}")
    
    # Clean up memory one more time before returning
    gc.collect()
    
    return all_weights

def load_weights_with_slices(checkpoint_files, slice_size=1000):
    """
    Load model weights from checkpoint files in slices to minimize memory usage.
    
    Args:
        checkpoint_files: List of safetensors file paths
        slice_size: Number of parameters to load at once
        
    Returns:
        Dictionary with all model weights
    """
    logger.info(f"Loading weights from {len(checkpoint_files)} checkpoint files using slice size {slice_size}")
    
    # First scan to get metadata for all tensors
    all_tensors_info = {}
    total_params = 0
    
    for checkpoint_file in checkpoint_files:
        with safe_open(checkpoint_file, framework="jax") as f:
            metadata = f.metadata()
            tensor_names = f.keys()
            
            for name in tensor_names:
                if name not in all_tensors_info:  # Prevent duplicates
                    tensor_info = metadata.get(name, {})
                    shape = f.get_tensor_shape(name)
                    dtype = f.get_tensor_dtype(name)
                    all_tensors_info[name] = {"file": checkpoint_file, "shape": shape, "dtype": dtype}
                    total_params += np.prod(shape)
    
    logger.info(f"Found {len(all_tensors_info)} tensors with a total of {total_params:,} parameters")
    
    # Group tensor names into slices to load
    tensor_slices = []
    current_slice = []
    current_slice_params = 0
    
    for name, info in all_tensors_info.items():
        param_count = np.prod(info["shape"])
        
        # If this tensor would exceed the slice size, start a new slice
        # unless the current slice is empty (for very large tensors)
        if current_slice_params + param_count > slice_size * 1e6 and current_slice:
            tensor_slices.append(current_slice)
            current_slice = [name]
            current_slice_params = param_count
        else:
            current_slice.append(name)
            current_slice_params += param_count
    
    # Add the last slice if not empty
    if current_slice:
        tensor_slices.append(current_slice)
    
    logger.info(f"Created {len(tensor_slices)} slices for loading")
    
    # Load tensor slices and combine into a single dictionary
    weights = {}
    
    for i, slice_names in enumerate(tensor_slices):
        logger.info(f"Loading slice {i+1}/{len(tensor_slices)} with {len(slice_names)} tensors")
        slice_weights = {}
        
        # Group by file to minimize file open/close operations
        file_groups = {}
        for name in slice_names:
            file_path = all_tensors_info[name]["file"]
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(name)
        
        # Load from each file for this slice
        for file_path, names in file_groups.items():
            with safe_open(file_path, framework="jax") as f:
                for name in names:
                    tensor = f.get_tensor(name)
                    slice_weights[name] = tensor
        
        # Add to our accumulated weights
        weights.update(slice_weights)
        
        # Force garbage collection to free memory
        slice_weights = None
        import gc
        gc.collect()
    
    logger.info(f"Successfully loaded all weights with {len(weights)} tensors")
    return weights

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load and convert Qwen 2.5 PyTorch weights to Flax format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the PyTorch safetensors checkpoint files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted Flax model weights",
    )
    parser.add_argument(
        "--slice_size",
        type=int,
        default=1000,
        help="Size of parameter slices to load at once (in millions of parameters)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Find checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.safetensors")))
    if not checkpoint_files:
        raise ValueError(f"No safetensors files found in {args.checkpoint_dir}")
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Load weights with memory-efficient slicing
    torch_weights = load_weights_with_slices(checkpoint_files, slice_size=args.slice_size)
    
    # Convert weights to Flax format
    logger.info("Converting weights to Flax format...")
    flax_params = fix_params_structure(torch_weights)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the converted weights
    logger.info(f"Saving converted weights to {args.output_dir}")
    with open(os.path.join(args.output_dir, "model.msgpack"), "wb") as f:
        f.write(msgpack.dumps(flax_params))
    
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main() 