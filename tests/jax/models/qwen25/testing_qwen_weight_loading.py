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

memory_efficient
python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method all --test_forward --memory_efficient

diagnose
python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method diagnose

python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method all --test_forward

direct with forward
python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method direct --test_forward --memory_efficient

New parameter structure verification:
python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method param_verify

Using new conversion method:
python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method direct --test_forward --use_new_conversion --memory_efficient

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
        
        # Clear memory before loading
        import gc
        import jax
        gc.collect()
        jax.clear_caches()
        
        # Load the weights with optimized memory usage
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
    
    # Clear caches before loading to free memory
    import gc
    gc.collect()
    
    try:
        # Use cached weights if available
        params = get_cached_weights(model_path, config, dtype)
        logger.info(f"Successfully loaded weights in {time.time() - start_time:.2f} seconds")
        
        # Verify parameter structure
        logger.info("Analyzing parameter structure...")
        analysis = analyze_param_structure(params)
        logger.info(f"Loaded {analysis['total_params']/1e9:.2f}B parameters across {analysis['num_tensors']} tensors")
        logger.info(f"Detected {analysis['num_layers_detected']} transformer layers")
        
        # Force garbage collection before returning
        gc.collect()
        return params
    except Exception as e:
        logger.error(f"Direct safetensors loading failed: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Clear memory before starting
    jax.clear_caches()
    gc.collect()
    
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
            # Clean up original model
            del model_or_params
            gc.collect()
        else:
            # We got back parameters - need to fix structure and update model
            logger.info("Got parameters instead of model - fixing structure and updating model")
            
            # Process in steps to avoid holding too much in memory at once
            logger.info("Fixing parameter structure...")
            params = fix_params_structure(model_or_params)
            
            # Clear references to old parameter structure
            del model_or_params
            gc.collect()
            jax.clear_caches()
            
            # Update model parameters
            logger.info("Updating model parameters...")
            model.params = params
            
            # Clear references
            del params
            gc.collect()
            jax.clear_caches()
        
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
    
    # Import for memory management
    import gc
    import jax
    
    try:
        # Clear caches before forward pass
        gc.collect()
        jax.clear_caches()
        
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
                ],
                # Format 3: With nested layers_0 structure
                [
                    ('params', 'model', 'embed_tokens', 'embedding'),
                    ('params', 'lm_head', 'kernel'),
                    ('params', 'model', 'layers', 'layers_0', 'attention', 'q_proj', 'kernel')
                ]
            ]
            
            # For each parameter set, check if any are missing
            format_found = False
            missing_patterns = {}
            for param_set in essential_param_patterns:
                missing_params = []
                for param_path in param_set:
                    if param_path not in flat_params:
                        missing_params.append('/'.join(param_path))
                
                # If all required params for this format exist, we found a match
                if not missing_params:
                    format_found = True
                    break
                else:
                    # Fix: use safe_path_join to handle different parameter path types
                    format_pattern = ""
                    try:
                        # Get a key pattern from parameters that exist (aren't missing)
                        existing_params = [param_path for param_path in param_set if safe_path_join(param_path) not in missing_params]
                        if existing_params:
                            format_pattern = safe_path_join(existing_params[0])
                        else:
                            # If none exist, just use the first one
                            format_pattern = safe_path_join(param_set[0]) if param_set else "unknown"
                        
                        missing_patterns[format_pattern] = missing_params
                    except Exception as e:
                        logger.error(f"Error processing parameter paths: {e}")
                        debug_param_path_types(param_set)  # Debug the structure
                        # Use a fallback approach
                        missing_patterns[f"format-{len(missing_patterns)}"] = missing_params
            
            if not format_found:
                logger.error(f"Missing essential parameters for all supported formats")
                for format_pattern, missing in missing_patterns.items():
                    logger.info(f"Format pattern {format_pattern} missing: {missing}")
                
                # Log available paths to help diagnose the issue
                logger.info("Available parameter patterns:")
                
                # Sample a few keys to see what's available
                sample_paths = set()
                for key in flat_params.keys():
                    path_parts = []
                    for part in key[:3]:  # Just use the first few parts of the path
                        path_parts.append(part)
                        pattern = '/'.join(path_parts)
                        sample_paths.add(pattern)
                    if len(sample_paths) >= 10:
                        break
                
                for pattern in sorted(sample_paths):
                    logger.info(f"  {pattern}...")
                
                # If no matching format, analyze largest tensors to help diagnose the issue
                log_tensor_sizes(model.params)
                
                # Clear memory
                del flat_params
                gc.collect()
                jax.clear_caches()
                
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
            
            # Clean up
            del flat_params
            gc.collect()
            
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
        gc.collect()
        jax.clear_caches()
        
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
        
        # Clear outputs to free memory
        del outputs
        gc.collect()
        jax.clear_caches()
        
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
            
            # Clean up
            del logits
            gc.collect()
            jax.clear_caches()
            
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
            if f"layers/{i}" in key or f"layers_(\d+)/" in key or f"layers\.(\d+)\." in key or f"layers\.(\d+)/" in key or f"layers/layers_(\d+)/" in key:
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
    
    # Import garbage collection to manage memory
    import gc
    
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
    
    # Use direct layer format - from code inspection we know FlaxQwen25LayerCollection
    # in qwen25_full_implementation.py uses the "layers_X" format
    # This is the critical part that was causing issues
    layer_pattern = 'layers_{}'
    logger.info(f"Using layer pattern 'layers_X' from code inspection")
    
    # Mapping from PyTorch parameter names to Flax parameter structure
    mappings = {
        # Embeddings
        "model.embed_tokens.weight": ("model", "embed_tokens", "embedding"),
        
        # Layer mappings (will be formatted with layer number)
        # Using the "layers_X" format with "attention" not "self_attn"
        "model.layers.{}.self_attn.q_proj.weight": ("model", "layers", layer_pattern, "attention", "q_proj", "kernel"),
        "model.layers.{}.self_attn.k_proj.weight": ("model", "layers", layer_pattern, "attention", "k_proj", "kernel"),
        "model.layers.{}.self_attn.v_proj.weight": ("model", "layers", layer_pattern, "attention", "v_proj", "kernel"),
        "model.layers.{}.self_attn.o_proj.weight": ("model", "layers", layer_pattern, "attention", "o_proj", "kernel"),
        
        # Add bias terms if they exist
        "model.layers.{}.self_attn.q_proj.bias": ("model", "layers", layer_pattern, "attention", "q_proj", "bias"),
        "model.layers.{}.self_attn.k_proj.bias": ("model", "layers", layer_pattern, "attention", "k_proj", "bias"),
        "model.layers.{}.self_attn.v_proj.bias": ("model", "layers", layer_pattern, "attention", "v_proj", "bias"),
        "model.layers.{}.self_attn.o_proj.bias": ("model", "layers", layer_pattern, "attention", "o_proj", "bias"),
        
        "model.layers.{}.mlp.gate_proj.weight": ("model", "layers", layer_pattern, "mlp", "gate_proj", "kernel"),
        "model.layers.{}.mlp.up_proj.weight": ("model", "layers", layer_pattern, "mlp", "up_proj", "kernel"),
        "model.layers.{}.mlp.down_proj.weight": ("model", "layers", layer_pattern, "mlp", "down_proj", "kernel"),
        
        # Add bias terms if they exist
        "model.layers.{}.mlp.gate_proj.bias": ("model", "layers", layer_pattern, "mlp", "gate_proj", "bias"),
        "model.layers.{}.mlp.up_proj.bias": ("model", "layers", layer_pattern, "mlp", "up_proj", "bias"),
        "model.layers.{}.mlp.down_proj.bias": ("model", "layers", layer_pattern, "mlp", "down_proj", "bias"),
        
        "model.layers.{}.input_layernorm.weight": ("model", "layers", layer_pattern, "input_layernorm", "scale"),
        "model.layers.{}.post_attention_layernorm.weight": ("model", "layers", layer_pattern, "post_attention_layernorm", "scale"),
        
        # Final layer norm - IMPORTANT: This was moved to be under "layers" instead of directly under "model"
        # From: "model.norm.weight": ("model", "norm", "scale"),
        # To:
        "model.norm.weight": ("model", "layers", "norm", "scale"),
        
        # Language model head
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    # Process every weight from the source dictionary
    processed_keys = set()
    
    # First, handle the layer-specific weights
    for layer_idx in range(layer_count):
        # Process each layer and free memory after
        layer_keys_to_process = []
        
        # Find all keys for this layer
        for src_pattern in mappings:
            if "{}" in src_pattern:
                src_key = src_pattern.format(layer_idx)
                if src_key in weights:
                    layer_keys_to_process.append((src_key, src_pattern))
        
        # Process this layer's keys
        for src_key, src_pattern in layer_keys_to_process:
            dst_path = mappings[src_pattern]
            
            # Create path in the params dictionary
            current = fixed_params["params"]
            for part in dst_path[:-1]:
                # Ensure correct layer path format
                if part == layer_pattern:
                    part_name = part.format(layer_idx)
                else:
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
        
        # Force garbage collection after each layer is processed
        if layer_idx % 5 == 0:  # Every 5 layers
            gc.collect()
    
    # Handle non-layer specific weights
    non_layer_keys = [k for k in mappings if "{}" not in k and k in weights]
    for src_key in non_layer_keys:
        dst_path = mappings[src_key]
        
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
    
    # Add debug output for norm-related parameters
    flat_params = flatten_dict(fixed_params)
    norm_params = [k for k in flat_params.keys() if 'norm' in '/'.join(k)]
    logger.info("Norm-related parameters after mapping:")
    for k in norm_params:
        logger.info(f"  {'/'.join(k)}: {flat_params[k].shape}")
    
    # Check for the critical parameter that was causing issues
    critical_param = ('params', 'model', 'layers', 'norm', 'scale')
    if critical_param in flat_params:
        logger.info(f"✓ Critical parameter {'/'.join(critical_param)} exists in mapped structure")
    else:
        logger.warning(f"❌ Critical parameter {'/'.join(critical_param)} is MISSING in mapped structure")
    
    # Force garbage collection before returning
    gc.collect()
    
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

def report_structure_mismatch(expected, actual):
    """
    Report differences between expected and actual parameter structures.
    
    Args:
        expected: Expected parameter structure (dictionary of parameter paths → shapes)
        actual: Actual parameter structure (flattened)
    """
    missing_keys = set(expected.keys()) - set(actual.keys())
    extra_keys = set(actual.keys()) - set(expected.keys())
    
    print(f"Missing {len(missing_keys)} parameters, extra {len(extra_keys)} parameters")
    
    # Identify patterns in missing keys
    missing_patterns = {}
    for key in missing_keys:
        # Create pattern by replacing numbers with X
        import re
        pattern = re.sub(r'\d+', 'X', '/'.join(key) if isinstance(key, tuple) else str(key))
        if pattern not in missing_patterns:
            missing_patterns[pattern] = []
        missing_patterns[pattern].append(key)
    
    print("\nMissing parameter patterns:")
    for pattern, examples in missing_patterns.items():
        print(f"  {pattern}: {len(examples)} parameters")
        print(f"    Example: {examples[0]}")
    
    # Check for shape mismatches in parameters that exist in both
    common_keys = set(expected.keys()) & set(actual.keys())
    shape_mismatches = []
    
    for key in common_keys:
        expected_shape = expected[key]
        actual_shape = actual[key].shape if hasattr(actual[key], 'shape') else None
        
        if expected_shape is not None and actual_shape is not None and expected_shape != actual_shape:
            shape_mismatches.append((key, expected_shape, actual_shape))
    
    if shape_mismatches:
        print("\nShape mismatches:")
        for key, expected_shape, actual_shape in shape_mismatches:
            print(f"  {key}: expected {expected_shape}, got {actual_shape}")
    
    return missing_keys, extra_keys

def get_expected_parameter_structure(config):
    """
    Create a model instance and extract its expected parameter structure.
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary of expected parameter structure information
    """
    logger.info("Creating reference model to get expected parameter structure...")
    
    # Import necessary modules for memory management
    import gc
    import jax
    
    # Clear memory before creating model
    jax.clear_caches()
    gc.collect()
    
    # Create a model with minimal configuration for memory efficiency
    # Use smallest valid values to minimize memory usage
    minimal_config = Qwen25Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    )
    
    try:
        # Create a reference model but with no parameter initialization
        reference_model = FlaxQwen25ForCausalLM(minimal_config, _do_init=False)
        
        # Create empty parameter structure with correct paths but minimal values
        from jax import random
        rng = random.PRNGKey(0)
        # Only initialize with tiny dummy inputs (1x1) to minimize memory
        variables = reference_model.module.init(rng, jnp.ones((1, 1), dtype=jnp.int32))
        
        # Extract structure but not actual tensors
        flat_params = flatten_dict(variables)
        
        # Create structure dictionary with just paths and shapes
        structure_info = {}
        for key, value in flat_params.items():
            structure_info[key] = getattr(value, "shape", None)
        
        # Clear out variables to free memory
        del variables
        del reference_model
        gc.collect()
        jax.clear_caches()
        
        # Log a sample of the structure
        sample_keys = list(structure_info.keys())[:5]
        logger.info(f"Sample expected parameters (first 5):")
        for key in sample_keys:
            key_str = '/'.join(key)
            logger.info(f"  {key_str}: {structure_info[key]}")
        
        return structure_info
    except Exception as e:
        logger.error(f"Error getting expected parameter structure: {e}")
        import traceback
        traceback.print_exc()
        return {}

def print_model_expected_structure(config):
    """
    Create an empty model and print its expected parameter structure.
    
    Args:
        config: Model configuration
    """
    logger.info("Creating model to inspect expected parameter structure...")
    
    # Import necessary modules for memory management
    import gc
    import jax
    
    try:
        # Create a model with minimal initialization
        model = FlaxQwen25ForCausalLM(config, _do_init=True)
        
        # Get flattened parameters
        flat_params = flatten_dict(model.params)
        
        # Print all norm-related parameters to help debug
        logger.info("Expected norm-related parameters:")
        for key in sorted(['/'.join(k) for k in flat_params.keys() if 'norm' in '/'.join(k)]):
            logger.info(f"  {key}")
        
        # Print a sample of parameters from different components
        param_categories = {
            'embed': [],
            'attention': [],
            'mlp': [],
            'layernorm': [],
            'lm_head': []
        }
        
        for key in flat_params.keys():
            key_str = '/'.join(key)
            for category in param_categories:
                if category in key_str.lower():
                    param_categories[category].append(key_str)
                    break
        
        logger.info("Sample parameters from different components:")
        for category, params in param_categories.items():
            if params:
                logger.info(f"  {category}: {params[0]}")
        
        # Check for specific problematic parameter
        if ('params', 'model', 'layers', 'norm', 'scale') in flat_params:
            logger.info("✓ Critical parameter 'params/model/layers/norm/scale' exists in expected structure")
        else:
            logger.warning("❌ Critical parameter 'params/model/layers/norm/scale' NOT FOUND in expected structure")
            # Try to find similar parameters
            similar_params = ['/'.join(k) for k in flat_params.keys() if 'norm' in '/'.join(k) and 'scale' in '/'.join(k)]
            logger.info(f"Similar parameters: {similar_params}")
        
        # Clean up
        del model
        del flat_params
        gc.collect()
        jax.clear_caches()
        
        return True
    except Exception as e:
        logger.error(f"Error inspecting model structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_param_structures(reference_model, fixed_params):
    """
    Compare parameter structures between reference model and fixed params.
    
    Args:
        reference_model: Reference model with expected parameter structure
        fixed_params: Parameter dictionary to compare
        
    Returns:
        Tuple of (missing_keys, extra_keys)
    """
    logger.info("Comparing parameter structures...")
    
    # Flatten both parameter structures
    ref_flat = flatten_dict(reference_model.params)
    actual_flat = flatten_dict(fixed_params)
    
    # Convert keys to strings for easier comparison and display
    ref_keys = set(['/'.join(k) for k in ref_flat.keys()])
    actual_keys = set(['/'.join(k) for k in actual_flat.keys()])
    
    # Find differences
    missing_keys = ref_keys - actual_keys
    extra_keys = actual_keys - ref_keys
    
    # Report statistics
    logger.info(f"Total expected parameters: {len(ref_keys)}")
    logger.info(f"Total actual parameters: {len(actual_keys)}")
    logger.info(f"Missing parameters: {len(missing_keys)}")
    logger.info(f"Extra parameters: {len(extra_keys)}")
    
    # Show specific missing parameters (limit to first 10)
    if missing_keys:
        logger.info("Missing parameters (first 10):")
        for key in sorted(list(missing_keys)[:10]):
            logger.info(f"  Missing: {key}")
    
    # Return the raw key tuples for potential fixes
    return (
        [k for k in ref_flat.keys() if '/'.join(k) in missing_keys],
        [k for k in actual_flat.keys() if '/'.join(k) in extra_keys]
    )

def fix_missing_parameters(params, model_class, config):
    """
    Fix parameter structure for compatibility with model_class expectations.
    
    Args:
        params: Parameter dictionary to fix
        model_class: Model class to check against
        config: Model configuration
        
    Returns:
        Fixed parameter dictionary
    """
    logger.info("Fixing missing parameters...")
    
    # Import for memory management
    import gc
    import jax
    
    # Create a copy of params to modify
    fixed_params = unfreeze(params) if hasattr(params, 'unfreeze') else params.copy()
    
    try:
        # Flatten the parameters
        flat_params = flatten_dict(fixed_params)
        
        # Create a minimal model to get expected structure
        tmp_model = model_class(config, _do_init=True)
        expected_flat = flatten_dict(tmp_model.params)
        
        # Get lists of keys
        expected_keys = set(expected_flat.keys())
        actual_keys = set(flat_params.keys())
        
        # Find missing keys
        missing_keys = expected_keys - actual_keys
        
        if missing_keys:
            logger.info(f"Found {len(missing_keys)} missing parameters to fix")
            
            # Create mappings for parameter name patterns
            # This maps potential source keys to destination keys
            potential_mappings = {
                # Key format issue: missing 'layers' in the path for norm
                ('params', 'model', 'norm', 'scale'): ('params', 'model', 'layers', 'norm', 'scale'),
                
                # Other potential mappings
                ('params', 'norm', 'scale'): ('params', 'model', 'layers', 'norm', 'scale'),
            }
            
            # Check all pre-defined mappings
            fixes_applied = 0
            for src_key, dst_key in potential_mappings.items():
                if src_key in flat_params and dst_key in missing_keys:
                    logger.info(f"Fixing: Moving {'/'.join(src_key)} -> {'/'.join(dst_key)}")
                    # Copy the parameter to the correct location
                    flat_params[dst_key] = flat_params[src_key]
                    fixes_applied += 1
            
            # If no predefined mappings worked, try to find by parameter name similarity
            if fixes_applied == 0:
                logger.info("No predefined mappings worked, trying similarity matching")
                
                # Group parameters by their last path component (parameter name)
                param_name_map = {}
                for key in flat_params.keys():
                    param_name = key[-1]
                    if param_name not in param_name_map:
                        param_name_map[param_name] = []
                    param_name_map[param_name].append(key)
                
                # For each missing key, try to find a matching parameter by name
                for missing_key in missing_keys:
                    missing_name = missing_key[-1]
                    if missing_name in param_name_map:
                        # Found potential matches by parameter name
                        candidates = param_name_map[missing_name]
                        
                        # Check if shapes are compatible
                        for candidate_key in candidates:
                            if flat_params[candidate_key].shape == expected_flat[missing_key].shape:
                                logger.info(f"Fixing by similarity: {'/'.join(candidate_key)} -> {'/'.join(missing_key)}")
                                flat_params[missing_key] = flat_params[candidate_key]
                                fixes_applied += 1
                                break
            
            logger.info(f"Applied {fixes_applied} parameter fixes")
        
        # Check if the specific problematic norm parameter was fixed
        critical_param = ('params', 'model', 'layers', 'norm', 'scale')
        if critical_param in missing_keys and critical_param not in flat_params:
            logger.warning(f"Critical parameter {'/'.join(critical_param)} still missing")
            
            # Last resort: try extracting from similar sources
            norm_sources = [k for k in flat_params.keys() if 'norm' in '/'.join(k) and 'scale' in '/'.join(k)]
            if norm_sources:
                src_key = norm_sources[0]
                logger.info(f"Last resort fix: Using {'/'.join(src_key)} for {'/'.join(critical_param)}")
                flat_params[critical_param] = flat_params[src_key]
        
        # Unflatten the parameters
        result_params = unflatten_dict(flat_params)
        
        # Clean up
        del tmp_model
        del flat_params
        del expected_flat
        gc.collect()
        jax.clear_caches()
        
        return result_params
    except Exception as e:
        logger.error(f"Error fixing parameters: {e}")
        import traceback
        traceback.print_exc()
        
        # Return original params if fixing failed
        return params

def safe_path_join(path_component):
    """
    Safely join path components regardless of type.
    
    Args:
        path_component: Path component to convert to string
        
    Returns:
        Path component as a string
    """
    if isinstance(path_component, (list, tuple)):
        return '/'.join(str(p) for p in path_component)
    return str(path_component)

def debug_param_path_types(param_set):
    """
    Debug parameter path types by examining their structure.
    
    Args:
        param_set: Parameter set to debug
        
    Returns:
        None
    """
    logger.info(f"Parameter set type: {type(param_set)}")
    
    if isinstance(param_set, (list, tuple, set)):
        logger.info(f"Parameter set contents: {param_set}")
        for i, item in enumerate(param_set):
            logger.info(f"  Item {i} type: {type(item)}, value: {item}")
            
            if isinstance(item, (list, tuple)):
                logger.info(f"    Nested item contents: {item}")
                for j, subitem in enumerate(item):
                    logger.info(f"      Subitem {j} type: {type(subitem)}, value: {subitem}")
    
    # If it's a dict or similar
    elif hasattr(param_set, 'items'):
        logger.info(f"Dict-like object with keys: {list(param_set.keys())}")

def analyze_dict_structure(d, name="dict", level=0):
    """
    Analyze dictionary structure recursively to identify issues.
    
    Args:
        d: Dictionary to analyze
        name: Name of the dictionary
        level: Current recursion level
        
    Returns:
        None
    """
    indent = "  " * level
    logger.info(f"{indent}{name} ({type(d)}):")
    
    if isinstance(d, dict) or isinstance(d, FrozenDict):
        for k, v in d.items():
            if isinstance(v, dict) or isinstance(v, FrozenDict):
                analyze_dict_structure(v, f"[{k}]", level+1)
            else:
                shape_info = getattr(v, 'shape', None)
                type_info = type(v)
                logger.info(f"{indent}  [{k}]: {type_info} (shape: {shape_info})")

def convert_params_to_jax_structure(weights_dict):
    """
    Convert loaded weights to the exact structure expected by JAX models.
    This completely rebuilds the parameter structure to match expectations.
    
    Args:
        weights_dict: Dictionary of weights loaded from safetensors
        
    Returns:
        Dictionary with restructured weights matching Flax model structure
    """
    logger.info("Converting parameters to exact JAX structure...")
    
    # Create the high-level structure first
    jax_params = {
        "params": {
            "model": {
                "embed_tokens": {},
                "layers": {}  # Will contain norm and layers_X subdicts
            },
            "lm_head": {}
        }
    }
    
    # Extract configuration parameters from weights to determine model structure
    config = {}
    if "model.embed_tokens.weight" in weights_dict:
        vocab_size = weights_dict["model.embed_tokens.weight"].shape[0]
        hidden_size = weights_dict["model.embed_tokens.weight"].shape[1]
        config["vocab_size"] = vocab_size
        config["hidden_size"] = hidden_size
    
    # Count number of layers based on pattern matching
    layer_count = 0
    for key in weights_dict:
        if "model.layers." in key:
            layer_num = int(key.split("model.layers.")[1].split(".")[0])
            layer_count = max(layer_count, layer_num + 1)
    
    logger.info(f"Detected model structure: vocab_size={config.get('vocab_size', 'unknown')}, "
                f"hidden_size={config.get('hidden_size', 'unknown')}, layers={layer_count}")
    
    # Map all parameters with explicit handling for each type
    for key, value in weights_dict.items():
        # Handle embeddings
        if key == "model.embed_tokens.weight":
            jax_params["params"]["model"]["embed_tokens"]["embedding"] = value
        
        # Handle final layer norm - CRITICAL FIX
        elif key == "model.norm.weight":
            # Ensure 'norm' exists under 'layers'
            if "norm" not in jax_params["params"]["model"]["layers"]:
                jax_params["params"]["model"]["layers"]["norm"] = {}
            jax_params["params"]["model"]["layers"]["norm"]["scale"] = value
        
        # Handle language model head
        elif key == "lm_head.weight":
            jax_params["params"]["lm_head"]["kernel"] = value.T  # Transpose for Flax
        
        # Handle per-layer parameters
        elif "model.layers." in key:
            # Extract layer index and parameter type
            parts = key.split(".")
            if len(parts) < 4:
                continue
                
            layer_idx = int(parts[2])
            layer_key = f"layers_{layer_idx}"
            
            # Ensure layer exists
            if layer_key not in jax_params["params"]["model"]["layers"]:
                jax_params["params"]["model"]["layers"][layer_key] = {}
            
            # Handle attention parameters
            if "self_attn" in key:
                if "attention" not in jax_params["params"]["model"]["layers"][layer_key]:
                    jax_params["params"]["model"]["layers"][layer_key]["attention"] = {}
                
                # Handle attention projections
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if f"self_attn.{proj}.weight" in key:
                        if proj not in jax_params["params"]["model"]["layers"][layer_key]["attention"]:
                            jax_params["params"]["model"]["layers"][layer_key]["attention"][proj] = {}
                        jax_params["params"]["model"]["layers"][layer_key]["attention"][proj]["kernel"] = value.T
                    elif f"self_attn.{proj}.bias" in key:
                        if proj not in jax_params["params"]["model"]["layers"][layer_key]["attention"]:
                            jax_params["params"]["model"]["layers"][layer_key]["attention"][proj] = {}
                        jax_params["params"]["model"]["layers"][layer_key]["attention"][proj]["bias"] = value
            
            # Handle MLP parameters
            elif "mlp" in key:
                if "mlp" not in jax_params["params"]["model"]["layers"][layer_key]:
                    jax_params["params"]["model"]["layers"][layer_key]["mlp"] = {}
                
                # Handle MLP projections
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    if f"mlp.{proj}.weight" in key:
                        if proj not in jax_params["params"]["model"]["layers"][layer_key]["mlp"]:
                            jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj] = {}
                        jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj]["kernel"] = value.T
                    elif f"mlp.{proj}.bias" in key:
                        if proj not in jax_params["params"]["model"]["layers"][layer_key]["mlp"]:
                            jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj] = {}
                        jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj]["bias"] = value
            
            # Handle layer norms
            elif "input_layernorm.weight" in key:
                if "input_layernorm" not in jax_params["params"]["model"]["layers"][layer_key]:
                    jax_params["params"]["model"]["layers"][layer_key]["input_layernorm"] = {}
                jax_params["params"]["model"]["layers"][layer_key]["input_layernorm"]["scale"] = value
                
            elif "post_attention_layernorm.weight" in key:
                if "post_attention_layernorm" not in jax_params["params"]["model"]["layers"][layer_key]:
                    jax_params["params"]["model"]["layers"][layer_key]["post_attention_layernorm"] = {}
                jax_params["params"]["model"]["layers"][layer_key]["post_attention_layernorm"]["scale"] = value
    
    # Validate expected structure after conversion
    if "embedding" not in jax_params["params"]["model"]["embed_tokens"]:
        logger.warning("❌ Embedding parameter missing in converted structure!")
    
    if "kernel" not in jax_params["params"]["lm_head"]:
        logger.warning("❌ LM head parameter missing in converted structure!")
    
    if "norm" not in jax_params["params"]["model"]["layers"]["norm"]:
        logger.warning("❌ Final norm parameter missing in converted structure!")
    
    # Important: ensure the params dict is properly frozen
    logger.info("Freezing parameter structure...")
    return freeze(jax_params)

def verify_parameter_structure(params):
    """
    Verify parameter structure matches JAX expectations.
    
    Args:
        params: Parameter dictionary to verify
        
    Returns:
        Boolean indicating whether the structure is valid
    """
    # Define expected parameter paths
    critical_params = [
        ("params", "model", "embed_tokens", "embedding"),
        ("params", "model", "layers", "norm", "scale"),  # Final layer norm
        ("params", "lm_head", "kernel")
    ]
    
    # Check if params is properly frozen
    if not isinstance(params, FrozenDict):
        logger.warning("Parameters are not properly frozen as FrozenDict")
        params = freeze(params)
    
    # Flatten the parameter dict
    flat_params = flatten_dict(params)
    
    # Check for critical parameters
    missing = []
    for path in critical_params:
        if path not in flat_params:
            missing.append(path)
            # Try to find similar paths for debugging
            similar = [k for k in flat_params.keys() if len(set(path) & set(k)) > 0]
            logger.error(f"Missing critical parameter: {'/'.join(path)}")
            logger.error(f"Similar parameters found: {similar[:5]}")
    
    if missing:
        logger.error(f"Critical parameters missing: {missing}")
        return False
    
    # Check some layer parameters are present (at least layer 0)
    has_layer_0 = False
    for k in flat_params.keys():
        if len(k) >= 4 and k[0] == 'params' and k[1] == 'model' and k[2] == 'layers' and k[3] == 'layers_0':
            has_layer_0 = True
            break
    
    if not has_layer_0:
        logger.error("Missing layer 0 parameters in structure")
        return False
    
    logger.info("✅ Parameter structure verification passed")
    return True

def validate_final_norm_parameter(params):
    """
    Validate the final norm parameter specifically.
    
    Args:
        params: Parameter dictionary to verify
        
    Returns:
        Boolean indicating whether the final norm parameter is valid
    """
    # Check if the parameter exists at the expected path
    try:
        # Try different possible paths
        if "params" in params and "model" in params["params"]:
            model_params = params["params"]["model"]
            
            # Check the corrected path
            if "layers" in model_params and "norm" in model_params["layers"]:
                norm_param = model_params["layers"]["norm"]
                if "scale" in norm_param:
                    logger.info(f"✅ Final norm parameter found at correct path with shape: {norm_param['scale'].shape}")
                    return True
            
            # Check the original incorrect path
            if "norm" in model_params:
                norm_param = model_params["norm"]
                if "scale" in norm_param:
                    logger.warning("⚠️ Final norm parameter found at INCORRECT path")
                    # Suggest fix
                    logger.info("Suggested fix: Move from params/model/norm/scale to params/model/layers/norm/scale")
                    return False
            
            logger.error("❌ Final norm parameter not found in any expected location")
            
            # Try to find any norm-related parameters
            flat_params = flatten_dict(params)
            norm_related = [k for k in flat_params.keys() if "norm" in str(k)]
            logger.info(f"Norm-related parameters found: {norm_related}")
            
            return False
        
    except Exception as e:
        logger.error(f"Error validating final norm parameter: {e}")
        return False

def initialize_and_load_model(config, weights, memory_efficient=True):
    """
    Initialize model and load weights with proper structure.
    
    Args:
        config: Model configuration
        weights: Model weights
        memory_efficient: Whether to use memory-efficient loading
        
    Returns:
        Initialized model with weights loaded
    """
    # Create model with _do_init=True to get proper initialization
    logger.info("Creating model with proper initialization...")
    model = FlaxQwen25ForCausalLM(config, _do_init=True)
    
    # Get the exact reference structure
    reference_params = model.params
    
    # Now convert our weights to the exact same structure
    logger.info("Converting weights to exact JAX structure...")
    if memory_efficient:
        # Process weights incrementally
        jax_params = convert_params_to_jax_structure(weights)
    else:
        # Convert all at once
        jax_params = convert_params_to_jax_structure(weights)
    
    # Verify the structure is correct
    logger.info("Verifying parameter structure...")
    if not verify_parameter_structure(jax_params):
        logger.warning("Parameter structure verification failed - attempting to continue")
    
    # Replace the model's parameters
    logger.info("Setting model parameters...")
    model.params = jax_params
    
    return model

def convert_params_to_jax_structure_efficient(weights_dict):
    """
    Convert parameters with memory efficiency.
    
    Args:
        weights_dict: Dictionary of weights loaded from safetensors
        
    Returns:
        Dictionary with restructured weights matching Flax model structure
    """
    logger.info("Converting parameters to JAX structure with memory efficiency...")
    
    # Import garbage collection to manage memory
    import gc
    
    # Start with an empty structure but with all required keys
    jax_params = {
        "params": {
            "model": {
                "embed_tokens": {},
                "layers": {
                    # Ensure this exists from the start
                    "norm": {}
                }
            },
            "lm_head": {}
        }
    }
    
    # Process parameters incrementally grouped by components
    # Start with embedding
    if "model.embed_tokens.weight" in weights_dict:
        jax_params["params"]["model"]["embed_tokens"]["embedding"] = weights_dict["model.embed_tokens.weight"]
        gc.collect()
    
    # Final norm - CRITICAL FIX
    if "model.norm.weight" in weights_dict:
        jax_params["params"]["model"]["layers"]["norm"]["scale"] = weights_dict["model.norm.weight"]
        gc.collect()
    
    # LM head 
    if "lm_head.weight" in weights_dict:
        jax_params["params"]["lm_head"]["kernel"] = weights_dict["lm_head.weight"].T
        gc.collect()
    
    # Count number of layers based on pattern matching
    layer_count = 0
    for key in weights_dict:
        if "model.layers." in key:
            layer_num = int(key.split("model.layers.")[1].split(".")[0])
            layer_count = max(layer_count, layer_num + 1)
    
    # Process layers one at a time to save memory
    for layer_idx in range(layer_count):
        layer_key = f"layers_{layer_idx}"
        
        # Initialize layer structure
        if layer_key not in jax_params["params"]["model"]["layers"]:
            jax_params["params"]["model"]["layers"][layer_key] = {
                "attention": {},
                "mlp": {},
                "input_layernorm": {},
                "post_attention_layernorm": {}
            }
        
        # Process attention parameters
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            bias_key = f"model.layers.{layer_idx}.self_attn.{proj}.bias"
            
            if weight_key in weights_dict:
                if proj not in jax_params["params"]["model"]["layers"][layer_key]["attention"]:
                    jax_params["params"]["model"]["layers"][layer_key]["attention"][proj] = {}
                jax_params["params"]["model"]["layers"][layer_key]["attention"][proj]["kernel"] = weights_dict[weight_key].T
            
            if bias_key in weights_dict:
                if proj not in jax_params["params"]["model"]["layers"][layer_key]["attention"]:
                    jax_params["params"]["model"]["layers"][layer_key]["attention"][proj] = {}
                jax_params["params"]["model"]["layers"][layer_key]["attention"][proj]["bias"] = weights_dict[bias_key]
            
            # Free memory after processing each projection
            gc.collect()
        
        # Process MLP parameters
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            weight_key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
            bias_key = f"model.layers.{layer_idx}.mlp.{proj}.bias"
            
            if weight_key in weights_dict:
                if proj not in jax_params["params"]["model"]["layers"][layer_key]["mlp"]:
                    jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj] = {}
                jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj]["kernel"] = weights_dict[weight_key].T
            
            if bias_key in weights_dict:
                if proj not in jax_params["params"]["model"]["layers"][layer_key]["mlp"]:
                    jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj] = {}
                jax_params["params"]["model"]["layers"][layer_key]["mlp"][proj]["bias"] = weights_dict[bias_key]
            
            # Free memory after processing each projection
            gc.collect()
        
        # Process layer norms
        input_norm_key = f"model.layers.{layer_idx}.input_layernorm.weight"
        if input_norm_key in weights_dict:
            jax_params["params"]["model"]["layers"][layer_key]["input_layernorm"]["scale"] = weights_dict[input_norm_key]
        
        post_norm_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        if post_norm_key in weights_dict:
            jax_params["params"]["model"]["layers"][layer_key]["post_attention_layernorm"]["scale"] = weights_dict[post_norm_key]
        
        # Free memory after processing this layer
        gc.collect()
    
    # Verify expected structure after conversion
    if "embedding" not in jax_params["params"]["model"]["embed_tokens"]:
        logger.warning("❌ Embedding parameter missing in converted structure!")
    
    if "kernel" not in jax_params["params"]["lm_head"]:
        logger.warning("❌ LM head parameter missing in converted structure!")
    
    if "scale" not in jax_params["params"]["model"]["layers"]["norm"]:
        logger.warning("❌ Final norm parameter missing in converted structure!")
    
    # Important: ensure the params dict is properly frozen
    logger.info("Freezing parameter structure...")
    return freeze(jax_params)

def match_parameter_format(src_params, ref_params):
    """
    Match parameter format exactly based on reference.
    
    Args:
        src_params: Source parameters to reformat
        ref_params: Reference parameters with the desired format
        
    Returns:
        Reformatted parameters
    """
    # Recursively match structure
    result = {}
    
    # Handle nested dictionaries
    for key, value in ref_params.items():
        if key in src_params:
            if isinstance(value, dict) and isinstance(src_params[key], dict):
                result[key] = match_parameter_format(src_params[key], value)
            else:
                # Copy the parameter but ensure it has right type/format
                result[key] = jnp.array(src_params[key], dtype=value.dtype)
    
    return result

def main():
    """Main entry point for testing Qwen 2.5 weight loading."""
    parser = argparse.ArgumentParser(description="Test Qwen 2.5 weight loading")
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Directory containing the weight files",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="direct",
        choices=["direct", "index", "model", "all", "diagnose", "check_structure", "param_verify"],
        help="Weight loading method to test, or diagnostic modes",
    )
    parser.add_argument(
        "--test_forward",
        action="store_true",
        help="Test a forward pass with the loaded weights",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to use for weights",
    )
    parser.add_argument(
        "--slice_size",
        type=int,
        default=1000,
        help="Size of parameter slices to load at once (in millions of parameters)",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Enable more aggressive memory management (slower but uses less RAM)",
    )
    parser.add_argument(
        "--alt_loading",
        action="store_true",
        help="Use alternative loading method from qwen25_full_implementation.py",
    )
    parser.add_argument(
        "--auto_fix",
        action="store_true",
        help="Attempt to automatically fix parameter structure issues",
    )
    parser.add_argument(
        "--param_structure_check",
        action="store_true",
        help="Perform detailed parameter structure check",
    )
    parser.add_argument(
        "--use_new_conversion",
        action="store_true",
        help="Use the new parameter conversion method",
    )
    args = parser.parse_args()

    # Import necessary modules for memory management
    import gc
    import jax
    
    # Clear memory at the start
    gc.collect()
    jax.clear_caches()

    # Map dtype string to jnp.dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Load config from weights directory
    config_path = os.path.join(args.weights_dir, "config.json")
    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        config = load_config_from_json(config_path)
    else:
        logger.warning(f"Config file not found at {config_path}, using default config")
        config = Qwen25Config()

    # Special structure check mode
    if args.method == "check_structure":
        logger.info("=== Checking expected model parameter structure ===")
        print_model_expected_structure(config)
        return
    
    # Special parameter verification mode
    if args.method == "param_verify":
        logger.info("=== Running parameter structure verification ===")
        
        # Create a minimal model to get expected structure
        logger.info("Creating minimal model for reference structure...")
        ref_model = FlaxQwen25ForCausalLM(config, _do_init=True)
        
        # Analyze reference structure
        logger.info("Analyzing reference parameter structure...")
        analyze_dict_structure(ref_model.params, "reference_params")
        
        # Load weights
        logger.info("Loading weights for verification...")
        weights = test_direct_safetensors(args.weights_dir, config, dtype)
        
        if weights is not None:
            # Convert using new method
            logger.info("Converting weights using new parameter conversion...")
            if args.memory_efficient:
                converted_params = convert_params_to_jax_structure_efficient(weights)
            else:
                converted_params = convert_params_to_jax_structure(weights)
            
            # Clear original weights to free memory
            del weights
            gc.collect()
            
            # Verify structure
            logger.info("Verifying converted parameter structure...")
            verify_parameter_structure(converted_params)
            
            # Check final norm specifically
            logger.info("Validating final layer norm parameter...")
            validate_final_norm_parameter(converted_params)
            
            # Try with a real model
            logger.info("Testing parameter compatibility with model...")
            try:
                test_model = FlaxQwen25ForCausalLM(config, _do_init=False)
                test_model.params = converted_params
                logger.info("✅ Successfully set parameters on model")
                
                # Cleanup
                del test_model
                del converted_params
                gc.collect()
                jax.clear_caches()
                
            except Exception as e:
                logger.error(f"❌ Failed to set parameters on model: {e}")
                import traceback
                traceback.print_exc()
        
        return

    # Special diagnostic mode
    if args.method == "diagnose":
        logger.info("=== Running parameter structure diagnostics ===")
        
        # Log first what we're expecting
        logger.info("Intended parameter structure from code inspection:")
        logger.info("- Top level: params/model/layers/layers_X/attention/q_proj/kernel")
        logger.info("- Final norm: params/model/layers/norm/scale (moved from params/model/norm/scale)")
        logger.info("- Note the 'layers/layers_X' double nesting and 'attention' (not 'self_attn')")
        
        # Check expected structure from a minimal model
        logger.info("Checking structure from a minimal model...")
        print_model_expected_structure(config)
        
        # Load weights
        logger.info("Loading weights for structure analysis...")
        weights = test_direct_safetensors(args.weights_dir, config, dtype)
        
        if weights is not None:
            # Try out our conversion without expensive validation
            try:
                logger.info("Testing parameter structure conversion...")
                
                # Use new conversion method if requested
                if args.use_new_conversion:
                    logger.info("Using new parameter conversion method...")
                    if args.memory_efficient:
                        fixed_params = convert_params_to_jax_structure_efficient(weights)
                    else:
                        fixed_params = convert_params_to_jax_structure(weights)
                else:
                    # Use original fix_params_structure function
                    fixed_params = fix_params_structure(weights)
                
                # Clear original weights to free memory
                del weights
                gc.collect()
                
                # Do a simple check for expected keys
                flat_params = flatten_dict(fixed_params)
                has_embedding = any('embed_tokens' in '/'.join(str(k)) for k in flat_params.keys())
                has_layer_0 = any('layers_0' in '/'.join(str(k)) for k in flat_params.keys())
                has_lm_head = any('lm_head' in '/'.join(str(k)) for k in flat_params.keys())
                
                # Check for final norm specifically
                has_final_norm = ('params', 'model', 'layers', 'norm', 'scale') in flat_params
                
                if has_embedding and has_layer_0 and has_lm_head and has_final_norm:
                    logger.info("✅ Basic structure validation passed")
                    
                    # Sample some parameters for inspection
                    sample_keys = []
                    for k in flat_params.keys():
                        key_str = '/'.join(str(x) for x in k)
                        if 'layers_0' in key_str or 'embed_tokens' in key_str or 'lm_head' in key_str or 'norm' in key_str:
                            sample_keys.append(k)
                        if len(sample_keys) >= 10:
                            break
                            
                    logger.info("Sample parameter paths:")
                    for k in sample_keys:
                        logger.info(f"  {'/'.join(str(x) for x in k)}: {flat_params[k].shape}")
                else:
                    logger.error("❌ Basic structure validation failed!")
                    logger.info("Missing critical components in structure:")
                    logger.info(f"  Embeddings present: {has_embedding}")
                    logger.info(f"  Layer 0 present: {has_layer_0}")
                    logger.info(f"  LM head present: {has_lm_head}")
                    logger.info(f"  Final norm present: {has_final_norm}")
                
                # Try loading into an uninitialized model
                try:
                    logger.info("Testing parameter compatibility with model...")
                    test_model = FlaxQwen25ForCausalLM(config, _do_init=False)
                    test_model.params = fixed_params
                    logger.info("✅ Successfully set parameters on model")
                    
                    # Clear model to free memory
                    del test_model
                    gc.collect()
                    jax.clear_caches()
                except Exception as e:
                    logger.error(f"❌ Failed to set parameters on model: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Clear structures to free memory
                del flat_params
                del fixed_params
                gc.collect()
            except Exception as e:
                logger.error(f"Error during structure conversion: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("Diagnostic mode completed.")
        return

    # Load tokenizer if available for forward pass testing
    tokenizer = None
    if args.test_forward:
        try:
            logger.info(f"Loading tokenizer from {args.weights_dir}")
            tokenizer = AutoTokenizer.from_pretrained(args.weights_dir)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")

    # Use alternative loading if specified
    if args.alt_loading:
        logger.info("=== Using alternative loading method ===")
        try:
            from qwen25_full_implementation import load_model_and_weights
            
            # Clear memory before loading
            gc.collect()
            jax.clear_caches()
            
            logger.info(f"Loading model directly using load_model_and_weights...")
            model = load_model_and_weights(args.weights_dir, dtype=dtype)
            
            # Perform parameter structure check if requested
            if args.param_structure_check and model is not None:
                logger.info("Analyzing loaded model parameter structure...")
                analyze_dict_structure(model.params, "alt_loaded_params")
                validate_final_norm_parameter(model.params)
            
            if model is not None and args.test_forward:
                logger.info("Testing forward pass with alternatively loaded weights...")
                test_forward_pass(model, config, tokenizer)
            
            return
        except Exception as e:
            logger.error(f"Alternative loading failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with regular loading if alternative fails

    # Test loading with the selected method(s)
    if args.method in ["direct", "all"]:
        logger.info("=== Testing direct safetensors loading ===")
        params = test_direct_safetensors(args.weights_dir, config, dtype)
        
        if params is not None:
            if args.test_forward:
                # Need to create a model with the params for forward pass testing
                logger.info("Creating model and setting parameters...")
                
                # Create model with minimal memory usage
                jax.clear_caches()
                gc.collect()
                
                model = FlaxQwen25ForCausalLM(config, dtype=dtype)
                
                # Handle possible memory constraints
                if args.memory_efficient:
                    logger.info("Using memory-efficient parameter conversion...")
                    
                    # Use new conversion method if requested
                    if args.use_new_conversion:
                        logger.info("Using new parameter conversion method...")
                        fixed_params = convert_params_to_jax_structure_efficient(params)
                    else:
                        # Process params incrementally to avoid holding two full copies
                        fixed_params = fix_params_structure(params)
                    
                    # Apply auto fix if requested
                    if args.auto_fix:
                        logger.info("Applying automatic parameter structure fixes...")
                        fixed_params = fix_missing_parameters(fixed_params, FlaxQwen25ForCausalLM, config)
                    
                    # Clear original params to free memory
                    del params
                    gc.collect()
                    jax.clear_caches()
                    
                    # Set parameters
                    logger.info("Setting model parameters...")
                    model.params = fixed_params
                    
                    # Clear parameter copy to free memory
                    del fixed_params
                    gc.collect()
                    jax.clear_caches()
                else:
                    # Standard conversion
                    logger.info("Converting parameters to model format...")
                    
                    # Use new conversion method if requested
                    if args.use_new_conversion:
                        logger.info("Using new parameter conversion method...")
                        fixed_params = convert_params_to_jax_structure(params)
                    else:
                        fixed_params = fix_params_structure(params)
                    
                    # Apply auto fix if requested
                    if args.auto_fix:
                        logger.info("Applying automatic parameter structure fixes...")
                        fixed_params = fix_missing_parameters(fixed_params, FlaxQwen25ForCausalLM, config)
                    
                    # Verify parameter structure if requested
                    if args.param_structure_check:
                        logger.info("Verifying parameter structure...")
                        verify_parameter_structure(fixed_params)
                        validate_final_norm_parameter(fixed_params)
                    
                    model.params = fixed_params
                    
                    # Clear original params to free memory
                    del params
                    del fixed_params
                    gc.collect()
                    jax.clear_caches()
                
                logger.info("Testing forward pass with direct loaded weights...")
                test_forward_pass(model, config, tokenizer)

    if args.method in ["index", "all"]:
        # Clear memory before next method
        gc.collect()
        jax.clear_caches()
        
        logger.info("=== Testing loading from index file ===")
        model = test_direct_load_from_index(args.weights_dir, config, dtype)
        if model is not None and args.test_forward:
            logger.info("Testing forward pass with index loaded weights...")
            test_forward_pass(model, config, tokenizer)

    if args.method in ["model", "all"]:
        # Clear memory before next method
        gc.collect()
        jax.clear_caches()
        
        logger.info("=== Testing load_model_and_weights function ===")
        model = test_model_and_weights(args.weights_dir, config, dtype)
        if model is not None and args.test_forward:
            logger.info("Testing forward pass with model loaded weights...")
            test_forward_pass(model, config, tokenizer)

    logger.info("Weight loading tests completed.")


if __name__ == "__main__":
    main() 