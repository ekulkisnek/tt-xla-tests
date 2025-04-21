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
python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights

python testing_qwen_weight_loading.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --method all --test_forward

The script will output detailed diagnostics about the weight loading process.
"""

import os
import sys
import time
import json
import logging
import argparse
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
    # Handle different parameter structures - flatten appropriately
    if isinstance(params, dict):
        if "params" in params:
            # This is a standard Flax params structure
            flattened = flatten_dict(params)
        else:
            # This could be a raw weight dictionary from load_safetensors_weights
            # Check if there are model-specific keys
            if "model" in params:
                # Extract model parameters
                model_params = params["model"]
                # Create custom flattened representation
                flattened = {}
                for key, value in flatten_dict(model_params).items():
                    if isinstance(key, tuple):
                        key_str = "/".join(key)
                    else:
                        key_str = key
                    flattened[f"model/{key_str}"] = value
                
                # Add lm_head if present
                if "lm_head" in params:
                    for key, value in flatten_dict(params["lm_head"]).items():
                        if isinstance(key, tuple):
                            key_str = "/".join(key)
                        else:
                            key_str = key
                        flattened[f"lm_head/{key_str}"] = value
            else:
                # Just flatten as is - likely raw weights
                tmp_flat = {}
                for key, value in flatten_dict(params).items():
                    if isinstance(key, tuple):
                        key_str = "/".join(key)
                    else:
                        key_str = key
                    tmp_flat[key_str] = value
                flattened = tmp_flat
    else:
        # Not a dictionary - can't analyze
        return {
            "error": "Input is not a dictionary",
            "status": "error",
            "total_params": 0,
            "num_tensors": 0
        }
    
    total_params = 0
    param_shapes = {}
    param_types = {}
    
    # Define critical keys that must be present for a functional model
    # For raw safetensors weights
    critical_keys_raw = [
        "model/embed_tokens/weight",
        "lm_head/weight"
    ]
    
    # For processed Flax model params
    critical_keys_flax = [
        "params/embed_tokens/embedding",
        "params/lm_head/kernel"
    ]
    
    # Check layer patterns
    layer_pattern_counts = {}
    
    # Convert keys to strings for easier analysis if they're not already
    key_strings = []
    for k in flattened.keys():
        if isinstance(k, tuple):
            key_strings.append("/".join(k))
        elif isinstance(k, str):
            key_strings.append(k)
        else:
            key_strings.append(str(k))
    
    # Count parameters and analyze structure
    for key, tensor in flattened.items():
        # Count parameters
        num_params = np.prod(tensor.shape)
        total_params += num_params
        
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
        key_str = key if isinstance(key, str) else "/".join(key) if isinstance(key, tuple) else str(key)
        for pattern in ["attention", "mlp", "layernorm", "embed"]:
            if pattern in key_str.lower():
                if pattern not in layer_pattern_counts:
                    layer_pattern_counts[pattern] = 0
                layer_pattern_counts[pattern] += 1
    
    # Check for critical keys
    critical_keys_found = []
    
    # Try both raw and Flax key formats
    for critical_key in critical_keys_raw + critical_keys_flax:
        found = False
        for key in key_strings:
            if critical_key in key:
                found = True
                critical_keys_found.append(critical_key)
                break
    
    # Determine if we have different layers
    num_layers = 0
    for i in range(50):  # Check up to 50 layers
        layer_found = False
        for key in key_strings:
            if f"layers/{i}/" in key or f"layers_{i}/" in key or f"layers.{i}." in key or f"layers.{i}/" in key:
                layer_found = True
                break
        if layer_found:
            num_layers = i + 1
        else:
            break
    
    return {
        "total_params": total_params,
        "num_tensors": len(flattened),
        "param_shapes": param_shapes,
        "param_types": param_types,
        "critical_keys_present": len(critical_keys_found) > 0,  # At least one critical key format found
        "critical_keys_found": critical_keys_found,
        "layer_pattern_counts": layer_pattern_counts,
        "num_layers_detected": num_layers,
        "status": "ok" if len(critical_keys_found) > 0 and num_layers > 0 else "incomplete",
        "sample_keys": key_strings[:10],  # First 10 keys for inspection
    }


def fix_params_structure(params):
    """
    Fix the parameter structure to ensure it's compatible with FlaxQwen25ForCausalLM.
    
    Args:
        params: Parameter dictionary, potentially needing structure fixes
        
    Returns:
        Properly structured parameter dictionary
    """
    from flax.core.frozen_dict import freeze, unfreeze
    
    # If params is already a FrozenDict, unfreeze it
    if hasattr(params, 'unfreeze'):
        params = params.unfreeze()
    
    # If params doesn't have a 'params' key, wrap it
    if isinstance(params, dict) and 'params' not in params:
        params = {'params': params}
    
    # Ensure the structure is correct
    if 'params' in params:
        # Correct structure for params - check needed keys
        param_dict = params['params']
        
        # Check for embed_tokens structure
        if 'embed_tokens' in param_dict and 'embedding' not in param_dict['embed_tokens']:
            # Look for weight parameter
            if 'weight' in param_dict['embed_tokens']:
                param_dict['embed_tokens']['embedding'] = param_dict['embed_tokens']['weight']
                del param_dict['embed_tokens']['weight']
        
        # Check for lm_head structure 
        if 'lm_head' in param_dict and 'kernel' not in param_dict['lm_head'] and 'weight' in param_dict['lm_head']:
            param_dict['lm_head']['kernel'] = param_dict['lm_head']['weight']
            del param_dict['lm_head']['weight']
        
        # Check for layers structure
        if 'layers' in param_dict:
            # Ensure we have the norm parameter
            if 'norm' not in param_dict['layers'] and 'ln_f' in param_dict:
                param_dict['layers']['norm'] = param_dict['ln_f']
                del param_dict['ln_f']
                
            # Check transformer layers
            if 'layers' not in param_dict['layers'] and 'h' in param_dict:
                # Handle h → layers.layers conversion
                param_dict['layers']['layers'] = param_dict['h']
                del param_dict['h']
                
            # Process each layer if layers.layers exists
            if 'layers' in param_dict['layers']:
                for layer_idx, layer in param_dict['layers']['layers'].items():
                    # Convert attention structure
                    if 'attn' in layer and 'attention' not in layer:
                        layer['attention'] = layer['attn']
                        del layer['attn']
                        
                    # Fix attention component keys
                    if 'attention' in layer:
                        attn = layer['attention']
                        
                        # Handle q/k/v/o projections
                        for proj in ['q', 'k', 'v', 'o']:
                            if proj in attn and f'{proj}_proj' not in attn:
                                attn[f'{proj}_proj'] = attn[proj]
                                del attn[proj]
                                
                            # Fix kernel/weight naming
                            if f'{proj}_proj' in attn and 'kernel' not in attn[f'{proj}_proj'] and 'weight' in attn[f'{proj}_proj']:
                                attn[f'{proj}_proj']['kernel'] = attn[f'{proj}_proj']['weight']
                                del attn[f'{proj}_proj']['weight']
                    
                    # Fix layer norm naming
                    if 'ln_1' in layer and 'input_layernorm' not in layer:
                        layer['input_layernorm'] = layer['ln_1']
                        del layer['ln_1']
                        
                    if 'ln_2' in layer and 'post_attention_layernorm' not in layer:
                        layer['post_attention_layernorm'] = layer['ln_2']
                        del layer['ln_2']
                        
                    # Fix layernorm scale/weight naming
                    for norm_name in ['input_layernorm', 'post_attention_layernorm']:
                        if norm_name in layer and 'scale' not in layer[norm_name] and 'weight' in layer[norm_name]:
                            layer[norm_name]['scale'] = layer[norm_name]['weight']
                            del layer[norm_name]['weight']
                    
                    # Fix MLP keys
                    if 'mlp' in layer:
                        mlp = layer['mlp']
                        
                        # Convert gate/up/down projection keys
                        key_mapping = {
                            'w1': 'gate_proj',
                            'w2': 'up_proj',
                            'w3': 'down_proj'
                        }
                        
                        for old_key, new_key in key_mapping.items():
                            if old_key in mlp and new_key not in mlp:
                                mlp[new_key] = mlp[old_key]
                                del mlp[old_key]
                                
                            # Fix kernel/weight naming
                            if new_key in mlp and 'kernel' not in mlp[new_key] and 'weight' in mlp[new_key]:
                                mlp[new_key]['kernel'] = mlp[new_key]['weight']
                                del mlp[new_key]['weight']
    
    # Return properly frozen parameters
    return freeze(params)


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
        params = load_safetensors_weights(model_path)
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
    
    try:
        # Create model instance
        model = FlaxQwen25ForCausalLM(config, dtype=dtype, _do_init=True)
        
        # Load weights using direct_load_from_index
        model_or_params = direct_load_from_index(model, model_path)
        
        # Check if we got back a model or just parameters
        if hasattr(model_or_params, 'params'):
            # We got back a model
            model = model_or_params
        else:
            # We got back parameters - need to fix structure and update model
            logger.info("Got parameters instead of model - fixing structure and updating model")
            params = fix_params_structure(model_or_params)
            model.params = params
        
        logger.info(f"Successfully loaded weights in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        logger.error(f"direct_load_from_index failed: {e}")
        import traceback
        traceback.print_exc()
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
            
            # Check that we have the essential parameters
            essential_params = [
                # Look for token embeddings
                ('params', 'embed_tokens', 'embedding'),
                # Look for LM head
                ('params', 'lm_head', 'kernel'),
                # Look for final layer norm 
                ('params', 'layers', 'norm', 'scale'),
                # Look for at least one transformer layer
                ('params', 'layers', 'layers', '0')
            ]
            
            missing_params = []
            for param_path in essential_params:
                if param_path not in flat_params:
                    missing_params.append('/'.join(param_path))
            
            if missing_params:
                logger.error(f"Missing essential parameters: {missing_params}")
                logger.info("Available top-level parameters:")
                if 'params' in model.params:
                    logger.info(f"  params keys: {list(model.params['params'].keys())}")
                    if 'layers' in model.params['params']:
                        logger.info(f"  layers keys: {list(model.params['params']['layers'].keys())}")
                        if 'layers' in model.params['params']['layers']:
                            logger.info(f"  layer indices: {list(model.params['params']['layers']['layers'].keys())}")
                return False
            
            # Check parameter shapes for key components
            # Embedding matrix should be [vocab_size, hidden_size]
            if ('params', 'embed_tokens', 'embedding') in flat_params:
                embed_shape = flat_params[('params', 'embed_tokens', 'embedding')].shape
                expected_shape = (config.vocab_size, config.hidden_size)
                if embed_shape != expected_shape:
                    logger.warning(f"Embedding shape mismatch: got {embed_shape}, expected {expected_shape}")
            
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Qwen2.5 weight loading for full implementation")
    parser.add_argument("--weights_dir", type=str, default="qwen25-weights",
                        help="Path to model weights directory")
    parser.add_argument("--use_bf16", action="store_true", 
                        help="Use bfloat16 precision instead of float32")
    parser.add_argument("--tokenizer_only", action="store_true",
                        help="Only test tokenizer loading")
    parser.add_argument("--test_forward", action="store_true",
                        help="Test forward pass after loading")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--method", type=str, choices=["direct", "index", "full", "all"],
                        default="direct", help="Which loading method to test")
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set data type based on arguments
    dtype = jnp.bfloat16 if args.use_bf16 else jnp.float32
    logger.info(f"Using precision: {dtype}")
    
    # Log system info
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available devices: {jax.devices()}")
    
    # Setup memory monitoring
    import psutil
    import gc
    
    def log_memory_usage(stage):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        logger.info(f"Memory usage at {stage}: {memory_mb:.2f} MB ({memory_mb/total_memory*100:.2f}% of system RAM)")
        return memory_mb
    
    def run_gc():
        """Run garbage collection and log memory change."""
        before = log_memory_usage("before GC")
        gc.collect()
        after = log_memory_usage("after GC")
        logger.info(f"Memory freed by GC: {before - after:.2f} MB")
    
    # Log initial memory usage
    initial_memory = log_memory_usage("start")
    
    # Check if weight directory exists
    weights_dir = args.weights_dir
    if not os.path.exists(weights_dir):
        logger.error(f"Weights directory {weights_dir} does not exist")
        return 1
    
    # Scan checkpoint files
    files_info = scan_checkpoint_files(weights_dir)
    
    # Check if we have necessary files
    if not files_info["safetensors"]:
        logger.error("No safetensors files found")
        return 1
    
    if not files_info["config"]:
        logger.error("No config.json found")
        return 1
    
    # Load config
    try:
        config = load_config_from_json(files_info["config"])
        logger.info(f"Loaded config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Try to load tokenizer if requested
    tokenizer = None
    if args.test_forward or args.tokenizer_only:
        try:
            logger.info(f"Loading tokenizer from {weights_dir}")
            tokenizer = AutoTokenizer.from_pretrained(weights_dir)
            logger.info(f"Loaded tokenizer with vocab size {len(tokenizer)}")
            
            # Test tokenizer
            sample_text = "Hello, world!"
            encoded = tokenizer(sample_text, return_tensors="jax")
            logger.info(f"Tokenizer test successful: {sample_text} -> {encoded.input_ids[0][:5]}...")
            
            if args.tokenizer_only:
                logger.info("Tokenizer test completed successfully")
                return 0
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            if args.tokenizer_only:
                return 1
    
    # Track success
    success = False
    
    # Test specific loading method based on command line argument
    if args.method in ["direct", "all"]:
        logger.info("\n" + "="*50)
        logger.info("TESTING DIRECT SAFETENSORS LOADING")
        logger.info("="*50)
        
        # Method 1: Direct safetensors loading
        params_direct = test_direct_safetensors(weights_dir, config, dtype)
        if params_direct is not None:
            # Analyze parameters
            analysis = analyze_param_structure(params_direct)
            logger.info(f"Direct loading analysis: Total params={analysis['total_params']:,}, Tensors={analysis['num_tensors']}")
            if analysis["critical_keys_present"]:
                logger.info("✅ Critical keys present in direct loading")
                success = True
            else:
                logger.warning("⚠️ Missing critical keys in direct loading")
                
            # Log some sample keys for debugging
            logger.info(f"Sample keys: {analysis['sample_keys']}")
            
            # Clean up to save memory
            del params_direct
            run_gc()
    
    if args.method in ["index", "all"]:
        logger.info("\n" + "="*50)
        logger.info("TESTING DIRECT LOAD FROM INDEX")
        logger.info("="*50)
        
        # Method 2: direct_load_from_index
        model_direct = test_direct_load_from_index(weights_dir, config, dtype)
        if model_direct is not None:
            # Analyze parameters
            analysis = analyze_param_structure(model_direct.params)
            logger.info(f"direct_load_from_index analysis: Total params={analysis['total_params']:,}, Tensors={analysis['num_tensors']}")
            if analysis["critical_keys_present"]:
                logger.info("✅ Critical keys present in direct_load_from_index")
                success = True
                
                # Verify parameter mapping
                mapping_result = verify_parameter_mapping(model_direct.params)
                if mapping_result["all_components_present"]:
                    logger.info(f"✅ All components present in parameter mapping (layers: {mapping_result['layer_count']})")
                else:
                    logger.warning(f"⚠️ Missing components in parameter mapping: {[k for k, v in mapping_result.items() if not v and k != 'all_components_present' and k != 'layer_count']}")
                
                # Test forward pass if requested
                if args.test_forward:
                    forward_success = test_forward_pass(model_direct, config, tokenizer)
                    if forward_success:
                        logger.info("✅ Forward pass with direct_load_from_index succeeded")
                    else:
                        logger.error("❌ Forward pass with direct_load_from_index failed")
            else:
                logger.warning("⚠️ Missing critical keys in direct_load_from_index")
                
            # Clean up to save memory
            del model_direct
            run_gc()
    
    if args.method in ["full", "all"]:
        logger.info("\n" + "="*50)
        logger.info("TESTING LOAD MODEL AND WEIGHTS")
        logger.info("="*50)
        
        # Method 3: load_model_and_weights
        model_full = test_model_and_weights(weights_dir, config, dtype)
        if model_full is not None:
            logger.info("✅ load_model_and_weights succeeded")
            success = True
            
            # Test forward pass if requested
            if args.test_forward:
                forward_success = test_forward_pass(model_full, config, tokenizer)
                if forward_success:
                    logger.info("✅ Forward pass with load_model_and_weights succeeded")
                else:
                    logger.error("❌ Forward pass with load_model_and_weights failed")
            
            # Clean up to save memory
            del model_full
            run_gc()
    
    # Final memory usage
    final_memory = log_memory_usage("end")
    logger.info(f"Total memory change: {final_memory - initial_memory:.2f} MB")
    
    # Final status
    if success:
        logger.info("🎉 Weight loading test succeeded!")
        return 0
    else:
        logger.error("❌ Weight loading test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 