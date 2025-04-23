#!/usr/bin/env python3
"""
Simplified Qwen2.5 Weight Loading for Flax

This streamlined script loads Qwen2.5 weights from safetensors files into a Flax model.
It uses the true_streaming_conversion approach which has proven most reliable for handling
parameter structure issues while maintaining memory efficiency.

Usage:
python load_qwen25_weights.py --weights_dir /path/to/qwen25-weights
"""

import os
import sys
import time
import json
import glob
import logging
import argparse
import gc
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import FrozenDict, freeze

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qwen25_loader")

# Import from qwen25 implementation file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen25_full_implementation import (
    Qwen25Config, 
    FlaxQwen25ForCausalLM,
    load_config_from_json,
    make_causal_mask
)

# Import required dependencies
from safetensors import safe_open

def log_memory_usage(label=""):
    """Log the current memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss_gb = memory_info.rss / (1024 ** 3)
        logger.info(f"Memory Usage {label}: RSS={rss_gb:.2f} GB")
        return rss_gb
    except Exception as e:
        logger.warning(f"Failed to log memory usage: {e}")
        return None

def load_qwen25_model(weights_dir, dtype=jnp.bfloat16, max_layers=4):
    """
    Load Qwen2.5 model using the most reliable streaming approach.
    
    Args:
        weights_dir: Path to directory containing safetensors files
        dtype: Data type for model (default: jnp.bfloat16)
        max_layers: Maximum number of layers to load (for memory efficiency)
        
    Returns:
        Initialized FlaxQwen25ForCausalLM model with weights loaded
    """
    # Start timing
    start_time = time.time()
    log_memory_usage("Before loading")
    
    # Load config
    config_path = os.path.join(weights_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    logger.info(f"Loading config from {config_path}")
    config = load_config_from_json(config_path)
    
    # Memory-saving option: Modify config to use fewer layers
    if max_layers < config.num_hidden_layers:
        logger.info(f"Memory-saving: Reducing layers from {config.num_hidden_layers} to {max_layers}")
        # Save original number for reference
        original_layers = config.num_hidden_layers
        config.num_hidden_layers = max_layers
    else:
        original_layers = config.num_hidden_layers
    
    # Find safetensors files
    safetensors_files = glob.glob(os.path.join(weights_dir, "*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {weights_dir}")
    
    logger.info(f"Found {len(safetensors_files)} safetensors files")
    
    # Create model structure to fill
    jax_params = {
        "model": {
            "embed_tokens": {},
            "layers": {
                # Create the norm structure directly
                "norm": {}
            }
        },
        "lm_head": {}
    }
    
    # Map parameters to files
    param_file_map = {}
    layer_count = 0
    
    # Scan files to create parameter map
    logger.info("Scanning files to map parameters...")
    for file_path in safetensors_files:
        with safe_open(file_path, framework="jax") as f:
            keys = f.keys()
            for key in keys:
                param_file_map[key] = file_path
                # Count layers
                if "model.layers." in key:
                    parts = key.split(".")
                    if len(parts) >= 3:
                        try:
                            layer_num = int(parts[2])
                            layer_count = max(layer_count, layer_num + 1)
                        except ValueError:
                            pass
    
    logger.info(f"Detected {layer_count} layers in weights, loading {max_layers} layers")
    
    # Load critical parameters first
    critical_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight"
    ]
    
    for key in critical_keys:
        if key in param_file_map:
            file_path = param_file_map[key]
            logger.info(f"Loading critical parameter: {key}")
            
            with safe_open(file_path, framework="jax") as f:
                if key in f.keys():
                    value = f.get_tensor(key)
                    
                    # Cast to target dtype for memory efficiency
                    value = value.astype(dtype)
                    
                    if key == "model.embed_tokens.weight":
                        jax_params["model"]["embed_tokens"]["embedding"] = value
                    elif key == "model.norm.weight":
                        jax_params["model"]["layers"]["norm"]["scale"] = value
                    elif key == "lm_head.weight":
                        jax_params["lm_head"]["kernel"] = value.T
            
            # Force memory cleanup
            gc.collect()
    
    # Process only up to max_layers
    for layer_idx in range(min(max_layers, layer_count)):
        logger.info(f"Processing layer {layer_idx}/{min(max_layers, layer_count)-1}")
        layer_key = f"layers_{layer_idx}"
        
        # Initialize layer structure
        jax_params["model"]["layers"][layer_key] = {
            "attention": {},
            "mlp": {},
            "input_layernorm": {},
            "post_attention_layernorm": {}
        }
        
        # Define parameter patterns for this layer
        layer_patterns = [
            # Attention parameters
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            f"model.layers.{layer_idx}.self_attn.q_proj.bias",
            f"model.layers.{layer_idx}.self_attn.k_proj.bias",
            f"model.layers.{layer_idx}.self_attn.v_proj.bias",
            f"model.layers.{layer_idx}.self_attn.o_proj.bias",
            
            # MLP parameters
            f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight",
            f"model.layers.{layer_idx}.mlp.down_proj.weight",
            f"model.layers.{layer_idx}.mlp.gate_proj.bias",
            f"model.layers.{layer_idx}.mlp.up_proj.bias",
            f"model.layers.{layer_idx}.mlp.down_proj.bias",
            
            # Layernorm parameters
            f"model.layers.{layer_idx}.input_layernorm.weight",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        ]
        
        # Process each parameter for this layer
        for pattern in layer_patterns:
            if pattern in param_file_map:
                file_path = param_file_map[pattern]
                
                with safe_open(file_path, framework="jax") as f:
                    if pattern in f.keys():
                        value = f.get_tensor(pattern)
                        
                        # Cast to target dtype for memory efficiency
                        value = value.astype(dtype)
                        
                        # Map to the proper location in the structure
                        if "self_attn" in pattern:
                            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                                if f"self_attn.{proj}.weight" in pattern:
                                    if proj not in jax_params["model"]["layers"][layer_key]["attention"]:
                                        jax_params["model"]["layers"][layer_key]["attention"][proj] = {}
                                    # Transpose weights for Flax format
                                    jax_params["model"]["layers"][layer_key]["attention"][proj]["kernel"] = value.T
                                    break
                                elif f"self_attn.{proj}.bias" in pattern:
                                    if proj not in jax_params["model"]["layers"][layer_key]["attention"]:
                                        jax_params["model"]["layers"][layer_key]["attention"][proj] = {}
                                    jax_params["model"]["layers"][layer_key]["attention"][proj]["bias"] = value
                                    break
                        
                        elif "mlp" in pattern:
                            for proj in ["gate_proj", "up_proj", "down_proj"]:
                                if f"mlp.{proj}.weight" in pattern:
                                    if proj not in jax_params["model"]["layers"][layer_key]["mlp"]:
                                        jax_params["model"]["layers"][layer_key]["mlp"][proj] = {}
                                    jax_params["model"]["layers"][layer_key]["mlp"][proj]["kernel"] = value.T
                                    break
                                elif f"mlp.{proj}.bias" in pattern:
                                    if proj not in jax_params["model"]["layers"][layer_key]["mlp"]:
                                        jax_params["model"]["layers"][layer_key]["mlp"][proj] = {}
                                    jax_params["model"]["layers"][layer_key]["mlp"][proj]["bias"] = value
                                    break
                        
                        elif "input_layernorm.weight" in pattern:
                            jax_params["model"]["layers"][layer_key]["input_layernorm"]["scale"] = value
                        
                        elif "post_attention_layernorm.weight" in pattern:
                            jax_params["model"]["layers"][layer_key]["post_attention_layernorm"]["scale"] = value
                        
                        # Free memory immediately
                        del value
                
        # Garbage collect after each layer
        gc.collect()
        jax.clear_caches()
        log_memory_usage(f"After layer {layer_idx}")
    
    # Freeze parameters and create model
    logger.info("Creating model with loaded parameters...")
    
    # First verify all critical parameters are present
    for path, expected in [
        (["model", "embed_tokens", "embedding"], "Embedding"),
        (["model", "layers", "norm", "scale"], "Final norm"),
        (["lm_head", "kernel"], "LM head")
    ]:
        current = jax_params
        for part in path:
            if part not in current:
                raise ValueError(f"Critical parameter missing: {expected}")
            current = current[part]
    
    # Create model with _do_init=True 
    logger.info("Creating model with proper initialization...")
    model = FlaxQwen25ForCausalLM(config, dtype=dtype, _do_init=True)
    
    # Freeze parameters first
    params_frozen = freeze(jax_params)
    
    # Log timing
    total_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {total_time:.2f} seconds")
    log_memory_usage("After loading")
    
    # Return the model, parameters and original config
    return model, params_frozen, original_layers

def prepare_attention_mask_for_forward(input_ids, dtype=jnp.float32):
    """
    Create properly formatted attention mask for the model's forward pass.
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        dtype: Data type for the mask
        
    Returns:
        Attention mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_length = input_ids.shape
    
    # Create a causal mask manually
    # First, create a square matrix where each position (i,j) contains j
    idxs = jnp.broadcast_to(jnp.arange(seq_length), (seq_length, seq_length))
    
    # Then create a matrix where each position (i,j) contains i
    idxs_t = jnp.transpose(idxs)
    
    # This creates a mask where position (i,j) is 1 if j<=i, and 0 otherwise
    causal_mask = idxs >= idxs_t
    
    # Convert to float32 and adjust for proper masking in attention (1.0 for positions to attend to, -1e10 for positions to mask)
    causal_mask = causal_mask.astype(dtype)
    causal_mask = jnp.where(causal_mask, 0.0, -1e10)
    
    # Reshape to (1, 1, seq_len, seq_len) for broadcasting across batch size and heads
    causal_mask = causal_mask.reshape(1, 1, seq_length, seq_length)
    
    # Broadcast to (batch_size, 1, seq_len, seq_len)
    causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_length, seq_length))
    
    return causal_mask

def main():
    parser = argparse.ArgumentParser(description="Load Qwen2.5 model for Flax")
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Directory containing the weight files",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to use for weights",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=4,
        help="Maximum number of layers to load (for memory efficiency)",
    )
    parser.add_argument(
        "--test_forward",
        action="store_true",
        help="Test a forward pass with dummy input",
    )
    args = parser.parse_args()

    # Map dtype string to jnp.dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load model
    try:
        # Get model and parameters
        model, params, original_layers = load_qwen25_model(args.weights_dir, dtype, args.max_layers)
        print(f"✅ Model loaded successfully! Using {args.max_layers} of {original_layers} layers")
        
        # Test forward pass if requested
        if args.test_forward:
            print("Testing forward pass...")
            # Create dummy input - use minimal size
            dummy_input = jnp.ones((1, 5), dtype=jnp.int32)
            
            # Create proper attention mask with causal mask
            attention_mask = prepare_attention_mask_for_forward(dummy_input, dtype=dtype)
            
            # Run forward pass by passing params explicitly
            outputs = model(dummy_input, attention_mask=attention_mask, params=params)
            print(f"✅ Forward pass successful! Output shape: {outputs.logits.shape}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 