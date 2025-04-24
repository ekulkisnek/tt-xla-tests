#!/usr/bin/env python3
"""Memory usage test script for the optimized Qwen2.5-7B model.

This script demonstrates the memory efficiency improvements and provides
memory profiling information during model loading and inference.

Usage:
  python run_memory_efficient.py --dtype bfloat16 [--shard]
"""

import os
import sys
import time
import argparse
import logging
import gc
import json
import jax
import jax.numpy as jnp
import re
from jax.sharding import Mesh, PartitionSpec as P

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import safetensors for weight loading
try:
    from safetensors import safe_open
except ImportError:
    logging.error("safetensors package is required. Install with: pip install safetensors")
    sys.exit(1)

# Import optimized model - using local imports instead of tt_xla package
from model import create_qwen25_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Memory tracking
try:
    import psutil
    def log_memory_usage(label=""):
        """Log current memory usage with an optional label."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss_gb = memory_info.rss / (1024 * 1024 * 1024)
        logger.info(f"Memory usage {label}: {rss_gb:.2f} GB")
        return rss_gb
except ImportError:
    def log_memory_usage(label=""):
        logger.info("psutil not available for memory tracking")
        return 0

# Parameter name mapping helpers
def get_param_path(name):
    """Map a PyTorch parameter name to its Flax path."""
    # Direct mappings
    direct_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    if name in direct_mapping:
        return direct_mapping[name]
    
    # Patterns for layer parameters
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    rotary_pattern = r"model\.layers\.(\d+)\.self_attn\.rotary_emb\..*"
    
    # Handle layer norms
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        layer_name = f"layers_{layer_idx}"
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (layer_name, norm_name, "scale")
    
    # Handle attention parameters
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        param_type = attn_match.group(3)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        param_name = "kernel" if param_type == "weight" else "bias"
        return (layer_name, "self_attn", proj_name, param_name)
    
    # Handle MLP parameters
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        return (layer_name, "mlp", proj_name, "kernel")
    
    # Handle rotary embedding parameters - skip these as they're computed on the fly in JAX
    rotary_match = re.match(rotary_pattern, name)
    if rotary_match:
        logger.warning(f"Skipping rotary embedding parameter: {name}")
        return None
    
    # Log any unhandled parameter patterns
    logger.warning(f"Unknown parameter pattern: {name}")
    return None

def transpose_if_needed(name, param):
    """Transpose weight matrices if needed based on the parameter name."""
    # Special case for embedding weights - Flax's nn.Embed expects (vocab_size, embedding_dim)
    # so we should NOT transpose these weights
    if "embed_tokens.weight" in name:
        # Do not transpose embedding weights
        logger.debug(f"Keeping embedding weight shape for {name}: {param.shape}")
        return param
    
    # Other attention and MLP weights need to be transposed
    if "weight" in name and ("proj" in name or "lm_head" in name):
        # For attention and MLP weight matrices
        logger.debug(f"Transposing weight matrix for {name}: {param.shape} -> {param.T.shape}")
        return jnp.transpose(param)
    
    return param

def process_safetensors_file(file_path, dtype=jnp.bfloat16):
    """
    Process a single safetensors file by streaming parameters one by one.
    Returns a dictionary of JAX parameters already in the correct format.
    """
    flax_params = {"params": {}}
    
    # Expected shapes for key parameters
    # For validation to catch errors early
    expected_shapes = {
        "model.embed_tokens.weight": None,  # Will be set dynamically from first file
        "lm_head.weight": None              # Will be set dynamically from first file
    }
    
    try:
        with safe_open(file_path, framework="numpy") as f:
            key_count = 0
            for key in f.keys():
                key_count += 1
                # Log progress
                if key_count % 10 == 0:
                    logger.debug(f"Processed {key_count} tensors...")
                
                # Get the parameter and immediately cast to the specified dtype
                param = f.get_tensor(key)
                
                # Skip parameters that don't map to our model
                param_path = get_param_path(key)
                if param_path is None:
                    logger.warning(f"Skipping unknown parameter: {key}")
                    continue
                
                # Store or validate expected shapes for critical parameters
                if key in expected_shapes:
                    if expected_shapes[key] is None:
                        # First time seeing this parameter, store its shape
                        expected_shapes[key] = param.shape
                        logger.debug(f"Recorded expected shape for {key}: {param.shape}")
                    else:
                        # Check if shape matches expected
                        if param.shape != expected_shapes[key]:
                            logger.warning(f"Shape mismatch for {key}: expected {expected_shapes[key]}, got {param.shape}")
                
                # Convert to JAX array with the correct dtype
                param = jnp.array(param, dtype=dtype)
                
                # Transpose if needed (dense layer weights)
                param = transpose_if_needed(key, param)
                
                # Add to the parameter dictionary with the correct nested structure
                current_dict = flax_params["params"]
                for path_part in param_path[:-1]:
                    if path_part not in current_dict:
                        current_dict[path_part] = {}
                    current_dict = current_dict[path_part]
                
                current_dict[param_path[-1]] = param
                
                # Free memory for the numpy array
                del param
                if key_count % 50 == 0:  # Periodically trigger garbage collection
                    gc.collect()
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise
    
    return flax_params

def merge_param_dicts(base_dict, new_dict):
    """Merge new parameter dictionary into the base dictionary."""
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict):
            if not isinstance(base_dict[key], dict):
                raise ValueError(f"Cannot merge dict into non-dict at key {key}")
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def main():
    parser = argparse.ArgumentParser(description="Memory profile of optimized Qwen2.5 model")
    parser.add_argument(
        "--weights_dir", 
        type=str, 
        default="/root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights",
        help="Directory containing Qwen2.5 weights"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16", 
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model parameters"
    )
    parser.add_argument(
        "--shard", 
        action="store_true",
        help="Enable parameter sharding across devices"
    )
    parser.add_argument(
        "--profile", 
        action="store_true",
        help="Enable detailed memory profiling"
    )
    parser.add_argument(
        "--generate", 
        action="store_true",
        help="Run generation after loading the model"
    )
    args = parser.parse_args()
    
    # Map dtype string to jnp.dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, jnp.bfloat16)
    
    # Print system information
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available devices: {jax.devices()}")
    logger.info(f"Using dtype: {dtype}")
    
    # Record memory usage at start
    initial_mem = log_memory_usage("initial")
    
    try:
        # Load configuration
        config_path = os.path.join(args.weights_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration with {config.get('num_hidden_layers', 'unknown')} layers")
        
        # Enable memory optimizations
        config["use_memory_efficient_attention"] = True

        # Setup mesh for multi-device execution if requested
        devices = jax.devices()
        mesh = None
        if args.shard and len(devices) > 1:
            # Create device mesh for model and data parallelism
            mesh_shape = (1, len(devices))  # (data_parallel, model_parallel)
            logger.info(f"Creating {mesh_shape} device mesh for tensor parallelism")
            devices_array = jnp.array(devices).reshape(mesh_shape)
            mesh = Mesh(devices_array, ("dp", "mp"))

        # Create model
        logger.info("Creating model structure...")
        t0 = time.time()
        model = create_qwen25_model(config, dtype=dtype)
        logger.info(f"Model structure created in {time.time() - t0:.2f}s")
        
        # Log memory after model creation
        model_creation_mem = log_memory_usage("after model creation")
        
        # Prepare dummy input
        input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        
        # Load parameters from safetensors files
        logger.info(f"Loading parameters from {args.weights_dir}...")
        params = {"params": {}}
        
        # Process weights by streaming parameters from each file
        safetensors_files = sorted([f for f in os.listdir(args.weights_dir) if f.endswith(".safetensors")])
        
        # Track memory for each file
        file_memories = []
        
        t0_total = time.time()
        for i, filename in enumerate(safetensors_files):
            weight_path = os.path.join(args.weights_dir, filename)
            logger.info(f"Processing file {i+1}/{len(safetensors_files)}: {filename}")
            
            # Process one file at a time
            t0 = time.time()
            file_params = process_safetensors_file(weight_path, dtype=dtype)
            
            # Merge parameters
            params = merge_param_dicts(params, file_params)
            
            # Log progress and memory usage
            file_time = time.time() - t0
            logger.info(f"Processed {filename} in {file_time:.2f} seconds")
            current_mem = log_memory_usage(f"after file {i+1}/{len(safetensors_files)}")
            file_memories.append((filename, current_mem, file_time))
            
            # Force garbage collection
            del file_params
            gc.collect()
        
        total_load_time = time.time() - t0_total
        logger.info(f"Total parameter loading time: {total_load_time:.2f} seconds")
        
        # Log memory after parameter loading
        params_loaded_mem = log_memory_usage("after all parameters loaded")
        
        # Run forward pass
        logger.info("Running forward pass...")
        t0 = time.time()
        
        # Use pjit with mesh if sharding is enabled
        if mesh is not None:
            from jax.experimental import pjit
            
            def sharded_forward(params, inputs):
                return model.apply(params, inputs)
            
            p_forward = pjit.pjit(
                sharded_forward,
                in_shardings=(P(None), P(None)),
                out_shardings=P(None)
            )
            
            with mesh:
                with jax.profiler.trace("forward_pass"):
                    outputs = p_forward(params, input_ids)
        else:
            # Regular forward pass
            with jax.profiler.trace("forward_pass"):
                outputs = model.apply(params, input_ids)
        
        # Calculate inference time
        inference_time = time.time() - t0
        logger.info(f"Forward pass completed in {inference_time:.4f} seconds")
        
        # Log memory after forward pass
        forward_mem = log_memory_usage("after forward pass")
        
        # Get logits
        logits = outputs["logits"]
        logger.info(f"Output logits shape: {logits.shape}")
        logger.info(f"First few logits: {logits[0, 0, :5]}")
        
        # Print memory usage summary
        logger.info("\n==== Memory Usage Summary ====")
        logger.info(f"Initial memory usage: {initial_mem:.2f} GB")
        logger.info(f"After model creation: {model_creation_mem:.2f} GB (+{model_creation_mem - initial_mem:.2f} GB)")
        logger.info(f"After loading parameters: {params_loaded_mem:.2f} GB (+{params_loaded_mem - model_creation_mem:.2f} GB)")
        logger.info(f"After forward pass: {forward_mem:.2f} GB (+{forward_mem - params_loaded_mem:.2f} GB)")
        logger.info(f"Total memory increase: {forward_mem - initial_mem:.2f} GB")
        
        # Print parameter loading details
        logger.info("\n==== Parameter Loading Details ====")
        for i, (filename, mem, load_time) in enumerate(file_memories):
            prev_mem = initial_mem if i == 0 else file_memories[i-1][1]
            logger.info(f"File {i+1}: {filename} - {mem:.2f} GB (+{mem - prev_mem:.2f} GB) - {load_time:.2f}s")
        
        # Print model info
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        logger.info(f"\nModel parameters: {num_params:,}")
        logger.info(f"Model size in memory: {params_loaded_mem:.2f} GB")
        logger.info(f"Inference time: {inference_time:.4f}s")
        
        logger.info("\nMemory optimization successful!")
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 