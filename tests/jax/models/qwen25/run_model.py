#!/usr/bin/env python3
"""Simple test script for the Qwen2.5-7B model with real weights.

Usage

Once in qwen25 directory,
source venv/bin/activate

Weights can be found in
qwen25-weights/

Memory-optimized version that implements streaming parameter loading.  
"""

import os
import sys
import time
import json
import argparse
import logging
import gc  # For garbage collection
import re
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from safetensors import safe_open
from functools import partial

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import the model
from tt_xla.jax.models.qwen25.model import Qwen25ForCausalLM, create_qwen25_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Memory tracking
try:
    import psutil
    def log_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")
except ImportError:
    def log_memory_usage():
        logger.info("psutil not available for memory tracking")

def load_config(config_path):
    """Load model configuration from JSON file."""
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

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
                # Log the tensor being loaded
                logger.debug(f"Loading tensor: {key}")
                key_count += 1
                
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
                    # Log progress periodically but at debug level
                    logger.debug(f"Processed {key_count} tensors...")
    
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
    parser = argparse.ArgumentParser(description="Test Qwen2.5-7B model with real weights (memory optimized)")
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
        "--input_text", 
        type=str, 
        default="Hello, world!",
        help="Text to tokenize and process through the model"
    )
    args = parser.parse_args()
    
    # Map dtype string to jnp.dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, jnp.bfloat16)
    
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX version: {jax.__version__}")
    log_memory_usage()
    
    try:
        # Load the model configuration
        config_path = os.path.join(args.weights_dir, "config.json")
        if os.path.exists(config_path):
            config = load_config(config_path)
            logger.info(f"Loaded configuration with {config.get('num_hidden_layers', 'unknown')} layers")
        else:
            raise ValueError(f"No config.json found in {args.weights_dir}")
        
        # Set up parameter sharding if enabled
        devices = jax.devices()
        mesh = None
        if args.shard and len(devices) > 1:
            logger.info(f"Setting up parameter sharding across {len(devices)} devices")
            mesh = Mesh(devices, ("dp", "mp"))
        
        # Create model instance without initializing parameters
        logger.info("Creating model structure...")
        model = create_qwen25_model(config, dtype=dtype)
        
        # Initialize inputs for inference
        input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        
        # Set up dummy variables for parameter structure
        params = {"params": {}}
        
        # Process weights by streaming parameters from each file
        logger.info(f"Processing weights from {args.weights_dir}...")
        safetensors_files = sorted([f for f in os.listdir(args.weights_dir) if f.endswith(".safetensors")])
        
        if not safetensors_files:
            raise ValueError("No safetensors files found in weights directory")
        
        for filename in safetensors_files:
            weight_path = os.path.join(args.weights_dir, filename)
            logger.info(f"Processing weights from {weight_path}")
            
            # Process one file at a time
            start_time = time.time()
            file_params = process_safetensors_file(weight_path, dtype=dtype)
            
            # Merge parameters
            params = merge_param_dicts(params, file_params)
            
            # Log progress and memory usage
            logger.info(f"Processed {filename} in {time.time() - start_time:.2f} seconds")
            log_memory_usage()
            
            # Force garbage collection
            del file_params
            gc.collect()
        
        logger.info("All parameters loaded and converted")
        log_memory_usage()
        
        # Run the model
        logger.info("Running forward pass...")
        start_time = time.time()
        
        with jax.profiler.trace("forward_pass"):
            if mesh is not None:
                # Use pjit for sharded execution
                from jax.experimental import pjit
                
                def sharded_forward(params, inputs):
                    return model.apply(params, inputs)
                
                p_forward = pjit.pjit(
                    sharded_forward,
                    in_shardings=(P(None), P(None)),
                    out_shardings=P(None)
                )
                
                with mesh:
                    outputs = p_forward(params, input_ids)
            else:
                # Regular forward pass
                outputs = model.apply(params, input_ids)
        
        logits = outputs["logits"]
        
        logger.info(f"Model ran in {time.time() - start_time:.2f} seconds")
        logger.info(f"Input shape: {input_ids.shape}")
        logger.info(f"Output shape: {logits.shape}")
        logger.info(f"First logit values: {logits[0, 0, :5]}")
        logger.info("Success!")
        log_memory_usage()
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 