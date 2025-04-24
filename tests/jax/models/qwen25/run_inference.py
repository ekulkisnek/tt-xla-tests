#!/usr/bin/env python3
"""
Run inference with the Qwen25 JAX model.
"""

import os
import sys
import time
import logging
import argparse
import re
import gc
import json
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

# Add the directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))

# Import Qwen25 model and generation functions - change to use local imports
from model import load_qwen25_model, create_qwen25_model
from generate import generate_text

# Import safetensors for weight loading
try:
    from safetensors import safe_open
except ImportError:
    logging.error("safetensors package is required. Install with: pip install safetensors")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qwen25_inference")

# Parameter name mapping helpers from run_memory_efficient.py
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

def load_model_parameters(weights_dir, dtype=jnp.bfloat16):
    """
    Load model parameters from safetensors files in the given directory.
    Implements the successful loading strategy from run_memory_efficient.py.
    
    Args:
        weights_dir: Directory containing model weights
        dtype: Data type for model parameters
        
    Returns:
        tuple of (config, params) where config is the model configuration and
        params is the loaded and formatted parameters
    """
    try:
        # Load configuration
        config_path = os.path.join(weights_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration with {config.get('num_hidden_layers', 'unknown')} layers")
        
        # Enable memory optimizations
        config["use_memory_efficient_attention"] = True

        # Initialize parameters
        params = {"params": {}}
        
        # Process weights by streaming parameters from each file
        safetensors_files = sorted([f for f in os.listdir(weights_dir) if f.endswith(".safetensors")])
        
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {weights_dir}")
        
        logger.info(f"Loading parameters from {len(safetensors_files)} files")
        
        # Track memory for each file
        file_memories = []
        
        # Load each file and merge parameters
        t0_total = time.time()
        for i, filename in enumerate(safetensors_files):
            weight_path = os.path.join(weights_dir, filename)
            logger.info(f"Processing file {i+1}/{len(safetensors_files)}: {filename}")
            
            # Process one file at a time
            t0 = time.time()
            file_params = process_safetensors_file(weight_path, dtype=dtype)
            
            # Merge parameters
            params = merge_param_dicts(params, file_params)
            
            # Log progress
            file_time = time.time() - t0
            logger.info(f"Processed {filename} in {file_time:.2f} seconds")
            
            # Force garbage collection
            del file_params
            gc.collect()
        
        total_load_time = time.time() - t0_total
        logger.info(f"Total parameter loading time: {total_load_time:.2f} seconds")
        
        return config, params
    
    except Exception as e:
        logger.error(f"Error loading model parameters: {e}")
        raise

def print_stream(text):
    """Print text with streaming effect."""
    print(text, end="", flush=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Qwen25 JAX model")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to directory containing Qwen25 model weights"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Write a short story about a robot learning to paint:",
        help="Text prompt for generation"
    )
    
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=200,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (lower = more deterministic)"
    )
    
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Nucleus sampling probability threshold"
    )
    
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--mesh_shape", 
        type=str, 
        default=None,
        help="Device mesh shape for tensor parallelism (e.g., '1,8')"
    )
    
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model parameters"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--no_stream", 
        action="store_true",
        help="Disable streaming output"
    )
    
    return parser.parse_args()

def main():
    """Run Qwen25 inference."""
    # Parse command line arguments
    args = parse_args()
    
    # Set debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["QWEN_DEBUG"] = "1"
    
    # Print JAX devices info
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"Number of devices: {jax.device_count()}")
    
    # Set data type
    if args.dtype == "float32":
        dtype = jnp.float32
    elif args.dtype == "float16":
        dtype = jnp.float16
    else:
        dtype = jnp.bfloat16
    
    # Configure device mesh for tensor parallelism
    if args.mesh_shape:
        mesh_shape = tuple(map(int, args.mesh_shape.split(",")))
    else:
        # Use simple 1D mesh by default (all devices in a row)
        # Ensure we're creating a mesh shape that matches our device array
        device_count = jax.device_count()
        if device_count == 1:
            # For single device, use a simple 1D mesh
            mesh_shape = (1,)
        else:
            # For multiple devices, create a 2D mesh
            mesh_shape = (1, device_count)
    
    logger.info(f"Using mesh shape: {mesh_shape}")
    
    # Load tokenizer - try to use local files from the model path first
    try:
        from transformers import AutoTokenizer
        
        # Check if tokenizer files exist in the model path
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
        has_tokenizer_files = any(os.path.exists(os.path.join(args.model_path, f)) for f in tokenizer_files)
        
        if has_tokenizer_files:
            logger.info(f"Loading tokenizer from model path: {args.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            logger.info("Loaded tokenizer from model path")
        else:
            # Try from cache
            try:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", local_files_only=True)
                logger.info("Loaded tokenizer from local cache")
            except Exception as e:
                logger.info(f"Error loading tokenizer locally: {e}")
                logger.info("Downloading tokenizer from Hugging Face Hub")
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    # Create mesh for device parallelism
    devices = jax.devices()
    if len(mesh_shape) == 1:
        # For 1D mesh, use a single axis name
        mesh = Mesh(np.array(devices), ("data",))
    else:
        # For 2D mesh, reshape devices to 2D and use both axis names
        devices_reshaped = np.array(devices).reshape(mesh_shape)
        mesh = Mesh(devices_reshaped, ("data", "model"))
    
    # Load model and parameters
    start_time = time.time()
    try:
        logger.info(f"Loading Qwen25 model from {args.model_path}")
        
        # Use our parameter loading function that matches run_memory_efficient.py's approach
        config, params = load_model_parameters(args.model_path, dtype=dtype)
        
        # Create model with the loaded configuration
        logger.info("Creating model with loaded configuration")
        model = create_qwen25_model(config, dtype=dtype)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        # Print model configuration
        logger.info(f"Model configuration: hidden_size={config['hidden_size']}, "
                   f"num_heads={config['num_attention_heads']}, "
                   f"num_layers={config['num_hidden_layers']}")
                   
        # Log parameter structure for debugging
        if args.debug:
            param_keys = list(params.keys())
            logger.debug(f"Top-level parameter keys: {param_keys}")
            if "params" in params:
                params_keys = list(params["params"].keys())
                logger.debug(f"Model params keys: {params_keys}")
                
                # Log a few layer keys to verify structure
                for layer_key in [k for k in params["params"].keys() if k.startswith("layers_")][:2]:
                    layer_keys = list(params["params"][layer_key].keys())
                    logger.debug(f"Layer {layer_key} keys: {layer_keys}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        logger.error("Falling back to creating model without loading weights (for debugging)")
        
        # Create empty configuration for debugging
        config = {
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "vocab_size": 151936,
            "use_memory_efficient_attention": True,
        }
        
        # Create model with empty params
        model = create_qwen25_model(config, dtype=dtype)
        params = None
    
    # Print prompt
    print(f"\nPrompt: {args.prompt}\n")
    print("Generated Response:")
    
    # Run inference with streaming
    try:
        stream_handler = None if args.no_stream else print_stream
        
        full_response = generate_text(
            model=model, 
            tokenizer=tokenizer, 
            prompt_tokens=args.prompt,
            params=params,
            mesh=mesh,
            max_decode_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            streamer=stream_handler,
            debug=args.debug
        )
        
        # Print the full response if streaming was disabled
        if args.no_stream:
            print(full_response)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        
        if params is None:
            logger.error("Generation failed because model parameters were not loaded.")
            logger.error("Make sure the model weights are in the correct format (safetensors).")
        if args.debug:
            # Log additional information in debug mode
            logger.debug(f"Model: {type(model)}")
            logger.debug(f"Tokenizer: {type(tokenizer)}")
            if params is not None:
                param_keys = list(params.keys() if hasattr(params, 'keys') else ['<Nested Structure>'])
                logger.debug(f"Parameters keys: {param_keys}")
    
    print("\n\nGeneration completed!")

if __name__ == "__main__":
    main() 