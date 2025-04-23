#!/usr/bin/env python3
"""Simple test script for the Qwen2.5-7B model with real weights."""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import the model
from tt_xla.jax.models.qwen25.model import Qwen25ForCausalLM, create_qwen25_model, load_qwen25_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load model configuration from JSON file."""
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def convert_params_to_flax_format(raw_params):
    """Convert raw parameter dictionary to proper Flax nested dictionary structure."""
    flax_params = {"params": {}}
    
    # Map of PyTorch parameter names to Flax parameter paths
    name_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    # Handle layer parameters
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    
    import re
    
    for name, param in raw_params.items():
        # Handle embedding and final norm
        if name in name_mapping:
            module_name, param_name = name_mapping[name]
            if module_name not in flax_params["params"]:
                flax_params["params"][module_name] = {}
            flax_params["params"][module_name][param_name] = param
            continue
            
        # Handle layer norms
        layer_norm_match = re.match(layer_norm_pattern, name)
        if layer_norm_match:
            layer_idx = int(layer_norm_match.group(1))
            norm_type = layer_norm_match.group(2)
            
            # Get layer name
            layer_name = f"layers_{layer_idx}"
            if layer_name not in flax_params["params"]:
                flax_params["params"][layer_name] = {}
                
            # Get norm name
            norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
            if norm_name not in flax_params["params"][layer_name]:
                flax_params["params"][layer_name][norm_name] = {}
                
            # Set norm parameter - IMPORTANT: Flax expects 'scale', not 'weight'
            flax_params["params"][layer_name][norm_name]["scale"] = param
            continue
            
        # Handle attention parameters
        attn_match = re.match(attention_pattern, name)
        if attn_match:
            layer_idx = int(attn_match.group(1))
            proj_type = attn_match.group(2)
            param_type = attn_match.group(3)
            
            # Get layer name
            layer_name = f"layers_{layer_idx}"
            if layer_name not in flax_params["params"]:
                flax_params["params"][layer_name] = {}
                
            # Ensure self_attn exists
            if "self_attn" not in flax_params["params"][layer_name]:
                flax_params["params"][layer_name]["self_attn"] = {}
                
            # Set attention parameter
            proj_name = f"{proj_type}_proj"
            if proj_name not in flax_params["params"][layer_name]["self_attn"]:
                flax_params["params"][layer_name]["self_attn"][proj_name] = {}
                
            param_name = "kernel" if param_type == "weight" else "bias"
            flax_params["params"][layer_name]["self_attn"][proj_name][param_name] = param
            continue
            
        # Handle MLP parameters
        mlp_match = re.match(mlp_pattern, name)
        if mlp_match:
            layer_idx = int(mlp_match.group(1))
            proj_type = mlp_match.group(2)
            
            # Get layer name
            layer_name = f"layers_{layer_idx}"
            if layer_name not in flax_params["params"]:
                flax_params["params"][layer_name] = {}
                
            # Ensure mlp exists
            if "mlp" not in flax_params["params"][layer_name]:
                flax_params["params"][layer_name]["mlp"] = {}
                
            # Set mlp parameter
            proj_name = f"{proj_type}_proj"
            if proj_name not in flax_params["params"][layer_name]["mlp"]:
                flax_params["params"][layer_name]["mlp"][proj_name] = {}
                
            flax_params["params"][layer_name]["mlp"][proj_name]["kernel"] = param
            
    return flax_params


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-7B model with real weights")
    parser.add_argument(
        "--weights_dir", 
        type=str, 
        default="/root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights",
        help="Directory containing Qwen2.5 weights"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float32", 
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model parameters"
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
    dtype = dtype_map.get(args.dtype, jnp.float32)
    
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX version: {jax.__version__}")
    
    try:
        # Load the real config if available
        config_path = os.path.join(args.weights_dir, "config.json")
        if os.path.exists(config_path):
            config = load_config(config_path)
            logger.info(f"Loaded configuration with {config.get('num_hidden_layers', 'unknown')} layers")
        else:
            # Fallback to a dummy config
            logger.warning(f"No config.json found in {args.weights_dir}, using dummy config")
            config = {
                "vocab_size": 32000,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "layer_norm_epsilon": 1e-5,
            }
        
        # Create basic model
        logger.info("Creating base model...")
        base_model = create_qwen25_model(config, dtype=dtype)
        
        # Load weights if available
        logger.info(f"Loading weights from {args.weights_dir}...")
        raw_params = {}
        prev_count = 0
        
        try:
            # Try to import safetensors if available
            from safetensors import safe_open
            
            # Check for model files
            for filename in os.listdir(args.weights_dir):
                if filename.endswith(".safetensors"):
                    weight_path = os.path.join(args.weights_dir, filename)
                    logger.info(f"Loading weights from {weight_path}")
                    
                    with safe_open(weight_path, framework="flax") as f:
                        for key in f.keys():
                            logger.info(f"Loading tensor: {key}")
                            raw_params[key] = f.get_tensor(key)
                            
                    logger.info(f"Loaded {len(raw_params) - prev_count} parameters from {filename}")
                    prev_count = len(raw_params)
                    
            if not raw_params:
                logger.warning("No safetensors files found, initializing random weights")
                # Initialize with random parameters
                input_ids = jnp.ones((1, 16), dtype=jnp.int32)
                rng = jax.random.PRNGKey(0)
                params = base_model.init(rng, input_ids)
            else:
                # Convert parameters to Flax format
                logger.info(f"Converting {len(raw_params)} raw parameters to Flax format...")
                params = convert_params_to_flax_format(raw_params)
                logger.info("Parameter conversion complete")
                
        except ImportError:
            logger.warning("safetensors not installed, initializing random weights")
            # Initialize with random parameters
            input_ids = jnp.ones((1, 16), dtype=jnp.int32)
            rng = jax.random.PRNGKey(0)
            params = base_model.init(rng, input_ids)
        
        # Create a dummy input
        input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        
        # Initialize parameters to understand the expected structure
        logger.info("Initializing a set of dummy parameters to understand the structure...")
        rng = jax.random.PRNGKey(0)
        init_params = base_model.init(rng, input_ids)
        
        # Print sample of initialized params to understand structure
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("==== Flax initialized parameter structure ====")
        
        # Print params structure (top level only)
        pp.pprint({k: type(v) for k, v in init_params.items()})
        
        # Print sample of first layer params
        if 'params' in init_params:
            logger.info("Params['params'] keys:")
            pp.pprint(list(init_params['params'].keys()))
            
            # Print structure of embed_tokens
            if 'embed_tokens' in init_params['params']:
                logger.info("Params['params']['embed_tokens'] keys:")
                pp.pprint(list(init_params['params']['embed_tokens'].keys()))
                
            # Print structure of the first layer
            if 'layers_0' in init_params['params']:
                logger.info(f"Params['params']['layers_0'] keys:")
                pp.pprint(list(init_params['params']['layers_0'].keys()))
                
                if 'input_layernorm' in init_params['params']['layers_0']:
                    logger.info(f"Params['params']['layers_0']['input_layernorm'] keys:")
                    pp.pprint(list(init_params['params']['layers_0']['input_layernorm'].keys()))
                
                # Show actual shape of the expected parameter
                if 'input_layernorm' in init_params['params']['layers_0'] and 'scale' in init_params['params']['layers_0']['input_layernorm']:
                    logger.info(f"Scale parameter shape: {init_params['params']['layers_0']['input_layernorm']['scale'].shape}")
        
        # Now use parameters from the previous weight loading
        logger.info("Now using loaded parameters...")
        
        # Run the model
        logger.info("Running forward pass...")
        start_time = time.time()
        outputs = base_model.apply(params, input_ids)
        logits = outputs["logits"]
        
        logger.info(f"Model ran in {time.time() - start_time:.2f} seconds")
        logger.info(f"Input shape: {input_ids.shape}")
        logger.info(f"Output shape: {logits.shape}")
        logger.info(f"First logit values: {logits[0, 0, :5]}")
        logger.info("Success!")
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 