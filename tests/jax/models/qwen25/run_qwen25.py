#!/usr/bin/env python3
"""Simple script to run the Qwen2.5-7B model."""

import os
import sys
import time
import argparse
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import the model
from tt_xla.jax.models.qwen25.model import (
    Qwen25ForCausalLM,
    create_qwen25_model,
    load_qwen25_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-7B model")
    parser.add_argument(
        "--weights_dir", 
        type=str, 
        required=True, 
        help="Directory containing Qwen2.5 weights"
    )
    parser.add_argument(
        "--mesh_shape", 
        type=str, 
        default="1,1", 
        help="Device mesh shape (e.g., '1,8', '2,4')"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16", 
        choices=["float32", "float16", "bfloat16"], 
        help="Data type for model parameters"
    )
    
    args = parser.parse_args()
    
    # Map dtype string to jnp.dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, jnp.bfloat16)
    
    # Create mesh shape from string
    mesh_shape = tuple(map(int, args.mesh_shape.split(",")))
    
    # Log JAX device and build information
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX runtime: {jax.lib.xla_bridge.get_backend().platform}")
    logger.info(f"Using mesh shape: {mesh_shape}")
    
    try:
        # Create a dummy config for testing
        config = {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "layer_norm_epsilon": 1e-5,
        }
        
        # Create model instance
        start_time = time.time()
        model = create_qwen25_model(config, dtype=dtype)
        logger.info(f"Model created in {time.time() - start_time:.2f} seconds")
        
        # Create a dummy input
        input_ids = jnp.ones((1, 10), dtype=jnp.int32)
        
        # Initialize parameters (normally we would load from checkpoint)
        logger.info("Initializing model parameters...")
        rng = jax.random.PRNGKey(0)
        try:
            params = model.init(rng, input_ids)
            logger.info("Parameters initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing parameters: {e}")
            return 1
        
        # Run a forward pass
        logger.info("Running forward pass...")
        try:
            start_time = time.time()
            outputs = model.apply(params, input_ids)
            logger.info(f"Forward pass completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Output logits shape: {outputs['logits'].shape}")
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            return 1
        
        logger.info("Model ran successfully!")
        
    except Exception as e:
        logger.error(f"Error running model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 