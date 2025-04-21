#!/usr/bin/env python3
"""
Test script to verify model initialization and basic forward pass.
"""

import os
import logging
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_init_test")

def test_model_init(model_path, mesh_shape=(1, 8)):
    """
    Test model initialization and a simple forward pass.
    
    Args:
        model_path: Path to model weights
        mesh_shape: Shape of the device mesh (batch, model)
    
    Returns:
        True if initialization succeeds, False otherwise
    """
    try:
        # Import tensor parallel model
        from tensor_parallel import (
            TensorParallelQwen2ForCausalLM, 
            create_device_mesh
        )
        
        # Load model configuration
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Use default configuration
            config = {
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "intermediate_size": 14336,
                "num_key_value_heads": 4,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "rope_theta": 10000.0,
            }
        
        # Print key model configuration
        logger.info(f"Model configuration:")
        logger.info(f"  hidden_size: {config['hidden_size']}")
        logger.info(f"  num_hidden_layers: {config['num_hidden_layers']}")
        logger.info(f"  num_attention_heads: {config['num_attention_heads']}")
        logger.info(f"  num_key_value_heads: {config.get('num_key_value_heads', config['num_attention_heads'])}")
        
        # Create device mesh
        logger.info(f"Creating device mesh with shape {mesh_shape}...")
        mesh = create_device_mesh(mesh_shape)
        logger.info(f"Device mesh created successfully")
        
        # Create model
        logger.info("Initializing model...")
        model = TensorParallelQwen2ForCausalLM(
            config=config,
            mesh=mesh,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16
        )
        
        # Load parameters
        logger.info(f"Loading parameters from {model_path}...")
        start_time = time.time()
        params = model.params_from_checkpoint(model_path)
        loading_time = time.time() - start_time
        logger.info(f"Parameters loaded in {loading_time:.2f} seconds")
        
        # Create sample input
        logger.info("Creating sample input and running a forward pass...")
        input_ids = jnp.ones((1, 8), dtype=jnp.int32)
        
        # Run a simple forward pass
        with mesh:
            start_time = time.time()
            outputs = model.apply(params, input_ids=input_ids)
            jax.block_until_ready(outputs)
            inference_time = time.time() - start_time
        
        # Check outputs
        if 'logits' in outputs:
            logits = outputs['logits']
            logger.info(f"Forward pass successful! Output logits shape: {logits.shape}")
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logger.info(f"Forward pass successful! Output shape: {logits.shape}")
            
        logger.info(f"Inference time for 8 tokens: {inference_time*1000:.1f} ms")
        
        return True
    except Exception as e:
        logger.error(f"Error in model initialization or forward pass: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test model initialization")
    parser.add_argument("model_path", help="Path to model weights")
    parser.add_argument("--mesh-rows", type=int, default=1, help="Number of rows in device mesh")
    parser.add_argument("--mesh-cols", type=int, default=8, help="Number of columns in device mesh")
    args = parser.parse_args()
    
    # Test model initialization
    mesh_shape = (args.mesh_rows, args.mesh_cols)
    success = test_model_init(args.model_path, mesh_shape)
    
    if success:
        logger.info("🎉 Model initialization and forward pass test completed successfully!")
        return 0
    else:
        logger.error("❌ Model initialization or forward pass test failed")
        return 1

if __name__ == "__main__":
    exit(main()) 