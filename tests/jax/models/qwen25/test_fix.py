#!/usr/bin/env python3
"""
Test script to verify our fix for the Qwen25 model works correctly.
"""
import os
import sys
import time
import logging
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25_test")

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "qwen25-weights")

def main():
    """Test the model loading and a simple forward pass."""
    print("JAX info:")
    print(f"- JAX version: {jax.__version__}")
    print(f"- Device count: {jax.device_count()}")
    print(f"- Devices: {jax.devices()[0].device_kind}")
    
    # Import the Qwen25 model after making sure paths are set up
    from model import load_qwen25_model
    
    # Test parameters
    dtype = jnp.bfloat16
    mesh_shape = (1, jax.device_count())
    
    print("\nLoading model...")
    start_time = time.time()
    
    try:
        # Load the model
        model, config, params, mesh = load_qwen25_model(model_path, mesh_shape, dtype=dtype)
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        print(f"\nModel configuration:")
        print(f"- Hidden size: {config['hidden_size']}")
        print(f"- Attention heads: {config['num_attention_heads']}")
        print(f"- KV heads: {config.get('num_key_value_heads', config['num_attention_heads'])}")
        print(f"- Layers: {config['num_hidden_layers']}")
        print(f"- Vocab size: {config['vocab_size']}")
        
        # Create a small test input
        input_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        
        print("\nRunning forward pass...")
        forward_start = time.time()
        
        # Run a forward pass
        outputs = model.apply(
            {"params": params},
            input_ids=input_ids,
            return_dict=True
        )
        
        # Print results
        print(f"Forward pass completed in {time.time() - forward_start:.2f} seconds")
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"First few logit values: {outputs['logits'][0, 0, :5]}")
        
        print("\nTest completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 