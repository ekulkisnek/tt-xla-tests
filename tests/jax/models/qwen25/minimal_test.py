#!/usr/bin/env python3
"""
Minimal test script for the Qwen25 model.
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

def main():
    """Test a simple model initialization and forward pass."""
    print("JAX info:")
    print(f"- JAX version: {jax.__version__}")
    print(f"- Device count: {jax.device_count()}")
    print(f"- Devices: {jax.devices()[0].device_kind}")
    
    # Import the Qwen25 model
    from model import create_qwen25_model
    
    # Create a minimal configuration
    config = {
        "hidden_size": 256,        # Small for testing
        "num_attention_heads": 8,  # Small for testing
        "num_hidden_layers": 2,    # Small for testing
        "num_key_value_heads": 2,  # Small for testing
        "vocab_size": 1000,        # Small for testing
        "use_memory_efficient_attention": True,
    }
    
    # Test parameters
    dtype = jnp.float32  # Use float32 for testing
    
    print("\nCreating minimal model...")
    start_time = time.time()
    
    try:
        # Create a minimal model for testing
        model = create_qwen25_model(config, dtype=dtype)
        
        print(f"Model created in {time.time() - start_time:.2f} seconds")
        
        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        input_shape = (1, 5)  # Batch size 1, sequence length 5
        params = model.init(rng, jnp.ones(input_shape, dtype=jnp.int32))
        
        print(f"Parameters initialized")
        
        # Create a small test input
        input_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        
        print("\nRunning forward pass...")
        forward_start = time.time()
        
        # Run a forward pass
        outputs = model.apply(
            params,
            input_ids=input_ids,
            return_dict=True
        )
        
        # Print results
        print(f"Forward pass completed in {time.time() - forward_start:.2f} seconds")
        print(f"Output logits shape: {outputs['logits'].shape}")
        
        print("\nTest completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 