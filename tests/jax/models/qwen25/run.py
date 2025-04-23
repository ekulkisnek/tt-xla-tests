#!/usr/bin/env python3
"""Run the Qwen model from the correct location."""

import os
import sys
import time
import logging
import numpy as np
import jax
import jax.numpy as jnp

# Make sure tt_xla is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Now import the model
from tt_xla.jax.models.qwen25.model import Qwen25ForCausalLM, create_qwen25_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX version: {jax.__version__}")
    
    try:
        # Create a basic config for testing
        config = {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "layer_norm_epsilon": 1e-5,
        }
        
        # Create model
        logger.info("Creating model...")
        model = create_qwen25_model(config, dtype=jnp.float32)
        
        # Create input
        input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        
        # Initialize parameters
        logger.info("Initializing parameters...")
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, input_ids)
        
        # Run model
        logger.info("Running model...")
        start_time = time.time()
        outputs = model.apply(params, input_ids)
        
        # Check output
        logits = outputs["logits"]
        logger.info(f"Model ran in {time.time() - start_time:.2f} seconds")
        logger.info(f"Output shape: {logits.shape}")
        logger.info("Success!")
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 