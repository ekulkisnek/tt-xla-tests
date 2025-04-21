#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Verification script to test the parameter structure mapping and fixes.

Usage:
cd to qwen25
source venv/bin/activate
export XLA_FLAGS="--xla_force_host_platform_device_count=8"
python verify_parameter_fixes.py --model_path /path/to/qwen25-weights
"""

import os
# Force JAX to use simulated devices for tensor parallelism testing
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import argparse
import time
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import logging
from functools import partial
import gc
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("qwen25-param-fixes")

# Import model components
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
    map_parameter_paths
)
from config import load_qwen_config, get_qwen2_7b_config, get_small_config
from weight_loading import load_qwen_weights, init_model_from_weights
from weight_diagnostics import (
    diagnose_parameter_structure_mismatch,
    create_parameter_structure_report,
    fix_parameter_structure
)

def test_parameter_fixes(
    model_path: str,
    mesh_shape: Tuple[int, int] = (1, 8),
    use_small_model: bool = True
):
    """
    Test the parameter mapping and structure fixes.
    
    Args:
        model_path: Path to the model weights
        mesh_shape: Device mesh shape (batch, model)
        use_small_model: Whether to use a small model for testing
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Testing parameter fixes with mesh shape {mesh_shape}")
    
    try:
        # Step 1: Create device mesh
        logger.info("\n[1/5] Creating device mesh...")
        mesh = create_device_mesh(mesh_shape)
        logger.info(f"✅ Created mesh with shape {mesh_shape}")
        
        # Step 2: Load config
        logger.info("\n[2/5] Loading model configuration...")
        if use_small_model:
            config = get_small_config(hidden_size=128, num_layers=2)
            logger.info("Using small model config for testing")
        else:
            if os.path.exists(os.path.join(model_path, "config.json")):
                config = load_qwen_config(model_path)
            else:
                config = get_qwen2_7b_config()
        logger.info(f"✅ Loaded config with hidden_size={config['hidden_size']}, layers={config['num_hidden_layers']}")
        
        # Step 3: Initialize model
        logger.info("\n[3/5] Initializing model...")
        model = TensorParallelQwen2ForCausalLM(
            config=config,
            mesh=mesh,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16
        )
        logger.info("✅ Model initialized")
        
        # Step 4: Load parameters
        logger.info("\n[4/5] Loading parameters...")
        
        # Initialize with random inputs
        batch_size = 1
        seq_length = 32
        input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        empty_params = model.init(jax.random.PRNGKey(0), input_ids)
        
        # Use model's built-in loading if available, otherwise fallback
        try:
            # Load parameters
            if hasattr(model, 'load_params_from_checkpoint') and not use_small_model:
                logger.info("Using model's built-in parameter loading")
                params = model.load_params_from_checkpoint(model_path)
            elif use_small_model:
                logger.info("Using random parameters for small model")
                params = empty_params
            else:
                logger.info("Using weight_loading utility")
                params = load_qwen_weights(model_path, model, config, mesh)
                
            # Create diagnostic report
            logger.info("\nRunning parameter structure diagnostics...")
            structure_report = create_parameter_structure_report(params)
            
            # Log diagnostic info
            logger.info(f"Parameter structure report:")
            logger.info(f"  Total parameters: {structure_report['total_parameters']}")
            logger.info(f"  Parameter categories: {structure_report['parameter_categories']}")
            
            # Display recommendations
            if structure_report["recommendations"]:
                logger.info(f"\nRecommendations:")
                for i, rec in enumerate(structure_report["recommendations"]):
                    logger.info(f"  {i+1}. {rec}")
            
            # Apply parameter mapping
            logger.info("\nApplying parameter mapping...")
            mapped_params = map_parameter_paths(params)
            logger.info(f"✅ Parameter mapping applied")
            
            # Apply parameter fixes
            logger.info("\nApplying parameter structure fixes...")
            fixed_params = fix_parameter_structure(mapped_params)
            logger.info(f"✅ Parameter fixes applied")
            
            # Step 5: Test forward pass
            logger.info("\n[5/5] Testing forward pass with fixed parameters...")
            
            # Apply model with fixed parameters
            with mesh:
                # Run forward pass
                outputs = model.apply(
                    fixed_params, 
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True
                )
                
                # Check output shape
                logits_shape = outputs.logits.shape
                logger.info(f"✅ Forward pass successful!")
                logger.info(f"   Output logits shape: {logits_shape}")
                
                # Try to run a second pass to check stability
                logger.info("\nRunning second forward pass to verify stability...")
                outputs2 = model.apply(
                    fixed_params, 
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True
                )
                logger.info(f"✅ Second forward pass successful!")
                
                # Check if outputs are consistent
                logits_similar = jnp.allclose(
                    outputs.logits, 
                    outputs2.logits, 
                    rtol=1e-2, 
                    atol=1e-2
                )
                logger.info(f"   Outputs consistent: {logits_similar}")
                
            logger.info("\n==========================")
            logger.info("✅ Parameter fixes verified successfully!")
            logger.info("==========================")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during parameter loading/testing: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test parameter structure fixes for Qwen 2.5")
    parser.add_argument("--model_path", type=str, help="Path to the model weights")
    parser.add_argument("--mesh_shape", type=str, default="1x8", help="Mesh shape (e.g., '1x8', '2x4')")
    parser.add_argument("--small_model", action="store_true", help="Use a small model for testing")
    args = parser.parse_args()
    
    # Parse mesh shape
    mesh_shape = tuple(map(int, args.mesh_shape.split("x")))
    
    # Run the test
    success = test_parameter_fixes(
        model_path=args.model_path,
        mesh_shape=mesh_shape,
        use_small_model=args.small_model
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 