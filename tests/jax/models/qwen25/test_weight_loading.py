#!/usr/bin/env python3
"""
Test script to verify the weight loading fixes.

python test_weight_loading.py /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights
"""

import os
import logging
import json
import jax
import numpy as np
from jax.sharding import Mesh
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("weight_loading_test")

def test_loading(model_path, test_direct=True, test_standard=True):
    """
    Test the weight loading functions.
    
    Args:
        model_path: Path to model weights
        test_direct: Whether to test direct safetensors loading
        test_standard: Whether to test standard loading path
    
    Returns:
        True if any loading method succeeds, False otherwise
    """
    # Import after fixing the code
    from weight_loading import (
        load_safetensors_only, 
        load_qwen_weights
    )
    from weight_diagnostics import (
        analyze_param_structure,
        scan_checkpoint_files
    )
    
    # First scan the checkpoint files
    logger.info(f"Scanning checkpoint files in {model_path}")
    files_info = scan_checkpoint_files(model_path)
    
    # Check if we have safetensors files
    if not files_info["safetensors"]:
        logger.error(f"No safetensors files found in {model_path}")
        return False
    
    # Set up minimal config for testing
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
    
    # Create a simple mesh for testing - a single device is fine for verification
    devices = jax.devices()
    mesh = Mesh(np.array(devices[:1]).reshape(1, 1), ('batch', 'model'))
    
    success = False
    
    if test_direct:
        try:
            logger.info("Testing direct safetensors loading...")
            params = load_safetensors_only(
                model_path=model_path,
                config=config,
                mesh=mesh
            )
            
            # Analyze the loaded parameters
            analysis = analyze_param_structure(params)
            logger.info(f"Direct loading analysis: {json.dumps(analysis, indent=2, default=str)}")
            
            if analysis.get("status") == "ok" or analysis.get("critical_keys_present", False):
                logger.info("✅ Direct loading succeeded!")
                success = True
            else:
                logger.warning("⚠️ Direct loading completed but may have issues")
        except Exception as e:
            logger.error(f"❌ Direct loading failed: {str(e)}")
    
    if test_standard:
        try:
            # Create a dummy model class to test with load_qwen_weights
            class DummyModel:
                def __init__(self):
                    self.config = config
            
            logger.info("Testing standard loading path...")
            params = load_qwen_weights(
                model_path=model_path,
                model=DummyModel(),
                config=config,
                mesh=mesh,
                debug=True
            )
            
            # Analyze the loaded parameters
            analysis = analyze_param_structure(params)
            logger.info(f"Standard loading analysis: {json.dumps(analysis, indent=2, default=str)}")
            
            if analysis.get("status") == "ok" or analysis.get("critical_keys_present", False):
                logger.info("✅ Standard loading succeeded!")
                success = True
            else:
                logger.warning("⚠️ Standard loading completed but may have issues")
        except Exception as e:
            logger.error(f"❌ Standard loading failed: {str(e)}")
    
    return success

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test weight loading")
    parser.add_argument("model_path", help="Path to model weights")
    parser.add_argument("--direct-only", action="store_true", help="Only test direct loading")
    parser.add_argument("--standard-only", action="store_true", help="Only test standard loading")
    args = parser.parse_args()
    
    # Determine which loading methods to test
    test_direct = not args.standard_only
    test_standard = not args.direct_only
    
    if test_direct and test_standard:
        logger.info("Testing both direct and standard loading paths")
    elif test_direct:
        logger.info("Testing only direct loading path")
    else:
        logger.info("Testing only standard loading path")
    
    success = test_loading(
        args.model_path,
        test_direct=test_direct,
        test_standard=test_standard
    )
    
    if success:
        logger.info("🎉 Weight loading test completed successfully!")
        return 0
    else:
        logger.error("❌ All weight loading methods failed")
        return 1

if __name__ == "__main__":
    exit(main()) 