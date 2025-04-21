"""
Weight loading diagnostics and verification utilities for Qwen25 model.

This module provides functions to:
1. Diagnose weight loading issues
2. Verify loaded weights against expected structure
3. Log information about loaded weights for debugging
"""

import os
import glob
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

logger = logging.getLogger(__name__)

def scan_checkpoint_files(model_path: str) -> Dict[str, List[str]]:
    """
    Scan a checkpoint directory for different types of weight files.
    
    Args:
        model_path: Path to the checkpoint directory
        
    Returns:
        Dictionary with file types as keys and lists of files as values
    """
    results = {
        "safetensors": [],
        "bin": [],
        "pt": [],
        "index": [],
        "json": [],
        "other": []
    }
    
    if not os.path.exists(model_path):
        logger.error(f"Checkpoint path does not exist: {model_path}")
        return results
    
    # Check if it's a file or directory
    if os.path.isfile(model_path):
        # Single file
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".safetensors":
            results["safetensors"].append(model_path)
        elif ext == ".bin":
            results["bin"].append(model_path)
        elif ext == ".pt":
            results["pt"].append(model_path)
        elif ext == ".json":
            results["json"].append(model_path)
        else:
            results["other"].append(model_path)
        return results
    
    # Directory - collect all weight files
    for ext, key in [
        ("*.safetensors", "safetensors"),
        ("*.bin", "bin"),
        ("*.pt", "pt"),
        ("*.json", "json"),
        ("*index*", "index")
    ]:
        files = glob.glob(os.path.join(model_path, ext))
        files.sort()  # Sort for consistent reporting
        results[key] = files
    
    # Count files
    total_files = sum(len(files) for files in results.values())
    
    # Log summary
    logger.info(f"Found {total_files} files in {model_path}:")
    for key, files in results.items():
        if files:
            logger.info(f"  - {key}: {len(files)} files")
    
    return results

def analyze_param_structure(params: Dict) -> Dict[str, Any]:
    """
    Analyze parameter structure and return information about it.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Dictionary with analysis results
    """
    if params is None:
        return {"status": "error", "error": "Parameters are None"}
    
    # Check for empty params
    if not params:
        return {"status": "error", "error": "Parameters dictionary is empty"}
    
    results = {
        "status": "ok",
        "has_params_key": "params" in params,
        "top_level_keys": list(params.keys())[:10],  # First 10 keys
        "key_counts": {},
        "parameter_info": {},
        "critical_keys_present": False
    }
    
    # Flatten and check parameter structure
    flat_params = flatten_dict(params)
    
    # Get key structure patterns
    key_patterns = {}
    for key_tuple in flat_params.keys():
        # Get the first two components of each key path
        if len(key_tuple) >= 2:
            prefix = key_tuple[0:2]
            prefix_str = "/".join(prefix)
            key_patterns[prefix_str] = key_patterns.get(prefix_str, 0) + 1
    
    results["key_patterns"] = key_patterns
    
    # Check for embed_tokens.embedding
    has_embedding = False
    embed_key = None
    for key_tuple in flat_params.keys():
        key_str = ".".join([str(k) for k in key_tuple])
        if "embed_tokens.embedding" in key_str:
            has_embedding = True
            embed_key = key_str
            break
    
    results["has_embedding"] = has_embedding
    if has_embedding:
        results["embedding_key"] = embed_key
    
    # Check for critical keys
    critical_keys = [
        "transformer", 
        "model", 
        "embed_tokens", 
        "lm_head"
    ]
    
    # Count first-level keys in params or params['params']
    if results["has_params_key"]:
        params_content = params["params"]
        results["params_keys"] = list(params_content.keys())[:10]
        
        # Check for critical keys in params['params']
        for key in critical_keys:
            if key in params_content:
                results["critical_keys_present"] = True
                break
    else:
        # Check for critical keys at top level
        for key in critical_keys:
            if key in params:
                results["critical_keys_present"] = True
                break
    
    # Get parameter shapes and types
    shapes = {}
    for key, value in flat_params.items():
        if hasattr(value, "shape"):
            key_str = "/".join([str(k) for k in key])
            shape_info = {
                "shape": str(value.shape),
                "dtype": str(value.dtype),
                "size_mb": round(np.prod(value.shape) * value.dtype.itemsize / (1024 * 1024), 2)
            }
            shapes[key_str] = shape_info
    
    # Only include 10 largest parameters by size
    largest_params = sorted(
        [(k, v) for k, v in shapes.items()], 
        key=lambda x: float(x[1]["size_mb"]), 
        reverse=True
    )[:10]
    
    results["largest_params"] = {k: v for k, v in largest_params}
    
    # Total parameter count and size
    total_params = sum(np.prod(value.shape) for value in flat_params.values() if hasattr(value, "shape"))
    total_size_mb = sum(
        np.prod(value.shape) * value.dtype.itemsize / (1024 * 1024) 
        for value in flat_params.values() 
        if hasattr(value, "shape")
    )
    
    results["total_params"] = total_params
    results["total_size_mb"] = round(total_size_mb, 2)
    
    return results

def verify_loaded_weights(params: Dict, config=None) -> Dict[str, Any]:
    """
    Verify that loaded weights match expected structure.
    
    Args:
        params: Parameter dictionary
        config: Model configuration (optional)
        
    Returns:
        Dictionary with verification results
    """
    results = {
        "status": "ok",
        "issues": [],
        "checks_passed": 0,
        "checks_failed": 0
    }
    
    # Get parameter analysis
    analysis = analyze_param_structure(params)
    
    # Check for any errors in analysis
    if analysis.get("status") == "error":
        results["status"] = "error"
        results["issues"].append(analysis.get("error", "Unknown error in parameter analysis"))
        results["checks_failed"] += 1
        return results
    
    # Check 1: Must have critical keys
    if not analysis.get("critical_keys_present", False):
        results["issues"].append("No critical model keys found (transformer, model, embed_tokens, or lm_head)")
        results["checks_failed"] += 1
    else:
        results["checks_passed"] += 1
    
    # Check 2: Must have embedding
    if not analysis.get("has_embedding", False):
        results["issues"].append("No embedding parameter found (embed_tokens.embedding)")
        results["checks_failed"] += 1
    else:
        results["checks_passed"] += 1
    
    # Check 3: Parameters size should be reasonable
    if analysis.get("total_size_mb", 0) < 10:
        results["issues"].append(f"Parameter size suspiciously small: {analysis.get('total_size_mb')} MB")
        results["checks_failed"] += 1
    else:
        results["checks_passed"] += 1
    
    # Add summary info to results
    results["structure_summary"] = {
        "params_structure": "Has 'params' key" if analysis.get("has_params_key", False) else "Flat structure",
        "total_params": analysis.get("total_params", 0),
        "total_size_mb": analysis.get("total_size_mb", 0),
        "has_embedding": analysis.get("has_embedding", False),
        "critical_keys_present": analysis.get("critical_keys_present", False)
    }
    
    # Overall status
    if results["checks_failed"] > 0:
        results["status"] = "issues_found"
    else:
        results["status"] = "ok"
    
    return results

def diagnose_weight_loading(model_path: str, load_func=None) -> Dict[str, Any]:
    """
    Diagnose weight loading issues by checking files and attempting various loading methods.
    
    Args:
        model_path: Path to the checkpoint
        load_func: Optional function to load weights for testing
        
    Returns:
        Dictionary with diagnostic results
    """
    from importlib import import_module
    
    results = {
        "status": "checking",
        "checkpoint_path": model_path,
        "files": None,
        "loading_attempts": [],
        "recommendations": []
    }
    
    # Step 1: Check files
    try:
        results["files"] = scan_checkpoint_files(model_path)
        
        # Check for minimum required files
        if not results["files"]["safetensors"] and not results["files"]["bin"] and not results["files"]["pt"]:
            results["status"] = "error"
            results["error"] = f"No weight files found in {model_path}"
            results["recommendations"].append(
                "Ensure the path contains .safetensors, .bin, or .pt weight files"
            )
            return results
    except Exception as e:
        results["status"] = "error"
        results["error"] = f"Error scanning checkpoint files: {str(e)}"
        return results
    
    # If no loading function is provided, we just report file structure
    if load_func is None:
        if results["files"]["safetensors"]:
            results["recommendations"].append(
                "Found safetensors files. Consider using load_safetensors_only() for loading"
            )
        else:
            results["recommendations"].append(
                "No safetensors files found. Consider converting model to safetensors format"
            )
        results["status"] = "ok"
        return results
    
    # Step 2: Try loading with provided function
    try:
        results["status"] = "loading"
        params = load_func(model_path)
        
        # Analyze loaded parameters
        param_analysis = analyze_param_structure(params)
        verify_results = verify_loaded_weights(params)
        
        results["loaded_params"] = {
            "analysis": param_analysis,
            "verification": verify_results
        }
        
        # Check for any issues
        if verify_results["status"] != "ok":
            results["status"] = "loaded_with_issues"
            results["recommendations"].extend([
                f"Issue: {issue}" for issue in verify_results["issues"]
            ])
        else:
            results["status"] = "ok"
            
    except Exception as e:
        results["status"] = "error"
        results["error"] = f"Error loading weights: {str(e)}"
        results["recommendations"].append(
            "Check the error message and ensure the weight files are compatible with the model"
        )
    
    return results

def weight_loading_tester(model_path: str) -> Dict[str, Any]:
    """
    Test all available weight loading methods on a checkpoint.
    
    Args:
        model_path: Path to the checkpoint
        
    Returns:
        Dictionary with test results
    """
    results = {
        "status": "running_tests",
        "checkpoint_path": model_path,
        "file_scan": None,
        "tests": []
    }
    
    # First scan files
    results["file_scan"] = scan_checkpoint_files(model_path)
    
    # Try to import weight loading functions
    try:
        from weight_loading import load_safetensors_only, load_qwen_weights
        from transformers import AutoConfig
    except ImportError as e:
        results["status"] = "error"
        results["error"] = f"Error importing required modules: {str(e)}"
        return results
    
    # Get model config
    try:
        config = AutoConfig.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"Could not load config from {model_path}: {e}")
        config = None
    
    # Test 1: Direct safetensors loading
    test_result = {
        "name": "load_safetensors_only",
        "status": "running"
    }
    
    try:
        params = load_safetensors_only(
            model_path=model_path,
            config=config,
            mesh=None,
            param_dtype=jnp.float16
        )
        
        # Verify loaded parameters
        verification = verify_loaded_weights(params, config)
        test_result["verification"] = verification
        test_result["status"] = verification["status"]
        
    except Exception as e:
        test_result["status"] = "error"
        test_result["error"] = str(e)
    
    results["tests"].append(test_result)
    
    # Test 2: Standard loading path (without model)
    test_result = {
        "name": "load_qwen_weights",
        "status": "running"
    }
    
    try:
        # Create a dummy model class just for testing
        class DummyModel:
            def __init__(self):
                self.config = config
        
        params = load_qwen_weights(
            model_path=model_path,
            model=DummyModel(),
            config=config,
            mesh=None,
            param_dtype=jnp.float16,
            debug=True
        )
        
        # Verify loaded parameters
        verification = verify_loaded_weights(params, config)
        test_result["verification"] = verification
        test_result["status"] = verification["status"]
        
    except Exception as e:
        test_result["status"] = "error"
        test_result["error"] = str(e)
    
    results["tests"].append(test_result)
    
    # Determine overall status
    successes = [test for test in results["tests"] if test["status"] == "ok"]
    if successes:
        results["status"] = "ok"
        results["best_method"] = successes[0]["name"]
    else:
        issues = [test for test in results["tests"] if test["status"] == "issues_found"]
        if issues:
            results["status"] = "issues_found"
            results["best_method"] = issues[0]["name"]
        else:
            results["status"] = "error"
            results["error"] = "All loading methods failed"
    
    return results

def fix_parameter_structure(params: Dict) -> Dict:
    """
    Fix common parameter structure issues.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Fixed parameter dictionary
    """
    if params is None:
        logger.error("Cannot fix None parameters")
        return {}
    
    # Check if we need to wrap in 'params'
    has_params_key = "params" in params
    
    # Critical top-level keys that would indicate we need a params wrapper
    critical_top_keys = ["transformer", "model", "lm_head"]
    needs_params_wrapper = any(key in params for key in critical_top_keys)
    
    if not has_params_key and needs_params_wrapper:
        logger.info("Wrapping parameters in 'params' key")
        params = {"params": params}
    
    # Check for embedding
    flat_params = flatten_dict(params)
    has_embedding = False
    
    # Check for embedding in tuple keys
    for key_tuple in flat_params.keys():
        # Check for embed_tokens.embedding pattern in the key tuple
        if len(key_tuple) >= 4:
            # Look for the sequence that would represent embed_tokens.embedding
            # This could be ('params', 'transformer', 'embed_tokens', 'embedding')
            # or similar pattern
            for i in range(len(key_tuple) - 1):
                if key_tuple[i] == "embed_tokens" and key_tuple[i+1] == "embedding":
                    has_embedding = True
                    logger.info(f"Found embedding parameter at key: {key_tuple}")
                    break
        if has_embedding:
            break
    
    if not has_embedding:
        logger.warning("No embedding parameter found (embed_tokens.embedding)")
        
        # Try to fix by finding embedding-like parameters
        for key_tuple, value in list(flat_params.items()):
            # Look for keys containing 'embedding' or 'embed'
            contains_embed = False
            for part in key_tuple:
                if isinstance(part, str) and ("embedding" in part or "embed" in part):
                    contains_embed = True
                    break
                
            if contains_embed and hasattr(value, "shape"):
                logger.info(f"Found potential embedding parameter: {key_tuple} with shape {value.shape}")
                
                # Try to create embed_tokens.embedding
                if has_params_key:
                    # Check if 'transformer' exists in structure
                    has_transformer = False
                    for k in flat_params.keys():
                        if len(k) > 1 and k[0] == "params" and k[1] == "transformer":
                            has_transformer = True
                            break
                    
                    if has_transformer:
                        # Add under transformer
                        new_key = ("params", "transformer", "embed_tokens", "embedding")
                    else:
                        # Add at top level
                        new_key = ("params", "embed_tokens", "embedding")
                    
                    flat_params[new_key] = value
                    logger.info(f"Added embedding parameter at {new_key}")
                    has_embedding = True
                    break
    
    # Reconstruct parameters from flat dictionary
    if flat_params != flatten_dict(params):
        params = unflatten_dict(flat_params)
        logger.info("Reconstructed parameters with modified structure")
    
    return params

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose weight loading issues")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--scan-only", action="store_true", help="Only scan files, don't attempt loading")
    parser.add_argument("--test-all", action="store_true", help="Test all loading methods")
    args = parser.parse_args()
    
    if args.scan_only:
        files = scan_checkpoint_files(args.model_path)
        print(json.dumps(files, indent=2))
    elif args.test_all:
        results = weight_loading_tester(args.model_path)
        print(json.dumps(results, indent=2))
    else:
        results = diagnose_weight_loading(args.model_path)
        print(json.dumps(results, indent=2)) 