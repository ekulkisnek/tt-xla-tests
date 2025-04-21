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
import re
from typing import Dict, List, Optional, Tuple, Union, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import freeze, unfreeze

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

def create_parameter_structure_report(params: Dict, prefix: str = '') -> Dict[str, Dict]:
    """
    Create a detailed report about parameter structure to diagnose issues.
    
    Args:
        params: Parameter dictionary
        prefix: Optional prefix for parameter paths
        
    Returns:
        Dictionary with parameter structure information
    """
    report = {
        'summary': {
            'total_params': 0,
            'total_size_mb': 0,
            'missing_params': [],
            'unusual_shapes': [],
            'unusual_dtypes': [],
        },
        'params': {},
    }
    
    # Check if params is directly available or needs extraction
    if hasattr(params, 'params'):
        params = params.params
    
    # Standard parameter names we expect to find
    expected_params = [
        'transformer.embed_tokens',
        'transformer.h',
        'transformer.ln_f',
        'lm_head'
    ]
    
    # Flatten the parameters for easier iteration
    flat_params = flatten_dict(params)
    
    # Check for expected parameters
    for exp_param in expected_params:
        found = False
        for param_path in flat_params.keys():
            path_str = '.'.join([str(p) for p in param_path])
            if exp_param in path_str:
                found = True
                break
        
        if not found:
            report['summary']['missing_params'].append(exp_param)
    
    # Process each parameter
    for param_path, param_value in flat_params.items():
        path_str = '.'.join([str(p) for p in param_path])
        
        if hasattr(param_value, 'shape'):
            shape = param_value.shape
            dtype = param_value.dtype
            size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            size_mb = size_bytes / (1024 * 1024)
            
            report['summary']['total_params'] += 1
            report['summary']['total_size_mb'] += size_mb
            
            param_info = {
                'shape': str(shape),
                'dtype': str(dtype),
                'size_mb': round(size_mb, 2),
            }
            
            # Check for unusual shapes (e.g., zero dimensions)
            if 0 in shape:
                report['summary']['unusual_shapes'].append(path_str)
                param_info['issue'] = 'Zero dimension'
                
            # Check for unusual dtypes
            if dtype not in [jnp.float32, jnp.float16, jnp.bfloat16, np.float32, np.float16]:
                report['summary']['unusual_dtypes'].append(path_str)
                param_info['issue'] = f'Unusual dtype: {dtype}'
            
            report['params'][path_str] = param_info
    
    # Round total size
    report['summary']['total_size_mb'] = round(report['summary']['total_size_mb'], 2)
    
    # Check for common structure issues
    report['structure_analysis'] = analyze_parameter_structure(params)
    
    return report

def analyze_parameter_structure(params: Dict) -> Dict[str, Any]:
    """
    Analyze the parameter structure for common issues.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'issues': [],
        'recommendations': [],
    }
    
    # Check if params is wrapped with a 'params' key
    if 'params' in params and isinstance(params['params'], dict):
        analysis['issues'].append('Parameters are wrapped with a "params" key, might need unwrapping')
        analysis['recommendations'].append('Consider unwrapping parameters: params = params["params"]')
    
    # Check for transformer structure
    if 'transformer' not in params and 'model' in params:
        analysis['issues'].append('Using "model" instead of "transformer" as the top-level key')
        analysis['recommendations'].append('Consider renaming "model" to "transformer" in parameter paths')
    
    # Check embedding structure
    flat_params = flatten_dict(params)
    has_embedding = False
    for path in flat_params.keys():
        path_str = '.'.join([str(p) for p in path])
        if 'embed_tokens' in path_str:
            has_embedding = True
            # Check for different embedding variants
            if 'embedding' not in path_str and 'weight' in path_str:
                analysis['issues'].append('Embedding uses "weight" instead of "embedding"')
                analysis['recommendations'].append('Map "weight" to "embedding" for embedding parameters')
    
    if not has_embedding:
        analysis['issues'].append('Missing embedding parameters')
        analysis['recommendations'].append('Check if embedding parameters are present under a different name')
    
    # Check layer naming convention
    layer_pattern_h = re.compile(r'transformer\.h\.(\d+)')
    layer_pattern_layers = re.compile(r'(transformer|model)\.layers\.(\d+)')
    layer_pattern_layers_underscore = re.compile(r'(transformer|model)\.layers_(\d+)')
    
    h_layers = []
    numeric_layers = []
    underscore_layers = []
    
    for path in flat_params.keys():
        path_str = '.'.join([str(p) for p in path])
        
        if layer_pattern_h.search(path_str):
            h_layers.append(path_str)
        
        if layer_pattern_layers.search(path_str):
            numeric_layers.append(path_str)
            
        if layer_pattern_layers_underscore.search(path_str):
            underscore_layers.append(path_str)
    
    if h_layers and (numeric_layers or underscore_layers):
        analysis['issues'].append('Mixed layer naming conventions (h.N vs layers.N vs layers_N)')
        analysis['recommendations'].append('Standardize layer naming to a single convention')
    
    # Check attention component naming
    attn_pattern_alt = re.compile(r'self_attn')
    attn_pattern_std = re.compile(r'attn')
    
    has_alt_attn = False
    has_std_attn = False
    
    for path in flat_params.keys():
        path_str = '.'.join([str(p) for p in path])
        
        if attn_pattern_alt.search(path_str):
            has_alt_attn = True
        
        if attn_pattern_std.search(path_str) and not attn_pattern_alt.search(path_str):
            has_std_attn = True
    
    if has_alt_attn and has_std_attn:
        analysis['issues'].append('Mixed attention naming conventions (self_attn vs attn)')
        analysis['recommendations'].append('Standardize attention naming to a single convention')
    
    return analysis

def fix_parameter_structure(params: Dict) -> Dict:
    """
    Fix common parameter structure issues.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Fixed parameter dictionary
    """
    # Make a modifiable copy
    params = unfreeze(params) if hasattr(params, 'unfreeze') else dict(params)
    
    # Unwrap params if nested
    if 'params' in params and isinstance(params['params'], dict) and len(params) == 1:
        params = params['params']
    
    # Handle model/transformer naming
    if 'model' in params and 'transformer' not in params:
        params['transformer'] = params['model']
        del params['model']
    
    # Fix embedding parameters
    if 'transformer' in params:
        if 'embed_tokens' in params['transformer']:
            embed = params['transformer']['embed_tokens']
            
            # Fix embedding weight vs embedding
            if 'weight' in embed and 'embedding' not in embed:
                embed['embedding'] = embed['weight']
                del embed['weight']
            
            # Ensure embedding exists in some form
            if 'embedding' not in embed and hasattr(embed, 'shape'):
                # Case where embed_tokens directly points to the tensor
                params['transformer']['embed_tokens'] = {'embedding': embed}
    
    # Fix layer naming conventions
    layer_keys = {}
    has_h_format = False
    has_layers_format = False
    has_layers_underscore_format = False
    
    if 'transformer' in params:
        tr_keys = list(params['transformer'].keys())
        
        # Detect layer format
        if 'h' in tr_keys:
            has_h_format = True
            layer_keys['h'] = params['transformer']['h']
        if 'layers' in tr_keys:
            has_layers_format = True
            layer_keys['layers'] = params['transformer']['layers']
        if any(k.startswith('layers_') for k in tr_keys):
            has_layers_underscore_format = True
            for k in tr_keys:
                if k.startswith('layers_'):
                    layer_keys[k] = params['transformer'][k]
    
    # Standardize to h format if mixed
    if (has_h_format and has_layers_format) or (has_h_format and has_layers_underscore_format):
        pass  # Keep h format, fix others later
    elif has_layers_format and not has_h_format:
        # Convert layers to h
        params['transformer']['h'] = params['transformer']['layers']
        del params['transformer']['layers']
    elif has_layers_underscore_format and not has_h_format:
        # Create h from layers_N
        if 'h' not in params['transformer']:
            params['transformer']['h'] = {}
            
        for k in list(params['transformer'].keys()):
            if k.startswith('layers_'):
                layer_idx = int(k.split('_')[1])
                params['transformer']['h'][layer_idx] = params['transformer'][k]
                del params['transformer'][k]
    
    # Fix attention component naming
    if 'transformer' in params and 'h' in params['transformer']:
        for layer_idx, layer in params['transformer']['h'].items():
            # Fix self_attn to attn
            if 'self_attn' in layer and 'attn' not in layer:
                layer['attn'] = layer['self_attn']
                del layer['self_attn']
            
            # Fix attention component naming
            if 'attn' in layer:
                attn = layer['attn']
                
                # q_proj -> q
                for proj_pair in [('q_proj', 'q'), ('k_proj', 'k'), ('v_proj', 'v'), ('o_proj', 'o')]:
                    old, new = proj_pair
                    if old in attn and new not in attn:
                        attn[new] = attn[old]
                        del attn[old]
                        
                    # Fix kernel/weight naming
                    for w_name in ['kernel', 'weight']:
                        if new in attn and w_name in attn[new]:
                            # Ensure consistent kernel naming
                            if w_name == 'weight' and 'kernel' not in attn[new]:
                                attn[new]['kernel'] = attn[new]['weight']
                                del attn[new]['weight']
    
    # Fix MLP component naming
    if 'transformer' in params and 'h' in params['transformer']:
        for layer_idx, layer in params['transformer']['h'].items():
            if 'mlp' in layer:
                mlp = layer['mlp']
                
                # Fix MLP component naming
                for mlp_pair in [
                    ('gate_proj', 'w1'), 
                    ('up_proj', 'w2'), 
                    ('down_proj', 'w3')
                ]:
                    old, new = mlp_pair
                    if old in mlp and new not in mlp:
                        mlp[new] = mlp[old]
                        del mlp[old]
                    
                    # Fix kernel/weight naming
                    for w_name in ['kernel', 'weight']:
                        comp = old if old in mlp else new if new in mlp else None
                        if comp and w_name in mlp[comp]:
                            # Ensure consistent kernel naming
                            if w_name == 'weight' and 'kernel' not in mlp[comp]:
                                mlp[comp]['kernel'] = mlp[comp]['weight']
                                del mlp[comp]['weight']
    
    # Fix layernorm naming
    if 'transformer' in params and 'h' in params['transformer']:
        for layer_idx, layer in params['transformer']['h'].items():
            # input_layernorm -> ln_1
            if 'input_layernorm' in layer and 'ln_1' not in layer:
                layer['ln_1'] = layer['input_layernorm']
                del layer['input_layernorm']
            
            # post_attention_layernorm -> ln_2
            if 'post_attention_layernorm' in layer and 'ln_2' not in layer:
                layer['ln_2'] = layer['post_attention_layernorm']
                del layer['post_attention_layernorm']
    
    # Fix final layernorm
    if 'transformer' in params:
        if 'norm' in params['transformer'] and 'ln_f' not in params['transformer']:
            params['transformer']['ln_f'] = params['transformer']['norm']
            del params['transformer']['norm']
    
    # Return potentially frozen parameters
    return freeze(params) if hasattr(params, 'freeze') else params

def map_parameter_paths(params: Dict) -> Dict:
    """
    Apply parameter mapping to handle different path conventions.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Mapped parameter dictionary
    """
    # Unfreeze parameters for modification
    params = unfreeze(params) if hasattr(params, 'unfreeze') else dict(params)
    
    # Flatten the parameters
    flat_params = flatten_dict(params)
    
    # Mapping templates
    param_path_mappings = {
        # Embedding weight mappings
        ('transformer', 'embed_tokens', 'weight'): ('transformer', 'embed_tokens', 'embedding'),
        ('model', 'embed_tokens', 'weight'): ('transformer', 'embed_tokens', 'embedding'),
        ('params', 'transformer', 'embed_tokens', 'weight'): ('params', 'transformer', 'embed_tokens', 'embedding'),
        
        # Attention component mappings - with layer index handled separately
        # For standard Transformer path format
    }
    
    # Apply static mappings
    mapped_params = {}
    for old_path, value in flat_params.items():
        if old_path in param_path_mappings:
            new_path = param_path_mappings[old_path]
            mapped_params[new_path] = value
        else:
            mapped_params[old_path] = value
    
    # Handle layer-specific patterns with dynamic layer indices
    dynamic_mapped_params = {}
    for path, value in mapped_params.items():
        new_path = None
        
        # Extract path as strings for easier pattern matching
        path_str = '.'.join(str(p) for p in path)
        
        # Handle model.layers.N vs transformer.h.N
        if 'model.layers.' in path_str:
            layer_match = re.search(r'model\.layers\.(\d+)', path_str)
            if layer_match:
                layer_idx = layer_match.group(1)
                new_path_str = path_str.replace(f'model.layers.{layer_idx}', f'transformer.h.{layer_idx}')
                new_path = tuple(new_path_str.split('.'))
        
        # Handle transformer.layers.N vs transformer.h.N
        elif 'transformer.layers.' in path_str:
            layer_match = re.search(r'transformer\.layers\.(\d+)', path_str)
            if layer_match:
                layer_idx = layer_match.group(1)
                new_path_str = path_str.replace(f'transformer.layers.{layer_idx}', f'transformer.h.{layer_idx}')
                new_path = tuple(new_path_str.split('.'))
        
        # Handle transformer.layers_N vs transformer.h.N
        elif 'transformer.layers_' in path_str:
            layer_match = re.search(r'transformer\.layers_(\d+)', path_str)
            if layer_match:
                layer_idx = layer_match.group(1)
                new_path_str = path_str.replace(f'transformer.layers_{layer_idx}', f'transformer.h.{layer_idx}')
                new_path = tuple(new_path_str.split('.'))
                
        # Handle attention naming differences
        if 'self_attn' in path_str:
            # q_proj, k_proj, v_proj, o_proj
            for old_name, new_name in [
                ('self_attn.q_proj', 'attn.q'),
                ('self_attn.k_proj', 'attn.k'),
                ('self_attn.v_proj', 'attn.v'),
                ('self_attn.o_proj', 'attn.o')
            ]:
                if old_name in path_str:
                    if new_path is None:
                        new_path_str = path_str
                    else:
                        new_path_str = '.'.join(str(p) for p in new_path)
                    
                    new_path_str = new_path_str.replace(old_name, new_name)
                    new_path = tuple(new_path_str.split('.'))
        
        # Handle MLP naming differences
        if 'mlp.' in path_str:
            # gate_proj -> w1, up_proj -> w2, down_proj -> w3
            for old_name, new_name in [
                ('mlp.gate_proj', 'mlp.w1'),
                ('mlp.up_proj', 'mlp.w2'),
                ('mlp.down_proj', 'mlp.w3')
            ]:
                if old_name in path_str:
                    if new_path is None:
                        new_path_str = path_str
                    else:
                        new_path_str = '.'.join(str(p) for p in new_path)
                    
                    new_path_str = new_path_str.replace(old_name, new_name)
                    new_path = tuple(new_path_str.split('.'))
        
        # Handle layernorm naming differences
        for old_name, new_name in [
            ('input_layernorm', 'ln_1'),
            ('post_attention_layernorm', 'ln_2'),
            ('norm', 'ln_f')
        ]:
            if old_name in path_str:
                if new_path is None:
                    new_path_str = path_str
                else:
                    new_path_str = '.'.join(str(p) for p in new_path)
                
                new_path_str = new_path_str.replace(old_name, new_name)
                new_path = tuple(new_path_str.split('.'))
        
        # Handle weight vs kernel naming
        if '.weight' in path_str and not any(ln in path_str for ln in ['ln_1.weight', 'ln_2.weight', 'ln_f.weight']):
            if new_path is None:
                new_path_str = path_str
            else:
                new_path_str = '.'.join(str(p) for p in new_path)
            
            new_path_str = new_path_str.replace('.weight', '.kernel')
            new_path = tuple(new_path_str.split('.'))
        
        # Use the new path if it was changed, otherwise use the original path
        if new_path is not None:
            dynamic_mapped_params[new_path] = value
        else:
            dynamic_mapped_params[path] = value
    
    # Convert the mapped paths back to a dictionary
    mapped_dict = unflatten_dict(dynamic_mapped_params)
    
    # Return as frozen dict if the input was frozen
    return freeze(mapped_dict) if hasattr(params, 'freeze') else mapped_dict

def combine_partial_params(params_list: List[Dict], config=None) -> Dict:
    """
    Combine parameters from multiple partial models into a single parameter dictionary.
    
    Args:
        params_list: List of parameter dictionaries to combine
        config: Optional model configuration to ensure consistent structure
        
    Returns:
        Combined parameter dictionary
    """
    if not params_list:
        return {}
    
    # Unfreeze all parameter dictionaries
    params_list = [unfreeze(p) if hasattr(p, 'unfreeze') else dict(p) for p in params_list]
    
    # Start with the first parameter set
    combined_params = dict(params_list[0])
    
    # Flatten all parameter dictionaries for easier merging
    flat_combined = flatten_dict(combined_params)
    
    # Merge in the remaining parameter sets
    for params in params_list[1:]:
        flat_params = flatten_dict(params)
        
        # Add any missing parameters
        for path, value in flat_params.items():
            if path not in flat_combined:
                flat_combined[path] = value
    
    # Unflatten back to a dictionary
    result = unflatten_dict(flat_combined)
    
    # Fix any structure issues
    result = fix_parameter_structure(result)
    
    # Return as frozen dict
    return freeze(result) if hasattr(params_list[0], 'freeze') else result

def check_parameter_shapes(params: Dict, config: Dict) -> List[str]:
    """
    Check if parameter shapes match the expected shapes based on the configuration.
    
    Args:
        params: Parameter dictionary
        config: Model configuration
        
    Returns:
        List of parameter paths with shape mismatches
    """
    # Extract config values
    hidden_size = config.get('hidden_size', 4096)
    intermediate_size = config.get('intermediate_size', 14336)
    num_attention_heads = config.get('num_attention_heads', 32)
    head_dim = hidden_size // num_attention_heads
    
    # Expected shapes for different parameter types
    expected_shapes = {
        'embed_tokens': {
            'embedding': (config.get('vocab_size', 151936), hidden_size),
        },
        'attn': {
            'q.kernel': (hidden_size, num_attention_heads * head_dim),
            'k.kernel': (hidden_size, num_attention_heads * head_dim),
            'v.kernel': (hidden_size, num_attention_heads * head_dim),
            'o.kernel': (num_attention_heads * head_dim, hidden_size),
        },
        'mlp': {
            'w1.kernel': (hidden_size, intermediate_size),
            'w2.kernel': (hidden_size, intermediate_size),
            'w3.kernel': (intermediate_size, hidden_size),
        },
        'ln': {
            'weight': (hidden_size,),
        },
        'lm_head': {
            'weight': (config.get('vocab_size', 151936), hidden_size),
        }
    }
    
    # Flatten parameters
    flat_params = flatten_dict(params)
    
    # Check for shape mismatches
    mismatches = []
    
    for path, value in flat_params.items():
        path_str = '.'.join(str(p) for p in path)
        
        # Skip non-tensor values
        if not hasattr(value, 'shape'):
            continue
        
        # Check embed_tokens
        if 'embed_tokens' in path_str and path_str.endswith('embedding'):
            expected = expected_shapes['embed_tokens']['embedding']
            if value.shape != expected:
                mismatches.append(f"{path_str}: got {value.shape}, expected {expected}")
        
        # Check attention components
        elif 'attn' in path_str:
            for comp in ['q.kernel', 'k.kernel', 'v.kernel', 'o.kernel']:
                if path_str.endswith(comp):
                    expected = expected_shapes['attn'][comp]
                    if value.shape != expected:
                        mismatches.append(f"{path_str}: got {value.shape}, expected {expected}")
        
        # Check MLP components
        elif 'mlp' in path_str:
            for comp in ['w1.kernel', 'w2.kernel', 'w3.kernel']:
                if path_str.endswith(comp):
                    expected = expected_shapes['mlp'][comp]
                    if value.shape != expected:
                        mismatches.append(f"{path_str}: got {value.shape}, expected {expected}")
        
        # Check layer norms
        elif any(ln in path_str for ln in ['ln_1.weight', 'ln_2.weight', 'ln_f.weight']):
            if value.shape != expected_shapes['ln']['weight']:
                mismatches.append(f"{path_str}: got {value.shape}, expected {expected_shapes['ln']['weight']}")
        
        # Check lm_head
        elif path_str.endswith('lm_head.weight'):
            expected = expected_shapes['lm_head']['weight']
            if value.shape != expected:
                mismatches.append(f"{path_str}: got {value.shape}, expected {expected}")
    
    return mismatches

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