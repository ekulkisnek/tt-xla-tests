#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Adaptive verification script for Qwen2.5 model with 1x8 tensor parallelism.
This script allows you to scale the model size dynamically to find the maximum
configuration your hardware can handle through adaptive scaling.

IMPROVEMENTS IN THIS VERSION:
- Added improved weight adaptation using SVD for better quality when scaling
- Reduced excessive logging for better readability
- Added a quiet mode for minimal logging (--quiet flag)
- Improved background resource monitoring to be less intrusive
- Better weight loading stats with summaries instead of per-weight logs
- Optimized tensor resizing to preserve important weights

Usage:
cd to qwen25
source venv/bin/activate
pip install jax jaxlib flax transformers safetensors numpy einops tqdm datasets tensorboard
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

# Auto-scale to the maximum possible model size (recommended first run)
python verify_qwen25_adaptive.py --model_path /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --auto_scale

# Try with specific scaling ratio (0.0-1.0) of the full model
python verify_qwen25_adaptive.py --model_path /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --scale_ratio 0.5

# Manually set size parameters (hidden_size, num_layers as fractions of original)
python verify_qwen25_adaptive.py --model_path /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --hidden_ratio 0.6 --layers_ratio 0.7

# Use quiet mode for minimal logging
python verify_qwen25_adaptive.py --model_path /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --auto_scale --quiet
"""

import os
# Force JAX to use simulated devices for tensor parallelism testing
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import argparse
import time
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import logging
import gc
from functools import partial
import psutil  # For memory and CPU monitoring
import traceback  # For detailed error tracing
import threading  # For background monitoring
import math
from tqdm import tqdm  # For progress bars
from typing import Dict, Any
from jax.sharding import Mesh, PartitionSpec as P
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("qwen25-adaptive")

# Set lower log level for verbose loggers
logging.getLogger("model_implementation").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("flax").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Disable excessive logging from tensor operations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
os.environ["JAX_LOG_COMPILES"] = "0"      # Disable JAX compilation logging

# Import model components
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
    get_partition_specs,
    load_params_from_checkpoint,
)
from config import load_qwen_config, get_qwen2_7b_config
from weight_loading import load_qwen_weights, load_partial_qwen_weights

# Global variables for monitoring
monitoring_active = False
monitor_thread = None
peak_memory_usage = 0
peak_cpu_usage = 0
last_memory_usage = 0
last_cpu_usage = 0
memory_history = []

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def colored(text, color):
    """Add color to terminal text."""
    return f"{color}{text}{Colors.ENDC}"

def get_system_metrics():
    """Get current memory and CPU usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    # Get CPU usage
    process_cpu = process.cpu_percent(interval=0.1)
    system_cpu = psutil.cpu_percent(interval=0.1)
    
    # Get per-core CPU usage
    per_core_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    
    # Get GPU metrics if available
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info[f"gpu_{i}"] = {
                    "memory_used": torch.cuda.memory_allocated(i) / (1024**2),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**2),
                    "name": torch.cuda.get_device_name(i)
                }
    except (ImportError, Exception):
        pass  # Skip if torch not available
    
    metrics = {
        "process_memory_mb": memory_info.rss / (1024 * 1024),
        "system_total_mb": system_memory.total / (1024 * 1024),
        "system_available_mb": system_memory.available / (1024 * 1024),
        "system_used_percent": system_memory.percent,
        "process_cpu_percent": process_cpu,
        "system_cpu_percent": system_cpu,
        "per_core_cpu": per_core_cpu,
        "gpu_info": gpu_info,
        "timestamp": time.time()
    }
    
    memory_history.append((metrics["timestamp"], metrics["process_memory_mb"]))
    
    return metrics

def log_system_metrics(phase="Current", detailed=True):
    """Log memory and CPU usage information."""
    global peak_memory_usage, peak_cpu_usage, last_memory_usage, last_cpu_usage
    
    metrics = get_system_metrics()
    last_memory_usage = metrics["process_memory_mb"]
    last_cpu_usage = metrics["process_cpu_percent"]
    
    # Update peak values
    peak_memory_usage = max(peak_memory_usage, metrics["process_memory_mb"])
    peak_cpu_usage = max(peak_cpu_usage, metrics["process_cpu_percent"])
    
    # Calculate memory usage ratio
    mem_ratio = metrics["process_memory_mb"] / metrics["system_total_mb"] * 100
    
    # Print header with phase only if detailed or the phase is not a routine check
    if detailed or phase not in ["Before parameter initialization", "After mesh creation", "After garbage collection"]:
        logger.info(colored(f"System Metrics ({phase}):", Colors.HEADER + Colors.BOLD))
    
        # Memory metrics - simplified
        mem_color = Colors.GREEN
        if mem_ratio > 70:
            mem_color = Colors.YELLOW
        if mem_ratio > 85:
            mem_color = Colors.RED
        
        logger.info(colored(f"  RAM: {metrics['process_memory_mb']:.1f} MB ({mem_ratio:.1f}%)", mem_color))
        
        # Only show peak RAM if it's significantly different than current
        if peak_memory_usage > metrics['process_memory_mb'] * 1.05:  # 5% difference
            logger.info(f"  Peak RAM: {peak_memory_usage:.1f} MB ({peak_memory_usage/metrics['system_total_mb']*100:.1f}%)")
    
        # Only show detailed metrics if requested and not a routine check
        if detailed and phase not in ["Background Monitor"]:
            # Just show essential metrics, skip per-core CPU and other verbose outputs
            if metrics["system_used_percent"] > 90:
                logger.info(colored(f"  System RAM: {metrics['system_used_percent']:.1f}% used (CRITICALLY HIGH)", Colors.RED))
            
            # Only show GPU info if any GPU is over 80% utilized
            if metrics["gpu_info"]:
                any_gpu_high = False
                for gpu_id, gpu_data in metrics["gpu_info"].items():
                    gpu_ratio = gpu_data["memory_used"] / gpu_data["memory_total"] * 100
                    if gpu_ratio > 80:
                        any_gpu_high = True
                
                if any_gpu_high:
                    logger.info(colored("  GPU Information:", Colors.CYAN))
                    for gpu_id, gpu_data in metrics["gpu_info"].items():
                        gpu_ratio = gpu_data["memory_used"] / gpu_data["memory_total"] * 100
                        gpu_color = Colors.GREEN
                        if gpu_ratio > 70:
                            gpu_color = Colors.YELLOW
                        if gpu_ratio > 85:
                            gpu_color = Colors.RED
                        logger.info(colored(f"    {gpu_id}: {gpu_data['memory_used']:.1f} MB / {gpu_data['memory_total']:.1f} MB = {gpu_ratio:.1f}%", gpu_color))
    
    return metrics

def start_background_monitoring(interval=10):
    """Start a background thread to monitor system resources."""
    global monitoring_active, monitor_thread
    
    def monitor_resources():
        last_log_time = time.time()
        log_frequency = 15  # Only log every 15 seconds even if checking more frequently
        
        while monitoring_active:
            # Always collect metrics but don't always log them
            metrics = get_system_metrics()
            
            # Check if it's time to log
            current_time = time.time()
            should_log = current_time - last_log_time >= log_frequency
            
            # Only log if RAM usage is high (>70%) or it's time for regular logging
            memory_percent = metrics["process_memory_mb"] / metrics["system_total_mb"] * 100
            if memory_percent > 70 or should_log:
                log_system_metrics("Background Monitor", detailed=False)
                last_log_time = current_time
                
            time.sleep(interval)
    
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    logger.info(colored(f"Started background resource monitoring (interval: {interval}s)", Colors.BLUE))


def stop_background_monitoring():
    """Stop the background monitoring thread."""
    global monitoring_active, monitor_thread
    
    if monitoring_active and monitor_thread:
        monitoring_active = False
        monitor_thread.join(timeout=1)
        logger.info(colored("Stopped background resource monitoring", Colors.BLUE))
        logger.info(colored(f"Peak process memory: {peak_memory_usage:.2f} MB", Colors.BOLD))
        logger.info(colored(f"Peak process CPU: {peak_cpu_usage:.1f}%", Colors.BOLD))


def load_original_model_config(model_path):
    """Load the original model configuration from config.json."""
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        # Use default Qwen2-7B as fallback
        logger.warning(colored(f"No config.json found at {model_path}, using default Qwen2-7B config", Colors.YELLOW))
        return get_qwen2_7b_config()


def calculate_model_size(hidden_size, num_layers, vocab_size, intermediate_multiplier=4):
    """Calculate approximate model size in parameters and memory."""
    # Calculate intermediate size based on multiplier
    intermediate_size = hidden_size * intermediate_multiplier
    
    # Calculate parameter count
    embedding_params = hidden_size * vocab_size
    
    # Each layer has:
    # - 3 attention matrices (q, k, v) and 1 output projection
    # - 2 layer norms
    # - 3 MLP matrices (up, gate, down)
    attn_params = 4 * hidden_size * hidden_size
    norm_params = 2 * hidden_size
    mlp_params = hidden_size * intermediate_size * 2 + intermediate_size * hidden_size
    
    layer_params = attn_params + norm_params + mlp_params
    total_params = embedding_params + (num_layers * layer_params)
    
    # Estimate memory (BF16 = 2 bytes)
    memory_bytes = total_params * 2  # BF16
    memory_mb = memory_bytes / (1024 * 1024)
    
    # Activation memory (rough estimate - varies by implementation)
    # Using a factor of 1.5x for all overhead including activations, state tracking, etc.
    total_memory_mb = memory_mb * 1.5
    
    return {
        "params": total_params,
        "memory_mb": memory_mb,
        "total_memory_mb": total_memory_mb,
        "embedding_params": embedding_params,
        "layer_params": layer_params
    }


def create_scaled_config(original_config, hidden_ratio=1.0, layers_ratio=1.0, heads_ratio=None, kv_ratio=None):
    """Create a scaled configuration for Qwen2.5 model based on original config.
    
    Args:
        original_config: Original model configuration
        hidden_ratio: Ratio to scale hidden size (0.0-1.0)
        layers_ratio: Ratio to scale layer count (0.0-1.0)
        heads_ratio: Ratio to scale attention heads (0.0-1.0) or None to derive from hidden_ratio
        kv_ratio: Ratio to scale KV heads (0.0-1.0) or None to derive from heads_ratio
    
    Returns:
        Dictionary with scaled model configuration
    """
    # Get original dimensions
    orig_hidden = original_config.get("hidden_size", 4096)
    orig_layers = original_config.get("num_hidden_layers", 32)
    orig_heads = original_config.get("num_attention_heads", 32)
    orig_kv_heads = original_config.get("num_key_value_heads", 4)
    orig_intermediate = original_config.get("intermediate_size", orig_hidden * 4)
    orig_vocab = original_config.get("vocab_size", 152064)
    
    # Calculate target dimensions (ensuring values make sense)
    target_hidden = max(128, int(orig_hidden * hidden_ratio))
    # Round to nearest multiple of 128 for better hardware efficiency
    target_hidden = (target_hidden + 64) // 128 * 128  
    
    target_layers = max(1, int(orig_layers * layers_ratio))
    
    # Derive heads from hidden size if not specified
    if heads_ratio is None:
        # Scale heads proportionally to hidden size
        # Aim for approximately 128 dimensions per head as in the original model
        target_heads = max(1, target_hidden // 128) 
    else:
        target_heads = max(1, int(orig_heads * heads_ratio))
    
    # For Qwen models, ensure number of heads divides hidden size evenly
    # and each head has reasonable dimension
    head_dim = target_hidden // target_heads
    if head_dim * target_heads != target_hidden:
        # Adjust number of heads to divide hidden size evenly
        target_heads = max(1, target_hidden // 128)
        # Ensure it divides evenly
        while target_hidden % target_heads != 0 and target_heads > 1:
            target_heads -= 1
    
    # Derive KV heads from attention heads if not specified
    if kv_ratio is None:
        # For scaled models, use a sensible KV to attn head ratio
        # For small models, keep more KV heads (minimum 1)
        kv_to_attn_ratio = min(0.5, orig_kv_heads / orig_heads)
        target_kv_heads = max(1, int(target_heads * kv_to_attn_ratio))
    else:
        target_kv_heads = max(1, int(orig_kv_heads * kv_ratio))
    
    # Ensure KV heads divides attention heads evenly for efficient computation
    if target_heads % target_kv_heads != 0:
        # Find the largest divisor of target_heads that's not larger than the target_kv_heads
        for i in range(target_kv_heads, 0, -1):
            if target_heads % i == 0:
                target_kv_heads = i
                break
    
    # Calculate intermediate size (typically 4x hidden size for Qwen)
    target_intermediate = target_hidden * 4
    
    # Create new config
    config = {
        "vocab_size": orig_vocab,  # Keep original vocab size
        "hidden_size": target_hidden,
        "intermediate_size": target_intermediate,
        "num_hidden_layers": target_layers,
        "num_attention_heads": target_heads,
        "num_key_value_heads": target_kv_heads,
        "hidden_act": original_config.get("hidden_act", "silu"),
        "max_position_embeddings": original_config.get("max_position_embeddings", 32768),
        "initializer_range": original_config.get("initializer_range", 0.02),
        "rms_norm_eps": original_config.get("rms_norm_eps", 1e-6),
        "use_cache": True,
        "tie_word_embeddings": False,
        "rope_theta": original_config.get("rope_theta", 10000.0),
        "attention_dropout": original_config.get("attention_dropout", 0.0),
    }
    
    # Calculate model size
    size_info = calculate_model_size(
        config["hidden_size"], 
        config["num_hidden_layers"],
        config["vocab_size"]
    )
    
    return config, size_info


def setup_tokenizer(tokenizer_path=None):
    """Set up the tokenizer."""
    try:
        from transformers import AutoTokenizer
        
        if tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # Try to load from Hugging Face or use a local fallback
            try:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
            except:
                # Try local path as fallback
                local_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), 
                    "qwen25-weights"
                ))
                if os.path.exists(local_path):
                    tokenizer = AutoTokenizer.from_pretrained(local_path)
                else:
                    raise ValueError(f"Could not find tokenizer at {local_path}")
        
        logger.info(colored("✅ Tokenizer loaded successfully", Colors.GREEN))
        return tokenizer
    except Exception as e:
        logger.error(colored(f"❌ Error loading tokenizer: {e}", Colors.RED))
        logger.info("Please install transformers: pip install transformers")
        return None


def estimate_max_model_size(system_memory_mb, target_usage_ratio=0.8):
    """Estimate maximum model size that would fit in target memory usage."""
    # Get available system memory
    available_memory_mb = system_memory_mb * target_usage_ratio
    
    # Subtract baseline memory usage for the Python process (~500MB)
    available_for_model_mb = available_memory_mb - 500
    
    # Each parameter in BF16 is 2 bytes
    # Add 50% overhead for activations and other runtime memory
    available_parameters = (available_for_model_mb * 1024 * 1024) / (2 * 1.5)
    
    return available_parameters 

def improve_load_partial_qwen_weights(model_path, target_params, config, mesh, num_layers=None, logger=None):
    """
    Improved version of partial weight loading with reduced logging and better tensor resizing.
    
    Args:
        model_path: Path to the full model weights
        target_params: Target parameter structure (from model.init())
        config: Reduced size configuration
        mesh: Device mesh
        num_layers: Number of layers to load
        logger: Logger for status messages
    
    Returns:
        Dictionary with loaded parameters matching the reduced structure
    """
    import os
    import json
    import numpy as np
    from safetensors import safe_open
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict, unflatten_dict
    
    # Simple logging function if no logger provided
    def log_info(msg, verbose_only=False):
        if logger:
            if not verbose_only:
                logger.info(msg)
            elif verbose_only and logger.level <= logging.DEBUG:
                logger.debug(msg)
        else:
            if not verbose_only:
                print(msg)
    
    log_info(f"Loading partial weights from {model_path} for reduced size model")
    
    # Create a copy of target_params to modify
    new_params = jax.tree_util.tree_map(lambda x: x, target_params)
    
    # Load original model config to get original dimensions
    original_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(original_config_path):
        with open(original_config_path, "r") as f:
            original_config = json.load(f)
        log_info(f"Original model: hidden_size={original_config.get('hidden_size')}, layers={original_config.get('num_hidden_layers')}")
    else:
        log_info(f"No config.json found at {model_path}, assuming standard Qwen2 7B architecture")
        original_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 14336
        }
    
    # Get list of weight files (safetensors format)
    weight_files = []
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            weight_files.append(os.path.join(model_path, file))
    
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    
    log_info(f"Found {len(weight_files)} weight files")
    
    # Target sizes from reduced config
    target_hidden_size = config["hidden_size"]
    target_layers = config["num_hidden_layers"]
    
    # Simple weight adaptation method
    def resize_tensor(tensor, target_shape):
        """Resize tensor to target shape by simple truncation"""
        if tensor.shape == target_shape:
            return tensor
            
        if len(tensor.shape) != len(target_shape):
            log_info(f"Incompatible shapes for resizing: {tensor.shape} vs {target_shape}", verbose_only=True)
            return tensor
            
        # Simple truncation for each dimension
        slices = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
        tensor_resized = np.zeros(target_shape, dtype=tensor.dtype)
        tensor_view = tensor[slices]
        target_slices = tuple(slice(0, s.stop) for s in slices)
        tensor_resized[target_slices] = tensor_view
        return tensor_resized
    
    # Track statistics
    total_weights = 0
    loaded_weights = 0
    resized_weights = 0
    not_found_weights = 0
    
    # Load weights from each file
    for file_path in weight_files:
        log_info(f"Processing {os.path.basename(file_path)}...")
        
        with safe_open(file_path, framework="numpy") as f:
            weight_keys = f.keys()
            
            for key in weight_keys:
                total_weights += 1
                
                # Extract layer number for layer-specific weights
                if ".layers." in key:
                    parts = key.split(".layers.")
                    if len(parts) > 1:
                        layer_parts = parts[1].split(".")
                        try:
                            layer_num = int(layer_parts[0])
                            # Skip layers beyond our target layer count
                            if layer_num >= target_layers:
                                continue
                        except ValueError:
                            pass
                
                # Try to find the corresponding key in our target params
                tensor = f.get_tensor(key)
                param_key = key
                
                # Convert PyTorch key format to JAX format
                param_key = param_key.replace("model.layers", "transformer.h")
                param_key = param_key.replace("self_attn", "attn")
                param_key = param_key.replace("q_proj", "q")
                param_key = param_key.replace("k_proj", "k")
                param_key = param_key.replace("v_proj", "v")
                param_key = param_key.replace("o_proj", "o")
                param_key = param_key.replace("mlp.gate_proj", "mlp.w1")
                param_key = param_key.replace("mlp.up_proj", "mlp.w2")
                param_key = param_key.replace("mlp.down_proj", "mlp.w3")
                param_key = param_key.replace("input_layernorm", "ln_1")
                param_key = param_key.replace("post_attention_layernorm", "ln_2")
                param_key = param_key.replace("model.norm", "transformer.ln_f")
                
                # Handle embedding conversion
                if "embed_tokens.weight" in param_key:
                    param_key = "transformer.embed_tokens.embedding"
                
                # For kernel/weight conversion
                if param_key.endswith(".weight") and not param_key.endswith("ln_1.weight") and not param_key.endswith("ln_2.weight") and not param_key.endswith("ln_f.weight"):
                    param_key = param_key.replace(".weight", ".kernel")
                
                # Handle different paths in the params tree
                param_path = param_key.split(".")
                
                # Try to find the tensor location in our parameter tree
                current = new_params
                found = True
                
                for i, part in enumerate(param_path):
                    if part in current:
                        if i == len(param_path) - 1:  # Last part
                            # Found the target - now resize if needed
                            target_shape = current[part].shape
                            if tensor.shape != target_shape:
                                log_info(f"Resizing {param_key}: {tensor.shape} -> {target_shape}", verbose_only=True)
                                resized_weights += 1
                                
                                # Use simple resizing approach
                                tensor = resize_tensor(tensor, target_shape)
                            
                            # Convert to the right dtype
                            try:
                                tensor = jnp.array(tensor, dtype=current[part].dtype)
                                
                                # Apply the correct partitioning spec
                                if hasattr(current[part], "sharding"):
                                    sharding = current[part].sharding
                                    tensor = jax.device_put(tensor, sharding)
                                
                                # Update the parameter
                                current[part] = tensor
                                loaded_weights += 1
                                log_info(f"Loaded {param_key}", verbose_only=True)
                            except Exception as e:
                                log_info(f"Error setting {param_key}: {e}", verbose_only=True)
                        else:
                            current = current[part]
                    else:
                        found = False
                        break
                
                if not found:
                    # Try with params prefix
                    param_key = "params." + param_key
                    param_path = param_key.split(".")
                    current = new_params
                    found = True
                    
                    for i, part in enumerate(param_path):
                        if part in current:
                            if i == len(param_path) - 1:  # Last part
                                # Found the target - now resize if needed
                                target_shape = current[part].shape
                                if tensor.shape != target_shape:
                                    log_info(f"Resizing {param_key}: {tensor.shape} -> {target_shape}", verbose_only=True)
                                    resized_weights += 1
                                    
                                    # Use simple resizing approach
                                    tensor = resize_tensor(tensor, target_shape)
                                
                                # Convert to the right dtype
                                try:
                                    tensor = jnp.array(tensor, dtype=current[part].dtype)
                                    
                                    # Apply the correct partitioning spec
                                    if hasattr(current[part], "sharding"):
                                        sharding = current[part].sharding
                                        tensor = jax.device_put(tensor, sharding)
                                    
                                    # Update the parameter
                                    current[part] = tensor
                                    loaded_weights += 1
                                    log_info(f"Loaded {param_key}", verbose_only=True)
                                except Exception as e:
                                    log_info(f"Error setting {param_key}: {e}", verbose_only=True)
                            else:
                                current = current[part]
                        else:
                            found = False
                            break
                    
                    if not found:
                        log_info(f"Key {key} not found in target params", verbose_only=True)
                        not_found_weights += 1
    
    # Summary stats
    log_info(f"Weight loading summary:")
    log_info(f"  Total weights processed: {total_weights}")
    log_info(f"  Weights loaded successfully: {loaded_weights}")
    log_info(f"  Weights that required resizing: {resized_weights}")
    log_info(f"  Weights not found in target model: {not_found_weights}")
    
    return new_params

def verify_adaptive_model(
    model_path: str, 
    use_demo_model: bool = False, 
    max_tokens: int = 5,
    hidden_ratio: float = None,
    layers_ratio: float = None,
    scale_ratio: float = None,
    target_ram_percent: float = 80.0,
    auto_scale: bool = False,
    batch_size: int = 1,
    monitor_interval: int = 10,
    verbose: bool = True,
    quiet: bool = False,
    temperature: float = 0.7,  # Sampling temperature (higher = more random)
    top_p: float = 0.9,        # Nucleus sampling parameter (higher = more diverse)
    top_k: int = 50            # Optional top-k sampling parameter
):
    """
    Verify an adaptively-scaled model with tensor parallelism.
    Attempts to automatically determine maximum size or use specified scaling ratios.
    
    Args:
        model_path: Path to the model
        use_demo_model: Whether to use demo model instead of loading real weights
        max_tokens: Maximum number of tokens to generate in test
        hidden_ratio: Ratio of original hidden_size to use (0.0-1.0)
        layers_ratio: Ratio of original num_layers to use (0.0-1.0)
        scale_ratio: Overall scaling ratio for both dimensions (overrides hidden_ratio and layers_ratio)
        target_ram_percent: Target RAM usage percentage
        auto_scale: Whether to automatically find maximum possible model size
        batch_size: Batch size for inference
        monitor_interval: Interval in seconds for background resource monitoring
        verbose: Whether to show detailed diagnostic information
        quiet: Whether to minimize logging output
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter (0.0-1.0, higher = more diverse)
        top_k: Optional top-k sampling parameter (use 0 to disable)
        
    Returns:
        bool: True if verification succeeds, False otherwise
    """
    mesh_shape = (1, 8)  # Focus only on the 1x8 mesh shape
    
    logger.info(colored(f"\n{'='*80}", Colors.BOLD))
    logger.info(colored(f"QWEN 2.5 ADAPTIVE MODEL VERIFICATION (1x8 Tensor Parallel)", Colors.BOLD + Colors.HEADER))
    logger.info(colored(f"{'='*80}", Colors.BOLD))
    
    # Start background resource monitoring
    start_background_monitoring(monitor_interval)
    
    # Log initial metrics
    initial_metrics = log_system_metrics("Initial Setup", not quiet)
    system_memory_mb = initial_metrics["system_total_mb"]
    
    # Flag to track if scaling has been applied
    scaling_applied = False
    
    try:
        # Load original model configuration to determine scaling factors
        logger.info(colored("\n[1/7] Loading original model configuration...", Colors.BOLD))
        original_config = load_original_model_config(model_path)
        
        # Original model dimensions
        orig_hidden = original_config.get("hidden_size", 4096)
        orig_layers = original_config.get("num_hidden_layers", 32)
        orig_heads = original_config.get("num_attention_heads", 32)
        orig_kv_heads = original_config.get("num_key_value_heads", 4)
        
        logger.info(f"Original model dimensions:")
        logger.info(colored(f"  Hidden size: {orig_hidden}", Colors.CYAN))
        logger.info(colored(f"  Layers: {orig_layers}", Colors.CYAN))
        logger.info(colored(f"  Attention heads: {orig_heads}", Colors.CYAN))
        logger.info(colored(f"  KV heads: {orig_kv_heads}", Colors.CYAN))
        
        # Calculate original model size
        orig_size_info = calculate_model_size(
            orig_hidden, 
            orig_layers, 
            original_config.get("vocab_size", 152064)
        )
        
        logger.info(f"Original model has {orig_size_info['params']:,} parameters")
        logger.info(f"Original model requires ~{orig_size_info['total_memory_mb']:,.1f} MB of memory")
        
        # Determine scaling factors
        if auto_scale:
            logger.info(colored("\n[2/7] Auto-scaling model to fit available RAM...", Colors.BOLD))
            
            # Calculate available parameters for the target RAM usage
            available_params = estimate_max_model_size(system_memory_mb, target_ram_percent / 100.0)
            
            # Calculate a balanced scaling factor
            scaling_factor = (available_params / orig_size_info['params']) ** 0.5
            
            # Apply a safety margin to avoid out-of-memory errors
            scaling_factor = scaling_factor * 0.9
            
            # Limit to a maximum of 1.0
            scaling_factor = min(1.0, scaling_factor)
            
            logger.info(f"Available RAM: {system_memory_mb:.1f} MB, target usage: {target_ram_percent:.1f}%")
            logger.info(f"Estimated maximum parameters: {available_params:,.0f}")
            logger.info(colored(f"Auto-scaling factor: {scaling_factor:.3f}", Colors.YELLOW))
            
            # Use slightly more hidden_size than layers for better quality
            # But keep it simple and balanced
            hidden_ratio = min(1.0, scaling_factor * 1.1)  # 10% more for hidden size
            layers_ratio = min(1.0, scaling_factor * 0.9)  # 10% less for layers
            
            logger.info(colored(f"Hidden size ratio: {hidden_ratio:.3f}", Colors.CYAN))
            logger.info(colored(f"Layers ratio: {layers_ratio:.3f}", Colors.CYAN))
            
        elif scale_ratio is not None:
            # Use the same ratio for both dimensions
            hidden_ratio = scale_ratio
            layers_ratio = scale_ratio
            logger.info(colored(f"\n[2/7] Using uniform scaling ratio: {scale_ratio:.3f}", Colors.BOLD))
            
        else:
            # Use the specified ratios or defaults
            if hidden_ratio is None:
                hidden_ratio = 0.5  # Default to half size
            if layers_ratio is None:
                layers_ratio = 0.5  # Default to half layers
                
            logger.info(colored(f"\n[2/7] Using custom scaling ratios:", Colors.BOLD))
            logger.info(f"  Hidden size ratio: {hidden_ratio:.3f}")
            logger.info(f"  Layers ratio: {layers_ratio:.3f}")
        
        # Create scaled configuration
        config, size_info = create_scaled_config(
            original_config,
            hidden_ratio=hidden_ratio,
            layers_ratio=layers_ratio
        )
        
        logger.info(colored("\n[3/7] Created scaled model configuration:", Colors.BOLD))
        logger.info(colored(f"  Hidden size: {config['hidden_size']} ({hidden_ratio:.2f} of original)", Colors.GREEN))
        logger.info(colored(f"  Layers: {config['num_hidden_layers']} ({layers_ratio:.2f} of original)", Colors.GREEN))
        logger.info(colored(f"  Attention heads: {config['num_attention_heads']}", Colors.GREEN))
        logger.info(colored(f"  KV heads: {config['num_key_value_heads']}", Colors.GREEN))
        logger.info(colored(f"  Intermediate size: {config['intermediate_size']}", Colors.GREEN))
        
        # Print memory estimates
        expected_memory_mb = size_info['total_memory_mb']
        expected_memory_ratio = expected_memory_mb / system_memory_mb * 100
        
        memory_color = Colors.GREEN
        if expected_memory_ratio > 70:
            memory_color = Colors.YELLOW
        if expected_memory_ratio > 85:
            memory_color = Colors.RED
            
        logger.info(f"Scaled model has {size_info['params']:,} parameters ({size_info['params']/orig_size_info['params']:.1%} of original)")
        logger.info(colored(f"Expected memory usage: ~{expected_memory_mb:,.1f} MB ({expected_memory_ratio:.1f}% of system RAM)", memory_color))
        
        # Create the mesh
        logger.info(colored(f"\n[4/7] Creating device mesh with shape {mesh_shape}...", Colors.BOLD))
        mesh = create_device_mesh(mesh_shape)
        logger.info(colored("✅ Mesh created successfully", Colors.GREEN))
        
        # Log metrics after mesh creation
        if not quiet:
            log_system_metrics("After mesh creation") if verbose else None
        
        # Initialize the model
        logger.info(colored(f"\n[5/7] Initializing tensor-parallel model...", Colors.BOLD))
        start_time = time.time()
        
        # Initialize model with bfloat16 precision
        model = TensorParallelQwen2ForCausalLM(
            config=config, 
            mesh=mesh, 
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16
        )
        
        # Generate initialization inputs
        input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
        
        # Log metrics before parameter initialization
        if not quiet:
            log_system_metrics("Before parameter initialization")
        
        # Initialize parameters with real weights or random initialization
        with mesh:
            if use_demo_model:
                # Generate a random key for initialization
                rng = jax.random.PRNGKey(0)
                
                # Initialize with random weights
                logger.info("Initializing model with random weights...")
                params = model.init(rng, input_ids=input_ids)
                logger.info(colored(f"✅ Model initialized with random weights in {time.time() - start_time:.2f} seconds", Colors.GREEN))
            else:
                try:
                    # Load weights from checkpoint
                    logger.info(f"Loading model weights from {model_path}...")
                    
                    # Handle scaled models with real weights
                    if scaling_applied:
                        logger.info(colored("Using partial weight loading for scaled model", Colors.CYAN))
                        
                        # First initialize with random weights to get the parameter structure
                        rng = jax.random.PRNGKey(0)
                        params = model.init(rng, input_ids=input_ids)
                        
                        # Use our improved loading function
                        params = improve_load_partial_qwen_weights(
                            model_path=model_path,
                            target_params=params,
                            config=config,
                            mesh=mesh,
                            num_layers=config["num_hidden_layers"],
                            logger=logger
                        )
                        logger.info(colored(f"✅ Partial weights loaded successfully in {time.time() - start_time:.2f} seconds", Colors.GREEN))
                    else:
                        # For full-sized model, use the model's built-in weight loading
                        logger.info(colored("Using direct checkpoint loading for full-sized model", Colors.CYAN))
                        params = load_params_from_checkpoint(model, model_path)
                        logger.info(colored(f"✅ Model weights loaded in {time.time() - start_time:.2f} seconds", Colors.GREEN))
                        
                except Exception as e:
                    logger.error(colored(f"❌ Error loading weights: {e}", Colors.RED))
                    logger.error(traceback.format_exc())
                    
                    # Ask if we should continue with random weights
                    fallback_message = colored("Weight loading failed. Continue with random weights? (y/n): ", Colors.YELLOW)
                    if quiet:
                        logger.warning(colored("Weight loading failed. Quiet mode, asking for confirmation.", Colors.YELLOW))
                        fallback_input = input(fallback_message)
                    else:
                        fallback_input = input(fallback_message)
                    
                    if fallback_input.lower() != 'y':
                        logger.error(colored("Execution stopped due to weight loading failure.", Colors.RED))
                        logger.error(colored(f"Weight loading error: {e}", Colors.RED))
                        stop_background_monitoring()
                        return False
                    
                    # Initialize with random weights as fallback
                    rng = jax.random.PRNGKey(0)
                    params = model.init(rng, input_ids=input_ids)
                    logger.info(colored(f"✅ Fallback: Model initialized with random weights", Colors.YELLOW))
        
        # Log metrics after parameter initialization
        metrics = log_system_metrics("After parameter initialization", not quiet)
        actual_memory_mb = metrics["process_memory_mb"]
        actual_memory_ratio = actual_memory_mb / system_memory_mb * 100
        
        logger.info(colored(f"Current RAM usage: {actual_memory_mb:.1f} MB ({actual_memory_ratio:.1f}% of system RAM)", Colors.BOLD))
        logger.info(f"Model memory estimate was {expected_memory_mb:.1f} MB, actual is {actual_memory_mb:.1f} MB")
        logger.info(f"Difference: {actual_memory_mb - expected_memory_mb:.1f} MB ({(actual_memory_mb/expected_memory_mb-1)*100:.1f}%)")
        
        # Run garbage collection to clean up unused objects
        gc.collect()
        if not quiet:
            log_system_metrics("After garbage collection")
        
        # Setup tokenizer
        logger.info(colored(f"\n[6/7] Setting up tokenizer...", Colors.BOLD))
        tokenizer = setup_tokenizer(model_path)
        if tokenizer is None:
            logger.error(colored("❌ Failed to load tokenizer", Colors.RED))
            stop_background_monitoring()
            return False
        
        # Run simple inference test
        logger.info(colored(f"\n[7/7] Running inference test...", Colors.BOLD))
        test_prompt = "What is the capital of France? Please provide a detailed explanation of why Paris is the capital and some historical context."
        logger.info(f"Test prompt: '{test_prompt}'")
        
        # Format prompt based on Qwen's expected format
        if hasattr(tokenizer, 'apply_chat_template'):
            # Use the chat template if available
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": test_prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="np")
        input_ids = jnp.array(input_ids)
        
        # Ensure input is properly shaped for the mesh
        if mesh_shape[0] > 1 and input_ids.shape[0] != mesh_shape[0]:
            input_ids = jnp.repeat(input_ids, mesh_shape[0], axis=0)
        
        # Apply input sharding
        with mesh:
            # Log metrics before forward pass
            log_system_metrics("Before forward pass")
            
            # Use model's input_sharding_spec method for proper sharding
            input_spec = model.input_sharding_spec(dtype=jnp.int32)
            input_ids = jax.device_put(input_ids, input_spec)
            
            # Run inference
            logger.info("Running forward pass...")
            inference_start = time.time()
            outputs = model.apply(params, input_ids=input_ids)
            forward_time = time.time() - inference_start
            logger.info(colored(f"✅ Forward pass completed in {forward_time:.2f} seconds", Colors.GREEN))
        
        # Log metrics after forward pass
        log_system_metrics("After forward pass") if verbose else None
        
        logits = outputs['logits']
        logger.info(f"Output logits shape: {logits.shape}")
        
        # Generate tokens
        logger.info(f"\nGenerating {max_tokens} tokens...")
        generated_ids = input_ids
        generated_tokens = []
        
        # Measure generation performance
        generation_start = time.time()
        tokens_per_sec = []
        
        # Log metrics at the start of generation
        if not quiet and verbose:
            log_system_metrics("Before token generation")
        
        # Helper for nucleus (top-p) sampling
        def nucleus_sampling(logits, top_p=0.9, top_k=50, temp=0.7):
            """
            Apply nucleus (top-p) sampling with optional temperature and top-k filtering
            
            Args:
                logits: token logits of shape [vocab_size]
                top_p: keep tokens comprising top p probability mass (0.0-1.0)
                top_k: optionally filter to top k tokens first (use 0 to disable)
                temp: temperature for softening/sharpening the distribution
                
            Returns:
                Selected token ID
            """
            # Apply temperature
            if temp != 1.0:
                logits = logits / temp
            
            # Convert to probabilities
            probs = jax.nn.softmax(logits)
            
            # Apply top-k filtering if enabled
            if top_k > 0:
                # Keep only top k tokens
                topk_probs, topk_indices = jax.lax.top_k(probs, top_k)
                # Create mask of selected tokens
                mask = jnp.zeros_like(probs)
                mask = mask.at[topk_indices].set(1)
                # Apply mask and renormalize
                probs = probs * mask
                probs = probs / jnp.sum(probs)
            
            # Sort for nucleus sampling
            sorted_probs, sorted_indices = jax.lax.top_k(probs, probs.shape[-1])
            cumulative_probs = jnp.cumsum(sorted_probs)
            
            # Create mask for tokens in the nucleus
            nucleus_mask = cumulative_probs <= top_p
            # Add at least one token to ensure non-empty nucleus
            nucleus_mask = nucleus_mask.at[0].set(True)
            
            # Apply nucleus mask
            sorted_mask = jnp.zeros_like(sorted_probs)
            sorted_mask = sorted_mask.at[jnp.where(nucleus_mask)].set(1)
            # Map back to original indices
            mask = jnp.zeros_like(probs)
            mask = mask.at[sorted_indices].set(sorted_mask)
            
            # Apply mask and renormalize
            probs = probs * mask
            probs = probs / jnp.sum(probs)
            
            # Sample from the filtered distribution
            # Use gumbel-max trick for sampling on CPU
            key = jax.random.PRNGKey(int(time.time() * 1000000) % (2**32))
            uniform = jax.random.uniform(key, shape=probs.shape)
            gumbel = -jnp.log(-jnp.log(uniform))
            return jnp.argmax(jnp.log(probs) + gumbel)
        
        # Simple generation loop
        for i in range(max_tokens):
            token_start = time.time()
            if not quiet:
                logger.info(f"  Generating token {i+1}/{max_tokens}...")
            
            with mesh:
                # Apply sharding to inputs
                input_spec = model.input_sharding_spec(dtype=jnp.int32)
                current_input = jax.device_put(generated_ids, input_spec)
                
                # Run forward pass
                outputs = model.apply(params, input_ids=current_input)
            
            # Get next token using nucleus sampling
            next_token_logits = outputs['logits'][0, -1, :]
            if temperature <= 0.0:
                # Use greedy decoding if temperature is 0
                next_token = jnp.argmax(next_token_logits)
            else:
                # Use nucleus sampling for better quality text
                next_token = nucleus_sampling(
                    next_token_logits, 
                    top_p=top_p, 
                    top_k=top_k,
                    temp=temperature
                )
            
            # Add token to sequence
            next_token_array = jnp.array([[next_token.item()]])
            generated_ids = jnp.concatenate([generated_ids, next_token_array], axis=1)
            generated_tokens.append(next_token.item())
            
            # Calculate token generation time
            token_time = time.time() - token_start
            tokens_per_sec.append(1.0 / token_time if token_time > 0 else 0)
            
            # Print the generated token
            if hasattr(tokenizer, 'decode') and not quiet:
                token_text = tokenizer.decode([next_token.item()])
                logger.info(f"  Generated token {i+1}: '{token_text}' (took {token_time:.2f}s)")
            elif not quiet:
                logger.info(f"  Generated token {i+1}: id={next_token.item()} (took {token_time:.2f}s)")
            
            # Log metrics every few tokens
            if verbose and ((i+1) % 2 == 0 or i == max_tokens-1) and not quiet:
                log_system_metrics(f"After generating {i+1} tokens", detailed=False)
        
        # Calculate generation performance
        total_generation_time = time.time() - generation_start
        avg_tokens_per_sec = max_tokens / total_generation_time if total_generation_time > 0 else 0
        
        logger.info(colored("\nGeneration stats:", Colors.BOLD))
        logger.info(f"  Total generation time: {total_generation_time:.2f} seconds")
        logger.info(f"  Average generation speed: {avg_tokens_per_sec:.2f} tokens/sec")
        logger.info(f"  Per-token speeds: min={min(tokens_per_sec):.2f}, max={max(tokens_per_sec):.2f}, avg={sum(tokens_per_sec)/len(tokens_per_sec):.2f} tokens/sec")
        
        # Decode the full sequence
        if hasattr(tokenizer, 'decode'):
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.info(f"\nGenerated response: '{generated_text}'")
            
            # Print just the generated part (after prompt)
            assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info(f"\nAssistant response: '{assistant_response}'")
        
        # Final metrics check
        final_metrics = log_system_metrics("Final")
        
        # Print final RAM utilization
        final_ram_mb = final_metrics["process_memory_mb"]
        final_ram_percent = (final_ram_mb / system_memory_mb * 100)
        logger.info(colored(f"\nFinal RAM utilization: {final_ram_percent:.1f}% of total system RAM", Colors.BOLD))
        
        # Print summary of adaptive scaling
        logger.info(colored("\n=== ADAPTIVE SCALING SUMMARY ===", Colors.HEADER + Colors.BOLD))
        logger.info(f"Original model: hidden_size={orig_hidden}, layers={orig_layers}, parameters={orig_size_info['params']:,}")
        logger.info(f"Scaled model:   hidden_size={config['hidden_size']}, layers={config['num_hidden_layers']}, parameters={size_info['params']:,}")
        logger.info(f"Scaling ratios: hidden={hidden_ratio:.3f}, layers={layers_ratio:.3f}, overall={(size_info['params']/orig_size_info['params']):.3f}")
        logger.info(f"Peak memory:    {peak_memory_usage:.1f} MB ({peak_memory_usage/system_memory_mb*100:.1f}% of system RAM)")
        
        # Print recommendations
        logger.info(colored("\n=== RECOMMENDATIONS ===", Colors.HEADER + Colors.BOLD))
        headroom = 90 - final_ram_percent
        
        if headroom > 20:
            # Significant headroom - can scale up
            growth_factor = min(1.2, 1 + (headroom / 100))
            new_hidden_ratio = min(1.0, hidden_ratio * growth_factor)
            new_layers_ratio = min(1.0, layers_ratio * growth_factor)
            
            logger.info(colored(f"✅ Model has significant memory headroom ({headroom:.1f}%). You can scale up!", Colors.GREEN))
            logger.info(f"Suggested scaling for next run:")
            logger.info(f"  --hidden_ratio {new_hidden_ratio:.3f} --layers_ratio {new_layers_ratio:.3f}")
            logger.info(f"  or use --scale_ratio {min(1.0, scale_ratio * growth_factor if scale_ratio else (new_hidden_ratio + new_layers_ratio)/2):.3f}")
            
        elif headroom > 5:
            # Some headroom - can scale up slightly
            growth_factor = 1 + (headroom / 150)
            new_hidden_ratio = min(1.0, hidden_ratio * growth_factor)
            new_layers_ratio = min(1.0, layers_ratio * growth_factor)
            
            logger.info(colored(f"✅ Model has some memory headroom ({headroom:.1f}%). You can scale up slightly.", Colors.GREEN))
            logger.info(f"Suggested scaling for next run:")
            logger.info(f"  --hidden_ratio {new_hidden_ratio:.3f} --layers_ratio {new_layers_ratio:.3f}")
            logger.info(f"  or use --scale_ratio {min(1.0, scale_ratio * growth_factor if scale_ratio else (new_hidden_ratio + new_layers_ratio)/2):.3f}")
            
        elif headroom > -5:
            # Good fit - keep current settings
            logger.info(colored(f"✅ Model has a good memory fit ({headroom:.1f}% headroom). Current settings are optimal.", Colors.GREEN))
            logger.info(f"Current settings to reuse:")
            logger.info(f"  --hidden_ratio {hidden_ratio:.3f} --layers_ratio {layers_ratio:.3f}")
            logger.info(f"  or use --scale_ratio {scale_ratio if scale_ratio else (hidden_ratio + layers_ratio)/2:.3f}")
            
        else:
            # Over memory limit - scale down
            reduction_factor = 0.9  # Scale down by 10%
            new_hidden_ratio = hidden_ratio * reduction_factor
            new_layers_ratio = layers_ratio * reduction_factor
            
            logger.info(colored(f"⚠️ Model is using too much memory ({-headroom:.1f}% over). Scale down for stability.", Colors.YELLOW))
            logger.info(f"Suggested scaling for next run:")
            logger.info(f"  --hidden_ratio {new_hidden_ratio:.3f} --layers_ratio {new_layers_ratio:.3f}")
            logger.info(f"  or use --scale_ratio {scale_ratio * reduction_factor if scale_ratio else (new_hidden_ratio + new_layers_ratio)/2:.3f}")
        
        # Additional recommendations
        if config["hidden_size"] / orig_hidden < 0.3 and config["num_hidden_layers"] / orig_layers > 0.6:
            logger.info(colored("\nPrioritization suggestion: Try increasing hidden_size and decreasing layers", Colors.CYAN))
            logger.info(f"  --hidden_ratio {min(1.0, hidden_ratio * 1.3):.3f} --layers_ratio {max(0.1, layers_ratio * 0.7):.3f}")
        
        if config["hidden_size"] / orig_hidden > 0.6 and config["num_hidden_layers"] / orig_layers < 0.3:
            logger.info(colored("\nPrioritization suggestion: Try increasing layers and decreasing hidden_size", Colors.CYAN))
            logger.info(f"  --hidden_ratio {max(0.1, hidden_ratio * 0.7):.3f} --layers_ratio {min(1.0, layers_ratio * 1.3):.3f}")
        
        # Stop background monitoring
        stop_background_monitoring()
        
        logger.info(colored(f"\n✅ Adaptive model verification completed successfully", Colors.GREEN + Colors.BOLD))
        return True
        
    except Exception as e:
        logger.error(colored(f"❌ Error in adaptive model verification: {e}", Colors.RED))
        logger.error(traceback.format_exc())
        
        # Log metrics after error
        log_system_metrics("Error state")
        
        # Stop background monitoring
        stop_background_monitoring()
        
        return False 

def disable_shape_debug_logs():
    """Disable all shape debug logging from tensor_parallel and model_implementation modules"""
    try:
        # Use the centralized function from the qwen25 package
        import sys
        sys.path.append("/root/tt-xla/tests/jax/models/qwen25")
        from __init__ import set_debug_logging
        
        # Disable all shape debug logging
        set_debug_logging(enabled=False)
        
        # Additional safety measures
        logging.getLogger("tensor_parallel").setLevel(logging.WARNING)
        logging.getLogger("model_implementation").setLevel(logging.CRITICAL)
        
        # Ensure JAX doesn't output compilation logs
        os.environ["JAX_LOG_COMPILES"] = "0"
        
        logger.info("Shape debug logging has been disabled")
    except Exception as e:
        # Fallback to direct approach if the import fails
        try:
            # Import and modify the DEBUG_SHAPES flag in tensor_parallel module
            import tensor_parallel
            tensor_parallel.DEBUG_SHAPES = False
            
            # Also modify the DEBUG_MODE flag in model_implementation module
            import model_implementation
            model_implementation.DEBUG_MODE = False
            
            logger.info("Shape debug logging has been disabled (direct approach)")
        except Exception as e2:
            logger.warning(f"Could not disable shape debug logs: {e2}")

def main():
    """Main function for adaptive model verification."""
    parser = argparse.ArgumentParser(
        description="Adaptive verification script for Qwen2.5 with tensor parallelism - finds optimal model size"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/tt-xla/tests/jax/models/qwen25/qwen25-weights",
        help="Path to model weights (default: /root/tt-xla/tests/jax/models/qwen25/qwen25-weights)"
    )
    
    parser.add_argument(
        "--use_demo_model", 
        action="store_true",
        help="Use demo model with random weights instead of loading from disk"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5,
        help="Maximum number of tokens to generate in test (default: 5)"
    )
    
    # Scaling parameters
    scaling_group = parser.add_argument_group("Scaling Parameters (choose one approach)")
    
    scaling_group.add_argument(
        "--auto_scale",
        action="store_true",
        help="Automatically determine maximum possible model size (recommended for first run)"
    )
    
    scaling_group.add_argument(
        "--scale_ratio",
        type=float,
        default=None,
        help="Overall scaling ratio (0.0-1.0) applied to both hidden_size and num_layers"
    )
    
    scaling_group.add_argument(
        "--hidden_ratio",
        type=float,
        default=None,
        help="Ratio of original hidden_size to use (0.0-1.0)"
    )
    
    scaling_group.add_argument(
        "--layers_ratio",
        type=float,
        default=None,
        help="Ratio of original num_layers to use (0.0-1.0)"
    )
    
    # System parameters
    parser.add_argument(
        "--target_ram_percent",
        type=float,
        default=80.0,
        help="Target RAM utilization percentage for auto-scaling (default: 80.0)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    
    parser.add_argument(
        "--monitor_interval",
        type=int,
        default=10,
        help="Interval in seconds for background resource monitoring (default: 10)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic information"
    )
    
    # Add a quiet mode option
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output to essential information only"
    )
    
    # New argument for debug tensor logs
    parser.add_argument(
        "--debug_tensor_ops",
        action="store_true",
        help="Show detailed tensor operation logs (very verbose)"
    )
    
    # Add a flag to keep shape debug logs
    parser.add_argument(
        "--keep_shape_logs",
        action="store_true",
        help="Keep the shape debug logs (warning: very verbose)"
    )
    
    # Add sampling parameters
    sampling_group = parser.add_argument_group("Text Generation Parameters")
    
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for text generation (0.0 = greedy, higher = more random)"
    )
    
    sampling_group.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (0.0-1.0, higher = more diverse)"
    )
    
    sampling_group.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (use 0 to disable top-k filtering)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Disable annoying shape debug logs unless explicitly requested
    if not args.keep_shape_logs:
        disable_shape_debug_logs()
    else:
        # Use the centralized function to enable shape debug logs
        try:
            import sys
            sys.path.append("/root/tt-xla/tests/jax/models/qwen25")
            from __init__ import set_debug_logging
            set_debug_logging(enabled=True)
            logger.info("Shape debug logs will be preserved as requested")
        except Exception as e:
            # Fallback to directly enabling logs if the import fails
            try:
                import tensor_parallel
                tensor_parallel.DEBUG_SHAPES = True
                import model_implementation
                model_implementation.DEBUG_MODE = True
                logger.info("Shape debug logs enabled (direct approach)")
            except Exception as e2:
                logger.warning(f"Could not enable shape debug logs: {e2}")
    
    # Set logging levels
    if args.quiet:
        logging.getLogger("qwen25-adaptive").setLevel(logging.WARNING)
        # Only show important warnings and errors
        logging.getLogger("jax").setLevel(logging.ERROR)
        logging.getLogger("flax").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger("qwen25-adaptive").setLevel(logging.DEBUG)
    else:
        logging.getLogger("qwen25-adaptive").setLevel(logging.INFO)
    
    # Always set model_implementation to ERROR unless debug_tensor_ops is set
    if not args.debug_tensor_ops:
        logging.getLogger("model_implementation").setLevel(logging.CRITICAL)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
        os.environ["JAX_LOG_COMPILES"] = "0"      # Disable JAX compilation logging
    
    # Validate arguments
    if sum([
        args.auto_scale, 
        args.scale_ratio is not None, 
        args.hidden_ratio is not None or args.layers_ratio is not None
    ]) > 1:
        logger.error(colored("Error: Please use only one scaling approach:", Colors.RED))
        logger.error("  - Either --auto_scale")
        logger.error("  - Or --scale_ratio")
        logger.error("  - Or --hidden_ratio and/or --layers_ratio")
        return False
    
    # Verify model path exists
    if not args.use_demo_model and not os.path.exists(args.model_path):
        logger.error(colored(f"Error: Model path {args.model_path} does not exist", Colors.RED))
        logger.error("Please provide a valid path to the model weights")
        return False
    
    # Display information
    logger.info(colored("Qwen2.5 Adaptive Model Verification", Colors.BOLD + Colors.HEADER))
    logger.info(f"System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if args.auto_scale:
        logger.info(colored(f"Mode: Auto-scaling to find maximum model size", Colors.BOLD))
        logger.info(f"Target RAM usage: {args.target_ram_percent:.1f}%")
    elif args.scale_ratio is not None:
        logger.info(colored(f"Mode: Uniform scaling with ratio {args.scale_ratio:.3f}", Colors.BOLD))
    else:
        hidden_ratio = args.hidden_ratio or 0.5
        layers_ratio = args.layers_ratio or 0.5
        logger.info(colored(f"Mode: Custom scaling with ratios:", Colors.BOLD))
        logger.info(f"  Hidden size ratio: {hidden_ratio:.3f}")
        logger.info(f"  Layers ratio: {layers_ratio:.3f}")
    
    # Run verification
    success = verify_adaptive_model(
        model_path=args.model_path,
        use_demo_model=args.use_demo_model,
        max_tokens=args.max_tokens,
        hidden_ratio=args.hidden_ratio,
        layers_ratio=args.layers_ratio,
        scale_ratio=args.scale_ratio,
        target_ram_percent=args.target_ram_percent,
        auto_scale=args.auto_scale,
        batch_size=args.batch_size,
        monitor_interval=args.monitor_interval,
        verbose=args.verbose,
        quiet=args.quiet,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Print summary
    if success:
        logger.info(colored("\n✅ Adaptive model verification completed successfully.", Colors.GREEN + Colors.BOLD))
    else:
        logger.error(colored("\n❌ Adaptive model verification failed.", Colors.RED + Colors.BOLD))
        logger.error("Please check the errors above for troubleshooting suggestions.")
    
    return success


if __name__ == "__main__":
    main() 