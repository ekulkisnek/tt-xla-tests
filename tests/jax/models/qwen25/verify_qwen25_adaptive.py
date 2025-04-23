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
import re

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
    map_parameter_paths,
)
from config import load_qwen_config, get_qwen2_7b_config
from weight_loading import load_qwen_weights, load_partial_qwen_weights, init_model_from_weights
from weight_diagnostics import create_parameter_structure_report, fix_parameter_structure

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

# Memory tracking class for more proactive memory management
class MemoryTracker:
    """Memory tracker to monitor and manage memory usage during model operations."""
    
    def __init__(self, gc_threshold_mb=500, critical_threshold_mb=0, 
                polling_interval=5, logger=None):
        """
        Initialize memory tracker.
        
        Args:
            gc_threshold_mb: Memory increase threshold to trigger garbage collection (MB)
            critical_threshold_mb: Critical memory threshold (MB, 0 for auto-detection)
            polling_interval: Background polling interval in seconds
            logger: Logger instance for output
        """
        self.last_tracked_memory = 0
        self.gc_threshold_mb = gc_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.memory_log = []
        self.peak_memory = 0
        self.logger = logger or logging.getLogger("memory-tracker")
        self.tracking_active = False
        self.tracking_thread = None
        self.polling_interval = polling_interval
        
        # Auto-detect critical threshold if not set
        if self.critical_threshold_mb <= 0:
            system_memory = psutil.virtual_memory()
            # Set critical threshold to 85% of system memory
            self.critical_threshold_mb = int(system_memory.total / (1024 * 1024) * 0.85)
            self.logger.info(f"Auto-detected critical memory threshold: {self.critical_threshold_mb} MB")
    
    def start_tracking(self):
        """Start background memory tracking."""
        if self.tracking_active:
            return
            
        self.tracking_active = True
        self.tracking_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.tracking_thread.start()
        self.logger.info(f"Started memory tracking (GC threshold: {self.gc_threshold_mb} MB, "
                        f"critical threshold: {self.critical_threshold_mb} MB)")
    
    def stop_tracking(self):
        """Stop background memory tracking."""
        if not self.tracking_active:
            return
            
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1)
            self.tracking_thread = None
        
        self.logger.info(f"Stopped memory tracking. Peak memory: {self.peak_memory:.1f} MB")
        return self.memory_log
    
    def _monitor_memory(self):
        """Background thread to monitor memory usage."""
        while self.tracking_active:
            try:
                # Check current memory
                current_memory = self._get_current_memory()
                
                # Update peak memory
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                # Check if we're approaching critical memory
                if current_memory > self.critical_threshold_mb * 0.9:  # Within 90% of critical
                    self.logger.warning(f"Memory usage nearing critical threshold: "
                                      f"{current_memory:.1f} MB / {self.critical_threshold_mb} MB")
                    # Force garbage collection
                    collected = gc.collect()
                    self.logger.info(f"Emergency garbage collection freed {collected} objects")
                
                # Add to log
                self.memory_log.append((time.time(), current_memory))
                
                # Sleep for the polling interval
                time.sleep(self.polling_interval)
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.polling_interval * 2)  # Sleep longer on error
    
    def _get_current_memory(self):
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except Exception:
            return 0
    
    def check_memory(self, operation_name, force_gc=False):
        """
        Check current memory usage and optionally trigger garbage collection.
        
        Args:
            operation_name: Name of the current operation for logging
            force_gc: Whether to force garbage collection
            
        Returns:
            Current memory usage in MB
        """
        # Get current memory usage
        current_memory = self._get_current_memory()
        
        # Calculate memory delta since last check
        memory_delta = current_memory - self.last_tracked_memory
        
        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        # Log the memory check
        self.memory_log.append((time.time(), operation_name, current_memory, memory_delta))
        
        # Check if we should run garbage collection
        should_collect = force_gc or memory_delta > self.gc_threshold_mb
        emergency_collect = current_memory > self.critical_threshold_mb
        
        if should_collect or emergency_collect:
            gc_type = "Emergency" if emergency_collect else "Routine"
            self.logger.info(f"{gc_type} garbage collection triggered during {operation_name} "
                           f"(current: {current_memory:.1f} MB, delta: {memory_delta:+.1f} MB)")
            
            # Run garbage collection
            gc_start = time.time()
            collected = gc.collect(generation=2)  # Full collection
            gc_time = time.time() - gc_start
            
            # Check memory after collection
            new_memory = self._get_current_memory()
            freed_mb = current_memory - new_memory
            
            self.logger.info(f"Garbage collection freed {freed_mb:.1f} MB ({collected} objects) "
                           f"in {gc_time:.2f}s")
            
            # Update current memory
            current_memory = new_memory
        
        # Update last tracked memory
        self.last_tracked_memory = current_memory
        
        return current_memory

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

def improved_memory_estimation(config, system_memory_mb, target_usage_ratio=0.8):
    """
    More accurate memory estimation based on model structure and scaling.
    
    Args:
        config: Model configuration
        system_memory_mb: Total system memory in MB
        target_usage_ratio: Target memory usage ratio (0.0-1.0)
        
    Returns:
        Dictionary with memory estimates
    """
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    vocab_size = config.get("vocab_size", 152064)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    
    # Base process memory requirements (more conservative estimate)
    base_process_mb = 1000  # 1GB base process memory
    
    # Available memory for model
    available_memory_mb = system_memory_mb * target_usage_ratio - base_process_mb
    
    # Calculate parameter memory
    # Each parameter in BF16 is 2 bytes
    parameter_bytes = calculate_model_size(hidden_size, num_layers, vocab_size, 
                                         intermediate_size // hidden_size)["params"] * 2  # BF16
    parameter_memory_mb = parameter_bytes / (1024 * 1024)
    
    # Calculate overhead with dynamic scaling based on model size
    # Smaller models have relatively higher overhead
    if hidden_size <= 1024:
        # Small models have higher relative overhead
        activation_overhead = 2.0  # 100% overhead
        compilation_overhead = 0.3  # 30% additional for compilation
    elif hidden_size <= 2048:
        activation_overhead = 1.8  # 80% overhead
        compilation_overhead = 0.25  # 25% additional for compilation
    elif hidden_size <= 4096:
        activation_overhead = 1.6  # 60% overhead
        compilation_overhead = 0.2  # 20% additional for compilation
    else:
        activation_overhead = 1.5  # 50% overhead
        compilation_overhead = 0.15  # 15% additional for compilation
    
    # Mesh operations add additional overhead for tensor parallelism
    tensor_parallel_overhead = 0.1  # 10% additional overhead for tensor parallelism
    
    # Activation memory
    activation_memory_mb = parameter_memory_mb * activation_overhead
    
    # Compilation memory spike
    compilation_spike_mb = parameter_memory_mb * compilation_overhead
    
    # Tensor parallelism overhead
    tensor_parallel_mb = parameter_memory_mb * tensor_parallel_overhead
    
    # Total memory estimate
    total_estimated_mb = parameter_memory_mb + activation_memory_mb + compilation_spike_mb + tensor_parallel_mb
    
    # Peak memory estimate (during weight loading and compilation)
    peak_memory_mb = total_estimated_mb * 1.1  # Add 10% safety margin for peak usage
    
    # Check if we have enough memory
    has_enough_memory = peak_memory_mb < available_memory_mb
    
    # Calculate the maximum possible scaling factor
    max_scaling_factor = (available_memory_mb / peak_memory_mb) ** 0.5 if peak_memory_mb > 0 else 0
    
    # Apply a safety margin to the scaling factor
    safe_scaling_factor = max_scaling_factor * 0.9  # 10% safety margin
    
    return {
        "parameter_memory_mb": parameter_memory_mb,
        "activation_memory_mb": activation_memory_mb,
        "compilation_spike_mb": compilation_spike_mb,
        "tensor_parallel_mb": tensor_parallel_mb,
        "total_estimated_mb": total_estimated_mb,
        "peak_memory_mb": peak_memory_mb,
        "available_memory_mb": available_memory_mb,
        "has_enough_memory": has_enough_memory,
        "max_scaling_factor": safe_scaling_factor,
    }

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
    from flax.core.frozen_dict import freeze, unfreeze
    
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
    
    # Add diagnostic function to inspect weight files
    def inspect_weight_file(file_path, log_all=False):
        """Print parameter names in a weight file, focusing on lm_head related parameters."""
        try:
            with safe_open(file_path, framework="numpy") as f:
                weight_keys = list(f.keys())
                
                # Count how many keys contain certain patterns
                patterns = {
                    "lm_head": [],
                    "embed_tokens": [],
                    "norm.weight": []
                }
                
                # Categorize keys by pattern
                for key in weight_keys:
                    for pattern in patterns:
                        if pattern in key:
                            patterns[pattern].append(key)
                
                # Log summary
                log_info(f"Inspecting {os.path.basename(file_path)} - {len(weight_keys)} parameters:")
                for pattern, keys in patterns.items():
                    if keys:
                        log_info(f"  Found {len(keys)} keys matching '{pattern}':")
                        for key in keys:
                            tensor = f.get_tensor(key)
                            log_info(f"    - {key} (shape: {tensor.shape})")
                
                # Optionally log all keys
                if log_all:
                    log_info("  All keys:")
                    for key in weight_keys:
                        log_info(f"    - {key}")
                        
                return patterns
        except Exception as e:
            log_info(f"Error inspecting {file_path}: {e}")
            return {}
    
    # Inspect all weight files to understand what's available
    log_info(colored("Inspecting weight files for important parameters...", Colors.BOLD))
    all_lm_head_keys = []
    all_embed_tokens_keys = []
    
    for file_path in weight_files:
        patterns = inspect_weight_file(file_path, log_all=False)
        all_lm_head_keys.extend(patterns.get("lm_head", []))
        all_embed_tokens_keys.extend(patterns.get("embed_tokens", []))
    
    # Log summary of findings
    log_info(colored("Weight file inspection summary:", Colors.BOLD))
    log_info(f"  Found {len(all_lm_head_keys)} lm_head related parameters:")
    for key in all_lm_head_keys:
        log_info(f"    - {key}")
    log_info(f"  Found {len(all_embed_tokens_keys)} embed_tokens related parameters:")
    for key in all_embed_tokens_keys:
        log_info(f"    - {key}")
    
    # Target sizes from reduced config
    target_hidden_size = config["hidden_size"]
    target_layers = config["num_hidden_layers"]
    
    # More robust tensor resizing with SVD for weight matrices
    def resize_tensor(tensor, target_shape):
        """Resize tensor to target shape with better quality for weight matrices"""
        if tensor.shape == target_shape:
            return tensor
            
        if len(tensor.shape) != len(target_shape):
            log_info(f"Incompatible shapes for resizing: {tensor.shape} vs {target_shape}", verbose_only=True)
            return np.zeros(target_shape, dtype=tensor.dtype)
        
        # Special handling for 2D weight matrices (use SVD for important dimensions)
        if len(tensor.shape) == 2:
            src_rows, src_cols = tensor.shape
            tgt_rows, tgt_cols = target_shape
            
            # Memory-efficient approach for large matrices
            # Lower threshold from 10000 to 5000 to avoid memory issues
            if src_rows > 5000 or src_cols > 5000:
                # Use chunked SVD approach for very large matrices
                return chunked_matrix_resize(tensor, target_shape)
            else:
                try:
                    # For smaller matrices, use SVD for better quality
                    # Free memory before SVD calculation
                    gc.collect()
                    
                    # Compute SVD of the source matrix
                    try:
                        u, s, vh = np.linalg.svd(tensor, full_matrices=False)
                        
                        # Determine rank to keep
                        k = min(len(s), tgt_rows, tgt_cols)
                        
                        # Resize u, s, vh
                        u_resized = np.zeros((tgt_rows, k), dtype=u.dtype)
                        vh_resized = np.zeros((k, tgt_cols), dtype=vh.dtype)
                        
                        # Copy values
                        u_resized[:min(src_rows, tgt_rows), :k] = u[:min(src_rows, tgt_rows), :k]
                        vh_resized[:k, :min(src_cols, tgt_cols)] = vh[:k, :min(src_cols, tgt_cols)]
                        
                        # Clear original matrices to free memory
                        del u, s 
                        
                        # Reconstruct the matrix in chunks to save memory
                        tensor_resized = np.zeros(target_shape, dtype=tensor.dtype)
                        
                        # Process in smaller chunks to reduce peak memory
                        chunk_size = 1000
                        for i in range(0, tgt_rows, chunk_size):
                            end_i = min(i + chunk_size, tgt_rows)
                            tensor_resized[i:end_i, :] = np.matmul(
                                u_resized[i:end_i, :] * s[:k], 
                                vh_resized
                            )
                            
                        # Free temp variables
                        del u_resized, vh_resized
                        gc.collect()
                        
                        return tensor_resized.astype(tensor.dtype)
                        
                    except np.linalg.LinAlgError:
                        # Fallback to truncation if SVD fails
                        log_info(f"SVD failed, falling back to truncation for {tensor.shape} -> {target_shape}", verbose_only=True)
                        return simple_truncation_resize(tensor, target_shape)
                except Exception as e:
                    log_info(f"Error during SVD resizing: {e}, falling back to simple resizing", verbose_only=True)
                    # Fallback to simple resizing
                    return simple_truncation_resize(tensor, target_shape)
        
        # For non-2D tensors, use simple truncation/padding
        return simple_truncation_resize(tensor, target_shape)
    
    def chunked_matrix_resize(tensor, target_shape):
        """Memory-efficient matrix resizing for very large matrices using chunking"""
        src_rows, src_cols = tensor.shape
        tgt_rows, tgt_cols = target_shape
        
        # Pre-allocate result
        tensor_resized = np.zeros(target_shape, dtype=tensor.dtype)
        
        # For very large matrices, process in smaller chunks
        max_chunk_size = 3000  # Reduced from 5000 to be even more memory-efficient
        
        # Determine if we should chunk by rows, columns, or both
        chunk_rows = src_rows > max_chunk_size
        chunk_cols = src_cols > max_chunk_size
        
        # If only one dimension is large, chunk along that dimension
        if chunk_rows and not chunk_cols:
            log_info(f"Chunking large matrix ({src_rows}x{src_cols}) by rows", verbose_only=True)
            
            # Calculate chunk sizes
            num_chunks = (src_rows + max_chunk_size - 1) // max_chunk_size
            
            for i in range(num_chunks):
                # Calculate chunk boundaries
                start_row = i * max_chunk_size
                end_row = min(src_rows, (i + 1) * max_chunk_size)
                target_end_row = min(tgt_rows, end_row)
                
                if start_row >= tgt_rows:
                    break  # Beyond target size
                
                # Extract and process chunk
                chunk = tensor[start_row:end_row, :]
                
                # Resize chunk to target width but keep same height
                chunk_target_shape = (end_row - start_row, tgt_cols)
                if chunk.shape[1] != tgt_cols:
                    # Use simple truncation for the row chunks
                    chunk_resized = simple_truncation_resize(chunk, chunk_target_shape)
                else:
                    chunk_resized = chunk
                
                # Copy to result tensor
                tensor_resized[start_row:target_end_row, :] = chunk_resized[:target_end_row-start_row, :]
                
                # Force garbage collection after processing each chunk
                del chunk, chunk_resized
                gc.collect()
                
        # If columns need chunking
        elif chunk_cols and not chunk_rows:
            log_info(f"Chunking large matrix ({src_rows}x{src_cols}) by columns", verbose_only=True)
            
            # Calculate chunk sizes
            num_chunks = (src_cols + max_chunk_size - 1) // max_chunk_size
            
            for i in range(num_chunks):
                # Calculate chunk boundaries
                start_col = i * max_chunk_size
                end_col = min(src_cols, (i + 1) * max_chunk_size)
                target_end_col = min(tgt_cols, end_col)
                
                if start_col >= tgt_cols:
                    break  # Beyond target size
                
                # Extract and process chunk
                chunk = tensor[:, start_col:end_col]
                
                # Resize chunk to target height but keep same width
                chunk_target_shape = (tgt_rows, end_col - start_col)
                if chunk.shape[0] != tgt_rows:
                    # Use simple truncation for the column chunks
                    chunk_resized = simple_truncation_resize(chunk, chunk_target_shape)
                else:
                    chunk_resized = chunk
                
                # Copy to result tensor
                tensor_resized[:, start_col:target_end_col] = chunk_resized[:, :target_end_col-start_col]
                
                # Force garbage collection after processing each chunk
                del chunk, chunk_resized
                gc.collect()
                
        # If both dimensions are large, use a simple approach to avoid excessive chunking
        else:
            log_info(f"Matrix too large in both dimensions ({src_rows}x{src_cols}), using simple resize", verbose_only=True)
            tensor_resized = simple_truncation_resize(tensor, target_shape)
        
        return tensor_resized
    
    def simple_truncation_resize(tensor, target_shape):
        """Simple resize by truncation or padding"""
        tensor_resized = np.zeros(target_shape, dtype=tensor.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
        tensor_view = tensor[slices]
        target_slices = tuple(slice(0, s.stop) for s in slices)
        tensor_resized[target_slices] = tensor_view
        return tensor_resized
    
    # Create parameter name mapping dictionary
    param_name_mappings = {
        # Embedding mappings
        "model.embed_tokens.weight": ["transformer.embed_tokens.embedding", "model.embed_tokens.weight", "params.transformer.embed_tokens.weight"],
        
        # Attention mappings
        "model.layers.{}.self_attn.q_proj.weight": ["transformer.h.{}.attn.q.kernel", "model.layers.{}.self_attn.q_proj.kernel", "params.transformer.layers_{}.self_attn.q_proj.kernel"],
        "model.layers.{}.self_attn.k_proj.weight": ["transformer.h.{}.attn.k.kernel", "model.layers.{}.self_attn.k_proj.kernel", "params.transformer.layers_{}.self_attn.k_proj.kernel"],
        "model.layers.{}.self_attn.v_proj.weight": ["transformer.h.{}.attn.v.kernel", "model.layers.{}.self_attn.v_proj.kernel", "params.transformer.layers_{}.self_attn.v_proj.kernel"],
        "model.layers.{}.self_attn.o_proj.weight": ["transformer.h.{}.attn.o.kernel", "model.layers.{}.self_attn.o_proj.kernel", "params.transformer.layers_{}.self_attn.o_proj.kernel"],
        
        # MLP mappings
        "model.layers.{}.mlp.gate_proj.weight": ["transformer.h.{}.mlp.w1.kernel", "model.layers.{}.mlp.gate_proj.kernel", "params.transformer.layers_{}.mlp.gate_proj.kernel"],
        "model.layers.{}.mlp.up_proj.weight": ["transformer.h.{}.mlp.w2.kernel", "model.layers.{}.mlp.up_proj.kernel", "params.transformer.layers_{}.mlp.up_proj.kernel"],
        "model.layers.{}.mlp.down_proj.weight": ["transformer.h.{}.mlp.w3.kernel", "model.layers.{}.mlp.down_proj.kernel", "params.transformer.layers_{}.mlp.down_proj.kernel"],
        
        # Layer norm mappings 
        "model.layers.{}.input_layernorm.weight": ["transformer.h.{}.ln_1.weight", "model.layers.{}.input_layernorm.weight", "params.transformer.layers_{}.input_layernorm.weight"],
        "model.layers.{}.post_attention_layernorm.weight": ["transformer.h.{}.ln_2.weight", "model.layers.{}.post_attention_layernorm.weight", "params.transformer.layers_{}.post_attention_layernorm.weight"],
        "model.norm.weight": ["transformer.ln_f.weight", "model.norm.weight", "params.transformer.norm.weight"],
        
        # LM head mappings - Add multiple potential mappings for the lm_head
        "lm_head.weight": ["lm_head.kernel", "lm_head.weight", "params.lm_head.kernel"],
        # Try additional formats that might be in the weights
        "model.lm_head.weight": ["lm_head.kernel", "lm_head.weight", "params.lm_head.kernel"],
        "transformer.lm_head.weight": ["lm_head.kernel", "lm_head.weight", "params.lm_head.kernel"]
    }
    
    # For flattened parameter path lookup
    flat_target_params = flatten_dict(new_params)
    
    # Track statistics
    total_weights = 0
    loaded_weights = 0
    resized_weights = 0
    not_found_weights = 0
    param_stats = {
        "embed": {"found": 0, "not_found": 0},
        "layers": {"found": 0, "not_found": 0},
        "norm": {"found": 0, "not_found": 0},
        "lm_head": {"found": 0, "not_found": 0},
        "other": {"found": 0, "not_found": 0}
    }
    
    # Load weights from each file
    for file_path in weight_files:
        log_info(f"Processing {os.path.basename(file_path)}...")
        
        with safe_open(file_path, framework="numpy") as f:
            weight_keys = f.keys()
            
            for key in weight_keys:
                total_weights += 1
                
                # Determine weight category for stats
                category = "other"
                if "embed_tokens" in key:
                    category = "embed"
                elif "layers" in key:
                    category = "layers"
                elif "norm" in key:
                    category = "norm"
                elif "lm_head" in key:
                    category = "lm_head"
                
                # Extract layer number for layer-specific weights
                if ".layers." in key:
                    parts = key.split(".layers.")
                    if len(parts) > 1:
                        layer_parts = parts[1].split(".")
                        try:
                            layer_num = int(layer_parts[0])
                            # Skip layers beyond our target layer count
                            if layer_num >= target_layers:
                                log_info(f"Skipping {key} (layer {layer_num} > target {target_layers})", verbose_only=True)
                                param_stats[category]["not_found"] += 1
                                continue
                        except ValueError:
                            pass
                
                # Get the tensor
                tensor = f.get_tensor(key)
                
                # Try different parameter name mappings
                found = False
                
                # Check for layer-specific parameter
                if ".layers." in key:
                    # Extract layer number for formatting
                    layer_match = re.search(r"\.layers\.(\d+)\.", key)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        
                        # Generate template key without layer number
                        template_key = key.replace(f".layers.{layer_idx}.", ".layers.{}.")
                        
                        # Check if we have a mapping for this template
                        if template_key in param_name_mappings:
                            # Try each possible target mapping with the layer number filled in
                            for target_template in param_name_mappings[template_key]:
                                target_key = target_template.format(layer_idx)
                                
                                # Convert to path components
                                target_path = target_key.split(".")
                                
                                # Navigate to parameter location
                                current = new_params
                                path_found = True
                                for i, part in enumerate(target_path):
                                    if part in current:
                                        if i == len(target_path) - 1:  # Last part
                                            # Found the target - now resize if needed
                                            target_shape = current[part].shape
                                            if tensor.shape != target_shape:
                                                log_info(f"Resizing {key} -> {target_key}: {tensor.shape} -> {target_shape}", verbose_only=True)
                                                resized_weights += 1
                                                
                                                # Use improved resizing
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
                                                param_stats[category]["found"] += 1
                                                log_info(f"Loaded {key} -> {target_key}", verbose_only=True)
                                                found = True
                                                break  # Break out of the target_template loop
                                            except Exception as e:
                                                log_info(f"Error setting {target_key}: {e}", verbose_only=True)
                                        else:
                                            current = current[part]
                                    else:
                                        path_found = False
                                        break
                                
                                if found:
                                    break  # Break out of the target_template loop if we found a match
                
                # Non-layer specific parameters (embedding, final norm, lm_head)
                if not found:
                    # Check simple key as-is
                    if key in param_name_mappings:
                        for target_key in param_name_mappings[key]:
                            target_path = target_key.split(".")
                            
                            # Navigate to parameter location
                            current = new_params
                            path_found = True
                            for i, part in enumerate(target_path):
                                if part in current:
                                    if i == len(target_path) - 1:  # Last part
                                        # Found the target - now resize if needed
                                        target_shape = current[part].shape
                                        if tensor.shape != target_shape:
                                            log_info(f"Resizing {key} -> {target_key}: {tensor.shape} -> {target_shape}", verbose_only=True)
                                            resized_weights += 1
                                            
                                            # Use improved resizing
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
                                            param_stats[category]["found"] += 1
                                            log_info(f"Loaded {key} -> {target_key}", verbose_only=True)
                                            found = True
                                            break  # Break out of the target_key loop
                                        except Exception as e:
                                            log_info(f"Error setting {target_key}: {e}", verbose_only=True)
                                    else:
                                        current = current[part]
                                else:
                                    path_found = False
                                    break
                            
                            if found:
                                break  # Break out of the target_key loop if we found a match
                
                # Original parameter name conversion and lookup logic (as fallback)
                if not found:
                    # Try to find the corresponding key in our target params
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
                    path_found = True
                    
                    for i, part in enumerate(param_path):
                        if part in current:
                            if i == len(param_path) - 1:  # Last part
                                # Found the target - now resize if needed
                                target_shape = current[part].shape
                                if tensor.shape != target_shape:
                                    log_info(f"Resizing {key} -> {param_key}: {tensor.shape} -> {target_shape}", verbose_only=True)
                                    resized_weights += 1
                                    
                                    # Use improved resizing
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
                                    param_stats[category]["found"] += 1
                                    log_info(f"Loaded {key} -> {param_key}", verbose_only=True)
                                    found = True
                                except Exception as e:
                                    log_info(f"Error setting {param_key}: {e}", verbose_only=True)
                            else:
                                current = current[part]
                        else:
                            path_found = False
                            break
                
                # Try with params prefix
                if not found:
                    # This is already handled in the mappings above
                    not_found_weights += 1
                    log_info(f"Could not find parameter mapping for {key}", verbose_only=True)
    
    # Summary stats
    log_info(f"Weight loading summary:")
    log_info(f"  Total weights processed: {total_weights}")
    log_info(f"  Weights loaded successfully: {loaded_weights} ({(loaded_weights/total_weights*100):.1f}%)")
    log_info(f"  Weights that required resizing: {resized_weights}")
    log_info(f"  Weights not found in target model: {not_found_weights} ({(not_found_weights/total_weights*100):.1f}%)")
    
    # Detailed stats by category
    log_info(f"Weight loading by category:")
    for category, stats in param_stats.items():
        total_cat = stats["found"] + stats["not_found"]
        if total_cat > 0:
            found_pct = stats["found"] / total_cat * 100
            log_info(f"  {category}: {stats['found']}/{total_cat} loaded ({found_pct:.1f}%)")
    
    # Check if LM head is missing and initialize it from embeddings if needed
    lm_head_found = param_stats["lm_head"]["found"] > 0
    
    # If LM head not found, try to initialize from embeddings
    if not lm_head_found:
        try:
            flat_params = flatten_dict(new_params)
            log_info(colored("LM head weights not found, initializing from embedding weights", Colors.YELLOW))
            
            # Try to find embedding weights
            embed_weights = None
            for path, param in flat_params.items():
                path_str = '.'.join(str(p) for p in path)
                if 'embed_tokens' in path_str and ('embedding' in path_str or 'weight' in path_str):
                    embed_weights = param
                    log_info(f"Found embedding weights at {path_str}, shape: {embed_weights.shape}")
                    break
            
            if embed_weights is not None:
                # Create lm_head with embedding weights
                if 'lm_head' in new_params:
                    if 'kernel' in new_params['lm_head']:
                        log_info("Setting lm_head.kernel from embeddings")
                        new_params['lm_head']['kernel'] = embed_weights
                    else:
                        log_info("Creating lm_head.kernel from embeddings")
                        new_params['lm_head']['kernel'] = embed_weights
                else:
                    log_info("Creating full lm_head structure from embeddings")
                    new_params['lm_head'] = {'kernel': embed_weights}
                
                log_info(colored("Successfully initialized LM head from embeddings", Colors.GREEN))
            else:
                log_info(colored("Could not find embedding weights to initialize LM head", Colors.RED))
        except Exception as e:
            log_info(f"Error while checking/fixing LM head: {e}")
    else:
        # LM head found but might be in a different structure than expected
        # Check for common mapping issues and fix them
        log_info(colored("LM head found, checking structure...", Colors.BOLD))
        
        # Get the lm_head weights
        lm_head_weight = None
        src_path = None
        
        try:
            flat_params = flatten_dict(new_params)
            
            # Check all parameters for anything that looks like lm_head
            for path, param in flat_params.items():
                path_str = '.'.join(str(p) for p in path)
                if 'lm_head' in path_str and ('weight' in path_str or 'kernel' in path_str):
                    lm_head_weight = param
                    src_path = path_str
                    log_info(f"Found lm_head weights at {path_str}, shape: {param.shape}")
                    break
            
            if lm_head_weight is not None:
                # Ensure it's in the correct structure expected by TensorParallelQwen2ForCausalLM
                # The expected path is ['params', 'lm_head', 'kernel'] or directly ['lm_head', 'kernel']
                if 'params' in new_params and 'lm_head' not in new_params['params']:
                    log_info("Adding lm_head to params collection")
                    new_params['params']['lm_head'] = {'kernel': lm_head_weight}
                elif 'lm_head' not in new_params:
                    log_info("Adding lm_head to top-level params")
                    new_params['lm_head'] = {'kernel': lm_head_weight}
                elif 'kernel' not in new_params['lm_head']:
                    log_info("Adding kernel to lm_head")
                    new_params['lm_head']['kernel'] = lm_head_weight
                
                # Alternate fix: ensure params.params.lm_head.kernel exists (sometimes needed for flax)
                if 'params' in new_params and 'params' not in new_params['params']:
                    new_params['params']['params'] = {}
                
                if 'params' in new_params and 'params' in new_params['params'] and 'lm_head' not in new_params['params']['params']:
                    log_info("Adding lm_head to params.params collection")
                    new_params['params']['params']['lm_head'] = {'kernel': lm_head_weight}
                
                log_info(colored("Successfully restructured LM head", Colors.GREEN))
            else:
                log_info(colored("Could not find LM head weights despite loading success", Colors.RED))
        except Exception as e:
            log_info(f"Error fixing lm_head structure: {e}")
    
    # Add a direct fix for the most common problem pattern
    try:
        # Ensure the lm_head is in the correct location expected by the model
        if 'params' in new_params:
            if 'lm_head' in new_params and 'kernel' in new_params['lm_head']:
                log_info("Copying lm_head from top level to params collection")
                if 'lm_head' not in new_params['params']:
                    new_params['params']['lm_head'] = {}
                new_params['params']['lm_head']['kernel'] = new_params['lm_head']['kernel']
            
            # Additionally check the structure
            flat_params = flatten_dict(new_params)
            for path, param in flat_params.items():
                if 'lm_head' in str(path) and 'kernel' in str(path):
                    log_info(f"Verified lm_head.kernel exists at: {'.'.join(str(p) for p in path)}")
    except Exception as e:
        log_info(f"Error during direct parameter structure fix: {e}")

    # Check if embeddings are missing but lm_head is available
    embed_found = param_stats["embed"]["found"] > 0
    if not embed_found and lm_head_found:
        try:
            flat_params = flatten_dict(new_params)
            log_info(colored("Embeddings not found, initializing from lm_head weights", Colors.YELLOW))
            
            # Find lm_head weights
            lm_head_weight = None
            for path, param in flat_params.items():
                path_str = '.'.join(str(p) for p in path)
                if 'lm_head' in path_str and ('weight' in path_str or 'kernel' in path_str):
                    lm_head_weight = param
                    log_info(f"Found lm_head weights at {path_str}, shape: {lm_head_weight.shape}")
                    break
            
            if lm_head_weight is not None:
                # Create embeddings from lm_head weights
                if 'transformer' in new_params:
                    if 'embed_tokens' not in new_params['transformer']:
                        log_info("Creating transformer.embed_tokens from lm_head")
                        new_params['transformer']['embed_tokens'] = {'embedding': lm_head_weight}
                    else:
                        if 'embedding' not in new_params['transformer']['embed_tokens']:
                            log_info("Creating transformer.embed_tokens.embedding from lm_head")
                            new_params['transformer']['embed_tokens']['embedding'] = lm_head_weight
                
                # Also try params.transformer path
                if 'params' in new_params and 'transformer' in new_params['params']:
                    if 'embed_tokens' not in new_params['params']['transformer']:
                        log_info("Creating params.transformer.embed_tokens from lm_head")
                        new_params['params']['transformer']['embed_tokens'] = {'embedding': lm_head_weight}
                    else:
                        if 'embedding' not in new_params['params']['transformer']['embed_tokens']:
                            log_info("Creating params.transformer.embed_tokens.embedding from lm_head")
                            new_params['params']['transformer']['embed_tokens']['embedding'] = lm_head_weight
                
                log_info(colored("Successfully initialized embeddings from lm_head", Colors.GREEN))
            else:
                log_info(colored("Could not find lm_head weights to initialize embeddings", Colors.RED))
        except Exception as e:
            log_info(f"Error while initializing embeddings from lm_head: {e}")

    # Apply our parameter mapping fix
    try:
        from tensor_parallel import map_parameter_paths
        from weight_diagnostics import fix_parameter_structure
        
        # Make a final pass to ensure parameter paths are correctly mapped
        log_info("Applying parameter path mapping...")
        new_params = map_parameter_paths(new_params)
        
        # Fix any remaining structure issues
        log_info("Applying parameter structure fixes...")
        new_params = fix_parameter_structure(new_params)
    except Exception as e:
        log_info(f"Error applying parameter fixes: {e}")
    
    # Final verification of parameter structure
    def verify_param_structure(params, config):
        """Verify that parameter structure has all required components."""
        flat_params = flatten_dict(params)
        
        # Essential component paths that must exist
        essential_components = {
            "lm_head": False,
            "embed_tokens": False,
            "ln_f": False,  # Final layer norm
        }
        
        # Key parameter paths that should exist
        expected_paths = [
            "lm_head.kernel",
            "params.lm_head.kernel",
            "params.params.lm_head.kernel"
        ]
        found_expected_paths = {path: False for path in expected_paths}
        
        # Component-specific checks
        lm_head_shape = None
        embed_shape = None
        
        # Check for essential components
        for path, param in flat_params.items():
            path_str = '.'.join(str(p) for p in path)
            
            # Check for lm_head in various possible locations
            if "lm_head" in path_str and "kernel" in path_str:
                essential_components["lm_head"] = True
                lm_head_shape = param.shape
                
                # Check specific paths
                for expected_path in expected_paths:
                    if expected_path in path_str:
                        found_expected_paths[expected_path] = True
                
            if "embed_tokens" in path_str and ("embedding" in path_str or "weight" in path_str):
                essential_components["embed_tokens"] = True
                embed_shape = param.shape
                
            if ("ln_f" in path_str or "norm" in path_str) and "weight" in path_str:
                essential_components["ln_f"] = True
        
        # Check if all essential components exist
        all_components_found = all(essential_components.values())
        
        # Make sure dimensions match between lm_head and embeddings
        shape_match = True
        if lm_head_shape is not None and embed_shape is not None:
            # Check if shapes are compatible (allowing for transposition)
            if sorted(lm_head_shape) != sorted(embed_shape):
                log_info(colored(f"Warning: LM head shape {lm_head_shape} doesn't match embedding shape {embed_shape}", Colors.YELLOW))
                shape_match = False
        
        # Check existence of transformer layers up to num_hidden_layers
        layers_found = 0
        num_layers = config["num_hidden_layers"]
        for i in range(num_layers):
            layer_found = False
            for path_str in [p for p, _ in flat_params.items()]:
                path_str = '.'.join(str(p) for p in path_str)
                # Check for layer-specific patterns
                if f"h.{i}." in path_str or f"layers_{i}" in path_str or f"layers.{i}" in path_str:
                    layer_found = True
                    break
            
            if layer_found:
                layers_found += 1
        
        # Report findings
        log_info(colored("Parameter Structure Verification:", Colors.BOLD))
        for component, found in essential_components.items():
            status = colored("✓ Found", Colors.GREEN) if found else colored("✗ Missing", Colors.RED)
            log_info(f"  {component}: {status}")
        
        # Report on expected paths
        log_info(colored("Expected Parameter Paths:", Colors.BOLD))
        for path, found in found_expected_paths.items():
            status = colored("✓ Found", Colors.GREEN) if found else colored("✗ Missing", Colors.YELLOW)
            log_info(f"  {path}: {status}")
        
        log_info(f"  Transformer layers: {layers_found}/{num_layers} found")
        if layers_found < num_layers:
            log_info(colored(f"  Warning: Only {layers_found} of {num_layers} required layers were found", Colors.YELLOW))
        
        # Overall result - we consider it valid if lm_head exists in any location and the shapes are compatible
        if essential_components["lm_head"] and shape_match:
            log_info(colored("✓ LM head validation successful", Colors.GREEN + Colors.BOLD))
            
            # Show full validation result
            if all_components_found and layers_found == num_layers:
                log_info(colored("✓ Complete parameter structure validation successful", Colors.GREEN + Colors.BOLD))
                return True
            else:
                log_info(colored("⚠ Partial parameter structure validation - model may work with limitations", Colors.YELLOW + Colors.BOLD))
                return True
        else:
            log_info(colored("✗ Parameter structure validation failed - model will not work", Colors.RED + Colors.BOLD))
            return False
    
    # Perform structure verification
    structure_valid = verify_param_structure(new_params, config)
    if not structure_valid:
        log_info(colored("Warning: Parameter structure has issues that may affect model operation", Colors.YELLOW))
    
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
    
    # Create a memory tracker instance for this run
    memory_tracker = MemoryTracker(
        gc_threshold_mb=500,
        logger=logger,
        polling_interval=monitor_interval
    )
    
    # Start background memory tracking
    memory_tracker.start_tracking()
    
    # Start legacy monitoring for compatibility
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
            
            # Use our improved adaptive scaling algorithm
            hidden_ratio, layers_ratio, scaling_successful = adaptive_scaling_algorithm(
                original_config,
                system_memory_mb,
                starting_ratio=0.5,
                max_iterations=5,
                target_usage_ratio=target_ram_percent / 100.0,
                logger=logger
            )
            
            if not scaling_successful:
                logger.warning(colored("Adaptive scaling algorithm encountered issues. "
                                      "Proceeding with conservative scaling.", Colors.YELLOW))
            
            logger.info(colored(f"Auto-scaling results:", Colors.YELLOW))
            logger.info(colored(f"  Hidden size ratio: {hidden_ratio:.3f}", Colors.CYAN))
            logger.info(colored(f"  Layers ratio: {layers_ratio:.3f}", Colors.CYAN))
            
            # Mark that scaling has been applied
            scaling_applied = True
            
        elif scale_ratio is not None:
            # Use the same ratio for both dimensions
            hidden_ratio = scale_ratio
            layers_ratio = scale_ratio
            logger.info(colored(f"\n[2/7] Using uniform scaling ratio: {scale_ratio:.3f}", Colors.BOLD))
            
            # Mark that scaling has been applied if not using full model
            scaling_applied = scale_ratio < 1.0
            
        else:
            # Use the specified ratios or defaults
            if hidden_ratio is None:
                hidden_ratio = 0.5  # Default to half size
            if layers_ratio is None:
                layers_ratio = 0.5  # Default to half layers
                
            logger.info(colored(f"\n[2/7] Using custom scaling ratios:", Colors.BOLD))
            logger.info(f"  Hidden size ratio: {hidden_ratio:.3f}")
            logger.info(f"  Layers ratio: {layers_ratio:.3f}")
            
            # Mark that scaling has been applied if not using full model
            scaling_applied = hidden_ratio < 1.0 or layers_ratio < 1.0
        
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
        
        # Use improved memory estimation
        memory_estimate = improved_memory_estimation(
            config, 
            system_memory_mb,
            target_ram_percent / 100.0
        )
        
        # Print detailed memory estimates with color coding
        expected_memory_mb = memory_estimate["total_estimated_mb"]
        expected_memory_ratio = expected_memory_mb / system_memory_mb * 100
        
        memory_color = Colors.GREEN
        if expected_memory_ratio > 70:
            memory_color = Colors.YELLOW
        if expected_memory_ratio > 85:
            memory_color = Colors.RED
            
        logger.info(f"Scaled model has {size_info['params']:,} parameters ({size_info['params']/orig_size_info['params']:.1%} of original)")
        logger.info(colored(f"Expected memory usage: ~{expected_memory_mb:,.1f} MB ({expected_memory_ratio:.1f}% of system RAM)", memory_color))
        
        # Show detailed memory breakdown if verbose
        if verbose:
            logger.info(colored("Memory breakdown:", Colors.BOLD))
            logger.info(f"  Parameter memory: {memory_estimate['parameter_memory_mb']:.1f} MB")
            logger.info(f"  Activation memory: {memory_estimate['activation_memory_mb']:.1f} MB")
            logger.info(f"  Compilation overhead: {memory_estimate['compilation_spike_mb']:.1f} MB")
            logger.info(f"  Tensor parallelism overhead: {memory_estimate['tensor_parallel_mb']:.1f} MB")
            logger.info(f"  Peak memory (projected): {memory_estimate['peak_memory_mb']:.1f} MB")
        
        # Create the mesh
        logger.info(colored(f"\n[4/7] Creating device mesh with shape {mesh_shape}...", Colors.BOLD))
        memory_tracker.check_memory("Before mesh creation")
        mesh = create_device_mesh(mesh_shape)
        logger.info(colored("✅ Mesh created successfully", Colors.GREEN))
        
        # Log metrics after mesh creation
        memory_tracker.check_memory("After mesh creation")
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
        memory_tracker.check_memory("Before parameter initialization", force_gc=True)
        if not quiet:
            log_system_metrics("Before parameter initialization")
        
        # Initialize parameters with real weights or random initialization
        with mesh:
            if use_demo_model:
                # Generate a random key for initialization
                memory_tracker.check_memory("Before random weight initialization")
                logger.info("Initializing model with random weights...")
                
                # Use staged initialization for better memory efficiency
                params = staged_model_initialization(
                    model_class=TensorParallelQwen2ForCausalLM,
                    config=config,
                    mesh=mesh,
                    dtype=jnp.bfloat16,
                    param_dtype=jnp.bfloat16,
                    batch_size=batch_size,
                    logger=logger
                )
                
                logger.info(colored(f"✅ Model initialized with random weights in {time.time() - start_time:.2f} seconds", Colors.GREEN))
            else:
                try:
                    # Load weights from checkpoint
                    logger.info(f"Loading model weights from {model_path}...")
                    
                    # Handle scaled models with real weights
                    if scaling_applied:
                        logger.info(colored("Using partial weight loading for scaled model", Colors.CYAN))
                        memory_tracker.check_memory("Before partial weight initialization")
                        
                        # First initialize with random weights to get the parameter structure
                        # Use staged initialization for better memory efficiency
                        params = staged_model_initialization(
                            model_class=TensorParallelQwen2ForCausalLM,
                            config=config,
                            mesh=mesh,
                            dtype=jnp.bfloat16,
                            param_dtype=jnp.bfloat16,
                            batch_size=batch_size,
                            logger=logger
                        )
                        
                        memory_tracker.check_memory("After parameter structure initialization")
                        
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
                        try:
                            logger.info(colored("Using direct checkpoint loading for full-sized model", Colors.CYAN))
                            # Import parameter mapping utilities
                            from tensor_parallel import load_params_from_checkpoint, map_parameter_paths
                            from weight_diagnostics import create_parameter_structure_report, fix_parameter_structure
                            
                            # Load parameters using our improved loading function
                            params = load_params_from_checkpoint(model, model_path)
                            
                            # Apply parameter path mapping to ensure compatibility
                            logger.info("Applying parameter path mapping...")
                            params = map_parameter_paths(params)
                            
                            # Run diagnostics on the loaded parameters
                            logger.info("Diagnosing parameter structure...")
                            parameter_report = create_parameter_structure_report(params)
                            
                            # Apply fixes if needed
                            if parameter_report["recommendations"]:
                                logger.info(f"Found {len(parameter_report['recommendations'])} parameter structure issues:")
                                for i, rec in enumerate(parameter_report["recommendations"]):
                                    logger.info(f"  {i+1}. {rec}")
                                logger.info("Applying parameter structure fixes...")
                                params = fix_parameter_structure(params)
                            else:
                                logger.info("Parameter structure looks good!")
                                
                            logger.info(colored(f"✅ Model weights loaded in {time.time() - start_time:.2f} seconds", Colors.GREEN))
                        except Exception as e:
                            logger.error(colored(f"Error in standard loading: {e}", Colors.RED))
                            
                            # Try fallback loading method
                            logger.info(colored("Trying fallback loading method...", Colors.YELLOW))
                            try:
                                from weight_loading import init_model_from_weights
                                model, params = init_model_from_weights(
                                    model_class=TensorParallelQwen2ForCausalLM,
                                    model_path=model_path,
                                    config=config,
                                    mesh=mesh,
                                    param_dtype=jnp.bfloat16
                                )
                                logger.info(colored(f"✅ Fallback loading successful in {time.time() - start_time:.2f} seconds", Colors.GREEN))
                            except Exception as e2:
                                logger.error(colored(f"❌ Fallback loading failed: {e2}", Colors.RED))
                                raise ValueError(f"Failed to load weights: {e2}")
                        
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
            
            # Fix parameter structure - extract inner params if needed
            apply_params = params
            if "params" in params and isinstance(params["params"], dict):
                # Check for double nesting issue
                if "params" in params["params"]:
                    logger.info(colored("Detected double-nested params structure, using params['params']", Colors.YELLOW))
                    apply_params = params["params"]
            
            # Try to run the forward pass with the corrected params
            try:
                outputs = model.apply(apply_params, input_ids=input_ids)
                forward_time = time.time() - inference_start
                logger.info(colored(f"✅ Forward pass completed in {forward_time:.2f} seconds", Colors.GREEN))
            except Exception as e:
                # If that failed, try with just the inner params
                if "params" in params:
                    logger.info(colored("First attempt failed, trying with inner params structure", Colors.YELLOW))
                    try:
                        outputs = model.apply({"params": params["params"]}, input_ids=input_ids)
                        forward_time = time.time() - inference_start
                        logger.info(colored(f"✅ Forward pass completed in {forward_time:.2f} seconds", Colors.GREEN))
                    except Exception as e2:
                        # If even that failed, try with the right set of keys
                        logger.info(colored("Second attempt failed, trying with adjusted structure", Colors.YELLOW))
                        try:
                            # This is the most stripped-down structure
                            stripped_params = {"params": {}}
                            
                            # Get the lm_head kernel - this is critical
                            if "lm_head" in params:
                                stripped_params["params"]["lm_head"] = params["lm_head"]
                            elif "params" in params and "lm_head" in params["params"]:
                                stripped_params["params"]["lm_head"] = params["params"]["lm_head"]
                            
                            # Get transformer if available
                            if "transformer" in params:
                                stripped_params["params"]["transformer"] = params["transformer"]
                            elif "params" in params and "transformer" in params["params"]:
                                stripped_params["params"]["transformer"] = params["params"]["transformer"]
                            
                            outputs = model.apply(stripped_params, input_ids=input_ids)
                            forward_time = time.time() - inference_start
                            logger.info(colored(f"✅ Forward pass completed with stripped params in {forward_time:.2f} seconds", Colors.GREEN))
                        except Exception as e3:
                            # Re-raise the original error
                            raise e
                else:
                    # If no inner params, re-raise the exception
                    raise e
        
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

def adaptive_scaling_algorithm(original_config, system_memory_mb, 
                              starting_ratio=0.5, max_iterations=5,
                              target_usage_ratio=0.8, logger=None):
    """
    Find the optimal model scale through intelligent binary search.
    
    Args:
        original_config: Original model configuration
        system_memory_mb: Available system memory in MB
        starting_ratio: Initial scaling ratio to try (0.0-1.0)
        max_iterations: Maximum number of iterations
        target_usage_ratio: Target memory usage ratio (0.0-1.0)
        logger: Logger for output
        
    Returns:
        Tuple of (hidden_ratio, layers_ratio, is_successful)
    """
    if logger is None:
        logger = logging.getLogger("adaptive-scaling")
    
    lower_bound = 0.1  # Minimum scale ratio
    upper_bound = 1.0  # Maximum scale ratio
    current_ratio = starting_ratio
    
    best_ratio = None
    best_memory_efficiency = 0
    
    logger.info(f"Starting adaptive scaling search (max iterations: {max_iterations})")
    
    for i in range(max_iterations):
        logger.info(f"Iteration {i+1}/{max_iterations}: Testing scale ratio {current_ratio:.3f}")
        
        # Configure model with current ratio
        test_config, _ = create_scaled_config(
            original_config,
            hidden_ratio=current_ratio,
            layers_ratio=current_ratio
        )
        
        # Run memory estimation
        memory_estimate = improved_memory_estimation(
            test_config, 
            system_memory_mb,
            target_usage_ratio
        )
        
        estimated_usage = memory_estimate["peak_memory_mb"]
        available_memory = memory_estimate["available_memory_mb"]
        
        # Calculate efficiency (how close to target without exceeding)
        efficiency = estimated_usage / available_memory
        
        logger.info(f"  Estimated memory: {estimated_usage:.1f} MB / {available_memory:.1f} MB available "
                   f"({efficiency:.1%} of target)")
        
        if memory_estimate["has_enough_memory"]:
            # Model fits - record as potential best
            logger.info(colored(f"  ✓ Configuration with ratio {current_ratio:.3f} fits in memory", Colors.GREEN))
            
            # Check if this is more efficient than previous best
            if best_ratio is None or efficiency > best_memory_efficiency:
                best_ratio = current_ratio
                best_memory_efficiency = efficiency
                logger.info(colored(f"  → New best ratio: {best_ratio:.3f} with {efficiency:.1%} efficiency", 
                                   Colors.BOLD + Colors.GREEN))
            
            # Try a larger model if we're using less than 90% of target
            if efficiency < 0.9:
                logger.info(f"  Model is using only {efficiency:.1%} of available memory, trying larger")
                lower_bound = current_ratio
                current_ratio = min(1.0, (current_ratio + upper_bound) / 2)
            else:
                # We found a good fit, close to the target
                logger.info(f"  Found good fit at {efficiency:.1%} of available memory")
                break
        else:
            # Model doesn't fit - adjust upper bound
            logger.info(colored(f"  ✗ Configuration with ratio {current_ratio:.3f} exceeds available memory", 
                              Colors.YELLOW))
            upper_bound = current_ratio
            current_ratio = (lower_bound + current_ratio) / 2
    
    # If we couldn't find a good ratio, use the lower bound with a safety margin
    if best_ratio is None:
        best_ratio = lower_bound * 0.9  # Add safety margin
        logger.warning(colored(f"Could not find a good model size, using {best_ratio:.3f} with safety margin", 
                             Colors.YELLOW))
        return best_ratio, best_ratio, False
    
    # Fine-tune the hidden size and layer ratios
    # Prefer slightly more hidden size than layers for better model quality
    hidden_ratio = min(1.0, best_ratio * 1.1)  # 10% more hidden size
    layers_ratio = max(0.1, best_ratio * 0.9)  # 10% fewer layers
    
    # Final check to make sure this is still valid
    final_config, _ = create_scaled_config(
        original_config,
        hidden_ratio=hidden_ratio,
        layers_ratio=layers_ratio
    )
    
    final_estimate = improved_memory_estimation(
        final_config, 
        system_memory_mb,
        target_usage_ratio
    )
    
    if not final_estimate["has_enough_memory"]:
        # Fall back to simpler approach
        logger.warning(colored(f"Fine-tuned ratios exceed memory, using uniform ratio {best_ratio:.3f}", 
                             Colors.YELLOW))
        return best_ratio, best_ratio, True
    
    logger.info(colored(f"Final model sizing: hidden_ratio={hidden_ratio:.3f}, layers_ratio={layers_ratio:.3f}", 
                      Colors.BOLD + Colors.GREEN))
    logger.info(f"Estimated memory usage: {final_estimate['total_estimated_mb']:.1f} MB")
    
    return hidden_ratio, layers_ratio, True

def staged_model_initialization(model_class, config, mesh, dtype, param_dtype=None, 
                                batch_size=1, logger=None):
    """
    Initialize model in stages to reduce peak memory usage.
    
    Args:
        model_class: Model class to initialize
        config: Model configuration
        mesh: Device mesh
        dtype: Default dtype for model
        param_dtype: Parameter dtype (defaults to dtype if None)
        batch_size: Batch size for initialization
        logger: Logger for output
        
    Returns:
        Dictionary with model parameters
    """
    if logger is None:
        logger = logging.getLogger("model-init")
    
    if param_dtype is None:
        param_dtype = dtype
    
    logger.info(colored("Initializing model in stages to reduce peak memory usage", Colors.BOLD))
    
    # Create memory tracker
    memory_tracker = MemoryTracker(gc_threshold_mb=500, logger=logger)
    memory_tracker.start_tracking()
    
    # Keep track of all parameters
    all_params = {}
    
    try:
        # Step 1: Initialize only embedding layer
        logger.info(colored("Stage 1/3: Initializing embeddings", Colors.BLUE))
        embedding_config = dict(config)
        embedding_config["num_hidden_layers"] = 0  # No layers yet
        
        with mesh:
            memory_tracker.check_memory("Before embedding init", force_gc=True)
            
            # Initialize embedding layers only
            temp_model = model_class(
                config=embedding_config, 
                mesh=mesh, 
                dtype=dtype,
                param_dtype=param_dtype
            )
            
            # Create dummy input
            rng = jax.random.PRNGKey(0)
            input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
            
            # Initialize embedding parameters
            logger.info("Initializing embeddings parameters...")
            embed_params = temp_model.init(rng, input_ids=input_ids)
            
            memory_tracker.check_memory("After embedding init")
            
            # Extract and save embedding parameters
            embed_only_params = {}
            for path, param in flatten_dict(embed_params).items():
                path_str = '.'.join(str(p) for p in path)
                if 'embed_tokens' in path_str:
                    embed_only_params[path] = param
            
            # Save to combined params
            all_params.update(embed_only_params)
            
            # Clear memory
            del temp_model, embed_params, input_ids
            memory_tracker.check_memory("After embedding cleanup", force_gc=True)
        
        # Step 2: Initialize transformer layers in small batches
        logger.info(colored("Stage 2/3: Initializing transformer layers in batches", Colors.BLUE))
        
        # Determine how many layers to initialize at once based on model size
        if config["hidden_size"] >= 4096:
            layers_per_batch = 2  # Very large models
        elif config["hidden_size"] >= 2048:
            layers_per_batch = 4  # Large models
        else:
            layers_per_batch = 8  # Smaller models
        
        layers_per_batch = min(layers_per_batch, config["num_hidden_layers"])
        logger.info(f"Initializing {layers_per_batch} layers at a time")
        
        # Process layers in batches
        for start_layer in range(0, config["num_hidden_layers"], layers_per_batch):
            end_layer = min(start_layer + layers_per_batch, config["num_hidden_layers"])
            batch_size = end_layer - start_layer
            
            logger.info(f"Initializing layers {start_layer} to {end_layer-1} ({batch_size} layers)")
            
            # Create config with just these layers
            layer_config = dict(config)
            layer_config["num_hidden_layers"] = batch_size
            
            with mesh:
                memory_tracker.check_memory(f"Before layers {start_layer}-{end_layer-1}", force_gc=True)
                
                # Initialize model with just these layers
                temp_model = model_class(
                    config=layer_config, 
                    mesh=mesh, 
                    dtype=dtype,
                    param_dtype=param_dtype
                )
                
                # Create dummy input
                rng = jax.random.PRNGKey(start_layer)  # Different seed per batch
                input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
                
                # Initialize parameters
                logger.info(f"Initializing parameters for layers {start_layer}-{end_layer-1}...")
                batch_params = temp_model.init(rng, input_ids=input_ids)
                
                memory_tracker.check_memory(f"After layers {start_layer}-{end_layer-1} init")
                
                # Extract layer parameters and remap layer indices
                layer_params = {}
                for path, param in flatten_dict(batch_params).items():
                    path_str = '.'.join(str(p) for p in path)
                    
                    # If this contains a layer reference, reindex to the correct position
                    if '.h.' in path_str:
                        # Extract layer index
                        for i, part in enumerate(path):
                            if part == 'h' and i+1 < len(path) and isinstance(path[i+1], int):
                                # Remap layer index
                                old_layer_idx = path[i+1]
                                new_layer_idx = start_layer + old_layer_idx
                                
                                # Create a new path with the corrected layer index
                                new_path = list(path)
                                new_path[i+1] = new_layer_idx
                                layer_params[tuple(new_path)] = param
                                break
                    
                # Save layer parameters
                all_params.update(layer_params)
                
                # Clear memory
                del temp_model, batch_params, input_ids, layer_params
                memory_tracker.check_memory(f"After layers {start_layer}-{end_layer-1} cleanup", force_gc=True)
        
        # Step 3: Initialize LM head
        logger.info(colored("Stage 3/3: Initializing LM head", Colors.BLUE))
        lm_head_config = dict(config)
        lm_head_config["num_hidden_layers"] = 0
        
        with mesh:
            memory_tracker.check_memory("Before LM head init", force_gc=True)
            
            # Initialize just the LM head
            temp_model = model_class(
                config=lm_head_config, 
                mesh=mesh, 
                dtype=dtype,
                param_dtype=param_dtype
            )
            
            # Create dummy input
            rng = jax.random.PRNGKey(99)
            input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
            
            # Initialize parameters
            logger.info("Initializing LM head parameters...")
            lm_head_params = temp_model.init(rng, input_ids=input_ids)
            
            memory_tracker.check_memory("After LM head init")
            
            # Extract LM head parameters
            lm_head_only_params = {}
            for path, param in flatten_dict(lm_head_params).items():
                path_str = '.'.join(str(p) for p in path)
                if 'lm_head' in path_str:
                    lm_head_only_params[path] = param
                elif 'ln_f' in path_str:  # Final layer norm
                    lm_head_only_params[path] = param
                    
            # Save LM head parameters
            all_params.update(lm_head_only_params)
            
            # Clear memory
            del temp_model, lm_head_params, input_ids, lm_head_only_params
            memory_tracker.check_memory("After LM head cleanup", force_gc=True)
        
        # Unflatten the combined parameters
        logger.info("Combining all parameters...")
        combined_params = unflatten_dict(all_params)
        
        # Final garbage collection
        del all_params
        memory_tracker.check_memory("After parameter combination", force_gc=True)
        
        # Stop memory tracking
        memory_stats = memory_tracker.stop_tracking()
        logger.info(colored(f"Peak memory during staged initialization: {memory_tracker.peak_memory:.1f} MB", 
                           Colors.GREEN))
        
        return combined_params
        
    except Exception as e:
        logger.error(colored(f"Error during staged initialization: {e}", Colors.RED))
        memory_tracker.stop_tracking()
        # Print traceback
        logger.error(traceback.format_exc())
        raise

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