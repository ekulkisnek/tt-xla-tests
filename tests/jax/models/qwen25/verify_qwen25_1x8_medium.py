#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Verification script for the Qwen2.5-7B model with 1x8 tensor parallelism.
This script uses a medium-sized model configuration to utilize approximately 80% of RAM.
Enhanced with detailed CPU and RAM metrics.

Usage:

python -m venv venv

cd to qwen25
source venv/bin/activate
pip install jax jaxlib flax transformers safetensors numpy einops tqdm datasets tensorboard
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

# Run with medium model size
python verify_qwen25_1x8_medium.py --model_path /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --hidden_size 1024 --num_layers 10
"""

import os
# Force JAX to use simulated devices for tensor parallelism testing
# We set this here, but it can be overridden by the environment
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
import gc
from functools import partial
import psutil  # For memory and CPU monitoring
import traceback  # For detailed error tracing
import threading  # For background monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("qwen25-1x8-medium")

# Import model components
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
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

def get_system_metrics():
    """Get current memory and CPU usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    # Get CPU usage (1 second interval)
    process_cpu = process.cpu_percent(interval=0.1)
    system_cpu = psutil.cpu_percent(interval=0.1)
    
    # Get per-core CPU usage
    per_core_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    
    return {
        "process_memory_mb": memory_info.rss / (1024 * 1024),
        "system_total_mb": system_memory.total / (1024 * 1024),
        "system_available_mb": system_memory.available / (1024 * 1024),
        "system_used_percent": system_memory.percent,
        "process_cpu_percent": process_cpu,
        "system_cpu_percent": system_cpu,
        "per_core_cpu": per_core_cpu,
    }


def log_system_metrics(phase="Current"):
    """Log memory and CPU usage information."""
    global peak_memory_usage, peak_cpu_usage, last_memory_usage, last_cpu_usage
    
    metrics = get_system_metrics()
    last_memory_usage = metrics["process_memory_mb"]
    last_cpu_usage = metrics["process_cpu_percent"]
    
    # Update peak values
    peak_memory_usage = max(peak_memory_usage, metrics["process_memory_mb"])
    peak_cpu_usage = max(peak_cpu_usage, metrics["process_cpu_percent"])
    
    logger.info(f"System Metrics ({phase}):")
    logger.info(f"  Process Memory: {metrics['process_memory_mb']:.2f} MB (Peak: {peak_memory_usage:.2f} MB)")
    logger.info(f"  System Memory: {metrics['system_used_percent']:.1f}% used "
                f"({(metrics['system_total_mb'] - metrics['system_available_mb']):.2f} MB / {metrics['system_total_mb']:.2f} MB)")
    logger.info(f"  Available Memory: {metrics['system_available_mb']:.2f} MB")
    logger.info(f"  Process CPU: {metrics['process_cpu_percent']:.1f}% (Peak: {peak_cpu_usage:.1f}%)")
    logger.info(f"  System CPU: {metrics['system_cpu_percent']:.1f}%")
    logger.info(f"  RAM Usage Ratio: {(metrics['process_memory_mb'] / metrics['system_total_mb'] * 100):.1f}% of total")
    
    # Log per-core usage in groups
    core_data = metrics["per_core_cpu"]
    if len(core_data) > 8:
        # For many cores, summarize
        logger.info(f"  CPU Cores: Min: {min(core_data):.1f}%, Max: {max(core_data):.1f}%, Avg: {sum(core_data)/len(core_data):.1f}%")
    else:
        # For fewer cores, show individual
        logger.info(f"  CPU Cores: {', '.join([f'{c:.1f}%' for c in core_data])}")
    
    return metrics


def start_background_monitoring(interval=5):
    """Start a background thread to monitor system resources."""
    global monitoring_active, monitor_thread
    
    def monitor_resources():
        while monitoring_active:
            metrics = log_system_metrics("Background Monitor")
            time.sleep(interval)
    
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    logger.info(f"Started background resource monitoring (interval: {interval}s)")


def stop_background_monitoring():
    """Stop the background monitoring thread."""
    global monitoring_active, monitor_thread
    
    if monitoring_active and monitor_thread:
        monitoring_active = False
        monitor_thread.join(timeout=1)
        logger.info("Stopped background resource monitoring")
        logger.info(f"Peak process memory: {peak_memory_usage:.2f} MB")
        logger.info(f"Peak process CPU: {peak_cpu_usage:.1f}%")


def create_medium_config(hidden_size, num_layers, num_heads=None, kv_heads=None):
    """Create a medium-sized configuration for Qwen2.5 model."""
    # Scale number of heads with hidden size if not specified
    if num_heads is None:
        num_heads = max(1, hidden_size // 128)
    
    # Scale KV heads with number of heads if not specified
    if kv_heads is None:
        kv_heads = max(1, num_heads // 4)
    
    # Calculate intermediate size (typically 4x hidden size for Qwen models)
    intermediate_size = hidden_size * 4
    
    return {
        "vocab_size": 152064,  # Keep original vocab size
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
    } 

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
                    "../qwen25-weights"
                ))
                if os.path.exists(local_path):
                    tokenizer = AutoTokenizer.from_pretrained(local_path)
                else:
                    raise ValueError(f"Could not find tokenizer at {local_path}")
        
        logger.info(f"✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"❌ Error loading tokenizer: {e}")
        logger.info("Please install transformers: pip install transformers")
        return None


def verify_1x8_mesh_medium(
    model_path: str, 
    use_demo_model: bool = False, 
    max_tokens: int = 5,
    hidden_size: int = 1024, 
    num_layers: int = 10,
    num_heads: int = None,
    kv_heads: int = None,
    batch_size: int = 1,
    monitor_interval: int = 5
):
    """
    Verify that the medium-sized model works with the 1x8 mesh shape.
    Includes enhanced resource monitoring.
    
    Args:
        model_path: Path to the model
        use_demo_model: Whether to use demo model instead of loading real weights
        max_tokens: Maximum number of tokens to generate in test
        hidden_size: Hidden size for medium model
        num_layers: Number of layers for medium model
        num_heads: Number of attention heads (or None to auto-scale with hidden size)
        kv_heads: Number of KV heads (or None to auto-scale with num_heads)
        batch_size: Batch size for inference
        monitor_interval: Interval in seconds for background monitoring
        
    Returns:
        bool: True if verification succeeds, False otherwise
    """
    mesh_shape = (1, 8)  # Focus only on the 1x8 mesh shape
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Verifying 1x8 mesh with MEDIUM model configuration")
    logger.info(f"Model configuration: hidden_size={hidden_size}, layers={num_layers}")
    if num_heads is not None:
        logger.info(f"  Custom attention heads: {num_heads}, kv_heads: {kv_heads or 'auto'}")
    logger.info(f"{'='*80}")
    
    # Start background resource monitoring
    start_background_monitoring(monitor_interval)
    
    # Log initial metrics
    initial_metrics = log_system_metrics("Initial")
    
    try:
        # Check if we have enough devices (or if simulation is active)
        required_devices = mesh_shape[0] * mesh_shape[1]
        available_devices = len(jax.devices())
        
        if available_devices < required_devices and "XLA_FLAGS" not in os.environ:
            logger.warning(f"⚠️ Not enough devices for mesh shape {mesh_shape} ({available_devices}/{required_devices})")
            logger.warning(f"⚠️ Set XLA_FLAGS='--xla_force_host_platform_device_count={required_devices}' to simulate")
            return False
        
        # Create the mesh
        logger.info(f"[1/6] Creating device mesh with shape {mesh_shape}...")
        mesh = create_device_mesh(mesh_shape)
        logger.info(f"✅ Mesh created successfully")
        
        # Log metrics after mesh creation
        log_system_metrics("After mesh creation")
        
        # Load the configuration
        logger.info(f"\n[2/6] Creating medium model configuration...")
        
        # Create a medium-sized configuration
        config = create_medium_config(
            hidden_size=hidden_size, 
            num_layers=num_layers,
            num_heads=num_heads,
            kv_heads=kv_heads
        )
        logger.info(f"✅ Medium configuration created")
        
        # Display model information
        logger.info(f"\nMedium Model details:")
        logger.info(f"- Hidden size: {config['hidden_size']}")
        logger.info(f"- Layers: {config['num_hidden_layers']}")
        logger.info(f"- Attention heads: {config['num_attention_heads']}")
        logger.info(f"- KV heads: {config['num_key_value_heads']}")
        logger.info(f"- Intermediate size: {config['intermediate_size']}")
        logger.info(f"- Vocab size: {config['vocab_size']}")
        
        # Estimate memory footprint
        param_count = (
            config['hidden_size'] * config['vocab_size'] +  # Embeddings
            config['num_hidden_layers'] * (
                4 * config['hidden_size'] * config['hidden_size'] +  # Attention matrices
                8 * config['hidden_size'] * config['intermediate_size']  # MLP matrices
            )
        )
        # Rough estimate (bfloat16 = 2 bytes)
        estimated_memory_mb = param_count * 2 / (1024 * 1024)
        logger.info(f"- Estimated parameter count: {param_count:,}")
        logger.info(f"- Estimated memory (parameters only): {estimated_memory_mb:.2f} MB")
        
        # Log metrics after configuration
        log_system_metrics("After configuration creation")
        
        # Initialize the model
        logger.info(f"\n[3/6] Initializing tensor-parallel model...")
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
        log_system_metrics("Before parameter initialization")
        
        # Initialize parameters with real weights or random initialization
        with mesh:
            if use_demo_model:
                # Generate a random key for initialization
                rng = jax.random.PRNGKey(0)
                
                # Initialize with random weights
                logger.info("Initializing model with random weights...")
                params = model.init(rng, input_ids=input_ids)
                logger.info(f"✅ Model initialized with random weights in {time.time() - start_time:.2f} seconds")
            else:
                try:
                    # Load weights from checkpoint
                    logger.info(f"Loading model weights from {model_path}...")
                    
                    # First initialize with random weights to get parameter structure
                    rng = jax.random.PRNGKey(0)
                    params = model.init(rng, input_ids=input_ids)
                    
                    # Then load partial weights to fit medium configuration
                    logger.info(f"Loading partial weights for medium model configuration...")
                    params = load_partial_qwen_weights(
                        model_path=model_path,
                        target_params=params,
                        config=config,
                        mesh=mesh,
                        num_layers=num_layers,
                        logger=logger
                    )
                    logger.info(f"✅ Model weights loaded in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"❌ Error loading weights: {e}")
                    logger.error("Weight loading failed. See details below:")
                    logger.error(traceback.format_exc())
                    stop_background_monitoring()
                    return False
        
        # Log metrics after parameter initialization
        log_system_metrics("After parameter initialization")
        logger.info(f"RAM utilization ratio: {(last_memory_usage / initial_metrics['system_total_mb'] * 100):.1f}% of total system RAM")
        
        # Run garbage collection to clean up unused objects
        gc.collect()
        log_system_metrics("After garbage collection")
        
        # Setup tokenizer
        logger.info(f"\n[4/6] Setting up tokenizer...")
        tokenizer = setup_tokenizer(model_path)
        if tokenizer is None:
            logger.error("❌ Failed to load tokenizer")
            stop_background_monitoring()
            return False
        
        # Run simple inference test
        logger.info(f"\n[5/6] Running forward pass test...")
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
            logger.info(f"✅ Forward pass completed in {forward_time:.2f} seconds")
        
        # Log metrics after forward pass
        log_system_metrics("After forward pass")
        
        logits = outputs[0]
        logger.info(f"Output logits shape: {logits.shape}")
        
        # Generate tokens
        logger.info(f"\n[6/6] Generating {max_tokens} tokens...")
        generated_ids = input_ids
        generated_tokens = []
        
        # Measure generation performance
        generation_start = time.time()
        tokens_per_sec = []
        
        # Log metrics at the start of generation
        log_system_metrics("Before token generation")
        
        # Simple generation loop
        for i in range(max_tokens):
            token_start = time.time()
            logger.info(f"  Generating token {i+1}/{max_tokens}...")
            
            with mesh:
                # Apply sharding to inputs
                input_spec = model.input_sharding_spec(dtype=jnp.int32)
                current_input = jax.device_put(generated_ids, input_spec)
                
                # Run forward pass
                outputs = model.apply(params, input_ids=current_input)
            
            # Get next token (greedy)
            next_token_logits = outputs[0][0, -1, :]
            next_token = jnp.argmax(next_token_logits)
            
            # Add token to sequence
            next_token_array = jnp.array([[next_token.item()]])
            generated_ids = jnp.concatenate([generated_ids, next_token_array], axis=1)
            generated_tokens.append(next_token.item())
            
            # Calculate token generation time
            token_time = time.time() - token_start
            tokens_per_sec.append(1.0 / token_time if token_time > 0 else 0)
            
            # Print the generated token
            if hasattr(tokenizer, 'decode'):
                token_text = tokenizer.decode([next_token.item()])
                logger.info(f"  Generated token {i+1}: '{token_text}' (took {token_time:.2f}s)")
            else:
                logger.info(f"  Generated token {i+1}: id={next_token.item()} (took {token_time:.2f}s)")
            
            # Log metrics every few tokens
            if (i+1) % 2 == 0 or i == max_tokens-1:
                log_system_metrics(f"After generating {i+1} tokens")
        
        # Calculate generation performance
        total_generation_time = time.time() - generation_start
        avg_tokens_per_sec = max_tokens / total_generation_time if total_generation_time > 0 else 0
        
        logger.info(f"\nGeneration stats:")
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
        log_system_metrics("Final")
        
        # Print final RAM utilization
        final_ram_percent = (last_memory_usage / initial_metrics['system_total_mb'] * 100)
        logger.info(f"\nFinal RAM utilization: {final_ram_percent:.1f}% of total system RAM")
        
        # Stop background monitoring
        stop_background_monitoring()
        
        logger.info(f"\n✅ Medium model with 1x8 mesh verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing medium model with 1x8 mesh: {e}")
        logger.error(traceback.format_exc())
        
        # Log metrics after error
        log_system_metrics("Error state")
        
        # Stop background monitoring
        stop_background_monitoring()
        
        return False 

def main():
    """Main function to verify 1x8 tensor parallel implementation with medium-sized model."""
    parser = argparse.ArgumentParser(
        description="Verify 1x8 tensor parallel implementation for Qwen2.5 with medium-sized model (~80% RAM utilization)"
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
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="Hidden size for medium model (default: 1024)"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=10,
        help="Number of layers for medium model (default: 10)"
    )
    
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Number of attention heads (default: auto-scaled with hidden size)"
    )
    
    parser.add_argument(
        "--kv_heads",
        type=int,
        default=None,
        help="Number of KV heads (default: auto-scaled with num_heads)"
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
        default=5,
        help="Interval in seconds for background resource monitoring (default: 5)"
    )
    
    parser.add_argument(
        "--target_ram_percent",
        type=int,
        default=80,
        help="Target RAM utilization percentage (default: 80)"
    )
    
    args = parser.parse_args()
    
    # Verify model path exists
    if not args.use_demo_model and not os.path.exists(args.model_path):
        logger.error(f"Error: Model path {args.model_path} does not exist")
        logger.error("Please provide a valid path to the model weights")
        return False
    
    # Check available system RAM to adjust model size if requested
    system_memory = psutil.virtual_memory()
    total_ram_gb = system_memory.total / (1024**3)
    
    # Adjust model size based on target RAM utilization
    if args.target_ram_percent > 0:
        logger.info(f"System has {total_ram_gb:.1f} GB RAM, target utilization: {args.target_ram_percent}%")
        
        # Calculate a very rough estimate of model size that would use target % of RAM
        # This is a very approximate heuristic that can be fine-tuned
        target_ram_bytes = system_memory.total * (args.target_ram_percent / 100)
        target_ram_gb = target_ram_bytes / (1024**3)
        
        # Use a scaling factor (these are rough estimations)
        # For reference:
        # - A model with hidden_size=1024, layers=12 uses ~1-1.5GB
        # - A model with hidden_size=2048, layers=16 uses ~4-5GB
        # - A model with hidden_size=3072, layers=20 uses ~10-12GB
        
        # This is a simple heuristic based on available RAM
        if target_ram_gb < 2:
            # For very small systems (<2GB target)
            suggested_hidden = 768
            suggested_layers = 6
        elif target_ram_gb < 4:
            # For small systems (2-4GB target)
            suggested_hidden = 1024
            suggested_layers = 8
        elif target_ram_gb < 8:
            # For medium systems (4-8GB target)
            suggested_hidden = 1536
            suggested_layers = 12
        elif target_ram_gb < 16:
            # For larger systems (8-16GB target)
            suggested_hidden = 2048
            suggested_layers = 16
        else:
            # For high-RAM systems (>16GB target)
            suggested_hidden = 3072
            suggested_layers = 24
        
        if args.hidden_size == 1024 and args.num_layers == 10:
            # Only override if user didn't specify custom values
            logger.info(f"Adjusting model size to target {args.target_ram_percent}% of system RAM ({target_ram_gb:.1f} GB)")
            logger.info(f"Suggested configuration: hidden_size={suggested_hidden}, layers={suggested_layers}")
            args.hidden_size = suggested_hidden
            args.num_layers = suggested_layers
    
    # Display information
    logger.info(f"Verifying 1x8 tensor parallel implementation with MEDIUM-sized model")
    logger.info(f"System RAM: {total_ram_gb:.1f} GB")
    logger.info(f"Model: {'Demo model (random weights)' if args.use_demo_model else f'Weights from {args.model_path}'}")
    logger.info(f"Model size: hidden_size={args.hidden_size}, layers={args.num_layers}")
    logger.info(f"Attention heads: {args.num_heads or 'auto'}, KV heads: {args.kv_heads or 'auto'}")
    logger.info(f"JAX devices available: {len(jax.devices())}")
    logger.info(f"Max tokens to generate: {args.max_tokens}")
    logger.info(f"Background monitoring interval: {args.monitor_interval}s")
    
    # Log initial system metrics
    log_system_metrics("Startup")
    
    # Run verification
    success = verify_1x8_mesh_medium(
        args.model_path, 
        use_demo_model=args.use_demo_model,
        max_tokens=args.max_tokens,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kv_heads=args.kv_heads,
        batch_size=args.batch_size,
        monitor_interval=args.monitor_interval
    )
    
    # Print summary
    if success:
        logger.info("\n✅ Medium model 1x8 mesh verification successful.")
        logger.info(f"Peak memory usage: {peak_memory_usage:.2f} MB")
        logger.info(f"Peak CPU usage: {peak_cpu_usage:.1f}%")
    else:
        logger.error("\n❌ Medium model 1x8 mesh verification failed.")
        logger.error("Please check the errors above for troubleshooting suggestions.")
    
    return success


if __name__ == "__main__":
    main() 