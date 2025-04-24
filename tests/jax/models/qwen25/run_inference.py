#!/usr/bin/env python3
"""
Run inference with the Qwen25 JAX model.
"""

import os
import sys
import time
import logging
import argparse
import re
import gc
import json
import datetime
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
from functools import partial

# Add the directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))

# Import Qwen25 model and generation functions
from model import create_qwen25_model
from generate import generate_text

# Import safetensors for weight loading
try:
    from safetensors import safe_open
except ImportError:
    logging.error("safetensors package is required. Install with: pip install safetensors")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qwen25_inference")

# Memory tracking
try:
    import psutil
    def log_memory_usage(label=""):
        """Log current memory usage with an optional label."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss_gb = memory_info.rss / (1024 * 1024 * 1024)
        logger.info(f"Memory usage {label}: {rss_gb:.2f} GB")
        return rss_gb
except ImportError:
    def log_memory_usage(label=""):
        logger.info("psutil not available for memory tracking")
        return 0

# Parameter name mapping helpers from run_memory_efficient.py
def get_param_path(name):
    """Map a PyTorch parameter name to its Flax path."""
    # Direct mappings
    direct_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    if name in direct_mapping:
        return direct_mapping[name]
    
    # Patterns for layer parameters
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    rotary_pattern = r"model\.layers\.(\d+)\.self_attn\.rotary_emb\..*"
    
    # Handle layer norms
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        layer_name = f"layers_{layer_idx}"
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (layer_name, norm_name, "scale")
    
    # Handle attention parameters
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        param_type = attn_match.group(3)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        param_name = "kernel" if param_type == "weight" else "bias"
        return (layer_name, "self_attn", proj_name, param_name)
    
    # Handle MLP parameters
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        return (layer_name, "mlp", proj_name, "kernel")
    
    # Handle rotary embedding parameters - skip these as they're computed on the fly in JAX
    rotary_match = re.match(rotary_pattern, name)
    if rotary_match:
        logger.warning(f"Skipping rotary embedding parameter: {name}")
        return None
    
    # Log any unhandled parameter patterns
    logger.warning(f"Unknown parameter pattern: {name}")
    return None

def transpose_if_needed(name, param):
    """Transpose weight matrices if needed based on the parameter name."""
    # Special case for embedding weights - Flax's nn.Embed expects (vocab_size, embedding_dim)
    # so we should NOT transpose these weights
    if "embed_tokens.weight" in name:
        # Do not transpose embedding weights
        logger.debug(f"Keeping embedding weight shape for {name}: {param.shape}")
        return param
    
    # Other attention and MLP weights need to be transposed
    if "weight" in name and ("proj" in name or "lm_head" in name):
        # For attention and MLP weight matrices
        logger.debug(f"Transposing weight matrix for {name}: {param.shape} -> {param.T.shape}")
        return jnp.transpose(param)
    
    return param

def process_safetensors_file(file_path, dtype=jnp.bfloat16):
    """
    Process a single safetensors file by streaming parameters one by one.
    Returns a dictionary of JAX parameters already in the correct format.
    """
    flax_params = {"params": {}}
    
    # Expected shapes for key parameters
    # For validation to catch errors early
    expected_shapes = {
        "model.embed_tokens.weight": None,  # Will be set dynamically from first file
        "lm_head.weight": None              # Will be set dynamically from first file
    }
    
    try:
        with safe_open(file_path, framework="numpy") as f:
            key_count = 0
            for key in f.keys():
                key_count += 1
                # Log progress
                if key_count % 10 == 0:
                    logger.debug(f"Processed {key_count} tensors...")
                
                # Get the parameter and immediately cast to the specified dtype
                param = f.get_tensor(key)
                
                # Skip parameters that don't map to our model
                param_path = get_param_path(key)
                if param_path is None:
                    logger.warning(f"Skipping unknown parameter: {key}")
                    continue
                
                # Store or validate expected shapes for critical parameters
                if key in expected_shapes:
                    if expected_shapes[key] is None:
                        # First time seeing this parameter, store its shape
                        expected_shapes[key] = param.shape
                        logger.debug(f"Recorded expected shape for {key}: {param.shape}")
                    else:
                        # Check if shape matches expected
                        if param.shape != expected_shapes[key]:
                            logger.warning(f"Shape mismatch for {key}: expected {expected_shapes[key]}, got {param.shape}")
                
                # Convert to JAX array with the correct dtype
                param = jnp.array(param, dtype=dtype)
                
                # Transpose if needed (dense layer weights)
                param = transpose_if_needed(key, param)
                
                # Add to the parameter dictionary with the correct nested structure
                current_dict = flax_params["params"]
                for path_part in param_path[:-1]:
                    if path_part not in current_dict:
                        current_dict[path_part] = {}
                    current_dict = current_dict[path_part]
                
                current_dict[param_path[-1]] = param
                
                # Free memory for the numpy array
                del param
                if key_count % 50 == 0:  # Periodically trigger garbage collection
                    gc.collect()
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise
    
    return flax_params

def merge_param_dicts(base_dict, new_dict):
    """Merge new parameter dictionary into the base dictionary."""
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict):
            if not isinstance(base_dict[key], dict):
                raise ValueError(f"Cannot merge dict into non-dict at key {key}")
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def log_output_text(prompt, response, output_file=None):
    """
    Log the prompt and generated text to a file.
    
    Args:
        prompt: The input prompt
        response: The generated response
        output_file: Path to output file, if None a timestamped file will be created
    
    Returns:
        Path to the output file
    """
    if output_file is None:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"qwen25_output_{timestamp}.txt")
    
    with open(output_file, "w") as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Generated Response:\n{response}\n")
        
    logger.info(f"Output saved to: {output_file}")
    return output_file

def print_stream(text, output_file=None):
    """
    Print text with streaming effect and optionally append to a file.
    
    Args:
        text: Text to print and log
        output_file: Optional file path to append streaming output
    """
    print(text, end="", flush=True)
    
    # If output file is provided, append the streamed text
    if output_file:
        with open(output_file, "a") as f:
            f.write(text)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Qwen25 JAX model")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to directory containing Qwen25 model weights"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Write a short story about a robot learning to paint:",
        help="Text prompt for generation"
    )
    
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=200,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (lower = more deterministic)"
    )
    
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Nucleus sampling probability threshold"
    )
    
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--mesh_shape", 
        type=str, 
        default=None,
        help="Device mesh shape for tensor parallelism (e.g., '1,8')"
    )
    
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model parameters"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--no_stream", 
        action="store_true",
        help="Disable streaming output"
    )
    
    parser.add_argument(
        "--shard",
        action="store_true",
        help="Enable parameter sharding across devices"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed memory profiling"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save generated text to this file (if not specified, a timestamped file will be created)"
    )
    
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save generated text to a file"
    )
    
    return parser.parse_args()

def main():
    """Run Qwen25 inference with memory optimizations."""
    # Parse command line arguments
    args = parse_args()
    
    # Set debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["QWEN_DEBUG"] = "1"
    
    # Print JAX devices info
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"Number of devices: {jax.device_count()}")
    
    # Track initial memory usage
    initial_mem = log_memory_usage("initial")
    
    # Set data type
    if args.dtype == "float32":
        dtype = jnp.float32
    elif args.dtype == "float16":
        dtype = jnp.float16
    else:
        dtype = jnp.bfloat16
    
    logger.info(f"Using dtype: {dtype}")
    
    # Configure device mesh for tensor parallelism
    devices = jax.devices()
    mesh = None
    
    if args.mesh_shape:
        mesh_shape = tuple(map(int, args.mesh_shape.split(",")))
    else:
        # Use simple 1D mesh by default
        device_count = jax.device_count()
        if device_count == 1:
            mesh_shape = (1,)
        else:
            mesh_shape = (1, device_count)  # (data_parallel, model_parallel)
    
    logger.info(f"Using mesh shape: {mesh_shape}")
    
    # Create device mesh for tensor parallelism if sharding is enabled
    if args.shard and len(devices) > 1:
        logger.info(f"Creating device mesh for tensor parallelism: {mesh_shape}")
        if len(mesh_shape) == 1:
            # For 1D mesh, use a single axis name
            mesh = Mesh(np.array(devices), ("data",))
        else:
            # For 2D mesh, reshape devices to match the mesh shape
            devices_reshaped = np.array(devices).reshape(mesh_shape)
            mesh = Mesh(devices_reshaped, ("dp", "mp"))  # data_parallel, model_parallel
    else:
        if len(mesh_shape) == 1:
            # For 1D mesh, use a single axis name
            mesh = Mesh(np.array(devices), ("data",))
        else:
            # For 2D mesh, reshape devices to 2D and use both axis names
            devices_reshaped = np.array(devices).reshape(mesh_shape)
            mesh = Mesh(devices_reshaped, ("data", "model"))
    
    try:
        # Load model configuration first
        config_path = os.path.join(args.model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration with {config.get('num_hidden_layers', 'unknown')} layers")
        
        # Enable memory optimizations
        config["use_memory_efficient_attention"] = True
        
        # Create model BEFORE loading parameters
        logger.info("Creating model structure...")
        t0 = time.time()
        model = create_qwen25_model(config, dtype=dtype)
        logger.info(f"Model structure created in {time.time() - t0:.2f}s")
        
        # Log memory after model creation
        model_creation_mem = log_memory_usage("after model creation")
        
        # Load tokenizer - try to use local files from the model path first
        try:
            from transformers import AutoTokenizer
            
            # Check if tokenizer files exist in the model path
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
            has_tokenizer_files = any(os.path.exists(os.path.join(args.model_path, f)) for f in tokenizer_files)
            
            if has_tokenizer_files:
                logger.info(f"Loading tokenizer from model path: {args.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                logger.info("Loaded tokenizer from model path")
            else:
                # Try from cache
                try:
                    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", local_files_only=True)
                    logger.info("Loaded tokenizer from local cache")
                except Exception as e:
                    logger.info(f"Error loading tokenizer locally: {e}")
                    logger.info("Downloading tokenizer from Hugging Face Hub")
                    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
            
        # Tokenizer memory usage
        tokenizer_mem = log_memory_usage("after tokenizer loading")
            
        # Load parameters from safetensors files
        logger.info(f"Loading parameters from {args.model_path}...")
        params = {"params": {}}
        
        # Process weights by streaming parameters from each file
        safetensors_files = sorted([f for f in os.listdir(args.model_path) if f.endswith(".safetensors")])
        
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {args.model_path}")
        
        # Track memory for each file
        file_memories = []
        
        # Load parameters
        t0_total = time.time()
        for i, filename in enumerate(safetensors_files):
            weight_path = os.path.join(args.model_path, filename)
            logger.info(f"Processing file {i+1}/{len(safetensors_files)}: {filename}")
            
            # Process one file at a time
            t0 = time.time()
            file_params = process_safetensors_file(weight_path, dtype=dtype)
            
            # Merge parameters
            params = merge_param_dicts(params, file_params)
            
            # Log progress and memory usage
            file_time = time.time() - t0
            logger.info(f"Processed {filename} in {file_time:.2f} seconds")
            current_mem = log_memory_usage(f"after file {i+1}/{len(safetensors_files)}")
            file_memories.append((filename, current_mem, file_time))
            
            # Force garbage collection
            del file_params
            gc.collect()
        
        total_load_time = time.time() - t0_total
        logger.info(f"Total parameter loading time: {total_load_time:.2f} seconds")
        
        # Log memory after parameter loading
        params_loaded_mem = log_memory_usage("after all parameters loaded")
        
        # Print model configuration
        logger.info(f"Model configuration: hidden_size={config['hidden_size']}, "
                   f"num_heads={config['num_attention_heads']}, "
                   f"num_layers={config['num_hidden_layers']}")
                   
        # Log parameter structure for debugging
        if args.debug:
            param_keys = list(params.keys())
            logger.debug(f"Top-level parameter keys: {param_keys}")
            if "params" in params:
                params_keys = list(params["params"].keys())
                logger.debug(f"Model params keys: {params_keys}")
                
                # Log a few layer keys to verify structure
                for layer_key in [k for k in params["params"].keys() if k.startswith("layers_")][:2]:
                    layer_keys = list(params["params"][layer_key].keys())
                    logger.debug(f"Layer {layer_key} keys: {layer_keys}")
        
        # Prepare dummy input to warm up the model
        logger.info("Running model warm-up to compile functions...")
        dummy_input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        
        # Pre-compile the forward function
        @partial(jax.jit, static_argnums=(2,))
        def forward_fn(params, input_ids, return_dict=True):
            return model.apply(
                params,
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=return_dict,
                mesh=mesh,
            )
        
        # Run a warm-up pass to compile the model
        try:
            logger.info("Running warm-up forward pass...")
            t0 = time.time()
            with jax.profiler.trace("warmup_forward_pass"):
                _ = forward_fn(params, dummy_input_ids)
            logger.info(f"Warm-up completed in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning(f"Warm-up forward pass failed: {e}")
            logger.warning("Continuing without warm-up...")
        
        # Clear backend caches to free memory
        jax.clear_caches()
        gc.collect()
        
        # Log memory after warm-up
        warmup_mem = log_memory_usage("after warm-up")
        
        # Print prompt
        print(f"\nPrompt: {args.prompt}\n")
        print("Generated Response:")
        
        # Pre-compile generation
        if args.debug:
            logger.debug("Starting text generation...")
        
        # Run inference with streaming
        generation_start_time = time.time()
        try:
            # Set up output file for streaming if requested
            streaming_output_file = None
            if not args.no_save:
                # Create output directory if it doesn't exist
                output_dir = os.path.join(os.getcwd(), "outputs")
                os.makedirs(output_dir, exist_ok=True)
                
                # Create timestamped filename if not provided
                if args.output_file is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    streaming_output_file = os.path.join(output_dir, f"qwen25_output_{timestamp}.txt")
                else:
                    streaming_output_file = args.output_file
                
                # Write the prompt to the output file
                with open(streaming_output_file, "w") as f:
                    f.write(f"Prompt: {args.prompt}\n\n")
                    f.write("Generated Response:\n")
                
                logger.info(f"Streaming output to: {streaming_output_file}")
            
            # Configure stream handler with output file
            if args.no_stream:
                stream_handler = None
            else:
                # Create a partial function with the output file
                stream_handler = lambda text: print_stream(text, streaming_output_file)
            
            # Profile memory if requested
            if args.profile:
                start_gen_mem = log_memory_usage("before generation")
            
            full_response = generate_text(
                model=model, 
                tokenizer=tokenizer, 
                prompt_tokens=args.prompt,
                params=params,
                mesh=mesh,
                max_decode_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                streamer=stream_handler,
                debug=args.debug
            )
            
            generation_time = time.time() - generation_start_time
            logger.info(f"Generation completed in {generation_time:.2f}s")
            
            # Print the full response if streaming was disabled
            if args.no_stream:
                print(full_response)
                
                # Save the full response if not already streaming to file
                if not args.no_save and streaming_output_file:
                    with open(streaming_output_file, "a") as f:
                        f.write(full_response)
                
            # Log memory after generation
            end_mem = log_memory_usage("after generation")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            
            if args.debug:
                # Log additional information in debug mode
                logger.debug(f"Model: {type(model)}")
                logger.debug(f"Tokenizer: {type(tokenizer)}")
                param_keys = list(params.keys() if hasattr(params, 'keys') else ['<Nested Structure>'])
                logger.debug(f"Parameters keys: {param_keys}")
        
        # Print memory usage summary if profiling
        if args.profile or args.debug:
            # Measure final memory usage if not already done
            end_mem = log_memory_usage("after error or completion")
            
            logger.info("\n==== Memory Usage Summary ====")
            logger.info(f"Initial memory usage: {initial_mem:.2f} GB")
            logger.info(f"After model creation: {model_creation_mem:.2f} GB (+{model_creation_mem - initial_mem:.2f} GB)")
            logger.info(f"After tokenizer loading: {tokenizer_mem:.2f} GB (+{tokenizer_mem - model_creation_mem:.2f} GB)")
            logger.info(f"After loading parameters: {params_loaded_mem:.2f} GB (+{params_loaded_mem - tokenizer_mem:.2f} GB)")
            logger.info(f"After warm-up: {warmup_mem:.2f} GB (+{warmup_mem - params_loaded_mem:.2f} GB)")
            logger.info(f"After generation/error: {end_mem:.2f} GB (+{end_mem - warmup_mem:.2f} GB)")
            logger.info(f"Total memory increase: {end_mem - initial_mem:.2f} GB")
            
            # Print parameter loading details
            logger.info("\n==== Parameter Loading Details ====")
            for i, (filename, mem, load_time) in enumerate(file_memories):
                prev_mem = initial_mem if i == 0 else file_memories[i-1][1]
                logger.info(f"File {i+1}: {filename} - {mem:.2f} GB (+{mem - prev_mem:.2f} GB) - {load_time:.2f}s")
            
            # Print model info
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            logger.info(f"\nModel parameters: {num_params:,}")
            logger.info(f"Model size in memory: {params_loaded_mem:.2f} GB")
            logger.info(f"Generation time: {generation_time:.2f}s")
        
        print("\n\nGeneration completed!")
            
        # Save generated text to file if requested
        if not args.no_save:
            # Skip explicit logging since we already handled streaming to file
            if streaming_output_file:
                logger.info(f"Complete response saved to: {streaming_output_file}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 