#!/usr/bin/env python3
"""
Simple demonstration of using Qwen25 JAX model for inference.
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qwen25_demo")

def main():
    """Run a simple demonstration of Qwen25 inference."""
    # Print JAX info
    print("\n=== JAX Configuration ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].device_kind}")
    
    # Add the directory to path for local imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
        print(f"Added {project_dir} to sys.path")
    
    # Import Qwen25 model and generation functions
    print("\n=== Importing Qwen25 Modules ===")
    from tt_xla.jax.models.qwen25.model import load_qwen25_model, create_qwen25_model
    from tt_xla.jax.models.qwen25.generate import generate_text
    
    # Load tokenizer
    print("\n=== Loading Tokenizer ===")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", local_files_only=True)
        print("Loaded tokenizer from local cache")
    except Exception as e:
        print(f"Error loading tokenizer locally: {e}")
        print("Downloading tokenizer from Hugging Face Hub")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Set up model path - modify this to point to your Qwen25 model path
    model_path = "/path/to/Qwen2.5-7B"  # CHANGE THIS TO YOUR MODEL PATH
    if len(sys.argv) > 1:
        # Allow overriding model path from command line
        model_path = sys.argv[1]
    
    # Configure model settings
    dtype = jnp.bfloat16
    mesh_shape = (1, jax.device_count())
    
    # Load or create model
    print(f"\n=== Loading Model ===")
    print(f"Model path: {model_path}")
    print(f"Using dtype: {dtype}, mesh_shape: {mesh_shape}")
    
    start_time = time.time()
    try:
        # Load the model with weights
        model, config, params, mesh = load_qwen25_model(model_path, mesh_shape, dtype=dtype)
        print(f"Model loaded in {time.time() - start_time:.2f}s")
        
        # Print model configuration
        print(f"Model configuration:")
        print(f"  - Hidden size: {config['hidden_size']}")
        print(f"  - Attention heads: {config['num_attention_heads']}")
        print(f"  - KV heads: {config.get('num_key_value_heads', config['num_attention_heads'])}")
        print(f"  - Layers: {config['num_hidden_layers']}")
        print(f"  - Vocab size: {config['vocab_size']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating an empty model for demonstration purposes...")
        
        # Create a mock configuration
        config = {
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "vocab_size": 151936,
            "use_memory_efficient_attention": True,
        }
        
        # Create model without loading weights
        model = create_qwen25_model(config, dtype=dtype)
        params = None
        
        # Create a simple mesh
        devices = jax.devices()
        mesh = Mesh(devices, ("data", "model"))
    
    # Set up prompts for testing
    print("\n=== Text Generation Demo ===")
    prompts = [
        "Write a short story about a robot learning to paint:",
        "Explain how quantum computing works in simple terms:",
        "List the steps to solve a Rubik's cube:"
    ]
    
    # Define streaming callback
    def print_stream(text):
        print(text, end="", flush=True)
    
    # Generate text for each prompt
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt}")
        print("Generated response:")
        
        try:
            start_time = time.time()
            
            # Generate text with streaming
            full_response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt,
                params=params,
                mesh=mesh,
                max_decode_tokens=100,  # Short generation for demo
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                streamer=print_stream
            )
            
            # Calculate metrics
            generation_time = time.time() - start_time
            prompt_tokens = len(tokenizer.encode(prompt))
            total_tokens = len(tokenizer.encode(full_response))
            generated_tokens = total_tokens - prompt_tokens
            tokens_per_second = generated_tokens / generation_time
            
            print(f"\n\nGeneration stats:")
            print(f"  - Time: {generation_time:.2f}s")
            print(f"  - Generated tokens: {generated_tokens}")
            print(f"  - Speed: {tokens_per_second:.2f} tokens/sec")
            
        except Exception as e:
            print(f"\nError during generation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    main() 