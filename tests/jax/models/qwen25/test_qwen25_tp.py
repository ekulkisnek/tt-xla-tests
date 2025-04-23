#!/usr/bin/env python3
"""
Test script for the Qwen2.5-7B tensor parallel implementation.

Usage
python test_qwen25_tp.py \
  --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights \
  --mesh_shape 1,8 \
  --dtype bfloat16 \
  --prompt "Your prompt text here" \
  --test_all

"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# Add parent directory to path to help with imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from qwen25_tp_model import (
        TPQwenForCausalLM,
        Qwen25Config,
        create_mesh_from_string,
        load_qwen_model,
        generate_text,
    )
except ImportError as e:
    print(f"Error importing qwen25_tp_model: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_model_loading(weights_dir, mesh_shape, dtype=jnp.bfloat16):
    """Test model loading and parameter validation."""
    logger.info(f"Testing model loading with weights from {weights_dir}")
    logger.info(f"Using mesh shape {mesh_shape} and dtype {dtype}")
    
    start_time = time.time()
    mesh = create_mesh_from_string(mesh_shape)
    
    with mesh:
        model, params = load_qwen_model(
            weights_dir=weights_dir,
            mesh_shape=mesh_shape,
            dtype=dtype,
        )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    
    # Log model configuration
    logger.info(f"Model configuration: {model.config}")
    
    # Log parameter count
    param_size = sum(np.prod(param.shape) for param in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameter count: {param_size:,}")
    
    return model, params, mesh


def test_forward_pass(model, params, mesh, sequence_length=32, batch_size=1):
    """Test a forward pass through the model."""
    logger.info(f"Testing forward pass with sequence length {sequence_length}")
    
    # Create dummy input
    input_ids = jnp.ones((batch_size, sequence_length), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, sequence_length), dtype=jnp.int32)
    
    # Run forward pass
    with mesh:
        start_time = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            return_dict=True,
        )
        jax.block_until_ready(outputs)
        forward_time = time.time() - start_time
    
    logger.info(f"Forward pass completed in {forward_time:.2f} seconds")
    logger.info(f"Output logits shape: {outputs['logits'].shape}")
    
    return outputs


def test_generation(model, params, mesh, sequence_length=32, max_new_tokens=20):
    """Test text generation with the model."""
    logger.info(f"Testing generation with {max_new_tokens} new tokens")
    
    # Create dummy input
    input_ids = jnp.ones((1, sequence_length), dtype=jnp.int32)
    
    # Run generation
    with mesh:
        start_time = time.time()
        
        # Get initial logits
        outputs = model(
            input_ids=input_ids,
            params=params,
            return_dict=True,
        )
        logits = outputs["logits"]
        
        # Take the last logit for next token prediction
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1)
        
        # Add the new token
        new_sequence = jnp.concatenate([input_ids, next_token_id[:, None]], axis=1)
        
        jax.block_until_ready(new_sequence)
        gen_time = time.time() - start_time
    
    logger.info(f"Generation of first token completed in {gen_time:.2f} seconds")
    logger.info(f"Generated token ID: {next_token_id[0]}")
    
    return new_sequence


def test_tokenizer_loading(weights_dir):
    """Test tokenizer loading."""
    logger.info(f"Testing tokenizer loading from {weights_dir}")
    
    try:
        from transformers import AutoTokenizer
        
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        load_time = time.time() - start_time
        
        logger.info(f"Tokenizer loaded successfully in {load_time:.2f} seconds")
        
        # Test tokenization
        sample_text = "Hello, world! This is a test."
        tokens = tokenizer(sample_text, return_tensors="np")
        logger.info(f"Sample text tokenized to shape: {tokens.input_ids.shape}")
        
        # Test detokenization
        decoded = tokenizer.decode(tokens.input_ids[0])
        logger.info(f"Decoded text: {decoded}")
        
        return tokenizer
    except ImportError:
        logger.warning("Failed to import transformers. Skipping tokenizer test.")
        return None
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None


def run_all_tests(args):
    """Run all tests."""
    logger.info("Starting Qwen2.5-7B tensor parallel implementation tests")
    
    # Map dtype string to jnp.dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, jnp.bfloat16)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        seed = jax.random.PRNGKey(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    
    # Test model loading
    model, params, mesh = test_model_loading(args.weights_dir, args.mesh_shape, dtype)
    
    # Test forward pass
    if args.test_forward:
        outputs = test_forward_pass(model, params, mesh, args.seq_len, args.batch_size)
    
    # Test generation
    if args.test_generation:
        new_sequence = test_generation(model, params, mesh, args.seq_len, args.max_new_tokens)
    
    # Test tokenizer
    if args.test_tokenizer:
        tokenizer = test_tokenizer_loading(args.weights_dir)
        
        if tokenizer and args.prompt:
            logger.info(f"Testing end-to-end generation with prompt: {args.prompt}")
            with mesh:
                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    params=params,
                    prompt=args.prompt,
                    max_length=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    deterministic=args.deterministic,
                    mesh=mesh,
                )
            logger.info(f"Generated text: {output}")
    
    logger.info("All tests completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-7B tensor parallel implementation")
    parser.add_argument(
        "--weights_dir", 
        type=str, 
        required=True, 
        help="Directory containing Qwen2.5 weights"
    )
    parser.add_argument(
        "--mesh_shape", 
        type=str, 
        default="1,1", 
        help="Device mesh shape (e.g., '1,8', '2,4')"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16", 
        choices=["float32", "float16", "bfloat16"], 
        help="Data type for model parameters"
    )
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=32, 
        help="Sequence length for tests"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for tests"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=20, 
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="The meaning of life is", 
        help="Prompt for text generation"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Nucleus sampling probability"
    )
    parser.add_argument(
        "--deterministic", 
        action="store_true", 
        help="Use deterministic decoding"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test_forward", 
        action="store_true", 
        help="Test forward pass"
    )
    parser.add_argument(
        "--test_generation", 
        action="store_true", 
        help="Test generation"
    )
    parser.add_argument(
        "--test_tokenizer", 
        action="store_true", 
        help="Test tokenizer"
    )
    parser.add_argument(
        "--test_all", 
        action="store_true", 
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    # Set all test flags if test_all is specified
    if args.test_all:
        args.test_forward = True
        args.test_generation = True
        args.test_tokenizer = True
    
    # Run tests
    run_all_tests(args)


if __name__ == "__main__":
    main() 