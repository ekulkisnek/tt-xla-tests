"""
Text generation functionality for Qwen25 JAX model.
"""

import os
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from jax.sharding import Mesh
import numpy as np
import time
from functools import partial
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25_generate")

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

def sample_next_token(
    logits: jnp.ndarray, 
    temperature: float = 0.7, 
    top_k: int = 50, 
    top_p: float = 0.9,
    rng_key: jax.random.PRNGKey = None
) -> jnp.ndarray:
    """Sample next token from logits with temperature, top-k, and top-p sampling.
    
    Args:
        logits: Logits tensor of shape (batch, 1, vocab_size)
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling probability threshold
        rng_key: JAX PRNG key for sampling
    
    Returns:
        Sampled token IDs of shape (batch,)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        
    # Get final dimension (vocab size)
    vocab_size = logits.shape[-1]
    
    # Apply temperature
    if temperature > 0:
        logits = logits / jnp.maximum(temperature, 1e-7)
    
    # Apply top-k filtering
    if top_k > 0 and top_k < vocab_size:
        # Get top-k logits and corresponding indices
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        
        # Create mask for non-top-k values
        logits_mask = jnp.full_like(logits, True)
        top_k_one_hot = jax.nn.one_hot(top_k_indices, vocab_size, dtype=bool)
        top_k_mask = jnp.logical_or.reduce(top_k_one_hot, axis=-2)
        
        # Apply mask to logits
        logits = jnp.where(
            top_k_mask,
            logits,
            jnp.full_like(logits, -float("inf"))
        )
    
    # Apply top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        # Convert logits to probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = jax.lax.sort_key_val(
            -probs, jnp.arange(vocab_size, dtype=jnp.int32).reshape(1, 1, -1).repeat(logits.shape[0], axis=0)
        )
        sorted_probs = -sorted_probs
        
        # Calculate cumulative probabilities
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        # Find indices where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Keep first token above threshold to maintain minimum probability
        sorted_indices_to_remove = jnp.concatenate([
            jnp.zeros_like(sorted_indices_to_remove[..., :1]),
            sorted_indices_to_remove[..., :-1]
        ], axis=-1)
        
        # Convert back to vocabulary indices
        indices_to_remove = jnp.zeros_like(sorted_indices_to_remove)
        indices_to_remove = indices_to_remove.at[
            jnp.arange(logits.shape[0])[:, None, None],
            jnp.zeros((logits.shape[0], 1), dtype=jnp.int32),
            sorted_indices
        ].set(sorted_indices_to_remove)
        
        # Apply mask to logits
        logits = jnp.where(
            indices_to_remove,
            jnp.full_like(logits, -float("inf")),
            logits
        )
    
    # If temperature is close to zero, perform greedy sampling
    if temperature < 1e-5:
        next_token = jnp.argmax(logits, axis=-1)
    else:
        # Sample from the filtered distribution
        next_token = jax.random.categorical(rng_key, logits, axis=-1)
    
    return next_token

def generate_text(
    model,
    tokenizer,
    params,
    prompt_tokens,
    max_decode_tokens,
    mesh,
    streamer=None,
    debug=False,
    **kwargs,
):
    """Generate text using a JAX model.
    
    Args:
        model: The JAX model
        tokenizer: Huggingface tokenizer
        params: Model parameters
        prompt_tokens: Tokenized prompt text
        max_decode_tokens: Maximum number of tokens to generate
        mesh: Optional mesh for parallel computation
        streamer: Optional callback function for streaming text output
        debug: Enable debug output
        **kwargs: Additional kwargs for generation
        
    Returns:
        Complete generated text (prompt + continuation)
    """
    # Debug logging to understand parameter structure
    if debug:
        logger.info("=== DEBUG: Parameter Structure ===")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Params type: {type(params)}")
        if hasattr(params, "keys"):
            logger.info(f"Params keys: {list(params.keys())}")
        else:
            logger.info(f"Params doesn't have keys method, type: {type(params)}")
        logger.info(f"kwargs: {kwargs}")
        logger.info("=== END DEBUG ===")
    
    # Log initial memory usage
    generation_start_mem = log_memory_usage("start of generation") if debug else 0
    
    # Configuration settings
    generation_start_time = time.time()
    logger.info(f"Generating up to {max_decode_tokens} tokens")
    
    # Get generation hyperparameters
    temperature = kwargs.get("temperature", 0.7)
    top_p = kwargs.get("top_p", 0.9)
    top_k = kwargs.get("top_k", 50)
    
    # Initialize with prompt
    if isinstance(prompt_tokens, str):
        logger.info(f"Converting prompt string to tokens: '{prompt_tokens}'")
        input_ids = tokenizer(prompt_tokens, return_tensors="np").input_ids
    else:
        input_ids = prompt_tokens
    
    # Log the shape of input_ids
    if debug:
        logger.info(f"Input tokens shape: {input_ids.shape}")
    
    # Create attention mask - use 4D format for Qwen model
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
    
    # Position IDs - make sure to match the input_ids length
    position_ids = np.arange(input_ids.shape[1])[None, :]
    
    if debug:
        logger.info(f"Initial shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, position_ids={position_ids.shape}")
    
    # IMPORTANT: Ensure position_ids shape exactly matches input_ids shape
    # This prevents shape mismatch errors in apply_rotary_emb
    if position_ids.shape[1] != input_ids.shape[1]:
        logger.warning(f"Position IDs shape {position_ids.shape} doesn't match input_ids shape {input_ids.shape}. Adjusting...")
        # Create position_ids to match input_ids exactly
        position_ids = np.arange(input_ids.shape[1], dtype=np.int32)[None, :]
        position_ids = np.broadcast_to(position_ids, input_ids.shape)
        logger.info(f"Adjusted position_ids shape: {position_ids.shape}")
    
    # Initialize generation state
    state = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
    }
    
    # Track generated text for streaming
    generated_text = ""
    past_text = ""
    
    # Create a PRNG key for sampling
    rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
    
    # Define a compiled forward step function - use a pure JAX version for better memory performance
    @partial(jax.jit, static_argnums=(4,))
    def forward_step(params, ids, mask, pos_ids, return_dict):
        """
        Memory-optimized forward step function.
        """
        # Note: model.apply is the correct way to use the model
        return model.apply(
            params,  # Use params directly without further nesting
            input_ids=ids, 
            attention_mask=mask,
            position_ids=pos_ids,
            past_key_values=None,  # Handle past_key_values separately for better memory management
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
            mesh=mesh,
        )
    
    # First token generation compile step
    @partial(jax.jit, static_argnums=(5,))
    def forward_with_past(params, ids, mask, pos_ids, past_kv, return_dict):
        """
        Memory-optimized forward step function with past key values.
        """
        # Use past_key_values for tokens after the first one
        return model.apply(
            params,
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos_ids,  # Pass in the position IDs explicitly
            past_key_values=past_kv,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
            mesh=mesh,
        )
    
    # Start generation timer
    generation_start_time = time.time()
    
    # Handle generation with batched input
    batch_size = input_ids.shape[0]
    
    # Track EOS tokens
    eos_token_id = tokenizer.eos_token_id
    all_terminated = np.zeros(batch_size, dtype=bool)
    
    # Generation loop
    for i in range(max_decode_tokens):
        # Time the token generation
        token_start_time = time.time()
        
        # Run one step of inference
        is_first_token = state["past_key_values"] is None
        current_input_ids = state["input_ids"] if is_first_token else state["input_ids"][:, -1:]
        
        # Create appropriate position IDs based on current sequence length
        if is_first_token:
            # For the first token, use the original position_ids (has same length as input_ids)
            current_position_ids = state["position_ids"]
            # CRITICAL FIX: Ensure position_ids match input_ids shape for first token
            if current_position_ids.shape[1] != current_input_ids.shape[1]:
                logger.warning(f"Shape mismatch before first forward pass: position_ids {current_position_ids.shape} vs input_ids {current_input_ids.shape}")
                # Create position_ids with same shape as input_ids
                current_position_ids = np.arange(current_input_ids.shape[1], dtype=np.int32)[None, :]
                current_position_ids = np.broadcast_to(current_position_ids, current_input_ids.shape)
                # Update state
                state["position_ids"] = current_position_ids
            
            if debug:
                logger.info(f"First token: input_ids shape: {current_input_ids.shape}, position_ids shape: {current_position_ids.shape}")
        else:
            # For subsequent tokens, create position IDs for just the new token position
            seq_length = state["input_ids"].shape[1] - 1  # This is the position of the current token (0-indexed)
            # FIXED: Create position_ids with the EXACT SAME SHAPE as input_ids to avoid broadcasting issues
            current_position_ids = np.full_like(current_input_ids, seq_length, dtype=np.int32)
            if debug and i % 10 == 0:
                logger.info(f"Token {i}: input_ids shape: {current_input_ids.shape}, position_ids shape: {current_position_ids.shape}, pos_value: {seq_length}")
        
        try:
            # Try to run the model
            if is_first_token:
                # Initial pass without past_key_values
                # Force JIT compilation here by explicitly calling jax.jit
                if debug:
                    logger.info(f"First token forward pass with shapes: input_ids={current_input_ids.shape}, "
                               f"attention_mask={state['attention_mask'].shape}, position_ids={current_position_ids.shape}")
                outputs = forward_step(
                    params,
                    current_input_ids,
                    state["attention_mask"],
                    current_position_ids,
                    True  # return_dict
                )
            else:
                # Subsequent passes with past_key_values
                # This is faster as it reuses past key/values
                if debug and i % 10 == 0:
                    logger.info(f"Token {i} forward pass with shapes: input_ids={current_input_ids.shape}, "
                               f"attention_mask={state['attention_mask'].shape}, position_ids={current_position_ids.shape}, "
                               f"past_kv_len={len(state['past_key_values'])}")
                outputs = forward_with_past(
                    params,
                    current_input_ids,
                    state["attention_mask"],
                    current_position_ids,
                    state["past_key_values"],
                    True  # return_dict
                )
                
            # Clear JAX backend caches periodically to prevent memory growth
            if i % 10 == 0 and i > 0:
                jax.clear_caches()
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error during forward pass at token {i}: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Current input_ids shape: {current_input_ids.shape}")
            if not is_first_token:
                logger.error(f"Past key/values exist with length: {len(state['past_key_values'])}")
            raise
        
        # Get logits and update past key values
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            past_key_values = outputs.get("past_key_values")
        else:
            # Handle tuple output format
            logits = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
            
        # Get next token logits (last token only)
        next_token_logits = logits[:, -1:, :]
        
        # Memory usage after forward pass
        if debug and i % 10 == 0:
            forward_mem = log_memory_usage(f"after forward pass for token {i}")
            
        # Update PRNG key
        rng_key, sampling_key = jax.random.split(rng_key)
        
        # Sample next token
        next_token = sample_next_token(
            next_token_logits, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p,
            rng_key=sampling_key
        )
        
        # Convert to numpy for easier manipulation
        next_token_np = np.array(next_token)
        
        # Check for EOS token and update terminated flag
        for b in range(batch_size):
            if next_token_np[b, 0] == eos_token_id:
                all_terminated[b] = True
        
        # Check if all sequences are terminated
        if np.all(all_terminated):
            logger.info(f"All sequences terminated at step {i}")
            break
        
        # Measure token generation time
        token_time = time.time() - token_start_time
        if i % 10 == 0:  # Log every 10 tokens
            logger.info(f"Generated token {i} in {token_time:.4f}s")
        
        # Append to the input_ids
        state["input_ids"] = np.concatenate([state["input_ids"], next_token_np], axis=1)
        
        # Update attention mask for the new token
        # The attention mask should be 2D and the same shape as input_ids
        if len(state["attention_mask"].shape) == 2:
            state["attention_mask"] = np.pad(
                state["attention_mask"],
                ((0, 0), (0, 1)),
                mode="constant",
                constant_values=1
            )
        else:
            # If 4D mask with shape [batch, 1, 1, seq_len], update it correctly
            seq_len = state["input_ids"].shape[1]
            state["attention_mask"] = np.ones((batch_size, 1, 1, seq_len), dtype=np.int32)
        
        # Update past key values for next iteration
        state["past_key_values"] = past_key_values
        
        # Decode for streaming output
        if streamer is not None:
            # Decode only the newly generated token
            new_text = tokenizer.decode(next_token_np[0])
            
            # Stream the new text
            if new_text:
                generated_text += new_text
                streamer(new_text)
                
    # Final generated text
    if streamer is None:
        # Decode the entire sequence if not streaming
        generated_text = tokenizer.decode(state["input_ids"][0])
        
    # Calculate overall generation time
    generation_time = time.time() - generation_start_time
    tokens_generated = state["input_ids"].shape[1] - input_ids.shape[1]
    logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
                f"({tokens_generated / generation_time:.2f} tokens/sec)")
    
    # Log final memory usage
    if debug:
        end_mem = log_memory_usage("end of generation")
        logger.info(f"Memory change during generation: {end_mem - generation_start_mem:.2f} GB")
    
    # Return the generated text
    return generated_text


def stream_process_outputs(model, tokenizer, prompt, stream_handler=None, **generation_kwargs):
    """Process generated tokens one at a time with streaming capability.
    
    Args:
        model: The JAX model
        tokenizer: Huggingface tokenizer
        prompt: Input text prompt
        stream_handler: Optional callback function for streaming text output
        generation_kwargs: Additional kwargs for generation
        
    Returns:
        Complete generated text
    """
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        params=model.params,
        prompt_tokens=prompt,
        max_decode_tokens=100,
        mesh=None,
        streamer=stream_handler,
        **generation_kwargs
    ) 