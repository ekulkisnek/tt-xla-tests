"""
Text generation functionality for Qwen25 JAX model.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from jax.sharding import Mesh
import numpy as np
import time
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25_generate")

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
    logger.info("=== DEBUG: Parameter Structure ===")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Params type: {type(params)}")
    if hasattr(params, "keys"):
        logger.info(f"Params keys: {list(params.keys())}")
    else:
        logger.info(f"Params doesn't have keys method, type: {type(params)}")
    logger.info(f"kwargs: {kwargs}")
    logger.info("=== END DEBUG ===")
    
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
    
    # Create attention mask
    attention_mask = np.ones_like(input_ids)
    
    # Position IDs
    position_ids = np.arange(input_ids.shape[1])[None, :]
    
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
    
    # Define a compiled forward step function
    @partial(jax.jit, static_argnums=(4,))
    def forward_step(ids, mask, pos_ids, past_kv, return_dict):
        # This implementation consistently works in run_memory_efficient.py
        return model.apply(
            params,  # No additional wrapping needed
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos_ids,
            past_key_values=past_kv,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
            mesh=mesh,
        )
    
    # Start generation timer
    generation_start_time = time.time()
    
    # Generation loop
    for i in range(max_decode_tokens):
        # Time the token generation
        token_start_time = time.time()
        
        # Run one step of inference
        is_first_token = state["past_key_values"] is None
        current_input_ids = state["input_ids"] if is_first_token else state["input_ids"][:, -1:]
        current_position_ids = state["position_ids"] if is_first_token else state["position_ids"][:, -1:]
        
        # Forward pass through model
        try:
            outputs = forward_step(
                current_input_ids,
                state["attention_mask"],
                current_position_ids,
                state["past_key_values"],
                True  # return_dict
            )
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Current ids shape: {current_input_ids.shape}")
            logger.error(f"Current position_ids shape: {current_position_ids.shape}")
            logger.error(f"Attention mask shape: {state['attention_mask'].shape}")
            if state["past_key_values"] is not None:
                logger.error(f"Past KV is not None, count: {len(state['past_key_values'])}")
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
        
        # Update generation state
        state["input_ids"] = np.concatenate([state["input_ids"], next_token_np[:, None]], axis=1)
        state["attention_mask"] = np.concatenate([state["attention_mask"], np.ones_like(next_token_np[:, None])], axis=1)
        state["position_ids"] = np.concatenate([state["position_ids"], state["position_ids"][:, -1:] + 1], axis=1)
        state["past_key_values"] = past_key_values
        
        # Calculate token generation speed
        token_time = time.time() - token_start_time
        
        # Decode the new token and update generated text
        new_text = tokenizer.decode(state["input_ids"][0, input_ids.shape[1]:], skip_special_tokens=True)
        
        # Only process what's new since last token
        if new_text != past_text:
            added_text = new_text[len(past_text):]
            generated_text += added_text
            past_text = new_text
            
            # Call stream handler if provided
            if streamer:
                streamer(added_text)
                
            # Log token generation (only when there's actual new text)
            logger.debug(f"Generated token {i+1}/{max_decode_tokens} in {token_time:.4f}s ({1/token_time:.2f} tokens/sec)")
        
        # Check for EOS token
        if next_token_np[0] == tokenizer.eos_token_id:
            logger.info(f"Reached EOS token after generating {i+1} tokens")
            break
    
    # Log generation statistics
    total_time = time.time() - generation_start_time
    total_new_tokens = len(state["input_ids"][0]) - len(input_ids[0])
    logger.info(f"Generated {total_new_tokens} tokens in {total_time:.2f}s ({total_new_tokens/total_time:.2f} tokens/sec)")
    
    # Return the complete generated text (prompt + generation)
    return prompt_tokens + generated_text


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