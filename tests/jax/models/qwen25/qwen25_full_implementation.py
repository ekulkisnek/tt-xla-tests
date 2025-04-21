#!/usr/bin/env python3
# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen 2.5 Model Full Implementation in Flax

This file combines the model implementation and weight loading functionality
for the Qwen 2.5 model architecture in Flax.

If necessary
pip install jax jaxlib flax transformers safetensors numpy einops tqdm datasets tensorboard

Usage:
cd to qwen25
source venv/bin/activate


python qwen25_full_implementation.py --weights_dir /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --prompt "What color is the sky?"
"""

import os
import math
import json
import time
import logging
from pathlib import Path
from functools import partial
from typing import Dict, Optional, Tuple, Union, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights
from flax.linen.attention import dot_product_attention
from flax.traverse_util import flatten_dict, unflatten_dict
import numpy as np

# Import from transformers for weight loading utilities
try:
    from transformers import AutoConfig, FlaxPreTrainedModel, PretrainedConfig
    from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
    from transformers.utils.hub import get_checkpoint_shard_files, cached_file
    from transformers.utils import logging as transformers_logging
    from transformers.modeling_flax_utils import ACT2FN
    from safetensors import safe_open
    from safetensors.flax import load_file as safe_load_file
except ImportError:
    raise ImportError(
        "This script requires transformers and safetensors. "
        "Please install with: pip install transformers safetensors"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default path to weights directory (can be overridden)
DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen25-weights")

###################
# Model Definition
###################

class Qwen25Config(PretrainedConfig):
    """
    Configuration class for Qwen 2.5 model.
    """
    model_type = "qwen2"

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        sliding_window=131072,
        max_window_layers=28,
        use_sliding_window=False,
        use_mrope=False,
        bos_token_id=151643,
        eos_token_id=151643,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window
        self.use_mrope = use_mrope
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

def create_sinusoidal_positions(num_pos, dim, base=10000):
    """
    Create sinusoidal position embeddings.
    
    Args:
        num_pos: Number of positions
        dim: Dimension of the embeddings
        base: Base value for frequencies
        
    Returns:
        Sinusoidal position embeddings of shape (num_pos, 2, dim/2)
    """
    # Make sure dim is even
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension {dim} should be even")
    
    # Create position indices
    positions = jnp.arange(0, num_pos, dtype=jnp.float32)
    
    # Create frequency indices - each dimension gets a different frequency
    # For a dimension 'd', the frequency is base^(-2*d/dim)
    half_dim = dim // 2
    freq_indices = jnp.arange(0, half_dim, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (freq_indices / half_dim))
    
    # Outer product to get all combinations of positions and frequencies
    # Shape: (num_pos, dim/2)
    pos_emb = jnp.outer(positions, inv_freq)
    
    # Create sinusoids
    sin = jnp.sin(pos_emb)  # (num_pos, dim/2)
    cos = jnp.cos(pos_emb)  # (num_pos, dim/2)
    
    # Stack sin and cos together along a new dimension
    # Result shape: (num_pos, 2, dim/2)
    sincos = jnp.stack([sin, cos], axis=1)
    
    return sincos


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    
    Args:
        x: Input tensor of shape (..., d)
        
    Returns:
        Rotated tensor where the second half of hidden dimensions are negated and 
        swapped with the first half.
    """
    # Split the last dimension in half
    x1, x2 = jnp.split(x, 2, axis=-1)
    # Return the concatenation of [-x2, x1]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(tensor, sin, cos):
    """
    Apply rotary position embeddings to the input tensor.
    
    Args:
        tensor: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        sin: Sine position embeddings of shape (batch_size, 1 or num_heads, seq_len, head_dim)
        cos: Cosine position embeddings of shape (batch_size, 1 or num_heads, seq_len, head_dim)
        
    Returns:
        Tensor with rotary position embeddings applied
    """
    # Ensure sin and cos broadcast correctly if they have fewer heads
    if sin.shape[1] == 1 and tensor.shape[1] > 1:
        sin = jnp.broadcast_to(sin, tensor.shape)
        cos = jnp.broadcast_to(cos, tensor.shape)
    
    # Apply rotary embeddings: tensor * cos + rotate_half(tensor) * sin
    return tensor * cos + rotate_half(tensor) * sin


class FlaxQwen25RMSNorm(nn.Module):
    """Qwen 2.5 RMSNorm layer."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param("scale", lambda _, shape: jnp.ones(shape), (self.config.hidden_size,))
        self.variance_epsilon = self.config.rms_norm_eps

    def __call__(self, hidden_states):
        """
        Apply RMS normalization to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Normalized tensor of the same shape
        """
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * (1.0 / jnp.sqrt(variance + self.variance_epsilon))
        
        # Scale with learned parameters
        return self.weight * hidden_states

class FlaxQwen25RotaryEmbedding(nn.Module):
    """Implementation of RotaryEmbedding for Qwen 2.5."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.rope_theta = self.config.rope_theta
        self.max_position_embeddings = self.config.max_position_embeddings

    def __call__(self, key, query, position_ids):
        """
        Apply rotary position embeddings to key and query tensors.
        
        Args:
            key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            position_ids: Position IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            rotated_key, rotated_query: Key and query tensors with rotary position embeddings applied
        """
        # Get head dimension
        head_dim = query.shape[-1]
        half_head_dim = head_dim // 2
        
        # Create sinusoidal positions - shape (max_pos, 2, dim/2)
        sin_cos = create_sinusoidal_positions(
            self.max_position_embeddings, head_dim, self.rope_theta
        )
        
        # Extract the positions we need based on position_ids
        seq_length = position_ids.shape[-1]
        batch_size = position_ids.shape[0]
        
        # Reshape position_ids for proper batch and sequence dimensions
        position_ids_flat = position_ids.reshape(-1)
        
        # Ensure position ids don't exceed the precomputed values
        safe_position_ids = jnp.clip(position_ids_flat, 0, sin_cos.shape[0] - 1)
        
        # Get sin and cos for these positions
        # Extract positions from (num_pos, 2, dim/2) -> (batch*seq_len, 2, dim/2)
        sin_cos_pos = jnp.take(sin_cos, safe_position_ids, axis=0)
        
        # Split into sin and cos
        sin = sin_cos_pos[:, 0]  # Shape: (batch*seq_len, dim/2)
        cos = sin_cos_pos[:, 1]  # Shape: (batch*seq_len, dim/2)
        
        # Reshape to match the input tensors
        sin = sin.reshape(batch_size, seq_length, half_head_dim)
        cos = cos.reshape(batch_size, seq_length, half_head_dim)
        
        # Preparing sin/cos for rotary embeddings
        # We need to expand sin/cos from (batch, seq_len, dim/2) to work with (batch, n_heads, seq_len, dim)
        # First, add head dimension: (batch, 1, seq_len, dim/2)
        sin = sin[:, None, :, :]
        cos = cos[:, None, :, :]
        
        # Duplicate values to match the full head dimension
        sin = jnp.concatenate([sin, sin], axis=-1)  # (batch, 1, seq_len, dim)
        cos = jnp.concatenate([cos, cos], axis=-1)  # (batch, 1, seq_len, dim)
        
        # Apply rotary embeddings
        key_rotated = apply_rotary_pos_emb(key, sin, cos)
        query_rotated = apply_rotary_pos_emb(query, sin, cos)
        
        return key_rotated, query_rotated


class FlaxQwen25MLP(nn.Module):
    """
    Multi-layer perceptron for Qwen 2.5.
    
    Qwen 2.5 uses the SwiGLU activation function.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        
        # SwiGLU implementation: Compute gate and up projections separately
        # Following the pattern in modeling_qwen2.py
        
        # Gate projection weights and biases 
        # (up_proj in huggingface, gate_proj here)
        self.gate_proj = nn.Dense(
            intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        
        # Value projection weights and biases 
        # (down_proj in huggingface)
        self.down_proj = nn.Dense(
            hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        
        # Up projection weights and biases
        # (gate_proj in huggingface, up_proj here)
        self.up_proj = nn.Dense(
            intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        
        # Get the activation function (silu/swish)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        """
        Apply the MLP to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Project input to intermediate size (twice)
        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        
        # Apply activation to gate_output and multiply with up_output (SwiGLU)
        intermediate_output = self.act_fn(gate_output) * up_output
        
        # Project back to hidden size
        output = self.down_proj(intermediate_output)
        
        return output 

class FlaxQwen25Attention(nn.Module):
    """
    Multi-head attention for Qwen 2.5 model.
    
    Supports grouped-query attention (GQA) where the number of key/value heads
    can be smaller than the number of query heads.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    layer_idx: Optional[int] = None

    def setup(self):
        hidden_size = self.config.hidden_size
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = hidden_size // num_attention_heads
        
        # Initialize dropout value
        self.attention_dropout = self.config.attention_dropout
        
        # Initialize query, key, value projections
        self.q_proj = nn.Dense(
            num_attention_heads * head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        
        self.k_proj = nn.Dense(
            num_key_value_heads * head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        
        self.v_proj = nn.Dense(
            num_key_value_heads * head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        
        self.o_proj = nn.Dense(
            hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )
        
        # Create rotary embedding layer
        self.rotary_emb = FlaxQwen25RotaryEmbedding(
            config=self.config,
            dtype=self.dtype,
        )
        
        # Store dimensions for reference
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.use_sliding_window = self.config.use_sliding_window
        
        # Set sliding window for this layer if applicable
        if self.use_sliding_window and self.layer_idx is not None:
            self.sliding_window = self.config.sliding_window if self.layer_idx < self.config.max_window_layers else None
        else:
            self.sliding_window = None

    def _split_heads(self, hidden_states, num_heads):
        """
        Split hidden_states into separate heads
        
        Args:
            hidden_states: input tensor of shape (batch_size, seq_len, hidden_size)
            num_heads: number of attention heads
            
        Returns:
            tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        hidden_states = hidden_states.reshape(batch_size, seq_length, num_heads, self.head_dim)
        return hidden_states.transpose(0, 2, 1, 3)

    def _merge_heads(self, hidden_states):
        """
        Merge separate heads back to batch dimension
        
        Args:
            hidden_states: input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, _, seq_length, _ = hidden_states.shape
        hidden_states = hidden_states.transpose(0, 2, 1, 3)
        return hidden_states.reshape(batch_size, seq_length, self.hidden_size)

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        Concatenate key, value states from previous calls into the cache (for generation).
        
        Args:
            key: key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            value: value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            query: query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            attention_mask: mask tensor of shape (batch_size, 1, query_length, key_length)
            
        Returns:
            key, value, and attention_mask, potentially concatenated with cached key/value states.
        """
        # Check if we have the cache initialized
        is_initialized = self.has_variable("cache", "cached_key")
        
        # Shape of the cache
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
        
        if is_initialized:
            # If cache is initialized, we use the cached values and update them
            # This happens during autoregressive generation
            *batch_dims, max_length, num_heads, head_dim = cached_key.value.shape
            
            # Update the cache
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            
            # Update the index
            cache_index.value = cache_index.value + query.shape[1]
            
            # Update attention mask to include the cache length
            # This assumes attention_mask is of shape (batch_size, 1, query_length, key_length)
            attention_mask = jnp.broadcast_to(
                jnp.concatenate(
                    [
                        jnp.ones((*batch_dims, 1, 1, cur_index), dtype=jnp.bool_),
                        attention_mask,
                    ],
                    axis=-1,
                ),
                (*batch_dims, 1, query.shape[1], cur_index + query.shape[1]),
            )
        
        # Update cache variables with new values
        cached_key.value = key
        cached_value.value = value
        
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
    ):
        """
        Apply self-attention to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask tensor of shape (batch_size, 1, 1, seq_len)
            position_ids: Position IDs tensor of shape (batch_size, seq_len)
            deterministic: Whether to apply dropout (False) or not (True)
            init_cache: Whether to initialize the cache for generation
            output_attentions: Whether to return attention weights
            
        Returns:
            attn_output: Output tensor of shape (batch_size, seq_len, hidden_size)
            attn_weights: Optional attention weights
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        
        # Project hidden states to query, key, value tensors
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Split into heads
        query_states = self._split_heads(query_states, self.num_heads)
        key_states = self._split_heads(key_states, self.num_kv_heads)
        value_states = self._split_heads(value_states, self.num_kv_heads)
        
        # Apply rotary positional embeddings to query and key states
        if position_ids is not None:
            key_states, query_states = self.rotary_emb(key_states, query_states, position_ids)
        
        # If we're initializing the cache for generation, setup the cache
        if init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )
        
        # If key and value heads are fewer than query heads, repeat them to match
        if self.num_kv_heads < self.num_heads:
            # Calculate the repeat factor for key/value heads
            num_repeat = self.num_heads // self.num_kv_heads
            
            # Repeat the key and value states to match the number of query heads
            # Using JAX's repeat to tile the KV heads
            key_states = jnp.repeat(key_states, num_repeat, axis=1)
            value_states = jnp.repeat(value_states, num_repeat, axis=1)
        
        # Create sliding window attention mask if needed
        if self.sliding_window is not None and self.causal and self.sliding_window < seq_length:
            # Determine the size of the sliding window
            window_size = self.sliding_window
            
            # Generate a mask that limits context to the sliding window
            # The causal mask already ensures we don't look at future tokens,
            # so we just need to limit how far back we can look
            
            # First create a mask that is 1 for positions within the window
            mask_indices = jnp.arange(seq_length)
            mask_indices = jnp.expand_dims(mask_indices, axis=0) - jnp.expand_dims(mask_indices, axis=1)
            window_mask = (mask_indices > -window_size) & (mask_indices <= 0)
            window_mask = window_mask.astype(jnp.float32)
            
            # Combine with the original attention mask
            if attention_mask is not None:
                # Ensure window_mask has the right shape
                window_mask = window_mask.reshape(1, 1, seq_length, seq_length)
                attention_mask = attention_mask * window_mask
            else:
                attention_mask = window_mask.reshape(1, 1, seq_length, seq_length)
        
        # Compute attention weights and dropout
        dropout_rng = None
        if not deterministic and self.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")
        
        # Apply attention mechanism
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            attention_mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
        )
        
        # Save attention weights for output if requested
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        
        if output_attentions:
            outputs = outputs + (attn_weights,)
            
        return outputs 

class FlaxQwen25DecoderLayer(nn.Module):
    """
    Decoder layer for Qwen 2.5 model.
    
    This layer includes:
    1. Input normalization
    2. Self-attention
    3. Post-attention normalization
    4. MLP (feed-forward network)
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    layer_idx: Optional[int] = None

    def setup(self):
        # Normalization layers
        self.input_layernorm = FlaxQwen25RMSNorm(config=self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxQwen25RMSNorm(config=self.config, dtype=self.dtype)
        
        # Attention layer
        self.attention = FlaxQwen25Attention(
            config=self.config,
            dtype=self.dtype,
            layer_idx=self.layer_idx,
        )
        
        # MLP layer
        self.mlp = FlaxQwen25MLP(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        Apply the decoder layer to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask tensor of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position IDs tensor of shape (batch_size, seq_len)
            deterministic: Whether to apply dropout (False) or not (True)
            init_cache: Whether to initialize the cache for generation
            output_attentions: Whether to return attention weights
            
        Returns:
            hidden_states: Output tensor
            attn_weights: Optional attention weights
        """
        # Residual connection preparation
        residual = hidden_states
        
        # Normalize input (pre-layernorm architecture)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply self-attention
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        
        # Add attention output to residual (first residual connection)
        hidden_states = attn_output + residual
        
        # Second residual connection preparation
        residual = hidden_states
        
        # Normalize post-attention (pre-layernorm for MLP)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP
        feed_forward_output = self.mlp(hidden_states)
        
        # Add MLP output to residual (second residual connection)
        hidden_states = feed_forward_output + residual
        
        outputs = (hidden_states,)
        
        # Include attention weights if requested
        if output_attentions:
            outputs += (attn_outputs[1],)
            
        return outputs


class FlaxQwen25LayerCollection(nn.Module):
    """
    Collection of decoder layers making up the Qwen 2.5 Transformer encoder.
    
    This module stacks multiple decoder layers.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Create the decoder layers
        self.layers = [
            FlaxQwen25DecoderLayer(
                config=self.config,
                dtype=self.dtype,
                layer_idx=i,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        
        # Final layer norm after all the decoder layers
        self.norm = FlaxQwen25RMSNorm(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        """
        Apply the transformer layers to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask tensor
            position_ids: Position IDs tensor
            deterministic: Whether to apply dropout
            init_cache: Whether to initialize the cache for generation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            hidden_states: Output tensor
            all_hidden_states: Optional all hidden states
            all_attentions: Optional all attention weights
        """
        # Initialize lists to store attention weights and hidden states if needed
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Apply each decoder layer
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add the final hidden states if needed
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Return as dict or tuple based on return_dict flag
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs
        
        # Return as dictionary
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        } 

class FlaxQwen25Module(nn.Module):
    """
    The main Qwen 2.5 model module.
    
    This module includes:
    1. Token embeddings
    2. Layer collection (transformer layers)
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Token embeddings
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        # Layer collection (transformer layers)
        self.layers = FlaxQwen25LayerCollection(
            config=self.config,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Apply the Qwen 2.5 model to the input token IDs.
        
        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask tensor
            position_ids: Position IDs tensor
            deterministic: Whether to apply dropout
            init_cache: Whether to initialize the cache for generation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            last_hidden_state: Output tensor
            hidden_states: Optional all hidden states
            attentions: Optional all attention weights
        """
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]
        
        # Convert input IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        
        # Convert attention mask to causal mask
        # Shape will be (batch_size, 1, seq_len, seq_len)
        extended_attention_mask = make_causal_mask(attention_mask, dtype=self.dtype)
        
        # Apply transformer layers
        outputs = self.layers(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return outputs


class FlaxQwen25PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = Qwen25Config
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Qwen25Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        """
        Initialize a FlaxQwen25PreTrainedModel.
        
        Args:
            config: Model configuration
            input_shape: Input shape for initializing parameters
            seed: Random seed
            dtype: Data type
            _do_init: Whether to initialize parameters
            
        """
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        """
        Initialize weights for the model.
        
        Args:
            rng: Random number generator key
            input_shape: Shape of input for parameter initialization
            params: Existing parameters to update
            
        Returns:
            Initialized parameters
        """
        # Split RNG key
        rng, key = jax.random.split(rng)
        
        # Create random input IDs
        input_ids = jax.random.randint(
            key,
            shape=input_shape,
            minval=0,
            maxval=self.config.vocab_size,
        )
        
        # Initialize the attention mask
        attention_mask = jnp.ones_like(input_ids)
        
        # Same for position IDs
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
            input_shape
        )
        
        # Random key for parameter initialization
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        
        # Initialize or update parameters
        if params is None:
            return self.module.init(rngs, input_ids, attention_mask, position_ids)
        else:
            return params

    def init_cache(self, batch_size, max_length):
        """
        Initialize cache for autoregressive generation.
        
        Args:
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            Initialized cache
        """
        # Create a dict to hold the cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
            input_ids.shape
        )
        
        # Initialize the model with cache
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            init_cache=True,
        )
        
        return init_variables["cache"]


def make_causal_mask(attention_mask: jnp.ndarray, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """
    Create a causal mask for self-attention.
    
    Args:
        attention_mask: Attention mask of shape (batch_size, seq_len)
        dtype: Data type for the mask
        
    Returns:
        Causal mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, sequence_length = attention_mask.shape
    
    # Create a causal mask that prevents attending to future tokens
    # First create a matrix where each position (i,j) contains j
    idxs = jnp.broadcast_to(jnp.arange(sequence_length), (sequence_length, sequence_length))
    
    # Then create a matrix where each position (i,j) contains i
    idxs_t = jnp.transpose(idxs)
    
    # This creates a mask where position (i,j) is 1 if j<=i, and 0 otherwise
    causal_mask = idxs >= idxs_t
    
    # Convert to float and adjust for proper masking in attention
    causal_mask = causal_mask.astype(dtype)
    
    # We want positions with value 0 to be masked out, so they should correspond
    # to -inf before softmax. But we represent them as 0 now for proper broadcasting
    causal_mask = causal_mask[None, None, :, :]  # Add batch and head dimensions
    
    # Now use the original attention_mask to mask out padding tokens as well
    # First expand attention_mask to 4D
    extended_attention_mask = attention_mask[:, None, None, :]
    
    # 1 for tokens to attend to, 0 for tokens to ignore
    extended_attention_mask = extended_attention_mask.astype(dtype)
    
    # Combine causal mask and attention mask
    # Convert 0s in attention mask to -inf for masked positions
    extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(dtype).min
    
    # Combine with causal mask to ensure both causality and proper padding mask
    return causal_mask * extended_attention_mask 

class FlaxQwen25Model(FlaxQwen25PreTrainedModel):
    """
    The base Qwen 2.5 Model transformer outputting raw hidden-states without any specific head on top.
    """
    module_class = FlaxQwen25Module


class FlaxQwen25ForCausalLMModule(nn.Module):
    """
    Qwen 2.5 model with a language modeling head.
    
    This module includes the Qwen 2.5 base model and a linear layer for language modeling.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Base Qwen 2.5 model
        self.model = FlaxQwen25Module(
            config=self.config,
            dtype=self.dtype,
        )
        
        # Language modeling head
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(self.config.initializer_range),
            use_bias=False,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Apply the Qwen 2.5 language model to the input.
        
        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask tensor
            position_ids: Position IDs tensor
            deterministic: Whether to apply dropout
            init_cache: Whether to initialize the cache for generation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            logits: Output logits tensor
            hidden_states: Optional all hidden states
            attentions: Optional all attention weights
        """
        # Apply the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the last hidden state
        if return_dict:
            hidden_states = outputs["last_hidden_state"]
        else:
            hidden_states = outputs[0]
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            outputs = (logits,) + outputs[1:]
            return outputs
        
        return {
            "logits": logits,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }


class FlaxQwen25ForCausalLM(FlaxQwen25PreTrainedModel):
    """
    The Qwen 2.5 Model transformer with a language modeling head on top.
    """
    module_class = FlaxQwen25ForCausalLMModule

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        train=False,
        add_params_field=False,
        mutable=False,
    ):
        """
        Call the Qwen 2.5 language model.
        
        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask tensor
            position_ids: Position IDs tensor
            params: Model parameters
            past_key_values: Past key values for generation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary or tuple
            train: Whether in training mode (affects dropout)
            add_params_field: Whether to include parameters in output
            mutable: List of fields that should be mutable
            
        Returns:
            logits: Output logits tensor
            hidden_states: Optional all hidden states
            attentions: Optional all attention weights
        """
        # Extract configuration for default values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Handle past key values for efficient generation
        if past_key_values is not None:
            if len(past_key_values) != 0:
                input_ids = input_ids[:, -1:]
                mutable = ["cache"]  # make sure to mutate cache when using past key values
                
                # position_ids for the current token should be the position of the last token + 1
                if position_ids is None:
                    position_ids = jnp.ones((input_ids.shape[0], 1), dtype=jnp.int32) * past_key_values.cache_index
                else:
                    position_ids = position_ids[:, -1:]
        
        # Generate position IDs if not provided
        if position_ids is None:
            batch_size, seq_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))
        
        # Call the model
        mutable = mutable if mutable else False
        
        # Handle outputs when using past key values
        outputs = self.module.apply(
            {"params": params or self.params} if not add_params_field else {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4") if attention_mask is not None else None,
            jnp.array(position_ids, dtype="i4"),
            not train,  # deterministic: not training
            past_key_values is not None,  # initialize cache if past_key_values is not None
            output_attentions,
            output_hidden_states,
            return_dict,
            mutable=mutable,
            rngs={} if not train else {"dropout": self.key},
        )
        
        if mutable and past_key_values is not None:
            if return_dict:
                outputs, mutated = outputs
            else:
                output_arrays, mutated = outputs
                outputs = (output_arrays[0], ) + (output_arrays[1:] if len(output_arrays) > 1 else ())
                
            # Add the mutated states to the output
            past_key_values = {"cache": mutated["cache"]}
            
            if return_dict:
                outputs["past_key_values"] = past_key_values
            else:
                outputs = outputs + (past_key_values,)
        
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs
    ):
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Token ID tensor
            max_length: Maximum sequence length
            attention_mask: Attention mask tensor
            position_ids: Position IDs tensor
            past_key_values: Past key values for generation
            
        Returns:
            Dictionary of prepared inputs
        """
        batch_size, seq_length = input_ids.shape
        
        # Handle attention_mask
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        
        # past key values implies we're doing generation
        if past_key_values is not None:
            # attention mask for the decoder, where 1 is not masked and 0 is masked
            # for each input sequence, we need to extend the mask with the past key values
            if attention_mask is not None:
                # We need to extend the attention mask to account for the past key values
                if seq_length > 1:
                    # input_ids.shape = (batch_size, seq_length) -> (batch_size, 1)
                    input_ids = input_ids[:, -1:]
                    attention_mask = attention_mask[:, -1:]
                    if position_ids is not None:
                        position_ids = position_ids[:, -1:]
            
            # We create position ids that account for the past key values
            if position_ids is None:
                # Create position IDs, accounting for cached key/values
                if past_key_values.cache_index is not None:
                    # We need to create position IDs that account for the past key values
                    # to do this properly, we need to know how many items are in the cache
                    position_ids = jnp.ones((batch_size, 1), dtype=jnp.int32) * past_key_values.cache_index
                else:
                    # If there's no cache_index, assume we're at position 0
                    position_ids = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        
        # If position_ids is still None, create them for the full sequence
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length)[None, :],
                (batch_size, seq_length)
            )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "mutable": ["cache"],
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """
        Update inputs for generation.
        
        Args:
            model_outputs: Outputs from the model
            model_kwargs: Keyword arguments for the model
            
        Returns:
            Updated model keyword arguments
        """
        model_kwargs["past_key_values"] = model_outputs.get("past_key_values")
        
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            batch_size, seq_length = attention_mask.shape
            
            # If we're using past key values, we need to extend the attention mask
            if model_kwargs["past_key_values"] is not None:
                # Extend the attention mask to account for the new token
                extended_attention_mask = jnp.ones((batch_size, 1), dtype=attention_mask.dtype)
                model_kwargs["attention_mask"] = jnp.concatenate(
                    [attention_mask, extended_attention_mask], axis=1
                )
        
        # update position ids
        if "position_ids" in model_kwargs and model_kwargs["past_key_values"] is not None:
            position_ids = model_kwargs["position_ids"]
            batch_size, seq_length = position_ids.shape
            
            # Update position IDs for the next token
            if model_kwargs["past_key_values"].cache_index is not None:
                # We should increment the position IDs
                next_position = model_kwargs["past_key_values"].cache_index
                model_kwargs["position_ids"] = jnp.ones((batch_size, 1), dtype=position_ids.dtype) * next_position
            else:
                # If there's no cache_index, just increment position_ids by 1
                model_kwargs["position_ids"] = position_ids[:, -1:] + 1
        
        return model_kwargs


###################
# Weight Loading
###################

def load_config_from_json(config_file):
    """
    Load Qwen 2.5 config from a JSON file.
    
    Args:
        config_file: Path to the config file
        
    Returns:
        Qwen25Config object
    """
    # Check if file exists
    if not os.path.exists(config_file):
        raise ValueError(f"Config file not found at {config_file}")
    
    # Load JSON config
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    # Convert to Qwen25Config
    return Qwen25Config(**config_dict)


def load_safetensors_weights(model_path):
    """
    Load weights from safetensors files.
    
    Args:
        model_path: Path to the directory containing safetensors files
        
    Returns:
        Dictionary of tensor parameters
    """
    from safetensors import safe_open  # Import here to ensure it's available
    
    logger.info(f"Loading safetensors weights from {model_path}")
    
    # Check for safetensors index file
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    safetensors_files = []
    
    if os.path.exists(index_file):
        logger.info(f"Found safetensors index file: {index_file}")
        try:
            with open(index_file, "r") as f:
                index = json.load(f)
                if "weight_map" in index:
                    weight_map = index["weight_map"]
                    files = sorted(list(set(weight_map.values())))
                    safetensors_files = [os.path.join(model_path, f) for f in files]
                    logger.info(f"Found {len(safetensors_files)} files in weight map")
        except Exception as e:
            logger.warning(f"Error reading index file: {e}")
            safetensors_files = []
    
    # If no index file or no files found from index, look for safetensors files directly
    if not safetensors_files:
        import glob
        safetensors_files = sorted(glob.glob(os.path.join(model_path, "model-*-of-*.safetensors")))
        logger.info(f"No index file found or empty, searching for safetensors files directly: found {len(safetensors_files)} files")
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    logger.info(f"Found {len(safetensors_files)} safetensors files: {[os.path.basename(f) for f in safetensors_files]}")
    
    # Load all tensors into a flat dictionary
    all_params = {}
    
    # Try to load each file
    for file_path in safetensors_files:
        logger.info(f"Loading weights from {file_path}")
        try:
            with safe_open(file_path, framework="flax") as f:
                # Get a list of keys in this file
                keys = f.keys()
                logger.info(f"File {os.path.basename(file_path)} contains {len(keys)} keys")
                
                # Load each tensor
                for key in keys:
                    try:
                        all_params[key] = f.get_tensor(key)
                    except Exception as tensor_err:
                        logger.warning(f"Error loading tensor {key}: {tensor_err}")
        except Exception as file_err:
            logger.warning(f"Error loading file {file_path}: {file_err}")
    
    # Convert flat dictionary to nested dictionary
    nested_params = unflatten_dict(all_params, sep=".")
    
    logger.info(f"Loaded {len(all_params)} parameters from safetensors files")
    return nested_params


def load_model_and_weights(model_path, dtype=jnp.float32):
    """
    Load the Qwen 2.5 model and weights.
    
    Args:
        model_path: Path to the directory containing model files
        dtype: Data type for the model parameters
        
    Returns:
        FlaxQwen25ForCausalLM model with loaded weights
    """
    # Import necessary libraries
    from safetensors import safe_open
    import glob
    
    # Load config
    config_file = os.path.join(model_path, "config.json")
    config = load_config_from_json(config_file)
    
    # Skip transformers loading - Qwen2 models aren't fully supported in Flax yet
    logger.info("Using direct initialization and weight loading for Qwen2.5 model")
    
    # Create model with initialization
    model = FlaxQwen25ForCausalLM(config, dtype=dtype, _do_init=True)
    
    try:
        # First try using the direct_load_from_index method
        logger.info("Attempting to load weights using direct_load_from_index")
        return direct_load_from_index(model, model_path)
    except Exception as e1:
        logger.warning(f"direct_load_from_index failed: {e1}")
        logger.info("Falling back to load_safetensors_weights")
        
        try:
            # Load weights directly from safetensors
            logger.info("Loading weights from safetensors files directly")
            # Load weights into a flat dictionary
            params_dict = load_safetensors_weights(model_path)
            
            # Use our parameter remapping function to map to correct structure
            old_params = model.params.unfreeze() if hasattr(model.params, 'unfreeze') else model.params.copy()
            params = remap_parameter_structure(params_dict, old_params)
            
            # Freeze and update the model parameters
            model.params = freeze(params)
            logger.info("Successfully loaded weights using load_safetensors_weights")
            return model
            
        except Exception as e2:
            logger.error(f"load_safetensors_weights failed: {e2}")
            logger.info("Falling back to direct loading as a last resort")
            
            # If all else fails, try one last approach with simpler loading
            # Extract model parameters
            if "model" in params_dict:
                pt_params = params_dict["model"]
            else:
                pt_params = params_dict
            
            # Convert parameters to the proper format for JAX
            # This process handles any necessary conversions
            flax_params = {}
            
            # Process embed_tokens
            if "embed_tokens" in pt_params:
                flax_params["embed_tokens"] = {"embedding": pt_params["embed_tokens"]["weight"]}
            
            # Process norm
            if "norm" in pt_params:
                flax_params["layers"] = {"norm": {"scale": pt_params["norm"]["weight"]}}
            
            # Process layers
            if "layers" in pt_params:
                if "layers" not in flax_params:
                    flax_params["layers"] = {}
                
                layers = pt_params["layers"]
                
                # Initialize layer collection if not present
                if "layers" not in flax_params["layers"]:
                    flax_params["layers"]["layers"] = {}
                
                # Process each layer
                for i in range(config.num_hidden_layers):
                    layer_str = str(i)
                    if layer_str in layers:
                        layer = layers[layer_str]
                        
                        layer_dict = {}
                        
                        # Add input layernorm
                        if "input_layernorm" in layer and "weight" in layer["input_layernorm"]:
                            layer_dict["input_layernorm"] = {"scale": layer["input_layernorm"]["weight"]}
                        
                        # Add post attention layernorm
                        if "post_attention_layernorm" in layer and "weight" in layer["post_attention_layernorm"]:
                            layer_dict["post_attention_layernorm"] = {"scale": layer["post_attention_layernorm"]["weight"]}
                        
                        # Add MLP
                        mlp_dict = {}
                        if "mlp" in layer:
                            mlp = layer["mlp"]
                            
                            if "gate_proj" in mlp and "weight" in mlp["gate_proj"]:
                                mlp_dict["gate_proj"] = {"kernel": mlp["gate_proj"]["weight"].T}
                            
                            if "up_proj" in mlp and "weight" in mlp["up_proj"]:
                                mlp_dict["up_proj"] = {"kernel": mlp["up_proj"]["weight"].T}
                            
                            if "down_proj" in mlp and "weight" in mlp["down_proj"]:
                                mlp_dict["down_proj"] = {"kernel": mlp["down_proj"]["weight"].T}
                        
                        if mlp_dict:
                            layer_dict["mlp"] = mlp_dict
                        
                        # Add attention
                        attn_dict = {}
                        if "self_attn" in layer:
                            attn = layer["self_attn"]
                            
                            if "q_proj" in attn and "weight" in attn["q_proj"]:
                                attn_dict["q_proj"] = {"kernel": attn["q_proj"]["weight"].T}
                                if "bias" in attn["q_proj"]:
                                    attn_dict["q_proj"]["bias"] = attn["q_proj"]["bias"]
                            
                            if "k_proj" in attn and "weight" in attn["k_proj"]:
                                attn_dict["k_proj"] = {"kernel": attn["k_proj"]["weight"].T}
                                if "bias" in attn["k_proj"]:
                                    attn_dict["k_proj"]["bias"] = attn["k_proj"]["bias"]
                            
                            if "v_proj" in attn and "weight" in attn["v_proj"]:
                                attn_dict["v_proj"] = {"kernel": attn["v_proj"]["weight"].T}
                                if "bias" in attn["v_proj"]:
                                    attn_dict["v_proj"]["bias"] = attn["v_proj"]["bias"]
                            
                            if "o_proj" in attn and "weight" in attn["o_proj"]:
                                attn_dict["o_proj"] = {"kernel": attn["o_proj"]["weight"].T}
                        
                        if attn_dict:
                            layer_dict["attention"] = attn_dict
                        
                        # Add layer to the model if it has any parameters
                        if layer_dict:
                            flax_params["layers"]["layers"][layer_str] = layer_dict
            
            # Add language model head
            if "lm_head" in params_dict:
                flax_params["lm_head"] = {"kernel": params_dict["lm_head"]["weight"].T}
            
            # Wrap in params field for proper structure
            params = {"params": flax_params}
            
            # Freeze and set parameters
            model.params = freeze(params)
            logger.info("Successfully loaded weights using fallback approach")
            return model


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """
    Generate text using the model.
    
    Args:
        model: The Qwen 2.5 model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Text prompt to generate from
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Generated text
    """
    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="jax").input_ids
    
    # Set up generation config
    gen_kwargs = {
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    # Generate
    output_ids = model.generate(input_ids, **gen_kwargs)
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """
    Main function to demonstrate the model.
    """
    import argparse
    import glob
    from safetensors import safe_open  # Import here to ensure it's available
    
    parser = argparse.ArgumentParser(description="Run Qwen 2.5 model")
    parser.add_argument("--weights_dir", type=str, default=DEFAULT_WEIGHTS_DIR, help="Path to weights directory")
    parser.add_argument("--prompt", type=str, default="Hello, I am a language model", help="Text prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--load_only", action="store_true", help="Only load the model without generating")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--direct_load", action="store_true", help="Use direct loading method")
    args = parser.parse_args()
    
    # Set up debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify weights directory existence
    if not os.path.exists(args.weights_dir):
        logger.error(f"Weights directory {args.weights_dir} does not exist")
        return
    
    # List files in the directory for debugging
    logger.info(f"Contents of weights directory {args.weights_dir}:")
    weights_files = os.listdir(args.weights_dir)
    for file in weights_files:
        file_path = os.path.join(args.weights_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # size in MB
        logger.info(f"  {file} ({file_size:.2f} MB)")
    
    # Load config first to show progress
    config_file = os.path.join(args.weights_dir, "config.json")
    if os.path.exists(config_file):
        logger.info(f"Loading config from {config_file}")
        config = load_config_from_json(config_file)
        logger.info(f"Config loaded with {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    else:
        logger.error(f"Config file not found at {config_file}")
        return
    
    try:
        from transformers import AutoTokenizer
        
        # Load tokenizer first to show progress
        logger.info(f"Loading tokenizer from {args.weights_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.weights_dir)
        logger.info(f"Tokenizer loaded with vocabulary size {len(tokenizer)}")
        
        # Load model and weights
        logger.info(f"Loading model from {args.weights_dir}")
        
        # Try different loading methods
        model = None
        
        if args.direct_load:
            try:
                # Create model instance
                logger.info("Creating model instance")
                model = FlaxQwen25ForCausalLM(config, dtype=jnp.float32, _do_init=True)
                
                # Use direct loading from index
                model = direct_load_from_index(model, args.weights_dir)
                logger.info("Model loaded successfully with direct loading from index!")
            except Exception as e:
                logger.error(f"Direct loading failed: {e}")
                import traceback
                traceback.print_exc()
                model = None
        
        # If model is still None, try the standard loading method
        if model is None:
            try:
                logger.info("Trying standard loading method")
                model = load_model_and_weights(args.weights_dir)
                logger.info("Model loaded successfully with standard method!")
            except Exception as e:
                logger.error(f"Standard loading failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Last resort: try simple direct loading without using the index
                try:
                    logger.info("Trying direct loading without index as last resort")
                    
                    # Create model instance
                    model = FlaxQwen25ForCausalLM(config, dtype=jnp.float32, _do_init=True)
                    
                    # Load safetensors files directly
                    safetensors_files = sorted(glob.glob(os.path.join(args.weights_dir, "model-*-of-*.safetensors")))
                    
                    if not safetensors_files:
                        logger.error(f"No safetensors files found in {args.weights_dir}")
                        return
                    
                    # Simple structure to collect weights
                    all_weights = {}
                    
                    # Load each file
                    for file_path in safetensors_files:
                        logger.info(f"Loading weights from {file_path}")
                        with safe_open(file_path, framework="flax") as f:
                            for key in f.keys():
                                all_weights[key] = f.get_tensor(key)
                    
                    logger.info(f"Loaded a total of {len(all_weights)} parameter tensors")
                    
                    # Create an expected parameter structure for the model
                    params = model.params.unfreeze()
                    
                    # Process load and map weights directly
                    # This is a simplified parameter mapping - just to get the model to load
                    logger.info("Mapping weights to model parameters...")
                    
                    # Initialize needed structures
                    params["params"] = {
                        "embed_tokens": {"embedding": None},
                        "layers": {
                            "norm": {"scale": None},
                            "layers": {}
                        },
                        "lm_head": {"kernel": None}
                    }
                    
                    # Map parameters from the weights to the model parameters
                    # Embedding
                    if "model.embed_tokens.weight" in all_weights:
                        params["params"]["embed_tokens"]["embedding"] = all_weights["model.embed_tokens.weight"]
                    
                    # Final layer norm
                    if "model.norm.weight" in all_weights:
                        params["params"]["layers"]["norm"]["scale"] = all_weights["model.norm.weight"]
                    
                    # Language model head
                    if "lm_head.weight" in all_weights:
                        params["params"]["lm_head"]["kernel"] = all_weights["lm_head.weight"].T
                    
                    # Process all layers
                    for layer_idx in range(config.num_hidden_layers):
                        layer_key = f"model.layers.{layer_idx}"
                        
                        # Create layer dictionary
                        params["params"]["layers"]["layers"][str(layer_idx)] = {
                            "input_layernorm": {"scale": None},
                            "post_attention_layernorm": {"scale": None},
                            "attention": {
                                "q_proj": {"kernel": None},
                                "k_proj": {"kernel": None},
                                "v_proj": {"kernel": None},
                                "o_proj": {"kernel": None},
                            },
                            "mlp": {
                                "gate_proj": {"kernel": None},
                                "up_proj": {"kernel": None},
                                "down_proj": {"kernel": None},
                            }
                        }
                        
                        # Layer norms
                        if f"{layer_key}.input_layernorm.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["input_layernorm"]["scale"] = \
                                all_weights[f"{layer_key}.input_layernorm.weight"]
                            
                        if f"{layer_key}.post_attention_layernorm.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["post_attention_layernorm"]["scale"] = \
                                all_weights[f"{layer_key}.post_attention_layernorm.weight"]
                        
                        # Self-attention
                        if f"{layer_key}.self_attn.q_proj.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]["kernel"] = \
                                all_weights[f"{layer_key}.self_attn.q_proj.weight"].T
                            
                        if f"{layer_key}.self_attn.k_proj.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]["kernel"] = \
                                all_weights[f"{layer_key}.self_attn.k_proj.weight"].T
                            
                        if f"{layer_key}.self_attn.v_proj.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]["kernel"] = \
                                all_weights[f"{layer_key}.self_attn.v_proj.weight"].T
                            
                        if f"{layer_key}.self_attn.o_proj.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["o_proj"]["kernel"] = \
                                all_weights[f"{layer_key}.self_attn.o_proj.weight"].T
                        
                        # Add biases if present
                        if f"{layer_key}.self_attn.q_proj.bias" in all_weights:
                            if "bias" not in params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]:
                                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]["bias"] = None
                            params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]["bias"] = \
                                all_weights[f"{layer_key}.self_attn.q_proj.bias"]
                            
                        if f"{layer_key}.self_attn.k_proj.bias" in all_weights:
                            if "bias" not in params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]:
                                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]["bias"] = None
                            params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]["bias"] = \
                                all_weights[f"{layer_key}.self_attn.k_proj.bias"]
                            
                        if f"{layer_key}.self_attn.v_proj.bias" in all_weights:
                            if "bias" not in params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]:
                                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]["bias"] = None
                            params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]["bias"] = \
                                all_weights[f"{layer_key}.self_attn.v_proj.bias"]
                        
                        # MLP
                        if f"{layer_key}.mlp.gate_proj.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["mlp"]["gate_proj"]["kernel"] = \
                                all_weights[f"{layer_key}.mlp.gate_proj.weight"].T
                            
                        if f"{layer_key}.mlp.up_proj.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["mlp"]["up_proj"]["kernel"] = \
                                all_weights[f"{layer_key}.mlp.up_proj.weight"].T
                            
                        if f"{layer_key}.mlp.down_proj.weight" in all_weights:
                            params["params"]["layers"]["layers"][str(layer_idx)]["mlp"]["down_proj"]["kernel"] = \
                                all_weights[f"{layer_key}.mlp.down_proj.weight"].T
                    
                    # Remove any None values from params (where the weights weren't found)
                    def remove_none_values(d):
                        if isinstance(d, dict):
                            return {k: remove_none_values(v) for k, v in d.items() if v is not None}
                        return d
                    
                    params = remove_none_values(params)
                    
                    # Freeze parameters and update the model
                    model.params = freeze(params)
                    
                    logger.info("Model created with direct loaded weights")
                    
                except Exception as last_e:
                    logger.error(f"All loading methods failed: {last_e}")
                    import traceback
                    traceback.print_exc()
                    return
        
        # If load only, don't generate
        if args.load_only:
            logger.info("Model loaded successfully!")
            return
        
        # Generate text
        logger.info(f"Generating text for prompt: {args.prompt}")
        try:
            # Encode the prompt
            input_ids = tokenizer(args.prompt, return_tensors="jax").input_ids
            
            # Set up generation config
            gen_kwargs = {
                "max_length": args.max_length,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            # Generate
            logger.info("Starting generation...")
            output_ids = model.generate(input_ids, **gen_kwargs)
            
            # Decode
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            print(f"\nPrompt: {args.prompt}")
            print(f"Generated: {generated_text}")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure transformers and safetensors are installed.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 

def direct_load_from_index(model, model_path):
    """
    Load model weights directly using the model.safetensors.index.json file
    
    Args:
        model: The FlaxQwen25ForCausalLM model instance
        model_path: Path to the model directory
        
    Returns:
        Model with weights loaded
    """
    from safetensors import safe_open  # Import here to ensure it's available
    
    logger.info("Using direct loading from index file")
    
    # First check if the index file exists
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise ValueError(f"Index file not found at {index_file}")
    
    # Load the index file
    with open(index_file, "r") as f:
        index = json.load(f)
    
    # Extract the weight map
    if "weight_map" not in index:
        raise ValueError(f"Invalid index file format at {index_file}")
    
    weight_map = index["weight_map"]
    logger.info(f"Index file contains {len(weight_map)} parameter mappings")
    
    # Create a dict to store all parameters
    all_weights = {}
    
    # Keep track of loaded files to avoid loading the same file multiple times
    loaded_files = {}
    
    # Load weights using the weight map
    for param_name, file_name in weight_map.items():
        file_path = os.path.join(model_path, file_name)
        
        # Load the file if not already loaded
        if file_path not in loaded_files:
            logger.info(f"Loading weights from {file_path}")
            with safe_open(file_path, framework="flax") as f:
                loaded_files[file_path] = {k: f.get_tensor(k) for k in f.keys()}
        
        # Get the parameter from the loaded file
        if param_name in loaded_files[file_path]:
            all_weights[param_name] = loaded_files[file_path][param_name]
        else:
            logger.warning(f"Parameter {param_name} not found in {file_path}")
    
    logger.info(f"Loaded {len(all_weights)} parameters")
    
    # Get expected parameter structure from model
    # Handle both dictionary and FrozenDict
    if hasattr(model.params, 'unfreeze'):
        params = model.params.unfreeze()
    else:
        # It's already a dict, make a copy to avoid modifying original
        params = {k: v.copy() if isinstance(v, dict) else v for k, v in model.params.items()}
    
    # Create placeholders for the parameters we expect
    # This ensures we have the correct structure even if some weights are missing
    try:
        # Try using the parameter remapping function
        logger.info("Using parameter remapping function for correct structure")
        params = remap_parameter_structure(all_weights, params)
    except Exception as e:
        logger.warning(f"Parameter remapping failed: {e}, falling back to direct approach")
        
        # Initialize needed structures
        params["params"] = {
            "embed_tokens": {"embedding": None},
            "layers": {
                "norm": {"scale": None},
                "layers": {}
            },
            "lm_head": {"kernel": None}
        }
        
        # Map parameters from the weights to the model parameters
        # Embedding
        if "model.embed_tokens.weight" in all_weights:
            params["params"]["embed_tokens"]["embedding"] = all_weights["model.embed_tokens.weight"]
        
        # Final layer norm
        if "model.norm.weight" in all_weights:
            params["params"]["layers"]["norm"]["scale"] = all_weights["model.norm.weight"]
        
        # Language model head
        if "lm_head.weight" in all_weights:
            params["params"]["lm_head"]["kernel"] = all_weights["lm_head.weight"].T
        
        # Process all layers
        config = model.config
        for layer_idx in range(config.num_hidden_layers):
            layer_key = f"model.layers.{layer_idx}"
            
            # Create layer dictionary
            params["params"]["layers"]["layers"][str(layer_idx)] = {
                "input_layernorm": {"scale": None},
                "post_attention_layernorm": {"scale": None},
                "attention": {
                    "q_proj": {"kernel": None},
                    "k_proj": {"kernel": None},
                    "v_proj": {"kernel": None},
                    "o_proj": {"kernel": None},
                },
                "mlp": {
                    "gate_proj": {"kernel": None},
                    "up_proj": {"kernel": None},
                    "down_proj": {"kernel": None},
                }
            }
            
            # Layer norms
            if f"{layer_key}.input_layernorm.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["input_layernorm"]["scale"] = \
                    all_weights[f"{layer_key}.input_layernorm.weight"]
                
            if f"{layer_key}.post_attention_layernorm.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["post_attention_layernorm"]["scale"] = \
                    all_weights[f"{layer_key}.post_attention_layernorm.weight"]
            
            # Self-attention
            if f"{layer_key}.self_attn.q_proj.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]["kernel"] = \
                    all_weights[f"{layer_key}.self_attn.q_proj.weight"].T
                
            if f"{layer_key}.self_attn.k_proj.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]["kernel"] = \
                    all_weights[f"{layer_key}.self_attn.k_proj.weight"].T
                
            if f"{layer_key}.self_attn.v_proj.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]["kernel"] = \
                    all_weights[f"{layer_key}.self_attn.v_proj.weight"].T
                
            if f"{layer_key}.self_attn.o_proj.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["o_proj"]["kernel"] = \
                    all_weights[f"{layer_key}.self_attn.o_proj.weight"].T
            
            # Add biases if present
            if f"{layer_key}.self_attn.q_proj.bias" in all_weights:
                if "bias" not in params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]:
                    params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]["bias"] = None
                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["q_proj"]["bias"] = \
                    all_weights[f"{layer_key}.self_attn.q_proj.bias"]
                
            if f"{layer_key}.self_attn.k_proj.bias" in all_weights:
                if "bias" not in params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]:
                    params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]["bias"] = None
                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["k_proj"]["bias"] = \
                    all_weights[f"{layer_key}.self_attn.k_proj.bias"]
                
            if f"{layer_key}.self_attn.v_proj.bias" in all_weights:
                if "bias" not in params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]:
                    params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]["bias"] = None
                params["params"]["layers"]["layers"][str(layer_idx)]["attention"]["v_proj"]["bias"] = \
                    all_weights[f"{layer_key}.self_attn.v_proj.bias"]
            
            # MLP
            if f"{layer_key}.mlp.gate_proj.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["mlp"]["gate_proj"]["kernel"] = \
                    all_weights[f"{layer_key}.mlp.gate_proj.weight"].T
                
            if f"{layer_key}.mlp.up_proj.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["mlp"]["up_proj"]["kernel"] = \
                    all_weights[f"{layer_key}.mlp.up_proj.weight"].T
                
            if f"{layer_key}.mlp.down_proj.weight" in all_weights:
                params["params"]["layers"]["layers"][str(layer_idx)]["mlp"]["down_proj"]["kernel"] = \
                    all_weights[f"{layer_key}.mlp.down_proj.weight"].T
    
    # Remove any None values from params (where the weights weren't found)
    def remove_none_values(d):
        if isinstance(d, dict):
            return {k: remove_none_values(v) for k, v in d.items() if v is not None}
        return d
    
    params = remove_none_values(params)
    
    # Add verbose logging
    logger.info(f"Parameter structure after mapping: {list(flatten_dict(params).keys())[:5]} (showing first 5)")
    
    # Freeze parameters and update the model
    model.params = freeze(params)
    
    return model

def remap_parameter_structure(loaded_params, expected_model_structure):
    """
    Remap parameters from loaded format to expected model format.
    
    Args:
        loaded_params: Parameters loaded from safetensors files
        expected_model_structure: Structure expected by the model
        
    Returns:
        Dict with parameters remapped to expected structure
    """
    logger.info("Remapping parameter structure to match model expectations")
    
    # Define mappings between different parameter name formats
    # From PyTorch/Safetensors format to Flax format
    key_mappings = {
        # Embeddings
        "model.embed_tokens.weight": "params/embed_tokens/embedding",
        
        # LM Head
        "lm_head.weight": "params/lm_head/kernel",
        
        # Layer norms
        "model.norm.weight": "params/layers/norm/scale",
        ".input_layernorm.weight": "/input_layernorm/scale",
        ".post_attention_layernorm.weight": "/post_attention_layernorm/scale",
        
        # Attention
        ".self_attn.q_proj.weight": "/attention/q_proj/kernel",
        ".self_attn.k_proj.weight": "/attention/k_proj/kernel",
        ".self_attn.v_proj.weight": "/attention/v_proj/kernel",
        ".self_attn.o_proj.weight": "/attention/o_proj/kernel",
        
        # Attention biases
        ".self_attn.q_proj.bias": "/attention/q_proj/bias",
        ".self_attn.k_proj.bias": "/attention/k_proj/bias",
        ".self_attn.v_proj.bias": "/attention/v_proj/bias",
        
        # MLP
        ".mlp.gate_proj.weight": "/mlp/gate_proj/kernel",
        ".mlp.up_proj.weight": "/mlp/up_proj/kernel",
        ".mlp.down_proj.weight": "/mlp/down_proj/kernel"
    }
    
    # Flatten both structures for easier manipulation
    flattened_params = {}
    
    # Process loaded parameters
    if isinstance(loaded_params, dict):
        if "model" in loaded_params:
            # This is a nested dictionary with model and lm_head
            # Combine them with proper prefixes
            combined = {}
            
            # Add model parameters with model. prefix
            for key, value in flatten_dict(loaded_params["model"]).items():
                if isinstance(key, tuple):
                    key_str = "model." + ".".join(key)
                else:
                    key_str = f"model.{key}"
                combined[key_str] = value
            
            # Add lm_head parameters
            if "lm_head" in loaded_params:
                for key, value in flatten_dict(loaded_params["lm_head"]).items():
                    if isinstance(key, tuple):
                        key_str = "lm_head." + ".".join(key)
                    else:
                        key_str = f"lm_head.{key}"
                    combined[key_str] = value
            
            flattened_params = combined
        else:
            # Just flatten as is - likely raw weights
            tmp_flat = {}
            for key, value in flatten_dict(loaded_params).items():
                if isinstance(key, tuple):
                    key_str = ".".join(key)
                else:
                    key_str = key
                tmp_flat[key_str] = value
            flattened_params = tmp_flat
    
    # Now map to expected structure
    mapped_params = {}
    
    # Special handling for model layers
    for key, value in flattened_params.items():
        # First, check for direct matches in key_mappings
        mapped_key = key
        for old_pattern, new_pattern in key_mappings.items():
            if old_pattern in key:
                mapped_key = key.replace(old_pattern, new_pattern)
                break
        
        # Special handling for layer indices
        if "model.layers." in key:
            # Extract layer index
            import re
            layer_match = re.search(r"model\.layers\.(\d+)", key)
            if layer_match:
                layer_idx = layer_match.group(1)
                mapped_key = mapped_key.replace(f"model.layers.{layer_idx}", f"params/layers/layers/{layer_idx}")
        
        # Convert string keys to tuples for Flax parameter tree
        mapped_key_tuple = tuple(mapped_key.split("/"))
        mapped_params[mapped_key_tuple] = value
        
        # Transpose weight matrices if needed (key/value/query/output projections)
        if '.weight' in key and ('proj' in key or 'mlp.' in key) and 'norm' not in key:
            # Transpose weight matrices for linear projections
            # PyTorch: [out_dim, in_dim], JAX: [in_dim, out_dim]
            if len(value.shape) == 2:  # Only transpose 2D matrices
                mapped_params[mapped_key_tuple] = value.T
    
    # Unflatten back to nested structure
    return unflatten_dict(mapped_params, sep="/")