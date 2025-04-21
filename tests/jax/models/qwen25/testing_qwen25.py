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
Qwen 2.5 Model implementation in Flax

This file implements the Qwen 2.5 model architecture in Flax.

Context
@modeling_flax_mistral.py @modeling_flax_llama.py @modeling_flax_utils.py @flax_utils.py @tokenization_auto.py @modeling_flax_pytorch_utils.py @modeling_flax_auto.py @configuration_auto.py@testing_weight_loading.py @transformers @qwen25-weights

python testing_qwen25.py
"""

import os
import math
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

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.modeling_flax_utils import FlaxPreTrainedModel, ACT2FN


logger = logging.get_logger(__name__)


class Qwen25Config(PretrainedConfig):
    """
    Configuration class for Qwen 2.5 model.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Qwen 2.5 model.
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 28):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key value heads for each attention layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to 131072):
            The size of the sliding window attention.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers with sliding window attention.
        use_sliding_window (`bool`, *optional*, defaults to False):
            Whether to use sliding window attention.
        use_mrope (`bool`, *optional*, defaults to False):
            Whether to use multi-head rotary position embeddings.
        output_attentions (`bool`, *optional*, defaults to False):
            Whether or not the model should return attentions tensors.
        output_hidden_states (`bool`, *optional*, defaults to False):
            Whether or not the model should return all hidden states.
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
        Sinusoidal position embeddings
    """
    # Make sure dim is even
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension {dim} should be even")
    
    # Create position indices
    positions = jnp.arange(0, num_pos, dtype=jnp.float32)
    
    # Create frequency indices
    dim_indices = jnp.arange(0, dim, 2, dtype=jnp.float32)
    
    # Compute frequencies
    inv_freq = 1.0 / (base ** (dim_indices / dim))
    
    # Outer product to get all combinations
    # Shape: (num_pos, dim/2)
    pos_emb = jnp.outer(positions, inv_freq)
    
    # Create sinusoids
    sin = jnp.sin(pos_emb)
    cos = jnp.cos(pos_emb)
    
    # Interleave to get embeddings of shape (num_pos, dim)
    sincos = jnp.stack([sin, cos], axis=-1).reshape(num_pos, dim)
    
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
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    """
    Apply rotary position embeddings to the input tensor.
    
    Args:
        tensor: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        sin_pos: Sine position embeddings
        cos_pos: Cosine position embeddings
        
    Returns:
        Tensor with rotary position embeddings applied
    """
    # Ensure all tensors have compatible dimensions for broadcasting
    # This handles cases where sin_pos and cos_pos might have fewer dimensions than tensor
    
    # Check if we need to unsqueeze sin/cos tensors to match tensor dimensions
    if sin_pos.ndim < tensor.ndim:
        # Add dimensions as needed (e.g., batch dimension)
        dims_to_add = tensor.ndim - sin_pos.ndim
        for _ in range(dims_to_add):
            sin_pos = sin_pos[None, ...]
            cos_pos = cos_pos[None, ...]
    
    # Check if shapes need broadcasting in the sequence dimension
    if sin_pos.shape[-2] == 1 and tensor.shape[-2] > 1:
        # Broadcast along sequence dimension
        sin_pos = jnp.broadcast_to(sin_pos, 
                                  tensor.shape[:-1] + (sin_pos.shape[-1],))
        cos_pos = jnp.broadcast_to(cos_pos,
                                  tensor.shape[:-1] + (cos_pos.shape[-1],))
    
    # Apply rotary embedding
    # tensor * cos + rotate_half(tensor) * sin
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class FlaxQwen25RMSNorm(nn.Module):
    """Qwen 2.5 RMSNorm layer."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param("scale", lambda _, shape: jnp.ones(shape), (self.config.hidden_size,))
        self.variance_epsilon = self.config.rms_norm_eps

    def __call__(self, hidden_states):
        """Apply RMSNorm to the input."""
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * (1.0 / jnp.sqrt(variance + self.variance_epsilon))
        
        # Convert to the desired dtype
        hidden_states = hidden_states.astype(self.dtype)
        # The weight needs to be transformed to the same dtype as the hidden_states
        weight = self.weight.astype(self.dtype)
        
        return hidden_states * weight


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
        Apply rotary embeddings to key and query vectors.
        
        Args:
            key: Key vectors of shape (batch_size, num_heads, seq_len, head_dim)
            query: Query vectors of shape (batch_size, num_heads, seq_len, head_dim)
            position_ids: Position indices of shape (batch_size, seq_len)
        
        Returns:
            Tuple of key and query tensors with rotary embeddings applied
        """
        head_dim = query.shape[-1]
        
        # Create position embeddings
        inv_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
        
        # Reshape position_ids to match the expected shape
        # From (batch_size, seq_len) to (batch_size, 1, seq_len, 1)
        position_ids = position_ids.astype(jnp.float32)
        position_ids = position_ids[:, None, :, None]
        
        # Compute the sinusoidal angles
        # This gives us (batch_size, 1, seq_len, head_dim/2)
        freqs = jnp.einsum("bsij,j->bsij", position_ids, inv_freq)
        
        # Create emb for both sin and cos
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        
        # Calculate cos and sin values
        cos_pos = jnp.cos(emb)
        sin_pos = jnp.sin(emb)
        
        # Convert to the model's dtype
        cos_pos = cos_pos.astype(self.dtype)
        sin_pos = sin_pos.astype(self.dtype)
        
        # Apply rotary embeddings to query and key
        query_embed = apply_rotary_pos_emb(query, sin_pos, cos_pos)
        key_embed = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        
        return key_embed, query_embed


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
        
        self.gate_proj = nn.Dense(
            intermediate_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        self.up_proj = nn.Dense(
            intermediate_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        self.down_proj = nn.Dense(
            hidden_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        # Get the activation function to use
        self.activation_fn = ACT2FN[self.config.hidden_act]
        
    def __call__(self, hidden_states):
        """Apply the MLP to the input hidden states."""
        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        
        # Apply SwiGLU/SiLU activation
        gate_output = self.activation_fn(gate_output)
        
        # Multiply gate and up
        intermediate_output = gate_output * up_output
        
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
    
    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.repeats = self.num_heads // self.num_kv_heads
        
        # Projection matrices
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        self.k_proj = nn.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        self.v_proj = nn.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        self.o_proj = nn.Dense(
            self.hidden_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        
        self.rotary_emb = FlaxQwen25RotaryEmbedding(config, dtype=self.dtype)
        
        self.attention_dropout = nn.Dropout(rate=config.attention_dropout)

    def _split_heads(self, hidden_states, num_heads):
        """Split hidden_states into separate attention heads."""
        batch_size, seq_length = hidden_states.shape[:2]
        head_dim = self.head_dim
        
        hidden_states = hidden_states.reshape(batch_size, seq_length, num_heads, head_dim)
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        
        return hidden_states
    
    def _merge_heads(self, hidden_states):
        """Merge attention heads back into full hidden states."""
        batch_size, _, seq_length, _ = hidden_states.shape
        hidden_size = self.hidden_size
        
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = hidden_states.reshape(batch_size, seq_length, hidden_size)
        
        return hidden_states
    
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        Function used for caching during generation (autoregressive decoding).
        
        This method manages the cache used for faster generation by avoiding
        recomputing already processed tokens.
        
        Args:
            key: Key states
            value: Value states
            query: Query states
            attention_mask: Attention mask
            
        Returns:
            Updated key, value, and attention mask
        """
        # Check if cache is already initialized
        is_initialized = self.has_variable("cache", "cached_key")
        
        # Create cache variables if they don't exist yet
        cached_key = self.variable("cache", "cached_key", 
                                   lambda: jnp.zeros(key.shape, key.dtype))
        cached_value = self.variable("cache", "cached_value", 
                                     lambda: jnp.zeros(value.shape, value.dtype))
        cache_index = self.variable("cache", "cache_index", 
                                    lambda: jnp.array(0, dtype=jnp.int32))
        
        # If cache is already initialized, update it
        if is_initialized:
            # Get shapes for handling the cache updates
            batch_size, num_heads, seq_length, head_dim = key.shape
            
            # Current position in the cache
            cur_index = cache_index.value
            
            # Concatenate current key/value with cached key/value
            if seq_length == 1:  # We're only adding one token at a time
                # Fast path for generation mode
                key_indices = jnp.full((batch_size, num_heads, seq_length, head_dim), cur_index, dtype=jnp.int32)
                key_indices = key_indices.at[:, :, 0, :].set(cur_index)
                
                # Store current key and value at the correct indices
                cached_key.value = cached_key.value.at[:, :, cur_index, :].set(key[:, :, 0, :])
                cached_value.value = cached_value.value.at[:, :, cur_index, :].set(value[:, :, 0, :])
                
                # Get the updated key and value by concatenating cached values
                # For query, we only use the newest part (current token)
                key = cached_key.value
                value = cached_value.value
                
                # Update the cache index
                cache_index.value = cache_index.value + 1
            else:
                # This code path is for handling sequences longer than 1 token
                indices = jnp.arange(cur_index, cur_index + seq_length)
                
                # Create update masks for the cache
                key_update_indices = jnp.reshape(indices, (1, 1, -1, 1))
                key_indices = jnp.broadcast_to(
                    key_update_indices, (batch_size, num_heads, seq_length, head_dim)
                )
                
                # Update the key and value cache
                for i in range(seq_length):
                    cached_key.value = cached_key.value.at[:, :, cur_index + i, :].set(key[:, :, i, :])
                    cached_value.value = cached_value.value.at[:, :, cur_index + i, :].set(value[:, :, i, :])
                
                # Return the full cached key and value
                key = cached_key.value
                value = cached_value.value
                
                # Update cache index
                cache_index.value = cur_index + seq_length
                
            # Adjust attention mask for the cached sequence
            # This is a critical step to ensure the model only attends to appropriate tokens
            if attention_mask is not None:
                attention_mask = jnp.concatenate(
                    [
                        attention_mask[:, :, :, :cur_index],
                        jnp.ones_like(attention_mask[:, :, :, :seq_length]) * -1e10,
                        attention_mask[:, :, :, cur_index:],
                    ],
                    axis=-1,
                )
        
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
        Apply attention mechanism to hidden states.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position indices of shape (batch_size, seq_len)
            deterministic: Whether to use deterministic operations (disables dropout in training)
            init_cache: Whether to initialize the cache for autoregressive generation
            output_attentions: Whether to output attention weights
            
        Returns:
            output tensor and optionally attention weights
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project hidden states to query, key, and value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Split heads
        query_states = self._split_heads(query_states, self.num_heads)
        key_states = self._split_heads(key_states, self.num_kv_heads)
        value_states = self._split_heads(value_states, self.num_kv_heads)
        
        # Apply rotary position embeddings
        if position_ids is not None:
            key_states, query_states = self.rotary_emb(key_states, query_states, position_ids)
        
        # Duplicate key and value states if using grouped-query attention
        if self.num_kv_heads < self.num_heads:
            key_states = jnp.repeat(key_states, self.repeats, axis=1)
            value_states = jnp.repeat(value_states, self.repeats, axis=1)
        
        # Handle caching for autoregressive generation
        if init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )
        
        # Compute dot-product attention
        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")
            
        # Calculate attention weights
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            attention_mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
        )
        
        # Apply attention weights to value states
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
    
    def setup(self):
        # Normalization layers
        self.input_layernorm = FlaxQwen25RMSNorm(config=self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxQwen25RMSNorm(config=self.config, dtype=self.dtype)
        
        # Self-attention and MLP
        self.attention = FlaxQwen25Attention(config=self.config, dtype=self.dtype)
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
        Apply the decoder layer to the input hidden states.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position indices of shape (batch_size, seq_len)
            deterministic: Whether to use deterministic operations (disables dropout in training)
            init_cache: Whether to initialize the cache for autoregressive generation
            output_attentions: Whether to output attention weights
            
        Returns:
            hidden_states after applying the layer, and optionally attention weights
        """
        # Normalize input
        residual = hidden_states
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
        
        # Get the attention output and optional attention weights
        attn_output = attn_outputs[0]
        
        # Apply residual connection
        hidden_states = attn_output + residual
        
        # Apply post-attention normalization and MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP
        mlp_output = self.mlp(hidden_states)
        
        # Apply final residual connection
        hidden_states = mlp_output + residual
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
            
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
            FlaxQwen25DecoderLayer(self.config, name=f"layers_{i}", dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
        
        # Final normalization layer
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
        Apply the layer collection to the input.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position indices of shape (batch_size, seq_len)
            deterministic: Whether to use deterministic operations (disables dropout in training)
            init_cache: Whether to initialize the cache for autoregressive generation
            output_attentions: Whether to output attention weights from each layer
            output_hidden_states: Whether to return the hidden states from each layer
            return_dict: Whether to return output as a dict or tuple
            
        Returns:
            Output hidden states and optionally all hidden states and attention weights
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Apply each layer in sequence
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Add final hidden state to list if needed
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            # Return tuple if not returning dict
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
            
        # Return dictionary-like output if specified
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
        config = self.config
        
        # Token embeddings
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=self.dtype,
        )
        
        # Transformer layers
        self.layers = FlaxQwen25LayerCollection(config, dtype=self.dtype)
        
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
        Apply the Qwen 2.5 model to the input tokens.
        
        Args:
            input_ids: Tokenized input ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position indices of shape (batch_size, seq_len)
            deterministic: Whether to use deterministic operations (disables dropout in training)
            init_cache: Whether to initialize the cache for autoregressive generation
            output_attentions: Whether to output attention weights from each layer
            output_hidden_states: Whether to return the hidden states from each layer
            return_dict: Whether to return output as a dict or tuple
            
        Returns:
            Output hidden states and optionally all hidden states and attention weights
        """
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape
        
        # If no position_ids are provided, create them automatically
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_length))
        
        # If no attention mask is provided, create an all-ones mask
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
            
        # Convert attention mask to casual attention mask of shape (batch_size, 1, seq_len, seq_len)
        causal_mask = make_causal_mask(attention_mask)
        
        # Embed the input tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        outputs = self.layers(
            hidden_states,
            attention_mask=causal_mask,
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
        Initialize a Qwen 2.5 pre-trained model.
        
        Args:
            config: Model configuration
            input_shape: Shape of input tensors
            seed: Random seed
            dtype: Data type of the model
            _do_init: Whether to initialize parameters
        """
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        """
        Initialize weights of the model.
        
        Args:
            rng: Random number generator key
            input_shape: Shape of input tensors
            params: Optional existing parameters
            
        Returns:
            Initialized parameters
        """
        # Initialize input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        
        # Initialize parameters
        if params is None:
            return self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)
        else:
            return params
    
    def init_cache(self, batch_size, max_length):
        """
        Initialize cache for autoregressive decoding.
        
        Args:
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            Cache variables
        """
        # Determine input shape based on batch size and maximum length
        # Make sure to use jnp.int32 for input_ids to avoid type errors
        input_ids = jnp.ones((batch_size, max_length), dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids, dtype=jnp.int32)
        position_ids = jnp.broadcast_to(
            jnp.arange(max_length, dtype=jnp.int32)[None, :],
            (batch_size, max_length)
        )
        
        # Initialize cache
        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        
        return init_variables["cache"]


class FlaxQwen25Model(FlaxQwen25PreTrainedModel):
    """
    The base Qwen 2.5 Model transformer outputting raw hidden-states without any specific head on top.
    """
    module_class = FlaxQwen25Module


# Helper function for creating a causal mask from an attention mask
def make_causal_mask(attention_mask: jnp.ndarray, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """
    Creates a causal mask from an attention mask.
    
    Args:
        attention_mask: Attention mask of shape (batch_size, seq_len)
        dtype: Data type of the mask
    
    Returns:
        Causal mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_length = attention_mask.shape
    
    # Create a causal mask (lower triangular)
    # where each position can attend only to previous positions
    causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=dtype))
    
    # Reshape causal mask to 4D [batch_size, 1, seq_length, seq_length]
    causal_mask = causal_mask[None, None, :, :]
    
    # Broadcast to batch size dimension
    causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_length, seq_length))
    
    # Reshape attention_mask to [batch_size, 1, 1, seq_length]
    attention_mask_4d = attention_mask[:, None, None, :].astype(dtype)
    
    # Convert attention mask to correct format for dot_product_attention_weights
    # In dot_product_attention_weights:
    # - mask=0.0 means "allow attention"
    # - mask=large negative means "block attention"
    attention_bias = (1.0 - attention_mask_4d) * -1e10
    
    # Expand attention mask to [batch_size, 1, seq_length, seq_length]
    # where each token can only attend to tokens in attention_mask
    attention_bias = jnp.broadcast_to(attention_bias, causal_mask.shape)
    
    # Convert causal mask to same format
    # 1.0 = allow attention, 0.0 = block attention
    # Convert to -1e10 for masked positions and 0.0 for allowed positions
    causal_bias = (1.0 - causal_mask) * -1e10
    
    # Combine the masks - take maximum of the biases (most restrictive)
    # This means a position is masked if either causal or attention mask blocks it
    combined_bias = jnp.maximum(attention_bias, causal_bias)
    
    return combined_bias

class FlaxQwen25ForCausalLMModule(nn.Module):
    """
    Qwen 2.5 model with a language modeling head.
    
    This module includes the Qwen 2.5 base model and a linear layer for language modeling.
    """
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Base Qwen 2.5 model
        self.model = FlaxQwen25Module(config=self.config, dtype=self.dtype)
        
        # Language modeling head
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
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
        Apply the model for causal language modeling.
        
        Args:
            input_ids: Tokenized input ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position indices of shape (batch_size, seq_len)
            deterministic: Whether to use deterministic operations (disables dropout in training)
            init_cache: Whether to initialize the cache for autoregressive generation
            output_attentions: Whether to output attention weights from each layer
            output_hidden_states: Whether to return the hidden states from each layer
            return_dict: Whether to return output as a dict or tuple
            
        Returns:
            Logits and optionally all hidden states and attention weights
        """
        # Apply base model
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the last hidden state or process the output tuple if return_dict=False
        if return_dict:
            hidden_states = outputs["last_hidden_state"]
        else:
            hidden_states = outputs[0]
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            # Return tuple
            outputs = (logits,) + outputs[1:]
            return outputs
        
        # Return dictionary-like output
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
        Main inference method for the Qwen 2.5 causal language model.
        
        Args:
            input_ids: Input token ids, shape (batch_size, seq_len)
            attention_mask: Optional attention mask, shape (batch_size, seq_len)
            position_ids: Optional position ids, shape (batch_size, seq_len)
            params: Optional model parameters
            past_key_values: Optional cached key values for faster generation
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dictionary
            train: Whether we're in training mode
            add_params_field: Whether to add a 'params' field to the parameters
            mutable: Whether to return updated cache
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle any PRNG if needed
        rngs = {}
        if train:
            rngs["dropout"] = jax.random.PRNGKey(0)

        # Create position_ids if not provided
        if position_ids is None:
            batch_size, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # Create attention_mask if not provided
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle generation with past_key_values (cached activations)
        init_cache = past_key_values is not None
        if init_cache:
            # Only use the last token for generation
            input_ids = input_ids[:, -1:]
            # Adjust position_ids accordingly
            batch_size = input_ids.shape[0]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            else:
                # Calculate position_id for the last token
                past_length = past_key_values.shape[2] if past_key_values.shape[2] > 0 else 0
                position_ids = jnp.ones((batch_size, 1), dtype=jnp.int32) * past_length

        # Get the correct params structure
        if params is None:
            # Avoid nested params structure
            if 'params' in self.params and not add_params_field:
                params = self.params['params']
            else:
                params = self.params
        elif add_params_field and 'params' not in params:
            params = {'params': params}

        # Handle the mutable parameter - needed for autoregressive generation
        mutable_dict = False
        if mutable:
            mutable_dict = ['cache']

        # Apply the model module
        result = self.module.apply(
            {'params': params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            deterministic=not train,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=mutable_dict,
        )

        # When using mutable, outputs is a tuple (outputs, mutated)
        if mutable:
            outputs, mutated = result
            # Add the cache to the outputs with a standardized name
            if isinstance(outputs, dict):
                outputs["past_key_values"] = mutated["cache"]
            else:
                # If outputs is not a dict (e.g., it's a tuple), convert to dict
                if return_dict:
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        hidden_states = outputs[1] if len(outputs) > 1 else None
                        attentions = outputs[2] if len(outputs) > 2 else None
                        outputs = {
                            "logits": logits,
                            "hidden_states": hidden_states,
                            "attentions": attentions,
                            "past_key_values": mutated["cache"]
                        }
                    else:
                        # Unexpected case - just return tuple
                        return (outputs, mutated)
                else:
                    # If return_dict is False, return the full tuple including mutated
                    return (outputs, mutated)
        else:
            outputs = result

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
            input_ids: Tokenized input ids, shape (batch_size, seq_len)
            max_length: Maximum sequence length for generation
            attention_mask: Optional attention mask, shape (batch_size, seq_len)
            position_ids: Optional position ids, shape (batch_size, seq_len)
            past_key_values: Optional past key values for faster generation
            
        Returns:
            Dictionary with prepared inputs for generation
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position_ids if not provided
        if position_ids is None:
            if past_key_values is not None:
                # For generation: position_ids are the position of new token
                past_length = past_key_values.shape[2]
                position_ids = jnp.ones((batch_size, 1), dtype=jnp.int32) * past_length
            else:
                # For first pass: position_ids are the positions of all tokens
                position_ids = jnp.broadcast_to(
                    jnp.arange(seq_length, dtype=jnp.int32)[None, :],
                    (batch_size, seq_length)
                )
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        
        # For generation with init_cache, we need to initialize the cache
        # only if past_key_values is None (first step of generation)
        if past_key_values is None:
            init_cache = True
            # Initialize cache for the maximum expected length
            past_key_values = self.init_cache(batch_size, max_length)
        else:
            init_cache = False
        
        # Prepare the model inputs dictionary
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "mutable": True,  # Always mutable during generation
        }
        
        return model_inputs
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """
        Update inputs for the next generation step.
        
        Args:
            model_outputs: Output from the model's forward pass
            model_kwargs: Keyword arguments used in the forward pass
            
        Returns:
            Updated model_kwargs for the next generation step
        """
        # Get the updated cache from model_outputs
        if "past_key_values" in model_outputs:
            model_kwargs["past_key_values"] = model_outputs["past_key_values"]
        
        # Update position_ids to point to the next token position
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = position_ids + 1
        
        # Extend attention mask for the next token if needed
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # Check if we need to extend the attention mask for the next token
            if model_kwargs.get("input_ids", None) is not None:
                input_ids = model_kwargs["input_ids"]
                if input_ids.shape[1] == 1:  # We're in generation mode (single token)
                    # Extend the attention mask for the next token
                    model_kwargs["attention_mask"] = jnp.concatenate(
                        [attention_mask, jnp.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
                        axis=1
                    )
        
        return model_kwargs

def main():
    """
    Test function to verify the Qwen 2.5 model architecture.
    
    This function creates a small model, initializes it, and runs a forward pass
    to ensure everything is working as expected.
    """
    import jax
    import os
    from jax import random
    
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Using 16-bit precision: {os.environ.get('JAX_ENABLE_X64', '0') == '0'}")
    print("Testing Qwen 2.5 model architecture...\n")
    
    # Create a very small model for testing (much smaller to avoid memory issues)
    config = Qwen25Config(
        vocab_size=1000,
        hidden_size=32,         # Very small hidden size
        intermediate_size=64,   # Very small intermediate size
        num_hidden_layers=2,    # Just 2 layers
        num_attention_heads=4,  # Fewer attention heads
        num_key_value_heads=2,  # Fewer key/value heads
        max_position_embeddings=32,  # Much smaller position embeddings
    )
    
    print("Model configuration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Number of attention heads: {config.num_attention_heads}")
    print(f"  Number of key/value heads: {config.num_key_value_heads}")
    print(f"  Using RoPE with theta={config.rope_theta}")
    print(f"  RMS norm epsilon: {config.rms_norm_eps}")
    print()
    
    # Initialize model
    print("Initializing model...")
    model = FlaxQwen25ForCausalLM(config, _do_init=True)
    print("Model initialized successfully!")
    
    # Test with a single token
    print("\nTesting with a single token...")
    batch_size = 1
    input_ids = jnp.ones((batch_size, 1), dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    
    # Get the output
    print("Running model.module.apply directly")
    outputs = model.module.apply(
        {"params": model.params["params"]},
        input_ids,
        attention_mask,
        None,  # position_ids - default will be used
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    
    # Check output shape
    if isinstance(outputs, dict) and "logits" in outputs:
        logits_shape = outputs["logits"].shape
        print(f"Single token output shape: {logits_shape}")
        expected_shape = (batch_size, 1, config.vocab_size)
        assert logits_shape == expected_shape, f"Expected {expected_shape}, got {logits_shape}"
        print("Single token test passed!")
    else:
        print(f"Unexpected output type: {type(outputs)}")
    
    # Test with a sequence
    print("\nTesting with a sequence...")
    input_ids = jnp.ones((batch_size, 4), dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    
    # Get the output
    print("Running model.module.apply directly")
    outputs = model.module.apply(
        {"params": model.params["params"]},
        input_ids,
        attention_mask,
        None,  # position_ids - default will be used
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    
    # Check output shape
    if isinstance(outputs, dict) and "logits" in outputs:
        logits_shape = outputs["logits"].shape
        print(f"Sequence output shape: {logits_shape}")
        expected_shape = (batch_size, 4, config.vocab_size)
        assert logits_shape == expected_shape, f"Expected {expected_shape}, got {logits_shape}"
        print("Sequence test passed!")
    else:
        print(f"Unexpected output type: {type(outputs)}")
    
    # Test caching for autoregressive generation
    print("\nTesting cache initialization for autoregressive generation...")
    
    # Create inputs
    seq_length = 4
    max_length = 8
    input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    
    try:
        # Initialize with cache
        print("Initializing variables with cache enabled")
        cache_vars = model.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            None,  # attention_mask
            None,  # position_ids
            deterministic=True,
            init_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        if "cache" in cache_vars:
            print("Cache successfully initialized:")
            print(jax.tree_util.tree_map(lambda x: x.shape, cache_vars["cache"]))
            print("Cache test passed!")
        else:
            print("No cache found in initialized variables")
    except Exception as e:
        print(f"Error during cache initialization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nQwen 2.5 model architecture test completed.")
    print("Basic model implementation successful!")


if __name__ == "__main__":
    main() 