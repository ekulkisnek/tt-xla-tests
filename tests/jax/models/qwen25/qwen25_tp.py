#!/usr/bin/env python3
"""
Tensor-Parallel Qwen2.5-7B Implementation in JAX.

A simplified, self-contained implementation focused on tensor parallelism
across multiple devices with support for various mesh configurations.

Usage:

source venv/bin/activate


export XLA_FLAGS="--xla_force_host_platform_device_count=8"

python qwen25_tp.py --model_path /path/to/weights --prompt "Hello" --mesh_shape 1,8

python qwen25_tp.py --model_path /root/wrkdir/tt-xla/tests/jax/models/qwen25/qwen25-weights --prompt "Your prompt here" --mesh_shape 1,8

"""

import os
import time
import json
import logging
import argparse
import gc
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
import flax.linen as nn
from flax.core.frozen_dict import freeze
from safetensors import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("qwen25_tp")

# -----------------------------------------
# Tensor-Parallel Primitives
# -----------------------------------------

class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism support."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.normal(0.02)
    bias_init: Any = nn.initializers.zeros
    shard_axes: Tuple[Optional[str], Optional[str]] = (None, "model")  # Input, output axis sharding
    
    @nn.compact
    def __call__(self, x, mesh=None):
        """Apply tensor-parallel dense layer."""
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (x.shape[-1], self.features),
            self.dtype
        )
        
        # Apply sharding constraint if mesh is provided
        if mesh is not None:
            kernel = jax.lax.with_sharding_constraint(kernel, P(*self.shard_axes))
            
        y = jnp.matmul(x, kernel)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            if mesh is not None:
                bias = jax.lax.with_sharding_constraint(bias, P(self.shard_axes[1]))
            y = y + bias
            
        return y 

def compute_cos_sin_cache(
    position_ids: jnp.ndarray,
    head_dim: int,
    rope_theta: float = 10000.0,
    max_seq_len: int = 4096,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute cosine and sine cache for rotary position embeddings.
    
    Args:
        position_ids: Position IDs of shape [batch_size, seq_len]
        head_dim: Dimension of each attention head
        rope_theta: Base for rotary embeddings
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (cos, sin) with shape [batch_size, seq_len, head_dim/2]
    """
    # Half of head dimension for rotary embeddings
    dim = head_dim // 2
    
    # Create position embeddings
    # Shape: [max_seq_len, dim]
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, 2) / dim))
    
    # Instead of computing based on dynamic max_pos, use predefined max_seq_len
    # This fixes the JIT tracing error by using a static shape
    t = jnp.arange(0, max_seq_len, dtype=jnp.float32)
    
    # Compute sinusoidal embeddings for all possible positions
    # Shape: [max_seq_len, dim/2]
    freqs = jnp.outer(t, inv_freq)
    
    # Shape: [max_seq_len, dim/2]
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    
    # Gather embeddings based on position_ids, and ensure positions are clipped to valid range
    # First ensure position_ids are within bounds to avoid out-of-bounds errors
    position_ids_clipped = jnp.clip(position_ids, 0, max_seq_len - 1)
    
    # Shape: [batch_size, seq_len, dim/2]
    cos_cached = jnp.take(cos, position_ids_clipped, axis=0)
    sin_cached = jnp.take(sin, position_ids_clipped, axis=0)
    
    return cos_cached, sin_cached

def apply_rotary_embeddings(
    query: jnp.ndarray,
    key: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        query: Query tensor of shape [batch, heads, seq_len, head_dim]
        key: Key tensor of shape [batch, heads, seq_len, head_dim]
        cos: Cosine cache of shape [batch, seq_len, head_dim/2]
        sin: Sine cache of shape [batch, seq_len, head_dim/2]
        
    Returns:
        Transformed query and key tensors
    """
    # Reshape for broadcasting
    # [batch, seq_len, head_dim/2] -> [batch, 1, seq_len, head_dim/2]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    
    # Get dimensions
    head_dim = query.shape[-1]
    dim_half = head_dim // 2
    
    # Split the last dimension in half
    query_half1, query_half2 = jnp.split(query, 2, axis=-1)
    key_half1, key_half2 = jnp.split(key, 2, axis=-1)
    
    # Apply rotary embeddings
    # For query: [q_half1 * cos - q_half2 * sin, q_half2 * cos + q_half1 * sin]
    query_rot_half1 = query_half1 * cos - query_half2 * sin
    query_rot_half2 = query_half2 * cos + query_half1 * sin
    
    # For key: [k_half1 * cos - k_half2 * sin, k_half2 * cos + k_half1 * sin]
    key_rot_half1 = key_half1 * cos - key_half2 * sin
    key_rot_half2 = key_half2 * cos + key_half1 * sin
    
    # Concatenate back along the last dimension
    query_rotated = jnp.concatenate([query_rot_half1, query_rot_half2], axis=-1)
    key_rotated = jnp.concatenate([key_rot_half1, key_rot_half2], axis=-1)
    
    return query_rotated, key_rotated 

class QwenAttention(nn.Module):
    """Multi-head attention with tensor parallelism support for Qwen2.5."""
    config: Dict
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        hidden_size = self.config["hidden_size"]
        num_heads = self.config["num_attention_heads"]
        num_kv_heads = self.config.get("num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads
        
        # Query projection with tensor parallelism (sharded on output dimension)
        self.q_proj = TensorParallelDense(
            features=num_heads * head_dim,
            dtype=self.dtype,
            shard_axes=(None, "model"),
        )
        
        # Key projection with tensor parallelism (sharded on output dimension)
        self.k_proj = TensorParallelDense(
            features=num_kv_heads * head_dim, 
            dtype=self.dtype,
            shard_axes=(None, "model"),
        )
        
        # Value projection with tensor parallelism (sharded on output dimension)
        self.v_proj = TensorParallelDense(
            features=num_kv_heads * head_dim,
            dtype=self.dtype,
            shard_axes=(None, "model"),
        )
        
        # Output projection with tensor parallelism (sharded on input dimension)
        self.o_proj = TensorParallelDense(
            features=hidden_size,
            dtype=self.dtype,
            shard_axes=("model", None),
        )
        
        # Save dimensions for reference
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        mesh: Optional[Mesh] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Apply multi-head attention with tensor parallelism.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            position_ids: Position IDs of shape [batch, seq_len]
            past_key_value: Cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            mesh: Device mesh for tensor parallelism constraints
            
        Returns:
            Output tensor and optional key-value cache
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project query, key, and value
        query_states = self.q_proj(hidden_states, mesh=mesh)
        key_states = self.k_proj(hidden_states, mesh=mesh)
        value_states = self.v_proj(hidden_states, mesh=mesh)
        
        # Reshape query, key, and value for multi-head attention
        # [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention
        # [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))
        
        # Handle key-value cache for incremental decoding
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Concatenate past and current key-value states
            key_states = jnp.concatenate([past_key, key_states], axis=2)
            value_states = jnp.concatenate([past_value, value_states], axis=2)
        
        # Save key and value for future use
        key_value_cache = (key_states, value_states) if past_key_value is not None or seq_length == 1 else None
        
        # Compute rotary embeddings
        if position_ids is not None:
            cos, sin = compute_cos_sin_cache(
                position_ids, 
                self.head_dim, 
                rope_theta=getattr(self.config, "rope_theta", 10000.0)
            )
            
            # Apply rotary embeddings to query and key
            query_states, key_states = apply_rotary_embeddings(query_states, key_states, cos, sin)
        
        # Compute attention scores
        # [batch, heads, seq, head_dim] @ [batch, heads, head_dim, kv_seq] = [batch, heads, seq, kv_seq]
        attention_scores = jnp.matmul(query_states, jnp.transpose(key_states, (0, 1, 3, 2)))
        attention_scores = attention_scores / jnp.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention weights to values
        # [batch, heads, seq, kv_seq] @ [batch, heads, kv_seq, head_dim] = [batch, heads, seq, head_dim]
        attention_output = jnp.matmul(attention_weights, value_states)
        
        # Reshape back to original dimensions
        # [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_length, -1)
        
        # Apply output projection
        output = self.o_proj(attention_output, mesh=mesh)
        
        return output, key_value_cache 

class QwenMLP(nn.Module):
    """Feed-forward MLP module with tensor parallelism for Qwen2.5."""
    config: Dict
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        hidden_size = self.config["hidden_size"]
        intermediate_size = self.config.get("intermediate_size", 4 * hidden_size)
        
        # Gate and up projections with tensor parallelism (sharded on output dimension)
        self.gate_proj = TensorParallelDense(
            features=intermediate_size,
            dtype=self.dtype,
            shard_axes=(None, "model"),
        )
        
        self.up_proj = TensorParallelDense(
            features=intermediate_size,
            dtype=self.dtype,
            shard_axes=(None, "model"),
        )
        
        # Down projection with tensor parallelism (sharded on input dimension)
        self.down_proj = TensorParallelDense(
            features=hidden_size,
            dtype=self.dtype,
            shard_axes=("model", None),
        )
    
    def __call__(self, x: jnp.ndarray, mesh: Optional[Mesh] = None) -> jnp.ndarray:
        """
        Apply feed-forward network with tensor parallelism.
        
        Args:
            x: Input tensor of shape [batch, seq, hidden_size]
            mesh: Device mesh for tensor parallelism constraints
            
        Returns:
            Output tensor of shape [batch, seq, hidden_size]
        """
        # Apply SwiGLU activation
        # SwiGLU: gate * silu(up)
        gate_output = self.gate_proj(x, mesh=mesh)
        up_output = self.up_proj(x, mesh=mesh)
        
        # SiLU activation: x * sigmoid(x)
        silu_output = up_output * jax.nn.sigmoid(up_output)
        
        # Gated activation
        intermediate_output = gate_output * silu_output
        
        # Apply down projection
        output = self.down_proj(intermediate_output, mesh=mesh)
        
        return output

class QwenDecoderLayer(nn.Module):
    """Transformer decoder layer with tensor parallelism for Qwen2.5."""
    config: Dict
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # RMSNorm layers
        hidden_size = self.config["hidden_size"]
        rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
        
        # Input layernorm
        self.input_layernorm = nn.LayerNorm(
            epsilon=rms_norm_eps,
            dtype=self.dtype,
            use_bias=False,
        )
        
        # Post-attention layernorm
        self.post_attention_layernorm = nn.LayerNorm(
            epsilon=rms_norm_eps,
            dtype=self.dtype,
            use_bias=False,
        )
        
        # Self-attention
        self.self_attn = QwenAttention(
            config=self.config,
            dtype=self.dtype,
        )
        
        # MLP
        self.mlp = QwenMLP(
            config=self.config,
            dtype=self.dtype,
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        mesh: Optional[Mesh] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Apply transformer decoder layer with tensor parallelism.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq, hidden_size]
            attention_mask: Attention mask of shape [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            position_ids: Position IDs of shape [batch, seq_len]
            past_key_value: Cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            mesh: Device mesh for tensor parallelism constraints
            
        Returns:
            Output tensor and optional key-value cache
        """
        # Self Attention
        residual = hidden_states
        
        # Apply input layernorm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply self-attention with tensor parallelism
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            mesh=mesh,
        )
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        # Feed-forward network
        residual = hidden_states
        
        # Apply post-attention layernorm
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP with tensor parallelism
        hidden_states = self.mlp(hidden_states, mesh=mesh)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states, present_key_value 

class Qwen25ForCausalLM(nn.Module):
    """Qwen2.5 model with tensor parallelism for causal language modeling."""
    config: Dict
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Extract config parameters
        hidden_size = self.config["hidden_size"]
        vocab_size = self.config["vocab_size"]
        num_layers = self.config["num_hidden_layers"]
        
        # Token embeddings - not sharded
        self.embed_tokens = nn.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(0.02),
        )
        
        # Transformer layers
        self.layers = [
            QwenDecoderLayer(
                config=self.config,
                dtype=self.dtype,
            )
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.norm = nn.LayerNorm(
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            use_bias=False,
        )
        
        # LM head with tensor parallelism (sharded on input dimension)
        self.lm_head = TensorParallelDense(
            features=vocab_size,
            use_bias=False,
            dtype=self.dtype,
            shard_axes=("model", None),
        )
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        mesh: Optional[Mesh] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply Qwen2.5 model with tensor parallelism.
        
        Args:
            input_ids: Token IDs of shape [batch, seq_len]
            attention_mask: Attention mask of shape [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            position_ids: Position IDs of shape [batch, seq_len]
            past_key_values: List of cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary or tuple
            mesh: Device mesh for tensor parallelism constraints
            
        Returns:
            Dictionary or tuple with model outputs
        """
        batch_size, seq_length = input_ids.shape
        
        # If no position IDs provided, create them
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)
        
        # If no attention mask provided, create a causal mask
        if attention_mask is None:
            # Create causal mask of shape [batch, 1, seq_len, seq_len]
            causal_mask = nn.make_causal_mask(input_ids)
            attention_mask = causal_mask
        
        # Past key values length for incremental decoding
        past_kv_length = 0
        if past_key_values is not None and seq_length > 0:
            past_kv_length = past_key_values[0][0].shape[2]  # [batch, heads, kv_seq, head_dim]
            
            # Extend the attention mask to include past key values
            if attention_mask.shape[-1] < seq_length + past_kv_length:
                # Create attention mask of correct size
                attention_mask = jnp.concatenate([
                    jnp.ones((batch_size, 1, 1, past_kv_length), dtype=jnp.int32),
                    attention_mask
                ], axis=-1)
        
        # Apply token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize outputs for optional returns
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        present_key_values = () if past_key_values is not None or seq_length == 1 else None
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            # Get past key-value for this layer
            past_key_value = None
            if past_key_values is not None:
                past_key_value = past_key_values[i]
            
            # Save hidden states if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Apply layer with tensor parallelism
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                mesh=mesh,
            )
            
            # Save attention outputs if requested
            if output_attentions:
                all_self_attns = all_self_attns + (present_key_value[1],)
                
            # Save key-values if requested
            if present_key_values is not None:
                present_key_values = present_key_values + (present_key_value,)
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Save final hidden states if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Apply LM head with tensor parallelism
        logits = self.lm_head(hidden_states, mesh=mesh)
        
        # Return as dictionary or tuple based on request
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": present_key_values,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attns
            }
        else:
            return (logits, present_key_values, all_hidden_states, all_self_attns) 

# -----------------------------------------
# Parameter Loading Functions
# -----------------------------------------

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
    import re
    
    # Patterns for layer parameters
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    
    # Handle layer norms
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (f"layers_{layer_idx}", norm_name, "scale")
    
    # Handle attention parameters
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        param_type = attn_match.group(3)
        proj_name = f"{proj_type}_proj"
        param_name = "kernel" if param_type == "weight" else "bias"
        return (f"layers_{layer_idx}", "self_attn", proj_name, param_name)
    
    # Handle MLP parameters
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        proj_name = f"{proj_type}_proj"
        return (f"layers_{layer_idx}", "mlp", proj_name, "kernel")
    
    logger.warning(f"Unknown parameter: {name}")
    return None

def transpose_if_needed(name, param):
    """Transpose weight matrices if needed based on the parameter name."""
    # Special case for embedding weights
    if "embed_tokens.weight" in name:
        # Do not transpose embedding weights
        return param
    
    # Other attention and MLP weights need to be transposed
    if "weight" in name and ("proj" in name or "lm_head" in name):
        # For attention and MLP weight matrices
        return jnp.transpose(param)
    
    return param

def process_safetensors_file(file_path, dtype=jnp.bfloat16):
    """
    Process a single safetensors file by streaming parameters.
    
    Args:
        file_path: Path to the safetensors file
        dtype: Data type for parameters
        
    Returns:
        Dictionary with parameters in Flax format
    """
    flax_params = {"params": {}}
    
    try:
        with safe_open(file_path, framework="numpy") as f:
            key_count = 0
            for key in f.keys():
                key_count += 1
                
                # Get parameter path
                param_path = get_param_path(key)
                if param_path is None:
                    continue
                
                # Get parameter and convert to JAX array
                param = f.get_tensor(key)
                param = jnp.array(param, dtype=dtype)
                
                # Transpose if needed
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
                if key_count % 50 == 0:
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

def load_model_params(model_path, dtype=jnp.bfloat16):
    """
    Load all model parameters from safetensors files.
    
    Args:
        model_path: Path to directory containing model weights
        dtype: Data type for parameters
        
    Returns:
        Dictionary with parameters in Flax format
    """
    # Load configuration
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    # Process weights by streaming parameters from each file
    safetensors_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    # Load parameters in memory-efficient way
    logger.info(f"Loading parameters from {len(safetensors_files)} files...")
    params = {"params": {}}
    
    for i, filename in enumerate(safetensors_files):
        weight_path = os.path.join(model_path, filename)
        logger.info(f"Processing file {i+1}/{len(safetensors_files)}: {filename}")
        
        # Process one file at a time
        file_params = process_safetensors_file(weight_path, dtype=dtype)
        
        # Merge parameters
        params = merge_param_dicts(params, file_params)
        
        # Force garbage collection
        del file_params
        gc.collect()
    
    # Freeze parameters for Flax
    return freeze(params), config 

# -----------------------------------------
# Generation Functions
# -----------------------------------------

def generate_text(
    model, 
    params, 
    tokenizer, 
    prompt, 
    max_tokens=100, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=50,
    mesh=None
):
    """
    Generate text using tensor-parallel Qwen2.5 model.
    
    Args:
        model: Qwen25ForCausalLM model
        params: Model parameters
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to generate from
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, lower = less random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling parameter
        mesh: Device mesh for tensor parallelism
        
    Returns:
        Generated text including the prompt
    """
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="np").input_ids
    batch_size, seq_length = input_ids.shape
    
    # Create attention mask - 4D format for Qwen model [batch, 1, 1, seq_len]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
    
    # Create position IDs for initial prompt
    position_ids = np.arange(seq_length, dtype=np.int32)[None, :]
    
    # Initialize generation state
    past_key_values = None
    generated_ids = input_ids.copy()
    max_seq_len = 4096  # Maximum sequence length for RoPE embeddings
    
    # Create a PRNG key for sampling
    rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
    
    # JIT-compile the forward pass functions - one for first forward pass, one for subsequent steps
    @partial(jax.jit, static_argnums=(4,))
    def initial_forward(params, ids, attn_mask, pos_ids, return_dict=True):
        return model.apply(
            params,
            input_ids=ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=None,
            return_dict=return_dict,
            mesh=mesh,
        )
    
    @partial(jax.jit, static_argnums=(5,))
    def next_forward(params, ids, attn_mask, pos_ids, past_kv, return_dict=True):
        return model.apply(
            params,
            input_ids=ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=past_kv,
            return_dict=return_dict,
            mesh=mesh,
        )

    # Generation loop
    for i in range(max_tokens):
        # Get the current sequence length for position IDs
        current_length = generated_ids.shape[1]
        
        # Early stopping if we're approaching maximum sequence length
        if current_length >= max_seq_len:
            logger.warning(f"Stopping generation at {current_length} tokens to avoid exceeding max length {max_seq_len}")
            break
            
        # Get input for current step
        if past_key_values is None:
            # First forward pass with full input
            current_input_ids = input_ids
            current_position_ids = position_ids
            current_attention_mask = attention_mask
            
            # Run forward pass
            outputs = initial_forward(
                params, 
                current_input_ids, 
                current_attention_mask,
                current_position_ids
            )
        else:
            # For subsequent tokens, we only need the last token
            current_input_ids = generated_ids[:, -1:]
            
            # Create position ID for the new token
            current_position_ids = jnp.array([[current_length - 1]], dtype=jnp.int32)
            
            # Extend attention mask to include past key values
            past_length = past_key_values[0][0].shape[2]  # [batch, heads, kv_seq, head_dim]
            current_attention_mask = jnp.ones((batch_size, 1, 1, past_length + 1), dtype=jnp.int32)
            
            # Run forward pass with past key values
            outputs = next_forward(
                params, 
                current_input_ids, 
                current_attention_mask,
                current_position_ids,
                past_key_values
            )
        
        # Get logits and past key values
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        past_key_values = outputs["past_key_values"] if isinstance(outputs, dict) else outputs[1]
        
        # Get logits for the last token
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / max(temperature, 1e-5)
        
        # Split the random key
        rng_key, sample_key = jax.random.split(rng_key)
        
        # Apply top-k if specified
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, min(top_k, next_token_logits.shape[-1]))
            # Create distribution from top-k logits
            top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)
            # Sample from the distribution
            next_token_id = jax.random.choice(
                sample_key, top_k_indices[0], p=top_k_probs[0]
            )
        else:
            # Apply softmax to get probabilities
            probs = jax.nn.softmax(next_token_logits, axis=-1)
            # Sample from the distribution
            next_token_id = jax.random.choice(sample_key, jnp.arange(probs.shape[-1]), p=probs[0])
        
        # Add token to generated sequence
        next_token_id = jnp.array([[next_token_id]])
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)
        
        # Check if we hit the EOS token
        if next_token_id[0, 0] == tokenizer.eos_token_id:
            break
            
    # Decode the generated IDs
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# -----------------------------------------
# Main Function
# -----------------------------------------

def main():
    """Run tensor-parallel Qwen2.5 model."""
    parser = argparse.ArgumentParser(description="Tensor-Parallel Qwen2.5 JAX Model")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--mesh_shape", default="1,8", help="Device mesh shape (e.g., '1,8', '2,4')")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16", 
                        help="Data type for model parameters")
    parser.add_argument("--simulate_tp", action="store_true", help="Simulate tensor parallelism on a single device")
    args = parser.parse_args()
    
    # Set data type
    if args.dtype == "float32":
        dtype = jnp.float32
    elif args.dtype == "float16":
        dtype = jnp.float16
    else:
        dtype = jnp.bfloat16
    
    # Get available devices
    devices = jax.devices()
    logger.info(f"Available devices: {len(devices)}")
    
    # Parse mesh shape
    requested_mesh_shape = tuple(map(int, args.mesh_shape.split(",")))
    
    # Adjust mesh shape based on available devices
    if len(devices) < np.prod(requested_mesh_shape):
        logger.warning(f"Not enough devices for mesh shape {requested_mesh_shape}, adjusting to {len(devices)} available devices")
        
        if len(devices) == 1:
            # With a single device, use a 1D mesh (simulate or degenerate case)
            if args.simulate_tp:
                logger.info(f"Simulating tensor parallelism with shape {requested_mesh_shape} on a single device")
                mesh_shape = requested_mesh_shape
            else:
                logger.info("Using single device with 1D mesh")
                mesh_shape = (1,)
        else:
            # With multiple devices but not enough for the requested shape,
            # try to keep the model parallelism dimension (second dimension) 
            # as close as possible to what was requested
            if len(requested_mesh_shape) == 2:
                mp_size = min(requested_mesh_shape[1], len(devices))
                dp_size = len(devices) // mp_size
                mesh_shape = (dp_size, mp_size)
                logger.info(f"Adjusted mesh shape from {requested_mesh_shape} to {mesh_shape}")
            else:
                # If 1D requested, just use the available devices in 1D
                mesh_shape = (len(devices),)
                logger.info(f"Using 1D mesh with {len(devices)} devices")
    else:
        # We have enough devices for the requested mesh shape
        mesh_shape = requested_mesh_shape
        logger.info(f"Using requested mesh shape: {mesh_shape}")
    
    # Create the mesh for tensor parallelism
    try:
        # Single device case
        if len(devices) == 1 and not args.simulate_tp:
            # For single device without simulation, create a simple 1D mesh
            logger.info("Creating single-device mesh")
            mesh = Mesh(np.array(devices), ("model",))
        else:
            # For multi-device or simulation case, use jax.experimental.mesh_utils
            from jax.experimental import mesh_utils
            
            if args.simulate_tp and len(devices) == 1:
                # For simulation on single device, create a virtual mesh
                logger.info(f"Creating simulated mesh with shape {mesh_shape}")
                # This allows simulating tensor parallelism on a single device
                mesh = Mesh(np.array([devices[0]] * np.prod(mesh_shape)).reshape(mesh_shape), 
                           ("data", "model") if len(mesh_shape) == 2 else ("model",))
            else:
                # Standard multi-device mesh
                logger.info(f"Creating device mesh with shape {mesh_shape}")
                device_mesh = mesh_utils.create_device_mesh(mesh_shape)
                axis_names = ("data", "model") if len(mesh_shape) == 2 else ("model",)
                mesh = Mesh(device_mesh, axis_names)
    except Exception as e:
        logger.error(f"Error creating mesh: {e}")
        # Fallback to basic approach
        try:
            if len(devices) == 1:
                mesh = Mesh(np.array(devices), ("model",))
            else:
                devices_array = np.array(devices).reshape(mesh_shape)
                axis_names = ("data", "model") if len(mesh_shape) == 2 else ("model",)
                mesh = Mesh(devices_array, axis_names)
        except Exception as e2:
            logger.error(f"Fallback mesh creation also failed: {e2}")
            logger.warning("Running without tensor parallelism")
            mesh = None
    
    # Load model parameters and config
    logger.info(f"Loading model from {args.model_path}")
    params, config = load_model_params(args.model_path, dtype=dtype)
    
    # Create model
    logger.info("Creating model")
    model = Qwen25ForCausalLM(config, dtype=dtype)
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        # Try loading from model path
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        except Exception as e2:
            logger.error(f"Failed to load tokenizer from model path: {e2}")
            raise RuntimeError("Could not load tokenizer")
    
    # Generate text
    logger.info(f"Generating text with prompt: {args.prompt}")
    start_time = time.time()
    
    # Use mesh context manager if we have a mesh
    if mesh is not None:
        with mesh:
            output = generate_text(
                model=model,
                params=params,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                mesh=mesh
            )
    else:
        # Fall back to no mesh if mesh creation failed
        output = generate_text(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            mesh=None
        )
    
    elapsed = time.time() - start_time
    tokens_generated = len(tokenizer.encode(output)) - len(tokenizer.encode(args.prompt))
    
    print("\nGenerated text:")
    print("=" * 40)
    print(output)
    print("=" * 40)
    print(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.2f} tokens/sec)")

if __name__ == "__main__":
    main() 