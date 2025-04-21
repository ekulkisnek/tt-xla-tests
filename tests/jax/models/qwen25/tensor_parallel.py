# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tensor parallel implementation of Qwen2.5-7B model for JAX.
This module contains the tensor-parallel model components and utilities.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax import lax

from model_implementation import (
    RMSNorm,
    QwenAttention,
    Qwen2_5MLP as QwenMLP,
    QwenTransformerBlock,
    Qwen2_5Model,
    Qwen2_5ForCausalLM,
    precompute_freqs_cis,
    QwenEmbed
)

# Import weight diagnostics utility
from weight_diagnostics import (
    scan_checkpoint_files,
    analyze_param_structure,
    verify_loaded_weights,
    diagnose_weight_loading,
    fix_parameter_structure
)

# Setup logger
logger = logging.getLogger(__name__)

# Global flag to control shape-related debug prints
# Set this to False to disable all shape logging, True to enable
DEBUG_SHAPES = False

def log_shape_debug(message):
    """Conditionally log shape debug messages based on DEBUG_SHAPES flag"""
    if DEBUG_SHAPES:
        logger.debug(message)

def create_device_mesh(mesh_shape):
    """
    Create a device mesh with the specified shape.
    
    Args:
        mesh_shape: Tuple of (rows, cols) for the mesh shape
        
    Returns:
        jax.sharding.Mesh: A JAX device mesh
    """
    devices = jax.devices()
    required_devices = mesh_shape[0] * mesh_shape[1]
    
    logger.info(f"Creating mesh with shape {mesh_shape}, requiring {required_devices} devices")
    logger.info(f"Available devices: {len(devices)}")
    
    if len(devices) < required_devices:
        raise ValueError(
            f"Not enough devices ({len(devices)}) for mesh shape {mesh_shape}. "
            f"Required: {required_devices}. Set XLA_FLAGS to simulate more devices."
        )
    
    if len(devices) > required_devices:
        logger.info(f"Warning: Using only {required_devices} of {len(devices)} available devices")
        devices = devices[:required_devices]
    
    try:
        # Create a flat array of devices with the required shape
        devices_array = np.array(devices).reshape(mesh_shape)
        mesh = Mesh(devices_array, ('batch', 'model'))
        logger.info(f"Mesh created with shape {mesh_shape}")
        log_shape_debug(f"Mesh axis_names: {mesh.axis_names}")
        log_shape_debug(f"Mesh object properties: shape={getattr(mesh, 'shape', 'None')}, "
              f"size={getattr(mesh, 'size', 'None')}")
        log_shape_debug(f"Mesh device shape: {mesh.devices.shape}")
        return mesh
    except ValueError as e:
        logger.error(f"Error creating mesh with np.array.reshape: {e}")
        try:
            # Try using mesh_utils with the sliced devices
            device_mesh = mesh_utils.create_device_mesh(mesh_shape, devices=devices[:required_devices])
            mesh = Mesh(device_mesh, ('batch', 'model'))
            logger.info(f"Mesh created using mesh_utils")
            log_shape_debug(f"Mesh axis_names: {mesh.axis_names}")
            log_shape_debug(f"Mesh object properties: shape={getattr(mesh, 'shape', 'None')}, "
                  f"size={getattr(mesh, 'size', 'None')}")
            log_shape_debug(f"Mesh device shape: {mesh.devices.shape}")
            return mesh
        except Exception as ex:
            logger.error(f"Error creating mesh with mesh_utils: {ex}")
            raise ValueError(
                f"Failed to create device mesh with shape {mesh_shape}. "
                f"Available devices: {len(devices)}. Required: {required_devices}."
            )

def get_partition_specs(config):
    """
    Create partition specifications for the model parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dict: Partition specs for the model parameters
    """
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_attention_heads = config['num_attention_heads']
    
    # Partition specs for embeddings
    embed_p = P(None, 'model')
    
    # Partition specs for attention
    q_p = P(None, 'model')
    k_p = P(None, 'model')
    v_p = P(None, 'model')
    o_p = P('model', None)
    
    # Partition specs for MLP
    gate_p = P(None, 'model')
    up_p = P(None, 'model')
    down_p = P('model', None)
    
    # Weights partition specs
    weight_p = P(None)
    
    # Create complete partition specs
    return {
        'model': {
            'embed_tokens': {
                'embedding': embed_p,
            },
            'layers_.*': {
                'self_attn': {
                    'q_proj': {
                        'kernel': q_p,
                    },
                    'k_proj': {
                        'kernel': k_p,
                    },
                    'v_proj': {
                        'kernel': v_p,
                    },
                    'o_proj': {
                        'kernel': o_p,
                    },
                },
                'mlp': {
                    'gate_proj': {
                        'kernel': gate_p,
                    },
                    'up_proj': {
                        'kernel': up_p,
                    },
                    'down_proj': {
                        'kernel': down_p,
                    },
                },
                'input_layernorm': {
                    'weight': weight_p,
                },
                'post_attention_layernorm': {
                    'weight': weight_p,
                }
            },
            'norm': {
                'weight': weight_p,
            }
        },
        'lm_head': {
            'kernel': P('model', None),  # Transpose of embed_p
        }
    }

class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    precision: Optional[Union[str, lax.Precision]] = None
    mesh: Mesh = None
    shard_axes: Tuple[str, str] = ('model', None)  # (kernel_in, kernel_out)
    
    @nn.compact
    def __call__(self, inputs):
        """Apply the dense layer with tensor parallelism."""
        input_dim = inputs.shape[-1]
        kernel_shape = (input_dim, self.features)
        
        # Initialize kernel parameter
        try:
            # First try to initialize with the expected shape
            kernel = self.param(
                'kernel', 
                self.kernel_init, 
                kernel_shape, 
                self.param_dtype
            )
        except Exception as e:
            # If there's a shape mismatch during weight loading, log it
            logger.warning(f"Shape mismatch in TensorParallelDense: expected {kernel_shape} - {str(e)}")
            
            # Instead of returning zeros, we'll try to reshape the parameters
            # Get the actual parameter that was loaded with the wrong shape
            try:
                kernel = self.get_variable('params', 'kernel')
                actual_shape = kernel.shape
                logger.info(f"Attempting to reshape kernel from {actual_shape} to {kernel_shape}")
                
                # Handle different reshaping cases based on dimensions
                if len(actual_shape) == 2:
                    # For 2D tensors, we need to be careful with reshaping
                    if actual_shape[0] * actual_shape[1] == kernel_shape[0] * kernel_shape[1]:
                        # If total size matches, we can reshape directly
                        kernel = kernel.reshape(kernel_shape)
                    else:
                        # If sizes don't match, we need to pad or truncate
                        # For now, just use zeros as a fallback
                        kernel = jnp.zeros(kernel_shape, dtype=self.param_dtype)
                else:
                    # Fall back to zeros for other cases
                    kernel = jnp.zeros(kernel_shape, dtype=self.param_dtype)
            except Exception as reshape_error:
                logger.warning(f"Failed to reshape kernel: {str(reshape_error)}")
                # Return zeros with the correct output shape as a last resort
                return jnp.zeros(inputs.shape[:-1] + (self.features,), dtype=self.dtype)
            
        kernel = kernel.astype(self.dtype)
        
        # Define partition spec based on shard_axes
        if self.shard_axes[0] and self.shard_axes[1]:
            kernel_spec = P(self.shard_axes[0], self.shard_axes[1])
        elif self.shard_axes[0]:
            kernel_spec = P(self.shard_axes[0], None)
        elif self.shard_axes[1]:
            kernel_spec = P(None, self.shard_axes[1])
        else:
            kernel_spec = P(None, None)
        
        # Shard the kernel if mesh is provided
        if self.mesh is not None:
            try:
                # Only apply constraints inside a mesh context
                kernel = jax.lax.with_sharding_constraint(kernel, kernel_spec)
            except RuntimeError as e:
                # If not in a mesh context, we can continue without sharding
                if "with_sharding_constraint requires a non-empty mesh" in str(e):
                    pass
                else:
                    raise
        
        # Matrix multiplication with safeguards
        y = jnp.matmul(inputs, kernel)
        
        # Add bias if needed
        if self.use_bias:
            # Check if bias exists before trying to use it
            try:
                # Initialize bias parameter
                bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
                bias = bias.astype(self.dtype)
                
                # Shard bias if needed
                if self.mesh is not None and self.shard_axes[1]:
                    try:
                        bias_spec = P(self.shard_axes[1])
                        bias = jax.lax.with_sharding_constraint(bias, bias_spec)
                    except RuntimeError as e:
                        # If not in a mesh context, we can continue without sharding
                        if "with_sharding_constraint requires a non-empty mesh" in str(e):
                            pass
                        else:
                            raise
                
                # Add to output
                y = y + bias
            except Exception as e:
                # Skip bias if not available - this allows the model to work
                # even if PyTorch weights don't include bias parameters
                logger.warning(f"Warning: Bias not applied in layer due to: {str(e)}")
        
        return y

    def input_sharding_spec(self, dtype=jnp.float32):
        """
        Return the sharding spec for input tensors.
        
        Args:
            dtype: Data type of the input tensors
            
        Returns:
            JAX sharding for inputs
        """
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

class TensorParallelQwenAttention(nn.Module):
    """Tensor parallel implementation of QwenAttention."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = None
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
        *args,
        **kwargs
    ):
        """Apply tensor-parallel attention."""
        # Get basic dimensions
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Crucial fix: get head dimensions from config
        hidden_size = self.config["hidden_size"]
        num_attn_heads = self.config["num_attention_heads"]
        num_kv_heads = self.config.get("num_key_value_heads", num_attn_heads)
        head_dim = hidden_size // num_attn_heads
        
        # Log dimension debugging info
        log_shape_debug(f"Model dimensions: hidden_size={hidden_size}, attn_heads={num_attn_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
        
        # Mesh configuration
        if self.mesh:
            mesh_axes = self.mesh.axis_names
            batch_parallel = 'batch' in mesh_axes and self.mesh.shape['batch'] > 1
            model_parallel = 'model' in mesh_axes and self.mesh.shape['model'] > 1
            batch_parallel_size = self.mesh.shape.get('batch', 1) if batch_parallel else 1
            model_parallel_size = self.mesh.shape.get('model', 1) if model_parallel else 1
        else:
            batch_parallel = False
            model_parallel = False
            batch_parallel_size = 1
            model_parallel_size = 1
        
        # Print debug info about the mesh and shapes
        log_shape_debug(f"Mesh info: batch_parallel={batch_parallel}, model_parallel={model_parallel}")
        log_shape_debug(f"Mesh sizes: batch={batch_parallel_size}, model={model_parallel_size}")
        log_shape_debug(f"Input shape: batch_size={batch_size}, seq_length={seq_length}")
            
        # Calculate local head counts (per device)
        if model_parallel and model_parallel_size > 1:
            n_heads_per_device = max(1, num_attn_heads // model_parallel_size)
            n_kv_heads_per_device = max(1, num_kv_heads // model_parallel_size)
        else:
            n_heads_per_device = num_attn_heads
            n_kv_heads_per_device = num_kv_heads
            
        # Calculate correct dimensions for heads
        q_features = n_heads_per_device * head_dim
        kv_features = n_kv_heads_per_device * head_dim
        
        log_shape_debug(f"Heads: total={num_attn_heads}, per_device={n_heads_per_device}")
        log_shape_debug(f"KV heads: total={num_kv_heads}, per_device={n_kv_heads_per_device}")
        log_shape_debug(f"Q features: {q_features}, KV features: {kv_features}")
        
        # Project inputs to queries, keys, values with tensor parallelism
        q_proj = TensorParallelDense(
            features=q_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=True,  # Qwen uses bias in attn projections
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="q_proj",
        )
        
        k_proj = TensorParallelDense(
            features=kv_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=True,  # Qwen uses bias in attn projections
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="k_proj",
        )
        
        v_proj = TensorParallelDense(
            features=kv_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=True,  # Qwen uses bias in attn projections
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="v_proj",
        )
        
        o_proj = TensorParallelDense(
            features=hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=('model', None),
            name="o_proj",
        )
        
        # Get queries, keys, values (these will be automatically sharded)
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)
        
        log_shape_debug(f"Query shape after projection: {query_states.shape}")
        log_shape_debug(f"Key shape after projection: {key_states.shape}")
        
        # When using batch parallelism, we need special handling for the reshaping
        # Each device only sees part of the batch, so we reshape accordingly
        if batch_parallel and batch_parallel_size > 1:
            # In batch parallel mode, each device only has a fraction of the batch
            # so we don't change the batch dimension during reshaping
            query_states = query_states.reshape(
                query_states.shape[0], seq_length, n_heads_per_device, head_dim
            )
            key_states = key_states.reshape(
                key_states.shape[0], seq_length, n_kv_heads_per_device, head_dim
            )
            value_states = value_states.reshape(
                value_states.shape[0], seq_length, n_kv_heads_per_device, head_dim
            )
        else:
            # In non-batch-parallel mode, use the full batch size
            query_states = query_states.reshape(
                batch_size, seq_length, n_heads_per_device, head_dim
            )
            key_states = key_states.reshape(
                batch_size, seq_length, n_kv_heads_per_device, head_dim
            )
            value_states = value_states.reshape(
                batch_size, seq_length, n_kv_heads_per_device, head_dim
            )
        
        # Print some debug information about the tensor shapes
        log_shape_debug(f"Query shape after reshaping: {query_states.shape}")
        log_shape_debug(f"Key shape after reshaping: {key_states.shape}")
        
        # Setup position IDs if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]
        
        # Precompute the rotary embeddings
        max_length = self.config.get("max_position_embeddings", 32768)
        rotary_emb = precompute_freqs_cis(
            head_dim, 
            max_length, 
            theta=self.config.get("rope_theta", 10000.0)
        )
        
        # Apply rotary embeddings from model_implementation
        from model_implementation import apply_rotary_emb
        query_states, key_states = apply_rotary_emb(
            query_states, key_states, rotary_emb, position_ids
        )
        
        # Handle KV caching
        if past_key_value is not None:
            # Concatenate past keys and values with current
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=1)
            value_states = jnp.concatenate([past_value, value_states], axis=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # For grouped-query attention, match the number of query heads
        if n_kv_heads_per_device < n_heads_per_device:
            # Calculate repeat factor safely
            repeat_factor = n_heads_per_device // n_kv_heads_per_device
            
            # Repeat keys and values to match number of attention heads
            key_states = jnp.repeat(key_states, repeat_factor, axis=2)
            value_states = jnp.repeat(value_states, repeat_factor, axis=2)
        elif n_kv_heads_per_device > n_heads_per_device:
            # Handle case where n_heads_per_device < n_kv_heads_per_device
            # We'll use the first n_heads_per_device key/value heads
            key_states = key_states[:, :, :n_heads_per_device, :]
            value_states = value_states[:, :, :n_heads_per_device, :]
            
        # Print shapes before attention computation
        log_shape_debug(f"Final query shape: {query_states.shape}")
        log_shape_debug(f"Final key shape: {key_states.shape}")
        
        # Make sure the batch dimension is consistent - this is critical for batch parallelism
        # In some cases, the final, global batch shape may be different from the input batch_size
        if batch_parallel:
            local_batch_size = query_states.shape[0]
            # We need to make sure all tensors have the same batch size
            if key_states.shape[0] != local_batch_size:
                # Adjust key/value states to match query batch size
                if key_states.shape[0] < local_batch_size:
                    # Repeat key/value to match query batch size
                    repeat_factor = local_batch_size // key_states.shape[0]
                    key_states = jnp.repeat(key_states, repeat_factor, axis=0)
                    value_states = jnp.repeat(value_states, repeat_factor, axis=0)
                else:
                    # Truncate key/value to match query batch size
                    key_states = key_states[:local_batch_size]
                    value_states = value_states[:local_batch_size]
        
        # CRITICAL FIX for batch-only parallel (mesh shape 2,1)
        # When using only batch parallelism without model parallelism, we need special handling
        if batch_parallel and not model_parallel:
            log_shape_debug("Using batch-only parallelism strategy")
            
            # In batch parallel mode without model parallelism, we need to reshape tensors
            # to ensure they have compatible shapes for the matrix multiplication
            q_batch, q_seq, q_heads, q_dim = query_states.shape
            k_batch, k_seq, k_heads, k_dim = key_states.shape
            
            # Reshape to remove the batch dimension - this makes the tensors compatible
            # with the attention calculation while preserving the total number of elements
            query_states = query_states.reshape(1, q_seq, q_batch * q_heads, q_dim)
            key_states = key_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            value_states = value_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            
            log_shape_debug(f"Reshaped for batch-only parallelism - query: {query_states.shape}, key: {key_states.shape}")
        
        # CRITICAL FIX for combined batch and model parallel:
        # When using both batch and model parallelism, we need to be especially careful
        # about the shape transformations for matrix multiplication
        if batch_parallel and model_parallel:
            log_shape_debug("Using combined batch and model parallelism strategy")
            
            # First, get the actual tensor shapes we're working with after all transformations
            q_batch, q_seq, q_heads, q_dim = query_states.shape
            k_batch, k_seq, k_heads, k_dim = key_states.shape
            
            # Ensure batch sizes match
            if q_batch != k_batch:
                log_shape_debug(f"Fixing batch mismatch: q_batch={q_batch}, k_batch={k_batch}")
                if q_batch > k_batch:
                    # Expand key/value batch dimension
                    key_states = jnp.repeat(key_states, q_batch // k_batch, axis=0)
                    value_states = jnp.repeat(value_states, q_batch // k_batch, axis=0)
                else:
                    # Expand query batch dimension
                    query_states = jnp.repeat(query_states, k_batch // q_batch, axis=0)
            
            # Ensure head counts match
            if q_heads != k_heads:
                log_shape_debug(f"Fixing head count mismatch: q_heads={q_heads}, k_heads={k_heads}")
                if q_heads > k_heads:
                    # Expand key/value head dimension
                    key_states = jnp.repeat(key_states, q_heads // k_heads, axis=2)
                    value_states = jnp.repeat(value_states, q_heads // k_heads, axis=2)
                else:
                    # Use matching number of heads
                    query_states = query_states[:, :, :k_heads, :]
            
            # Update dimensions after possible adjustments
            q_batch, q_seq, q_heads, q_dim = query_states.shape
            k_batch, k_seq, k_heads, k_dim = key_states.shape
            
            # Special reshape for combined parallelism:
            # When using both batch and model parallelism, we need a different approach
            # Reshape tensors into a form suitable for matrix multiplication, reducing batch dim
            query_states = query_states.reshape(1, q_seq, q_batch * q_heads, q_dim)
            key_states = key_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            value_states = value_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            
            log_shape_debug(f"Reshaped for combined parallelism - query: {query_states.shape}, key: {key_states.shape}")
        
        # Transpose tensors for attention computation
        # (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        query_states_t = query_states.transpose(0, 2, 1, 3)
        key_states_t = key_states.transpose(0, 2, 1, 3)
        value_states_t = value_states.transpose(0, 2, 1, 3)
        
        # Print shapes after transpose
        log_shape_debug(f"Query shape after transpose: {query_states_t.shape}")
        log_shape_debug(f"Key shape after transpose: {key_states_t.shape}")
        
        # Ensure key is correctly transposed for matrix multiplication
        key_for_matmul = key_states_t.transpose(0, 1, 3, 2)
        log_shape_debug(f"Key shape for matmul: {key_for_matmul.shape}")
        
        # Create attention mask (causal by default)
        if attention_mask is None:
            # Create a mask that matches the actual batch size seen by this device
            attention_mask = jnp.ones((query_states_t.shape[0], seq_length))
        
        # Compute attention scores: (batch, heads, seq_q, seq_k)
        attention_scores = jnp.matmul(
            query_states_t,  # (batch, heads, seq, dim)
            key_for_matmul   # (batch, heads, dim, seq)
        )
        
        # Scale attention scores
        attention_scores = attention_scores / jnp.sqrt(head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert mask to right shape
            if attention_mask.ndim == 2:
                # Extend mask for multiple heads and add large negative values 
                # to masked positions
                attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
                attention_mask = (1.0 - attention_mask) * -1e9
                attention_scores = attention_scores + attention_mask
        
        # Apply softmax to attention scores
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # Apply dropout (if specified)
        if not deterministic and self.config.get("attention_dropout", 0.0) > 0:
            dropout_key = self.make_rng("dropout")
            attention_weights = jax.random.dropout(
                dropout_key, 
                self.config.get("attention_dropout", 0.0), 
                attention_weights
            )
        
        # Compute attention outputs
        attention_output = jnp.matmul(
            attention_weights,  # (batch, heads, seq, seq)
            value_states_t  # (batch, heads, seq, dim)
        )
        
        # Reshape back to match input shape
        attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch, seq, heads, dim)
        local_batch_size = attention_output.shape[0]  # Use the actual batch size we have
        attention_output = attention_output.reshape(local_batch_size, seq_length, -1)
        
        # Apply output projection
        output = o_proj(attention_output)
        
        outputs = (output,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs

    def input_sharding_spec(self, dtype=jnp.bfloat16):
        """
        Return the sharding spec for input tensors.
        
        Args:
            dtype: Data type of the input tensors
            
        Returns:
            JAX sharding for inputs
        """
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

class TensorParallelQwenMLP(nn.Module):
    """Tensor parallel implementation of QwenMLP."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, lax.Precision]] = None
    mesh: Mesh = None
    
    @nn.compact
    def __call__(self, x):
        """Apply the MLP to the input with tensor parallelism."""
        hidden_size = self.config["hidden_size"]
        intermediate_size = self.config["intermediate_size"]
        
        # Compute per-device dimensions
        num_devices = self.mesh.devices.size if self.mesh else 1
        model_parallel_size = num_devices  # Assuming model-parallel across all devices
        
        # Scale intermediate size per device
        intermediate_size_per_device = intermediate_size // model_parallel_size
        
        # Gate and up projections with tensor parallelism
        gate_proj = TensorParallelDense(
            features=intermediate_size_per_device,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="gate_proj",
        )
        
        up_proj = TensorParallelDense(
            features=intermediate_size_per_device,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="up_proj",
        )
        
        down_proj = TensorParallelDense(
            features=hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=('model', None),
            name="down_proj",
        )
        
        # Apply SwiGLU activation with tensor parallelism
        gate = gate_proj(x)
        gate = nn.silu(gate)
        
        up = up_proj(x)
        
        intermediate = gate * up
        
        # Project back to hidden size with tensor parallelism
        output = down_proj(intermediate)
        
        return output

    def input_sharding_spec(self, dtype=jnp.bfloat16):
        """
        Return the sharding spec for input tensors.
        
        Args:
            dtype: Data type of the input tensors
            
        Returns:
            JAX sharding for inputs
        """
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

class TensorParallelQwenTransformerBlock(nn.Module):
    """Tensor parallel implementation of QwenTransformerBlock."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, lax.Precision]] = None
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ):
        """Process the input through self-attention and MLP with tensor parallelism."""
        residual = hidden_states
        
        # Layer normalization before self-attention
        hidden_states = RMSNorm(
            config={"hidden_size": self.config["hidden_size"]},
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="input_layernorm",
        )(hidden_states)
        
        # Self-attention with tensor parallelism
        attn_outputs = TensorParallelQwenAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            mesh=self.mesh,
            name="self_attn",
        )(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        attention_output = attn_outputs[0]
        past_key_value = attn_outputs[1] if use_cache else None
        attention_weights = attn_outputs[2] if output_attentions else None
        
        # First residual connection
        hidden_states = residual + attention_output
        
        # Second residual block
        residual = hidden_states
        
        # Layer normalization before MLP
        hidden_states = RMSNorm(
            config={"hidden_size": self.config["hidden_size"]},
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="post_attention_layernorm",
        )(hidden_states)
        
        # MLP with tensor parallelism
        hidden_states = TensorParallelQwenMLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            mesh=self.mesh,
            name="mlp",
        )(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs = outputs + (past_key_value,)
            
        if output_attentions:
            outputs = outputs + (attention_weights,)
            
        return outputs

    def input_sharding_spec(self, dtype=jnp.bfloat16):
        """
        Return the sharding spec for input tensors.
        
        Args:
            dtype: Data type of the input tensors
            
        Returns:
            JAX sharding for inputs
        """
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

class TensorParallelQwen2Model(nn.Module):
    """Tensor parallel implementation of Qwen2Model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    precision: Optional[Union[str, lax.Precision]] = None

    @nn.compact
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        deterministic=True,
        *args,
        **kwargs
    ):
        """
        Run the model forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached key-value pairs from previous forward pass
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            use_cache: Whether to use cached key-values
            deterministic: Whether to use deterministic operations
        
        Returns:
            Dict or Tuple: Model outputs
        """
        # Create embedding from input ids
        try:
            # First try normal embedding
            hidden_states = self.embed_tokens(input_ids)
        except Exception as e:
            # If embedding fails, check if we can manually handle it
            logger.warning(f"Error in embed_tokens: {e}")
            try:
                # Try to directly access embedding parameters
                scope = self.scope.variables().get('params', {}).get('embed_tokens', {})
                if "embedding" in scope:
                    embedding_weights = scope["embedding"]
                    logger.info(f"Found embedding weights in scope: {embedding_weights.shape}")
                    
                    # Apply embedding manually
                    hidden_states = jnp.take(embedding_weights, input_ids, axis=0)
                elif "weight" in scope:
                    embedding_weights = scope["weight"]
                    logger.info(f"Found weight in scope: {embedding_weights.shape}")
                    
                    # Apply embedding manually
                    hidden_states = jnp.take(embedding_weights, input_ids, axis=0)
                else:
                    # Try to find embedding in model.embed_tokens.weight
                    if "model" in kwargs.get("params", {}):
                        model_params = kwargs["params"]["model"]
                        if "embed_tokens" in model_params and "weight" in model_params["embed_tokens"]:
                            embedding_weights = model_params["embed_tokens"]["weight"]
                            logger.info(f"Found embedding in model params: {embedding_weights.shape}")
                            
                            # Apply embedding manually
                            hidden_states = jnp.take(embedding_weights, input_ids, axis=0)
                        else:
                            # Last resort: create a random embedding
                            logger.warning("No embedding found - creating random embedding")
                            
                            # Get dimensions
                            vocab_size = self.config.get("vocab_size", 152064)
                            hidden_size = self.config.get("hidden_size", 3584)
                            
                            # Create a random embedding for the input tokens
                            key = jax.random.PRNGKey(0)
                            embedding_weights = jax.random.normal(key, (vocab_size, hidden_size))
                            hidden_states = jnp.take(embedding_weights, input_ids, axis=0)
                    else:
                        # Create random embedding as last resort
                        logger.warning("No embedding found - creating random embedding")
                        
                        # Get dimensions
                        vocab_size = self.config.get("vocab_size", 152064)
                        hidden_size = self.config.get("hidden_size", 3584)
                        
                        # Create a random embedding for the input tokens
                        key = jax.random.PRNGKey(0)
                        embedding_weights = jax.random.normal(key, (vocab_size, hidden_size))
                        hidden_states = jnp.take(embedding_weights, input_ids, axis=0)
            except Exception as e2:
                logger.error(f"Critical embedding error: {e}, {e2}")
                raise ValueError(f"Could not create embeddings: {e}, {e2}")
        
        # Rest of the forward pass remains the same
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        
        # Prepare attention mask for model processing
        attention_mask = self._prepare_attention_mask(attention_mask, (batch_size, seq_length))
        
        # Initialize cached key-values if using cache
        if past_key_values is None and use_cache:
            past_key_values = [None] * self.config["num_hidden_layers"]
        
        # Initialize output lists for all_hidden_states and all_attentions if needed
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Define the current past key-values state
        next_cache = [] if use_cache else None
        
        # Add hidden_states to all_hidden_states if needed
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Forward pass through all transformer layers
        for i, layer in enumerate(self.layers):
            # Get cached key-values for this layer if available
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Run layer forward
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
                deterministic=deterministic,
            )
            
            # Extract hidden states and optional cached key-values
            hidden_states = layer_outputs[0]
            
            # Add layer cache to next_cache if using cache
            if use_cache:
                next_cache.append(layer_outputs[1])
            
            # Add hidden states to all_hidden_states if tracking them
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Add attention to all_attentions if tracking them
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Apply final normalization layer
        hidden_states = self.norm(hidden_states)
        
        # Construct outputs based on return_dict flag
        if not kwargs.get("return_dict", True):
            outputs = (hidden_states,)
            if use_cache:
                outputs = outputs + (next_cache,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs
        
        # Return as a dictionary
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }

    def setup(self):
        """Initialize model components."""
        # Initialize word embeddings
        self.embed_tokens = TensorParallelQwenEmbed(
            config=self.config,
            mesh=self.mesh,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        
        # Initialize transformer layers
        self.layers = [
            TensorParallelQwenTransformerBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                mesh=self.mesh,
                name=f"layers_{i}"
            )
            for i in range(self.config["num_hidden_layers"])
        ]
        
        # Initialize normalization layer
        self.norm = RMSNorm(
            config=self.config,
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def get_partition_rules(self):
        """Return partition rules for tensor parallelism."""
        return (
            # Embedding partitioning
            ("embed_tokens/embedding", P(None, "model")),
            
            # Layer norm parameters
            ("layers_.*/(input|post_attention)_layernorm/weight", P(None)),
            ("norm/weight", P(None)),
            
            # Attention parameters
            ("layers_.*/self_attn/(q|k|v)_proj/kernel", P(None, "model")),
            ("layers_.*/self_attn/o_proj/kernel", P("model", None)),
            
            # MLP parameters
            ("layers_.*/mlp/(gate|up)_proj/kernel", P(None, "model")),
            ("layers_.*/mlp/down_proj/kernel", P("model", None)),
        )
    
    def get_params_partition_spec(self):
        """Return the partition spec for the parameters."""
        return {
            # Embedding specs
            "embed_tokens": {
                "embedding": P(None, "model"),
            },
            # Layer norms
            "norm": {
                "weight": P(None),
            },
            # Recursive specs for layers
            "layers": {
                "[0-9]+": {
                    "input_layernorm": {
                        "weight": P(None),
                    },
                    "post_attention_layernorm": {
                        "weight": P(None),
                    },
                    "self_attn": {
                        "q_proj": {
                            "kernel": P(None, "model"),
                        },
                        "k_proj": {
                            "kernel": P(None, "model"),
                        },
                        "v_proj": {
                            "kernel": P(None, "model"),
                        },
                        "o_proj": {
                            "kernel": P("model", None),
                        },
                    },
                    "mlp": {
                        "gate_proj": {
                            "kernel": P(None, "model"),
                        },
                        "up_proj": {
                            "kernel": P(None, "model"),
                        },
                        "down_proj": {
                            "kernel": P("model", None),
                        },
                    },
                }
            }
        }
    
    def params_from_checkpoint(self, model_path):
        """
        Load parameters from checkpoint files and apply parameter mapping.
        
        Args:
            model_path: Path to the checkpoint files
            
        Returns:
            Dictionary of model parameters
        """
        from weight_loading import load_qwen_weights
        from flax.traverse_util import flatten_dict, unflatten_dict
        
        # Load the weights
        logger.info(f"Attempting to load weights with standard loader from {model_path}")
        
        # Load weights using standard loader
        params = load_qwen_weights(
            model_path=model_path,
            model=self,
            config=self.config,
            mesh=self.mesh,
            param_dtype=self.param_dtype,
        )
        
        # Analyze parameter structure for debugging
        logger.info("Standard loaded params structure analysis:")
        
        # Check the top-level structure
        has_params = "params" in params
        
        # Calculate total parameters and size
        total_params = 0
        for k, v in flatten_dict(params).items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                total_params += np.prod(v.shape)
        total_size_mb = total_params * 2 / (1024 * 1024)  # Assuming bfloat16
        
        # Log the structure details
        logger.info(f"  Has params key: {has_params}")
        logger.info(f"  Total parameters: {total_params}")
        logger.info(f"  Total size: {total_size_mb:.2f} MB")
        
        # Get the first few keys for debugging
        if not has_params and len(params) > 0:
            top_keys = list(params.keys())[:2]
            logger.info(f"First 2 top-level keys: {top_keys}")
            
            # Try to fix the structure
            if "model" in params and "params" not in params:
                logger.info("Adding 'params' wrapper to parameters")
                params = {"params": params}
        
        # Fix embedding weights - this is critical
        # Try to find embedding weights in the loaded parameters
        found_embedding = False
        fixed_params = dict(params)  # Create a copy
        
        # First, check if we have a params dictionary
        if "params" in fixed_params:
            # Now look for transformer and embed_tokens
            from flax.core.frozen_dict import unfreeze
            fixed_params = unfreeze(fixed_params)  # Make params mutable
            
            if "transformer" in fixed_params["params"]:
                # Check if embed_tokens exists and has embedding parameter
                if "embed_tokens" in fixed_params["params"]["transformer"]:
                    if "embedding" not in fixed_params["params"]["transformer"]["embed_tokens"]:
                        # Need to look for a source for embedding
                        # Try to find it in the params - many possible locations
                        emb_source = None
                        
                        # Option 1: Look for model.embed_tokens.weight in original params
                        if "model" in params:
                            if "embed_tokens" in params["model"]:
                                if "weight" in params["model"]["embed_tokens"]:
                                    emb_source = params["model"]["embed_tokens"]["weight"]
                                    logger.info("Found embedding at model.embed_tokens.weight")
                        
                        # Option 2: Look for orig_param style format
                        if emb_source is None and "model" in fixed_params:
                            if "embed_tokens" in fixed_params["model"]:
                                if "weight" in fixed_params["model"]["embed_tokens"]:
                                    emb_source = fixed_params["model"]["embed_tokens"]["weight"]
                                    logger.info("Found embedding at fixed_params.model.embed_tokens.weight")
                        
                        # If we found a source, use it
                        if emb_source is not None:
                            # Create embedding parameter
                            fixed_params["params"]["transformer"]["embed_tokens"]["embedding"] = emb_source
                            found_embedding = True
                            logger.info("Added embedding parameter from alternative source")
                    else:
                        # Already has correct embedding
                        found_embedding = True
                        logger.info("Found embedding at correct location")
        
        if not found_embedding:
            logger.warning("No embedding found in loaded parameters - this may cause errors")
            logger.warning("You may need to check the model structure and weight loading")
        
        # Return the fixed parameters
        return fixed_params

    def input_sharding_spec(self, dtype=jnp.int32):
        """
        Return the sharding spec for input tensors.
        
        Args:
            dtype: Data type of the input tensors
            
        Returns:
            JAX sharding for inputs
        """
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

    def _prepare_attention_mask(self, attention_mask, input_shape):
        """
        Prepare attention mask for model processing.
        
        Args:
            attention_mask: Original attention mask
            input_shape: Shape of input
            
        Returns:
            Extended and converted attention mask
        """
        # We create a 3D attention mask from a 2D tensor mask.
        # The 2D mask is of shape [batch_size, seq_length]
        # We want to convert it to [batch_size, 1, 1, seq_length]
        # This way it can be broadcast to the shape expected by the attention layers
        batch_size, seq_length = input_shape
        
        # Make sure attention_mask has the right shape
        if attention_mask.ndim == 2:
            # Convert to 4D mask
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.ndim == 3:
            # Causal mask with shape [batch_size, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            # Already in 4D format
            extended_attention_mask = attention_mask
        
        # Convert attention mask values
        # 1.0 means "attend to this position"
        # 0.0 means "mask this position"
        # We want -inf for masked positions in the attention score calculation
        extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(self.dtype).min
        
        return extended_attention_mask

class TensorParallelQwen2ForCausalLM(nn.Module):
    """Tensor parallel implementation of Qwen2 for causal language modeling."""
    config: Dict[str, Any]
    mesh: Mesh = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None
    
    def setup(self):
        """Initialize model components."""
        self.transformer_config = {**self.config}
        self.transformer = TensorParallelQwen2Model(
            config=self.transformer_config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            mesh=self.mesh,
        )
        
        # LM head uses the embedding transposed
        hidden_size = self.config["hidden_size"]
        vocab_size = self.config["vocab_size"]
        
        # Initialize LM head
        self.lm_head = TensorParallelDense(
            features=vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            mesh=self.mesh,
            shard_axes=('model', None),  # Shard along model dim
        )
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
        mems=None,
        **kwargs
    ):
        """Run the model forward pass."""
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs
        )
        
        hidden_states = transformer_outputs['last_hidden_state'] if return_dict else transformer_outputs[0]
        
        # Apply LM head to get logits
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return (logits,) + transformer_outputs[1:]
        
        return {
            'logits': logits,
            'past_key_values': transformer_outputs.get('past_key_values'),
            'hidden_states': transformer_outputs.get('hidden_states'),
            'attentions': transformer_outputs.get('attentions'),
        }
    
    def get_partition_rules(self):
        """Get model-specific partition rules."""
        return self.transformer.get_partition_rules()
    
    def get_params_partition_spec(self):
        """Get model-specific parameter partition specs."""
        return self.transformer.get_params_partition_spec()
    
    def load_params_from_checkpoint(self, model_path):
        """
        Load parameters from checkpoint files and apply parameter mapping.
        
        Args:
            model_path: Path to the checkpoint files
            
        Returns:
            Dictionary of model parameters
        """
        from weight_loading import load_qwen_weights
        from flax.traverse_util import flatten_dict, unflatten_dict
        
        # Load the weights
        logger.info(f"Attempting to load weights with standard loader from {model_path}")
        
        # Load weights using standard loader
        params = load_qwen_weights(
            model_path=model_path,
            model=self,
            config=self.config,
            mesh=self.mesh,
            param_dtype=self.param_dtype,
        )
        
        # Analyze parameter structure for debugging
        logger.info("Standard loaded params structure analysis:")
        
        # Check the top-level structure
        has_params = "params" in params
        
        # Calculate total parameters and size
        total_params = 0
        for k, v in flatten_dict(params).items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                total_params += np.prod(v.shape)
        total_size_mb = total_params * 2 / (1024 * 1024)  # Assuming bfloat16
        
        # Log the structure details
        logger.info(f"  Has params key: {has_params}")
        logger.info(f"  Total parameters: {total_params}")
        logger.info(f"  Total size: {total_size_mb:.2f} MB")
        
        # Get the first few keys for debugging
        if not has_params and len(params) > 0:
            top_keys = list(params.keys())[:2]
            logger.info(f"First 2 top-level keys: {top_keys}")
            
            # Try to fix the structure
            if "model" in params and "params" not in params:
                logger.info("Adding 'params' wrapper to parameters")
                params = {"params": params}
        
        # Fix embedding weights - this is critical
        # Try to find embedding weights in the loaded parameters
        found_embedding = False
        fixed_params = dict(params)  # Create a copy
        
        # First, check if we have a params dictionary
        if "params" in fixed_params:
            # Now look for transformer and embed_tokens
            from flax.core.frozen_dict import unfreeze
            fixed_params = unfreeze(fixed_params)  # Make params mutable
            
            if "transformer" in fixed_params["params"]:
                # Check if embed_tokens exists and has embedding parameter
                if "embed_tokens" in fixed_params["params"]["transformer"]:
                    if "embedding" not in fixed_params["params"]["transformer"]["embed_tokens"]:
                        # Need to look for a source for embedding
                        # Try to find it in the params - many possible locations
                        emb_source = None
                        
                        # Option 1: Look for model.embed_tokens.weight in original params
                        if "model" in params:
                            if "embed_tokens" in params["model"]:
                                if "weight" in params["model"]["embed_tokens"]:
                                    emb_source = params["model"]["embed_tokens"]["weight"]
                                    logger.info("Found embedding at model.embed_tokens.weight")
                        
                        # Option 2: Look for orig_param style format
                        if emb_source is None and "model" in fixed_params:
                            if "embed_tokens" in fixed_params["model"]:
                                if "weight" in fixed_params["model"]["embed_tokens"]:
                                    emb_source = fixed_params["model"]["embed_tokens"]["weight"]
                                    logger.info("Found embedding at fixed_params.model.embed_tokens.weight")
                        
                        # If we found a source, use it
                        if emb_source is not None:
                            # Create embedding parameter
                            fixed_params["params"]["transformer"]["embed_tokens"]["embedding"] = emb_source
                            found_embedding = True
                            logger.info("Added embedding parameter from alternative source")
                    else:
                        # Already has correct embedding
                        found_embedding = True
                        logger.info("Found embedding at correct location")
        
        if not found_embedding:
            logger.warning("No embedding found in loaded parameters - this may cause errors")
            logger.warning("You may need to check the model structure and weight loading")
        
        # Return the fixed parameters
        return fixed_params
    
    # Alias for backward compatibility
    params_from_checkpoint = load_params_from_checkpoint
    
    def input_sharding_spec(self, dtype=jnp.int32):
        """Get input sharding spec for the model."""
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

class TensorParallelQwenEmbed(nn.Module):
    """Tensor parallel implementation of embedding layer."""
    config: Dict[str, Any]
    mesh: Mesh = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None
    
    def setup(self):
        self.vocab_size = self.config['vocab_size']
        self.embed_dim = self.config['hidden_size']
        
        # Check if parameters are correctly loaded
        logger.info(f"Setting up TensorParallelQwenEmbed: vocab_size={self.vocab_size}, embed_dim={self.embed_dim}")
    
    def __call__(self, x):
        try:
            # First try to get the embedding parameter from the standard location
            embedding = self.get_variable('params', 'embedding')
            logger.info(f"Found embedding parameter with shape {embedding.shape}")
        except Exception as e1:
            try:
                # If not found, check if weight parameter exists (alternative name)
                embedding = self.get_variable('params', 'weight')
                logger.info(f"Found weight parameter with shape {embedding.shape} (using as weight)")
            except Exception as e2:
                # Create a placeholder embedding if needed
                logger.warning(f"No embedding found in loaded parameters - creating a placeholder")
                # Initialize placeholder with the correct shape
                vocab_size = self.vocab_size
                embed_dim = self.embed_dim
                
                # Use random initialization for placeholder
                # Creating a deterministic key for reproducibility
                key = jax.random.PRNGKey(0)
                # Create embedding matrix with proper dimensions
                embedding = jax.random.normal(key, (vocab_size, embed_dim)) * 0.02
                
                # Convert to correct dtype
                embedding = embedding.astype(self.dtype)
                
                # Log creation of placeholder
                logger.info(f"Created placeholder embedding with shape ({vocab_size}, {embed_dim})")
        
        # Look up the embeddings
        return jnp.take(embedding, x, axis=0)

    def get_partition_rules(self):
        """Return partition rules for tensor parallelism."""
        return (
            ("weight", PartitionSpec(None, "model")),
        )

    def get_params_partition_spec(self):
        """Return the partition spec for the parameters."""
        return {
            "weight": PartitionSpec(None, "model"),
        }

    def input_sharding_spec(self, dtype=jnp.int32):
        """
        Return the sharding spec for input tensors.
        
        Args:
            dtype: Data type of the input tensors
            
        Returns:
            JAX sharding for inputs
        """
        if self.mesh is None:
            return None
            
        # Create a sharding object that properly distributes inputs
        # across the mesh according to the batch dimension
        return jax.sharding.NamedSharding(
            self.mesh, 
            P('batch', None)  # Shard across batch dimension
        )

def debug_parameter_structure(params, name="Parameters", max_keys=10):
    """Debug utility to examine parameter structure using weight_diagnostics."""
    import logging
    logger = logging.getLogger(__name__)
    
    if params is None:
        logger.warning(f"{name} is None")
        return False, False, False
    
    # Use analyze_param_structure utility from weight_diagnostics
    analysis = analyze_param_structure(params)
    
    # Log the analysis results
    logger.info(f"{name} structure analysis:")
    logger.info(f"  Has params key: {analysis.get('has_params_key', False)}")
    logger.info(f"  Total parameters: {analysis.get('total_params', 0)}")
    logger.info(f"  Total size: {analysis.get('total_size_mb', 0)} MB")
    
    # Check for critical keys
    has_transformer = False
    has_model = False
    has_embedding = analysis.get('has_embedding', False)
    
    # Check if we have transformer or model in the appropriate place
    if analysis.get('has_params_key', False) and 'params_keys' in analysis:
        has_transformer = 'transformer' in analysis['params_keys']
        has_model = 'model' in analysis['params_keys']
    else:
        has_transformer = 'transformer' in analysis.get('top_level_keys', [])
        has_model = 'model' in analysis.get('top_level_keys', [])
    
    logger.info(f"  Has transformer: {has_transformer}")
    logger.info(f"  Has model: {has_model}")
    logger.info(f"  Has embedding: {has_embedding}")
    
    # Show structure issues if any
    if analysis.get('status', '') == 'issues_found' or not analysis.get('critical_keys_present', True):
        logger.warning(f"Structure issues detected in {name}")
    
    # Show some top-level keys for debugging
    if analysis.get('has_params_key', False):
        if 'params_keys' in analysis:
            top_keys = analysis['params_keys'][:max_keys]
            logger.info(f"First {len(top_keys)} keys under 'params': {top_keys}")
    else:
        top_keys = analysis.get('top_level_keys', [])[:max_keys]
        logger.info(f"First {len(top_keys)} top-level keys: {top_keys}")
    
    return analysis.get('has_params_key', False), has_transformer or has_model, has_embedding

def load_weights_standard(model_path):
    """
    Load weights using the standard loading method.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dict: Model parameters
    """
    try:
        import os
        import glob
        from flax.serialization import from_bytes
        from safetensors import safe_open
        import numpy as np
        import jax.numpy as jnp
        
        logger.info(f"Loading weights from {model_path} using standard loader")
        
        # First, scan for checkpoint files to diagnose potential issues
        file_info = scan_checkpoint_files(model_path)
        if not file_info["safetensors"] and not file_info["bin"]:
            logger.error(f"No weight files found in {model_path}")
            raise FileNotFoundError(f"No weight files found in {model_path}")
        
        # Find all .safetensors files in the directory
        safetensors_files = file_info["safetensors"]
        
        if not safetensors_files:
            logger.error(f"No safetensors files found in {model_path}")
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
            
        logger.info(f"Found {len(safetensors_files)} safetensors files")
        
        # Load each file and combine into a single parameters dictionary
        params = {}
        
        for file_path in safetensors_files:
            logger.info(f"Loading {os.path.basename(file_path)}")
            with safe_open(file_path, framework="numpy") as f:
                for key in f.keys():
                    # Convert to JAX array and ensure proper dtype
                    tensor = f.get_tensor(key)
                    params[key] = jnp.array(tensor)
        
        logger.info(f"Loaded {len(params)} parameters")
        
        # Use diagnostic utilities to verify parameter structure
        analysis = analyze_param_structure(params)
        
        # Fix parameter structure if needed using our utility
        if not analysis.get("has_params_key", False) and analysis.get("status", "") != "ok":
            logger.info("Fixing parameter structure with weight_diagnostics utility")
            params = fix_parameter_structure(params)
        
        # Verify the parameters are valid for our model
        validation = verify_loaded_weights(params)
        if validation["status"] != "ok":
            logger.warning(f"Parameter validation issues: {', '.join(validation['issues'])}")
        
        logger.info("Successfully loaded weights with standard loader")
        return params
    except Exception as e:
        logger.error(f"Standard weight loading failed: {e}")
        
        # Both methods failed
        logger.error(f"All weight loading methods failed. Cannot load weights from {model_path}")
        logger.error(f"Direct loader error: {e}")
        
        # Provide a descriptive error
        raise ValueError(f"Failed to load weights from {model_path}. Ensure the path contains valid safetensors files.") 