# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Tuple, Union
import logging
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from flax.core.scope import FrozenVariableDict

# Create logger with a flag to control verbosity
logger = logging.getLogger(__name__)
# Set to True to enable debug logging of shapes and operations
DEBUG_MODE = False

def debug_log(message):
    """Only log debug messages if DEBUG_MODE is enabled"""
    if DEBUG_MODE:
        logger.debug(message)

# Add a custom embed class to handle parameter name differences
class QwenEmbed(nn.Module):
    """Token embeddings for Qwen."""
    vocab_size: int
    hidden_size: int
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.float32
    embedding_init: Callable[..., jnp.ndarray] = nn.initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        # Initialize the embedding table - use standard HF parameter name "weight" instead of "embedding"
        inputs_embeds = self.param(
            "weight",
            self.embedding_init,
            (self.vocab_size, self.hidden_size),
            self.param_dtype,
        )
        
        # Log init of embedding
        if input_ids.size == 0:
            logging.warning("Empty input_ids passed to QwenEmbed")
            return jnp.zeros((0, self.hidden_size), dtype=self.dtype)
        
        # Use standard take operation
        embeds = jnp.take(inputs_embeds, input_ids, axis=0)
        return embeds.astype(self.dtype)

class RMSNorm(nn.Module):
    """RMSNorm implementation."""
    config: Dict[str, Any]
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.hidden_size = self.config["hidden_size"]
        logger.info(f"Setup RMSNorm with hidden_size={self.hidden_size}")

    def __call__(self, hidden_states, params=None):
        """
        Apply RMS normalization to input hidden states.
        
        Args:
            hidden_states: Input tensor to normalize
            params: Dictionary containing the scale parameter
        
        Returns:
            Normalized tensor
        """
        # Get scale parameter from params or create a dummy one
        scale = None
        if params is not None:
            scale = params.get("scale")
        
        if scale is None:
            logger.warning(f"RMSNorm: Creating dummy scale parameter with shape ({self.hidden_size},)")
            scale = jnp.ones((self.hidden_size,), dtype=self.param_dtype)
        
        # Check scale shape
        if scale.shape != (self.hidden_size,):
            logger.error(f"RMSNorm: scale has wrong shape: {scale.shape}, expected: ({self.hidden_size},)")
            scale = jnp.ones((self.hidden_size,), dtype=self.param_dtype)
            
        # Use variance_epsilon for numerical stability
        input_dtype = hidden_states.dtype
        hidden_shape = hidden_states.shape
        
        # Check last dimension is hidden_size
        if hidden_shape[-1] != self.hidden_size:
            logger.error(f"RMSNorm: hidden_states has wrong shape: {hidden_shape}, expected last dimension to be {self.hidden_size}")
            
        # Calculate variance
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.epsilon)
        
        # Cast scale to match input dtype
        scale = scale.astype(input_dtype)
        
        return scale * hidden_states

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
    """Precompute the frequency tensor for complex exponentials with given dimension."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end)  # type: ignore
    freqs = jnp.outer(t, freqs)  # [end, dim // 2]
    
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    
    return jnp.concatenate([cos, sin], axis=-1)

def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    position_ids: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary embeddings to the query and key tensors."""
    # xq, xk: [batch, seq_len, n_heads, head_dim]
    # freqs_cis: [seq_len, dim//2]
    # position_ids: [batch_size, seq_len]
    
    batch_size, seq_len = position_ids.shape
    head_dim = xq.shape[-1]
    
    # Get the position embeddings for the positions we're interested in
    freqs_cis = jnp.take(freqs_cis, position_ids.reshape(-1), axis=0)
    freqs_cis = freqs_cis.reshape(batch_size, seq_len, head_dim // 2, 2)
    
    # Reshape query and key for interleaved complex multiplication
    xq_r = xq.reshape(batch_size, seq_len, -1, head_dim // 2, 2)
    xk_r = xk.reshape(batch_size, seq_len, -1, head_dim // 2, 2)
    
    # Extract real and imaginary parts
    xq_real, xq_imag = xq_r[..., 0], xq_r[..., 1]
    xk_real, xk_imag = xk_r[..., 0], xk_r[..., 1]
    
    # Extract cos and sin components
    freqs_cos = freqs_cis[..., 0]  # [batch, seq, dim//2]
    freqs_sin = freqs_cis[..., 1]  # [batch, seq, dim//2]
    
    # Reshape for broadcasting
    freqs_cos = freqs_cos[:, :, None, :]  # [batch, seq, 1, dim//2]
    freqs_sin = freqs_sin[:, :, None, :]  # [batch, seq, 1, dim//2]
    
    # Complex multiplication
    # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    out_q_real = xq_real * freqs_cos - xq_imag * freqs_sin
    out_q_imag = xq_real * freqs_sin + xq_imag * freqs_cos
    out_k_real = xk_real * freqs_cos - xk_imag * freqs_sin
    out_k_imag = xk_real * freqs_sin + xk_imag * freqs_cos
    
    # Stack real and imaginary parts
    out_q = jnp.stack([out_q_real, out_q_imag], axis=-1)
    out_k = jnp.stack([out_k_real, out_k_imag], axis=-1)
    
    # Reshape back to original shapes
    out_q = out_q.reshape(batch_size, seq_len, -1, head_dim)
    out_k = out_k.reshape(batch_size, seq_len, -1, head_dim)
    
    return out_q, out_k

class QwenAttention(nn.Module):
    """Multi-head attention for Qwen2.5."""
    config: Dict[str, Any]
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        logger.info(f"Setup QwenAttention with hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        deterministic=True,  # Add deterministic parameter but don't use it
        params=None,
        **kwargs
    ):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Log debug info on shapes
        debug_log(f"QwenAttention: hidden_states shape={hidden_states.shape}")
        if attention_mask is not None:
            debug_log(f"QwenAttention: attention_mask shape={attention_mask.shape}")
            
        # Access attention projection weights directly from flattened params
        q_proj = None
        k_proj = None
        v_proj = None
        o_proj = None
        
        if params is not None:
            q_proj = params.get("q_proj", {}).get("kernel")
            k_proj = params.get("k_proj", {}).get("kernel") 
            v_proj = params.get("v_proj", {}).get("kernel")
            o_proj = params.get("o_proj", {}).get("kernel")
        
        # Create dummy parameters if any are missing or not provided
        if q_proj is None:
            logger.warning(f"QwenAttention: Creating dummy q_proj with shape ({self.hidden_size}, {self.hidden_size})")
            q_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
        
        if k_proj is None:
            logger.warning(f"QwenAttention: Creating dummy k_proj with shape ({self.hidden_size}, {self.hidden_size})")
            k_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
        
        if v_proj is None:
            logger.warning(f"QwenAttention: Creating dummy v_proj with shape ({self.hidden_size}, {self.hidden_size})")
            v_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
        
        if o_proj is None:
            logger.warning(f"QwenAttention: Creating dummy o_proj with shape ({self.hidden_size}, {self.hidden_size})")
            o_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
        
        # Check dimensions to make sure they're correct
        if q_proj.shape[0] != self.hidden_size or q_proj.shape[1] != self.hidden_size:
            logger.error(f"QwenAttention: q_proj has wrong shape: {q_proj.shape}, expected: ({self.hidden_size}, {self.hidden_size})")
            q_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
            
        if k_proj.shape[0] != self.hidden_size or k_proj.shape[1] != self.hidden_size:
            logger.error(f"QwenAttention: k_proj has wrong shape: {k_proj.shape}, expected: ({self.hidden_size}, {self.hidden_size})")
            k_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
            
        if v_proj.shape[0] != self.hidden_size or v_proj.shape[1] != self.hidden_size:
            logger.error(f"QwenAttention: v_proj has wrong shape: {v_proj.shape}, expected: ({self.hidden_size}, {self.hidden_size})")
            v_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
            
        if o_proj.shape[0] != self.hidden_size or o_proj.shape[1] != self.hidden_size:
            logger.error(f"QwenAttention: o_proj has wrong shape: {o_proj.shape}, expected: ({self.hidden_size}, {self.hidden_size})")
            o_proj = jnp.ones((self.hidden_size, self.hidden_size), dtype=self.param_dtype)
        
        # Project hidden states to query, key, value
        # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, hidden_size]
        q = jnp.matmul(hidden_states, q_proj)
        k = jnp.matmul(hidden_states, k_proj)
        v = jnp.matmul(hidden_states, v_proj)
        
        # Reshape query, key, value for multi-head attention
        # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, num_heads, head_dim]
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
                
            # Transpose for attention computation
        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Compute attention scores
        # [batch_size, num_heads, seq_length, head_dim] @ [batch_size, num_heads, head_dim, seq_length]
        # -> [batch_size, num_heads, seq_length, seq_length]
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Extend attention mask for broadcasting
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        
        # Compute context vectors
        # [batch_size, num_heads, seq_length, seq_length] @ [batch_size, num_heads, seq_length, head_dim]
        # -> [batch_size, num_heads, seq_length, head_dim]
        context = jnp.matmul(attention_probs, v)
        
        # Reshape context
        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        context = jnp.transpose(context, (0, 2, 1, 3))
        
        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, seq_length, hidden_size]
        context = context.reshape(batch_size, seq_length, self.hidden_size)
        
        # Project to output
        # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, hidden_size]
        attn_output = jnp.matmul(context, o_proj)
        
        outputs = (attn_output,)
        
        if output_attentions:
            outputs = outputs + (attention_probs,)
            
        if use_cache:
            outputs = outputs + (None,)  # past_key_value
            
        return outputs[0] if len(outputs) == 1 else outputs

class Qwen2_5MLP(nn.Module):
    """MLP for Qwen2.5."""
    config: Dict[str, Any]
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.hidden_size = self.config["hidden_size"]
        self.intermediate_size = self.config["intermediate_size"]
        logger.info(f"Setup Qwen2_5MLP with hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}")

    def __call__(self, hidden_states, params=None):
        """
        Process hidden states through MLP.
        
        Args:
            hidden_states: Input tensor
            params: Dictionary containing gate_proj, up_proj, and down_proj parameters
            
        Returns:
            Processed hidden states
        """
        # Access MLP projection weights directly from flattened params
        if params is not None:
            gate_proj = params["gate_proj"]["kernel"] if "gate_proj" in params else None
            up_proj = params["up_proj"]["kernel"] if "up_proj" in params else None
            down_proj = params["down_proj"]["kernel"] if "down_proj" in params else None
        else:
            logger.warning(f"MLP: Creating dummy parameters for projections")
            gate_proj = None
            up_proj = None
            down_proj = None

        # Create dummy parameters if any are missing or not provided
        if gate_proj is None:
            logger.warning(f"MLP: Creating dummy gate_proj with shape ({self.hidden_size}, {self.intermediate_size})")
            gate_proj = jnp.ones((self.hidden_size, self.intermediate_size), dtype=self.param_dtype)
        
        if up_proj is None:
            logger.warning(f"MLP: Creating dummy up_proj with shape ({self.hidden_size}, {self.intermediate_size})")
            up_proj = jnp.ones((self.hidden_size, self.intermediate_size), dtype=self.param_dtype)
        
        if down_proj is None:
            logger.warning(f"MLP: Creating dummy down_proj with shape ({self.intermediate_size}, {self.hidden_size})")
            down_proj = jnp.ones((self.intermediate_size, self.hidden_size), dtype=self.param_dtype)

        # Check dimensions to make sure they're correct
        if gate_proj.shape[0] != self.hidden_size or gate_proj.shape[1] != self.intermediate_size:
            logger.error(f"MLP: gate_proj has wrong shape: {gate_proj.shape}, expected: ({self.hidden_size}, {self.intermediate_size})")
            gate_proj = jnp.ones((self.hidden_size, self.intermediate_size), dtype=self.param_dtype)
            
        if up_proj.shape[0] != self.hidden_size or up_proj.shape[1] != self.intermediate_size:
            logger.error(f"MLP: up_proj has wrong shape: {up_proj.shape}, expected: ({self.hidden_size}, {self.intermediate_size})")
            up_proj = jnp.ones((self.hidden_size, self.intermediate_size), dtype=self.param_dtype)
            
        if down_proj.shape[0] != self.intermediate_size or down_proj.shape[1] != self.hidden_size:
            logger.error(f"MLP: down_proj has wrong shape: {down_proj.shape}, expected: ({self.intermediate_size}, {self.hidden_size})")
            down_proj = jnp.ones((self.intermediate_size, self.hidden_size), dtype=self.param_dtype)

        # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, intermediate_size]
        gate_proj_output = jnp.matmul(hidden_states, gate_proj)
        up_proj_output = jnp.matmul(hidden_states, up_proj)

        # Apply SiLU activation to gate_proj_output and multiply with up_proj_output
        intermediate_output = jax.nn.silu(gate_proj_output) * up_proj_output

        # [batch_size, seq_length, intermediate_size] -> [batch_size, seq_length, hidden_size]
        down_proj_output = jnp.matmul(intermediate_output, down_proj)

        return down_proj_output

class QwenTransformerBlock(nn.Module):
    """Transformer block for Qwen2.5."""
    config: Dict[str, Any]
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    def setup(self):
        self.hidden_size = self.config["hidden_size"]
        logger.info(f"Setup QwenTransformerBlock with hidden_size={self.hidden_size}")
        
        # Initialize sub-modules
        self.attention = QwenAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        self.mlp = Qwen2_5MLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        self.input_layernorm = RMSNorm(
            config=self.config,
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        self.post_attention_layernorm = RMSNorm(
            config=self.config,
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        params=None,
        **kwargs
    ):
        """
        Process hidden states through transformer block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Cached key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use caching
            params: Dictionary containing parameters for all sub-modules
            
        Returns:
            Processed hidden states and optional outputs
        """
        if params is None:
            logger.warning("QwenTransformerBlock: No params provided, using dummy weights")
            block_params = None
            input_ln_params = None
            post_attn_ln_params = None
            attn_params = None
            mlp_params = None
        else:
            # Extract parameters for each sub-module
            input_ln_key = "input_layernorm"
            post_attn_ln_key = "post_attention_layernorm"
            attn_key = "self_attn"
            mlp_key = "mlp"
            
            # Check if parameters exist and log
            input_ln_params = params.get(input_ln_key, None)
            post_attn_ln_params = params.get(post_attn_ln_key, None)
            attn_params = params.get(attn_key, None)
            mlp_params = params.get(mlp_key, None)
            
            if input_ln_params is None:
                logger.warning(f"QwenTransformerBlock: Missing {input_ln_key} parameters")
            if post_attn_ln_params is None:
                logger.warning(f"QwenTransformerBlock: Missing {post_attn_ln_key} parameters")
            if attn_params is None:
                logger.warning(f"QwenTransformerBlock: Missing {attn_key} parameters")
            if mlp_params is None:
                logger.warning(f"QwenTransformerBlock: Missing {mlp_key} parameters")

        # Self Attention
        residual = hidden_states
        
        # Layer normalization before self-attention
        hidden_states = self.input_layernorm(hidden_states, params=input_ln_params)
        
        # Self-attention
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            params=attn_params,
            **kwargs
        )
        
        # Get attention output
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]
        else:
            attn_output = attn_outputs
            outputs = ()
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        
        # Layer normalization before MLP
        hidden_states = self.post_attention_layernorm(hidden_states, params=post_attn_ln_params)
        
        # MLP
        mlp_output = self.mlp(hidden_states, params=mlp_params)
        
        # Residual connection
        hidden_states = residual + mlp_output
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
            
        return outputs

class Qwen2_5Model(nn.Module):
    """Base model for Qwen2.5."""
    config: Dict[str, Any]
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_hidden_layers = self.config["num_hidden_layers"]
        
        logger.info(f"Setup Qwen2_5Model with vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, num_hidden_layers={self.num_hidden_layers}")
        
        # Initialize embedding
        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        
        # Initialize transformer layers
        self.layers = [
            QwenTransformerBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            for _ in range(self.num_hidden_layers)
        ]
        
        # Initialize final layer norm
        self.norm = RMSNorm(
            config=self.config,
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        params=None,
        **kwargs
    ):
        """
        Process input_ids through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached key/value states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            use_cache: Whether to use caching
            params: Dictionary containing parameters for all components
            
        Returns:
            Hidden states and optional outputs
        """
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)

        # Get embedding parameters from params dict if available
        if params is not None and "embed_tokens" in params:
            logger.debug(f"Qwen2_5Model: Using provided embed_tokens parameters")
            embedding_params = params["embed_tokens"]
            # Use embedding parameters directly
            if "embedding" in embedding_params:
                embedding_table = embedding_params["embedding"]
                hidden_states = jnp.take(embedding_table, input_ids, axis=0)
            else:
                logger.warning(f"Qwen2_5Model: embed_tokens params found but no embedding parameter")
                hidden_states = self.embed_tokens(input_ids)
        else:
            logger.warning(f"Qwen2_5Model: Using default embed_tokens")
            hidden_states = self.embed_tokens(input_ids)
            
        # Prepare outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Extract layer parameters if available
            layer_params = None
            if params is not None and "layers" in params:
                layers_params = params["layers"]
                if isinstance(layers_params, dict) and str(i) in layers_params:
                    layer_params = layers_params[str(i)]
                elif isinstance(layers_params, list) and i < len(layers_params):
                    layer_params = layers_params[i]
            
            # Pass to layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None if past_key_values is None else past_key_values[i],
                output_attentions=output_attentions,
                use_cache=use_cache,
                params=layer_params,
            )
            
            # Extract outputs
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                outputs = layer_outputs[1:]
            else:
                hidden_states = layer_outputs
                outputs = ()
                
            if output_attentions and len(outputs) > 0:
                all_self_attns = all_self_attns + (outputs[0],)
                
            if use_cache and len(outputs) > 0:
                next_cache = next_cache + (outputs[0],)
        
        # Final layer norm
        norm_params = None
        if params is not None and "norm" in params:
            norm_params = params["norm"]
            
        hidden_states = self.norm(hidden_states, params=norm_params)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Prepare output
        outputs = (hidden_states,)
        
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
            
        if output_attentions:
            outputs = outputs + (all_self_attns,)
            
        if use_cache:
            outputs = outputs + (next_cache,)
            
        return outputs

class Qwen2_5ForCausalLM(nn.Module):
    """Qwen2.5 model for causal language modeling."""
    config: Dict[str, Any]
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    def setup(self):
        self.transformer = Qwen2_5Model(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        self.vocab_size = self.config["vocab_size"]
        self.lm_head = nn.Dense(
            features=self.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )
        
        logger.info(f"Setup Qwen2_5ForCausalLM with vocab_size={self.vocab_size}")

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        params=None,
        **kwargs
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached key/value states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            use_cache: Whether to use caching
            params: Dictionary containing parameters for transformer and lm_head
            
        Returns:
            Tuple containing logits and optional hidden states/attentions/cache
        """
        # Extract parameters for transformer and lm_head
        transformer_params = None
        lm_head_params = None
        
        if params is not None:
            if "transformer" in params:
                transformer_params = params["transformer"]
            elif "params" in params and "transformer" in params["params"]:
                transformer_params = params["params"]["transformer"]
                
            if "lm_head" in params:
                lm_head_params = params["lm_head"]
            elif "params" in params and "lm_head" in params["params"]:
                lm_head_params = params["params"]["lm_head"]
        
        # Pass inputs to transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            params=transformer_params,
            **kwargs
        )
        
        # Extract transformer outputs
        if isinstance(transformer_outputs, tuple):
            hidden_states = transformer_outputs[0]
            outputs = transformer_outputs[1:]
        else:
            hidden_states = transformer_outputs
            outputs = ()
        
        # Apply lm_head
        if lm_head_params is not None:
            logits = self.lm_head.apply({"params": lm_head_params}, hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        
        # Create full outputs tuple: (logits, *outputs)
        outputs = (logits,) + outputs
        
        return outputs 

def build_sinusoidal_positions(max_sequence_length, dim, base=10000):
    """
    Build sinusoidal embeddings.
    
    Args:
        max_sequence_length: Maximum sequence length
        dim: Hidden dimension size
        base: Base value for angular frequency calculations
        
    Returns:
        Tuple of (sin_emb, cos_emb)
    """
    # Create position ids from 0 to max_sequence_length-1
    position_ids = jnp.arange(max_sequence_length)
    
    # Calculate frequencies for each position
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
    
    # Compute sin/cos positional embeddings
    sinusoid_inp = jnp.einsum("i,j->ij", position_ids, inv_freq)
    sin_embeddings = jnp.sin(sinusoid_inp)
    cos_embeddings = jnp.cos(sinusoid_inp)
    
    return sin_embeddings, cos_embeddings

def apply_rotary_pos_emb(
    x: jnp.ndarray,
    sin_emb: jnp.ndarray,
    cos_emb: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply rotary position embeddings to input tensors.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
        sin_emb: Sine embeddings
        cos_emb: Cosine embeddings
        
    Returns:
        Tensor with rotary position embeddings applied
    """
    # Ensure embeddings have at least 2 dimensions and slice safely
    seq_len = x.shape[1]
    half_head_dim = x.shape[-1] // 2
    
    # Make sure we don't try to index beyond the embeddings size
    seq_len = min(seq_len, sin_emb.shape[0])
    half_head_dim = min(half_head_dim, sin_emb.shape[1])
    
    # Reshape for easy manipulation
    sin_emb = sin_emb[:seq_len, :half_head_dim]  # [seq_len, dim//2]
    cos_emb = cos_emb[:seq_len, :half_head_dim]  # [seq_len, dim//2]
    
    # Add head dimension
    sin_emb = sin_emb[None, :, None, :]  # [1, seq_len, 1, dim//2]
    cos_emb = cos_emb[None, :, None, :]  # [1, seq_len, 1, dim//2]
    
    # Split input into even and odd dimensions
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
    
    # Apply rotation using the sin/cos values
    rotated = jnp.stack([
        x1 * cos_emb - x2 * sin_emb,  
        x1 * sin_emb + x2 * cos_emb
    ], axis=-1)
    
    # Reshape back to original shape
    rotated = rotated.reshape(*x.shape)
    
    return rotated 