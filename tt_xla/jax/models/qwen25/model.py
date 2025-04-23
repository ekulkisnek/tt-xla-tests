"""Qwen-2.5-7B model implementation with tensor parallelism support."""

import time
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from tt_xla.jax.partitioning import with_sharding_constraint
from tt_xla.jax.models.common import load_params_from_checkpoint


class QwenAttention(nn.Module):
    """Multi-head attention for Qwen2.5 model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=config.get("attention_bias", True),
            name="q_proj",
        )
        self.k_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=config.get("attention_bias", True),
            name="k_proj",
        )
        self.v_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=config.get("attention_bias", True),
            name="v_proj",
        )
        self.o_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=config.get("attention_bias", True),
            name="o_proj",
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        mesh: Optional[Mesh] = None,
    ) -> Union[Tuple[jnp.ndarray, ...], jnp.ndarray]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        if mesh is not None:
            # Apply tensor parallelism constraint
            query_states = with_sharding_constraint(query_states, P("batch", "length", "model", None), mesh)
            key_states = with_sharding_constraint(key_states, P("batch", "length", "model", None), mesh)
            value_states = with_sharding_constraint(value_states, P("batch", "length", "model", None), mesh)
        
        # Handle past key values if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=1)
            value_states = jnp.concatenate([past_value, value_states], axis=1)
            
        # Transpose for batched matrix multiplication
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))  # (batch, heads, seq_len, head_dim)
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))
        
        # Compute scaled dot-product attention
        attention_scores = jnp.matmul(query_states, jnp.transpose(key_states, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        
        # Compute attention output
        attn_output = jnp.matmul(attention_probs, value_states)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        if mesh is not None:
            attn_output = with_sharding_constraint(attn_output, P("batch", "length", "model"), mesh)
            
        outputs = (attn_output,)
        
        if output_attentions:
            outputs = outputs + (attention_probs,)
            
        return outputs


class QwenMLP(nn.Module):
    """MLP module for Qwen2.5 model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        
        self.gate_proj = nn.Dense(
            self.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="gate_proj",
        )
        self.up_proj = nn.Dense(
            self.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="up_proj",
        )
        self.down_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="down_proj",
        )
        
    def __call__(self, x: jnp.ndarray, mesh: Optional[Mesh] = None) -> jnp.ndarray:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        if mesh is not None:
            gate = with_sharding_constraint(gate, P("batch", "length", "model"), mesh)
            up = with_sharding_constraint(up, P("batch", "length", "model"), mesh)
        
        # SwiGLU activation
        intermediate = jax.nn.silu(gate) * up
        
        if mesh is not None:
            intermediate = with_sharding_constraint(intermediate, P("batch", "length", "model"), mesh)
            
        output = self.down_proj(intermediate)
        
        if mesh is not None:
            output = with_sharding_constraint(output, P("batch", "length", "model"), mesh)
            
        return output


class QwenDecoderLayer(nn.Module):
    """Transformer decoder layer for Qwen2.5 model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.hidden_size = config["hidden_size"]
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(
            epsilon=config.get("layer_norm_epsilon", 1e-5),
            dtype=self.dtype,
            name="input_layernorm",
        )
        self.post_attention_layernorm = nn.LayerNorm(
            epsilon=config.get("layer_norm_epsilon", 1e-5),
            dtype=self.dtype,
            name="post_attention_layernorm",
        )
        
        # Self-attention and MLP modules
        self.self_attn = QwenAttention(config, dtype=self.dtype)
        self.mlp = QwenMLP(config, dtype=self.dtype)
        
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        mesh: Optional[Mesh] = None,
    ) -> Union[Tuple[jnp.ndarray, ...], jnp.ndarray]:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            mesh=mesh,
        )
        
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, mesh=mesh)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
            
        return outputs


class Qwen25ForCausalLM(nn.Module):
    """Qwen2.5 model for causal language modeling with tensor parallelism support."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        
        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="embed_tokens",
        )
        
        self.layers = [
            QwenDecoderLayer(config, dtype=self.dtype, name=f"layers_{i}")
            for i in range(self.num_layers)
        ]
        
        self.norm = nn.LayerNorm(
            epsilon=config.get("layer_norm_epsilon", 1e-5),
            dtype=self.dtype,
            name="norm",
        )
        
        self.lm_head = nn.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="lm_head",
        )
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        mesh: Optional[Mesh] = None,
    ) -> Dict[str, jnp.ndarray]:
        batch_size, seq_length = input_ids.shape
        
        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        if mesh is not None:
            hidden_states = with_sharding_constraint(hidden_states, P("batch", "length", "model"), mesh)
        
        # Prepare attention masks for causal attention if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        
        # Extend attention mask for attention computation
        extended_attention_mask = jnp.expand_dims(jnp.expand_dims(attention_mask, axis=1), axis=1)
        extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(self.dtype).min
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                mesh=mesh,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        if mesh is not None:
            logits = with_sharding_constraint(logits, P("batch", "length", "vocab"), mesh)
        
        if not return_dict:
            return (logits, all_hidden_states, all_attentions)
        
        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


def create_qwen25_model(
    config: Dict[str, Any],
    dtype: jnp.dtype = jnp.float32,
) -> Qwen25ForCausalLM:
    """Create a Qwen2.5 model instance."""
    return Qwen25ForCausalLM(config=config, dtype=dtype)


def load_qwen25_model(
    weights_dir: str,
    mesh_shape: Tuple[int, ...],
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[Qwen25ForCausalLM, Dict[str, Any]]:
    """
    Load a Qwen2.5 model with tensor parallelism from checkpoint.
    
    Args:
        weights_dir: Directory containing the model weights
        mesh_shape: Shape of the device mesh for tensor parallelism
        dtype: Data type for model parameters
        
    Returns:
        model: Initialized model instance
        config: Model configuration
    """
    start_time = time.time()
    
    # Load model configuration
    import json
    import os
    
    config_path = os.path.join(weights_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create device mesh for tensor parallelism
    devices = jax.devices()
    mesh = Mesh(devices, ("data", "model"))
    
    # Create model instance
    model = create_qwen25_model(config, dtype=dtype)
    
    # Load model parameters
    params = load_params_from_checkpoint(weights_dir)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    return model, config, params, mesh 